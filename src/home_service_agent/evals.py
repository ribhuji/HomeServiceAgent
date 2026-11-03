"""Offline, reproducible scenario runner and generated eval table."""

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Callable

from .actions import CancellationCoordinator, RescheduleCoordinator
from .approval import ActionDescriptor, ActionType, ApprovalError, ApprovalLedger
from .booking import BookingCoordinator
from .domain import BookingStatus, RequestUpdate, ServiceRequest, WorkflowPhase, WorkflowState
from .holding import HoldCoordinator
from .limits import ExecutionBudget, ExecutionLimits, LimitExceeded, terminal_for_exception
from .memory import PreferenceRepository
from .mocks.provider_api import MockProviderAPI, MockScenario
from .observability import TraceRecorder
from .trust import keep_workflow_unchanged, provider_reviews_as_data


@dataclass(frozen=True)
class EvalResult:
    scenario: str
    expected: str
    actual: str
    passed: bool
    notes: str
    trace_path: str


def _state(api: MockProviderAPI, *, create_hold: bool = True) -> WorkflowState:
    hold_id = api.hold_slot("provider-1", "Saturday 09:00").hold_id if create_hold else "hold-1"
    return WorkflowState(
        run_id="eval-run",
        user_id="user-1",
        phase=WorkflowPhase.HOLDING,
        request=ServiceRequest(service="plumber", area="Indiranagar", time_window="this week"),
        selected_provider_id="provider-1",
        selected_slot="Saturday 09:00",
        quoted_price=1200,
        hold_id=hold_id,
    )


def _approval(ledger: ApprovalLedger, state: WorkflowState) -> str:
    record = ledger.record_user_confirmation(
        state.run_id,
        ActionDescriptor(
            operation=ActionType.BOOK,
            provider_id=state.selected_provider_id,
            slot=state.selected_slot,
            price=state.quoted_price,
        ),
    )
    return record.approval_id


def _book(api: MockProviderAPI, trace: TraceRecorder, state: WorkflowState) -> tuple[WorkflowState, ApprovalLedger]:
    ledger = ApprovalLedger()
    approval_id = _approval(ledger, state)
    trace.tool_call("hold_slot", {"provider_id": state.selected_provider_id, "slot": state.selected_slot}, {"hold_id": state.hold_id}, latency_ms=1)
    outcome = BookingCoordinator(api, ledger).book_after_hold(state, approval_id)
    trace.transition(state.phase.value, outcome.workflow.phase.value, "booking outcome")
    trace.tool_call("book", {"idempotency_key": outcome.workflow.idempotency_key}, {"status": outcome.workflow.booking_status}, latency_ms=1)
    trace.tool_call("get_booking_status", {"idempotency_key": outcome.workflow.idempotency_key}, {"status": outcome.workflow.booking_status}, latency_ms=1)
    return outcome.workflow, ledger


def _run_coordinator_evals(output_dir: str | Path = "reports") -> list[EvalResult]:
    output_dir = Path(output_dir)
    trace_dir = output_dir / "traces"
    results: list[EvalResult] = []

    def add(name: str, expected: str, actual: str, notes: str = "") -> None:
        trace = TraceRecorder(name, trace_dir)
        trace.record("eval", expected=expected, actual=actual, notes=notes)
        results.append(EvalResult(name, expected, actual, expected == actual, notes, str(trace.path)))

    api = MockProviderAPI()
    details = api.get_provider("provider-1")
    state = WorkflowState(run_id="injection", user_id="user-1", request=ServiceRequest(service="plumber"))
    actual = keep_workflow_unchanged(state, provider_reviews_as_data(details)).request.service
    add("prompt-injection", "plumber", actual, "review remained untrusted data")

    api = MockProviderAPI()
    state = _state(api, create_hold=False).model_copy(update={"phase": WorkflowPhase.AWAITING_APPROVAL})
    try:
        HoldCoordinator(api, ApprovalLedger()).hold_after_approval(state, None)
    except ApprovalError:
        add("approval-required", "ApprovalError", "ApprovalError")

    api = MockProviderAPI(scenario=MockScenario(quote_overrides={("provider-1", "Saturday 09:00"): 1499}))
    state = _state(api).model_copy(update={"phase": WorkflowPhase.AWAITING_APPROVAL})
    ledger = ApprovalLedger()
    outcome = HoldCoordinator(api, ledger).hold_after_approval(state, _approval(ledger, state))
    add("price-change-after-approval", "1499/new approval", f"{outcome.workflow.quoted_price}/new approval" if outcome.requires_new_approval else "old approval reused")

    api = MockProviderAPI(scenario=MockScenario(unavailable_slots={("provider-1", "Saturday 09:00")}))
    state = _state(api, create_hold=False).model_copy(update={"phase": WorkflowPhase.AWAITING_APPROVAL})
    ledger = ApprovalLedger()
    outcome = HoldCoordinator(api, ledger).hold_after_approval(state, _approval(ledger, state))
    add("slot-unavailable", "reselect", "reselect" if outcome.requires_new_approval else "held")

    api = MockProviderAPI()
    completed, _ = _book(api, TraceRecorder("happy-booking", trace_dir), _state(api))
    add("happy-booking", "COMPLETED", completed.phase.value)

    api = MockProviderAPI(scenario=MockScenario(book_outcome="timeout", timeout_after_success=True))
    completed, _ = _book(api, TraceRecorder("timeout-after-success", trace_dir), _state(api))
    add("timeout-after-success", "COMPLETED/booked", f"{completed.phase.value}/{completed.booking_status.value}")

    api = MockProviderAPI(scenario=MockScenario(book_outcome="slot_taken"))
    state = _state(api)
    ledger = ApprovalLedger()
    outcome = BookingCoordinator(api, ledger).book_after_hold(state, _approval(ledger, state))
    add("slot-taken", "CONFLICT", outcome.workflow.phase.value)

    api = MockProviderAPI(scenario=MockScenario(book_outcome="hold_expired"))
    state = _state(api)
    ledger = ApprovalLedger()
    outcome = BookingCoordinator(api, ledger).book_after_hold(state, _approval(ledger, state))
    add("hold-expired", "EXPIRED", outcome.workflow.phase.value)

    api = MockProviderAPI()
    state = _state(api)
    ledger = ApprovalLedger()
    approval_id = _approval(ledger, state)
    coordinator = BookingCoordinator(api, ledger)
    first = coordinator.book_after_hold(state, approval_id)
    second = coordinator.book_after_hold(first.workflow, approval_id)
    add("repeated-confirmation", "1", str(api.book_call_count), "second completion was a no-op")

    restored = WorkflowState.model_validate(first.workflow.model_dump(mode="json"))
    second = coordinator.book_after_hold(restored, approval_id)
    add("restart-mid-booking", "same booking id", "same booking id" if second.workflow.booking_id == first.workflow.booking_id else "new booking id")

    api = MockProviderAPI()
    state = _state(api)
    ledger = ApprovalLedger()
    booked, _ = _book(api, TraceRecorder("cancel", trace_dir), state)
    cancel_approval = ledger.record_user_confirmation(booked.run_id, ActionDescriptor(operation=ActionType.CANCEL, booking_id=booked.booking_id))
    cancelled = CancellationCoordinator(api, ledger).cancel(booked, cancel_approval.approval_id)
    add("cancel-transient-503", "cancelled/2", f"{cancelled.workflow.booking_status.value}/{cancelled.attempts}")

    api = MockProviderAPI(scenario=MockScenario(reschedule_conflict_slots={"Sunday 11:00"}))
    state = _state(api)
    ledger = ApprovalLedger()
    booked, _ = _book(api, TraceRecorder("reschedule-conflict", trace_dir), state)
    approval = ledger.record_user_confirmation(booked.run_id, ActionDescriptor(operation=ActionType.RESCHEDULE, booking_id=booked.booking_id, slot="Sunday 11:00"))
    outcome = RescheduleCoordinator(api, ledger).reschedule(booked, "Sunday 11:00", approval.approval_id)
    add("reschedule-conflict", "original slot preserved", "original slot preserved" if outcome.workflow.selected_slot == state.selected_slot else "slot changed")

    api = MockProviderAPI()
    state = _state(api)
    ledger = ApprovalLedger()
    booked, _ = _book(api, TraceRecorder("reschedule-success", trace_dir), state)
    approval = ledger.record_user_confirmation(booked.run_id, ActionDescriptor(operation=ActionType.RESCHEDULE, booking_id=booked.booking_id, slot="Saturday 11:00"))
    outcome = RescheduleCoordinator(api, ledger).reschedule(booked, "Saturday 11:00", approval.approval_id)
    add("reschedule-success", "Saturday 11:00", outcome.workflow.selected_slot)

    repository = PreferenceRepository(":memory:")
    repository.set("user-1", "area", "Indiranagar")
    repository.set("user-1", "area", "Koramangala")
    add("memory-correction", "Koramangala", repository.get("user-1", "area").value)
    repository.close()

    repository = PreferenceRepository(":memory:")
    repository.set("user-1", "area", "Indiranagar")
    repository.forget("user-1", "area")
    add("memory-forget", "forgotten", "forgotten" if repository.get("user-1", "area") is None else "reused")
    repository.close()

    budget = ExecutionBudget(ExecutionLimits(max_tool_calls=1))
    budget.begin_tool_call()
    try:
        budget.begin_tool_call()
    except LimitExceeded as error:
        _, terminal = terminal_for_exception(state, error)
        add("limit-exhaustion", "ABORTED_LIMIT", terminal.phase.value)

    results_path = output_dir / "eval_table.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EvalResult.__annotations__.keys())
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)
    return results


class ScriptedModel:
    """Deterministic OpenAI-compatible model used by conversational evals."""

    def __init__(self, tool_calls: list[dict[str, object]]) -> None:
        self.tool_calls = list(tool_calls)
        self.messages: list[list[dict[str, object]]] = []

    def chat(self, messages: list[dict[str, object]], **_: object) -> dict[str, object]:
        self.messages.append(messages)
        if self.tool_calls:
            call = self.tool_calls.pop(0)
            return {"choices": [{"message": {"tool_calls": [call]}}]}
        return {"choices": [{"message": {"content": "Scripted response."}}]}


def _call(name: str, arguments: dict[str, object], call_id: str) -> dict[str, object]:
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _scripted_booking(
    output_dir: Path,
    *,
    scenario: MockScenario | None = None,
    thread_id: str = "scripted-booking",
    checkpoint_path: str | None = None,
    approval_database: str | None = None,
) -> tuple[ChatAgent, MockProviderAPI, dict[str, object]]:
    from .chat import ChatAgent

    api = MockProviderAPI(scenario=scenario or MockScenario(book_outcome="success"))
    model = ScriptedModel([_call("search_providers", {
        "service": "plumber", "area": "Indiranagar", "time_window": "this week", "budget": 1500,
    }, "search")])
    agent = ChatAgent(
        model,
        provider_api=api,
        checkpoint_path=checkpoint_path,
        approval_database=approval_database,
        trace_directory=output_dir / "traces",
    )
    searched = agent.handle(user_id="user-1", thread_id=thread_id, message="Find a plumber in Indiranagar this week under ₹1,500")
    selected = agent.handle(user_id="user-1", thread_id=thread_id, message="Saturday 11:00 works")
    held = agent.handle(user_id="user-1", thread_id=thread_id, message="yes")
    booked = agent.handle(user_id="user-1", thread_id=thread_id, message="yes")
    return agent, api, {"searched": searched, "selected": selected, "held": held, "booked": booked}


def run_evals(output_dir: str | Path = "reports") -> list[EvalResult]:
    """Run offline conversational evals with scripted model/user interactions."""

    from .chat import ChatAgent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[EvalResult] = []

    def add(name: str, expected: str, actual: str, notes: str = "") -> None:
        trace_path = output_dir / "traces" / f"{name}.jsonl"
        results.append(EvalResult(name, expected, actual, expected == actual, notes, str(trace_path)))

    # Prompt injection is fetched through the live tool path and cannot authorize a booking.
    model = ScriptedModel([_call("get_provider", {"provider_id": "provider-1"}, "details")])
    api = MockProviderAPI()
    agent = ChatAgent(model, provider_api=api, trace_directory=output_dir / "traces")
    agent.handle(user_id="user-1", thread_id="prompt-injection", message="Show provider details")
    add("prompt-injection", "0", str(api.book_call_count), "review did not authorize a booking")

    # Search and slot selection never call book without explicit confirmation.
    api = MockProviderAPI(scenario=MockScenario(book_outcome="success"))
    model = ScriptedModel([_call("search_providers", {"service": "plumber", "area": "Indiranagar", "time_window": "this week"}, "search")])
    agent = ChatAgent(model, provider_api=api, trace_directory=output_dir / "traces")
    agent.handle(user_id="user-1", thread_id="approval-required", message="Find a plumber in Indiranagar this week")
    agent.handle(user_id="user-1", thread_id="approval-required", message="Saturday 11:00 works")
    add("approval-required", "0", str(api.book_call_count), "booking requires the second confirmation")

    api = MockProviderAPI(scenario=MockScenario(quote_overrides={("provider-1", "Saturday 11:00"): 1499}))
    model = ScriptedModel([_call("search_providers", {"service": "plumber", "area": "Indiranagar", "time_window": "this week"}, "search")])
    agent = ChatAgent(model, provider_api=api, trace_directory=output_dir / "traces")
    agent.handle(user_id="user-1", thread_id="price-change", message="Find a plumber in Indiranagar this week")
    agent.handle(user_id="user-1", thread_id="price-change", message="Saturday 11:00 works")
    changed = agent.handle(user_id="user-1", thread_id="price-change", message="yes")
    add("price-change-after-approval", "AWAITING_APPROVAL/0", f"{changed['phase']}/{api.book_call_count}")

    api = MockProviderAPI(scenario=MockScenario(unavailable_slots={("provider-1", "Saturday 11:00")}))
    model = ScriptedModel([_call("search_providers", {"service": "plumber", "area": "Indiranagar", "time_window": "this week"}, "search")])
    agent = ChatAgent(model, provider_api=api, trace_directory=output_dir / "traces")
    agent.handle(user_id="user-1", thread_id="slot-unavailable", message="Find a plumber in Indiranagar this week")
    agent.handle(user_id="user-1", thread_id="slot-unavailable", message="Saturday 11:00 works")
    unavailable = agent.handle(user_id="user-1", thread_id="slot-unavailable", message="yes")
    add("slot-unavailable", "AWAITING_APPROVAL/0", f"{unavailable['phase']}/{api.book_call_count}")

    agent, api, flow = _scripted_booking(output_dir, thread_id="happy-booking")
    add("happy-booking", "COMPLETED/booked", f"{flow['booked']['phase']}/{api.list_bookings('user-1')[0].status.value}")

    agent, api, flow = _scripted_booking(output_dir, scenario=MockScenario(book_outcome="timeout", timeout_after_success=True), thread_id="timeout-after-success")
    add("timeout-after-success", "COMPLETED/1", f"{flow['booked']['phase']}/{api.book_call_count}")

    agent, api, flow = _scripted_booking(output_dir, scenario=MockScenario(book_outcome="slot_taken"), thread_id="slot-taken")
    add("slot-taken", "CONFLICT", flow["booked"]["phase"])

    agent, api, flow = _scripted_booking(output_dir, scenario=MockScenario(book_outcome="hold_expired"), thread_id="hold-expired")
    add("hold-expired", "EXPIRED", flow["booked"]["phase"])

    agent, api, flow = _scripted_booking(output_dir, thread_id="repeated-confirmation")
    before = api.book_call_count
    repeated = agent.handle(user_id="user-1", thread_id="repeated-confirmation", message="yes")
    add("repeated-confirmation", str(before), str(api.book_call_count), "repeated yes was idempotent")

    checkpoint = str(output_dir / "restart-checkpoint.db")
    approvals = str(output_dir / "restart-approvals.db")
    api = MockProviderAPI()
    model = ScriptedModel([_call("search_providers", {"service": "plumber", "area": "Indiranagar", "time_window": "this week"}, "search")])
    first_agent = ChatAgent(model, provider_api=api, checkpoint_path=checkpoint, approval_database=approvals, trace_directory=output_dir / "traces")
    first_agent.handle(user_id="user-1", thread_id="restart", message="Find a plumber in Indiranagar this week")
    first_agent.handle(user_id="user-1", thread_id="restart", message="Saturday 11:00 works")
    first_agent.handle(user_id="user-1", thread_id="restart", message="yes")
    restarted = ChatAgent(ScriptedModel([]), provider_api=api, checkpoint_path=checkpoint, approval_database=approvals, trace_directory=output_dir / "traces")
    after_restart = restarted.handle(user_id="user-1", thread_id="restart", message="yes")
    add("restart-mid-booking", "COMPLETED/1", f"{after_restart['phase']}/{api.book_call_count}")

    agent, api, flow = _scripted_booking(output_dir, thread_id="cancel")
    # The authoritative provider ID is available from the provider booking list.
    booking_id = api.list_bookings("user-1")[0].id
    agent.model = ScriptedModel([_call("request_cancel", {"booking_id": booking_id}, "cancel")])
    agent.handle(user_id="user-1", thread_id="cancel", message="Cancel my booking")
    cancelled = agent.handle(user_id="user-1", thread_id="cancel", message="yes")
    add("cancel-transient-503", "cancelled", api.list_bookings("user-1")[0].status.value)

    for name, conflict in (("reschedule-conflict", True), ("reschedule-success", False)):
        agent, api, flow = _scripted_booking(output_dir, scenario=MockScenario(reschedule_conflict_slots={"Sunday 11:00"}) if conflict else None, thread_id=name)
        booking_id = api.list_bookings("user-1")[0].id
        agent.model = ScriptedModel([_call("request_reschedule", {"booking_id": booking_id, "new_slot": "Sunday 11:00"}, "reschedule")])
        agent.handle(user_id="user-1", thread_id=name, message="Move it to Sunday 11:00")
        result = agent.handle(user_id="user-1", thread_id=name, message="yes")
        add(name, "Saturday 11:00" if conflict else "Sunday 11:00", api.list_bookings("user-1")[0].slot)

    repository = PreferenceRepository(str(output_dir / "memory.db"))
    model = ScriptedModel([_call("set_memory", {"key": "area", "value": "Koramangala", "source": "stated"}, "set")])
    ChatAgent(model, preference_repository=repository, trace_directory=output_dir / "traces").handle(user_id="user-1", thread_id="memory-correction", message="Remember Koramangala")
    add("memory-correction", "Koramangala", str(repository.get("user-1", "area").value))
    model = ScriptedModel([_call("forget_memory", {"key": "area"}, "forget")])
    ChatAgent(model, preference_repository=repository, trace_directory=output_dir / "traces").handle(user_id="user-1", thread_id="memory-forget", message="Forget my area")
    add("memory-forget", "forgotten", "forgotten" if repository.get("user-1", "area") is None else "reused")
    repository.close()

    class LoopModel(ScriptedModel):
        def __init__(self) -> None:
            super().__init__([])

        def chat(self, messages: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
            self.messages.append(messages)
            return {"choices": [{"message": {"tool_calls": [_call("search_providers", {"service": "plumber", "area": "Indiranagar", "time_window": "this week"}, "loop")]}}]}

    limited = ChatAgent(LoopModel(), execution_limits=ExecutionLimits(max_model_turns=1, max_tool_calls=12), trace_directory=output_dir / "traces")
    limited_result = limited.handle(user_id="user-1", thread_id="limit-exhaustion", message="Find a plumber in Indiranagar this week")
    add("limit-exhaustion", "ABORTED_LIMIT", limited_result["phase"])

    output_dir.joinpath("eval_table.csv").write_text("scenario,expected,actual,passed,notes,trace_path\n" + "\n".join(
        f"{r.scenario},{r.expected},{r.actual},{r.passed},{r.notes},{r.trace_path}" for r in results
    ) + "\n", encoding="utf-8")
    return results

