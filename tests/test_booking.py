from home_service_agent.approval import ActionDescriptor, ActionType, ApprovalLedger
from home_service_agent.booking import BookingCoordinator
from home_service_agent.domain import (
    BookingStatus,
    ServiceRequest,
    WorkflowPhase,
    WorkflowState,
)
from home_service_agent.mocks.provider_api import MockProviderAPI, MockScenario


def held_state(api: MockProviderAPI) -> WorkflowState:
    hold = api.hold_slot("provider-1", "Saturday 09:00")
    return WorkflowState(
        run_id="run-1",
        user_id="user-1",
        phase=WorkflowPhase.HOLDING,
        request=ServiceRequest(service="plumber", area="Indiranagar", time_window="this week"),
        selected_provider_id="provider-1",
        selected_slot="Saturday 09:00",
        quoted_price=1200,
        hold_id=hold.hold_id,
    )


def approval_for(ledger: ApprovalLedger, state: WorkflowState) -> str:
    approval = ledger.record_user_confirmation(
        state.run_id,
        ActionDescriptor(
            operation=ActionType.BOOK,
            provider_id=state.selected_provider_id,
            slot=state.selected_slot,
            price=state.quoted_price,
        ),
    )
    return approval.approval_id


def test_success_is_reported_only_after_authoritative_verification() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = held_state(api)

    outcome = BookingCoordinator(api, ledger).book_after_hold(state, approval_for(ledger, state))

    assert outcome.workflow.phase is WorkflowPhase.COMPLETED
    assert outcome.workflow.booking_status is BookingStatus.BOOKED
    assert outcome.workflow.booking_id == "booking:run-1"
    assert outcome.workflow.idempotency_key == "book:run-1"
    assert outcome.attempts == 1


def test_timeout_after_success_is_reconciled_without_retrying() -> None:
    api = MockProviderAPI(
        scenario=MockScenario(book_outcome="timeout", timeout_after_success=True)
    )
    ledger = ApprovalLedger()
    state = held_state(api)

    outcome = BookingCoordinator(api, ledger).book_after_hold(state, approval_for(ledger, state))

    assert outcome.workflow.phase is WorkflowPhase.COMPLETED
    assert outcome.workflow.booking_status is BookingStatus.BOOKED
    assert outcome.attempts == 1
    assert api.book_call_count == 1


def test_timeout_without_success_retries_same_idempotency_key() -> None:
    api = MockProviderAPI(scenario=MockScenario(book_outcome="timeout"))
    ledger = ApprovalLedger()
    state = held_state(api)
    approval_id = approval_for(ledger, state)

    # Simulate the provider recovering after the first unknown result.
    original_book = api.book
    calls = 0

    def book_once_then_succeed(hold_id: str, booking_id: str, idempotency_key: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_book(hold_id, booking_id, idempotency_key)
        api.scenario.book_outcome = "success"
        return original_book(hold_id, booking_id, idempotency_key)

    api.book = book_once_then_succeed  # type: ignore[method-assign]
    outcome = BookingCoordinator(api, ledger).book_after_hold(state, approval_id)

    assert outcome.workflow.phase is WorkflowPhase.COMPLETED
    assert outcome.attempts == 2
    assert api.book_call_count == 2
    assert outcome.workflow.idempotency_key == "book:run-1"


def test_repeated_confirmation_after_completion_does_not_call_provider_again() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = held_state(api)
    approval_id = approval_for(ledger, state)
    coordinator = BookingCoordinator(api, ledger)

    first = coordinator.book_after_hold(state, approval_id)
    second = coordinator.book_after_hold(first.workflow, approval_id)

    assert first.workflow.booking_id == second.workflow.booking_id
    assert second.workflow.phase is WorkflowPhase.COMPLETED
    assert api.book_call_count == 1


def test_restart_reuses_persisted_booking_identity() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = held_state(api)
    approval_id = approval_for(ledger, state)
    coordinator = BookingCoordinator(api, ledger)

    first = coordinator.book_after_hold(state, approval_id)
    restored = WorkflowState.model_validate(first.workflow.model_dump(mode="json"))
    second = coordinator.book_after_hold(restored, approval_id)

    assert second.workflow.booking_id == "booking:run-1"
    assert second.workflow.idempotency_key == "book:run-1"
    assert api.book_call_count == 1

