from home_service_agent.actions import CancellationCoordinator, RescheduleCoordinator
from home_service_agent.approval import ActionDescriptor, ActionType, ApprovalLedger
from home_service_agent.domain import BookingStatus, ServiceRequest, WorkflowState
from home_service_agent.mocks.provider_api import MockProviderAPI, MockScenario


def booked_state(api: MockProviderAPI) -> WorkflowState:
    hold = api.hold_slot("provider-1", "Saturday 09:00")
    api.book(hold.hold_id, "book-idem-1")
    return WorkflowState(
        run_id="run-1",
        user_id="user-1",
        request=ServiceRequest(service="plumber", area="Indiranagar", time_window="this week"),
        selected_provider_id="provider-1",
        selected_slot="Saturday 09:00",
        booking_id="booking-1",
        booking_status=BookingStatus.BOOKED,
    )


def test_cancel_retries_transient_503_and_updates_state() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = booked_state(api)
    approval = ledger.record_user_confirmation(
        state.run_id,
        ActionDescriptor(operation=ActionType.CANCEL, booking_id=state.booking_id),
    )

    outcome = CancellationCoordinator(api, ledger).cancel(state, approval.approval_id)

    assert outcome.succeeded is True
    assert outcome.attempts == 2
    assert outcome.workflow.booking_status is BookingStatus.CANCELLED
    assert api.cancel_call_count == 2


def test_repeated_cancel_does_not_call_provider_again() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = booked_state(api)
    approval = ledger.record_user_confirmation(
        state.run_id,
        ActionDescriptor(operation=ActionType.CANCEL, booking_id=state.booking_id),
    )
    coordinator = CancellationCoordinator(api, ledger)

    first = coordinator.cancel(state, approval.approval_id)
    second = coordinator.cancel(first.workflow, approval.approval_id)

    assert second.succeeded is True
    assert second.attempts == 0
    assert api.cancel_call_count == 2


def test_reschedule_conflict_preserves_original_booking() -> None:
    api = MockProviderAPI(scenario=MockScenario(reschedule_conflict_slots={"Sunday 11:00"}))
    ledger = ApprovalLedger()
    state = booked_state(api)
    approval = ledger.record_user_confirmation(
        state.run_id,
        ActionDescriptor(
            operation=ActionType.RESCHEDULE,
            booking_id=state.booking_id,
            slot="Sunday 11:00",
        ),
    )

    outcome = RescheduleCoordinator(api, ledger).reschedule(
        state,
        "Sunday 11:00",
        approval.approval_id,
    )

    assert outcome.succeeded is False
    assert outcome.workflow.selected_slot == "Saturday 09:00"
    assert "unchanged" in outcome.message


def test_reschedule_success_updates_slot_and_uses_approval() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = booked_state(api)
    approval = ledger.record_user_confirmation(
        state.run_id,
        ActionDescriptor(
            operation=ActionType.RESCHEDULE,
            booking_id=state.booking_id,
            slot="Saturday 11:00",
        ),
    )

    outcome = RescheduleCoordinator(api, ledger).reschedule(
        state,
        "Saturday 11:00",
        approval.approval_id,
    )

    assert outcome.succeeded is True
    assert outcome.workflow.selected_slot == "Saturday 11:00"
    assert api.reschedule_call_count == 1

