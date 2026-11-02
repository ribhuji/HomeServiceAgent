import pytest

from home_service_agent.approval import ActionDescriptor, ActionType, ApprovalError, ApprovalLedger
from home_service_agent.domain import ServiceRequest, WorkflowPhase, WorkflowState
from home_service_agent.holding import HoldCoordinator
from home_service_agent.mocks.provider_api import MockProviderAPI, MockScenario


def approved_state(price: int = 1200, slot: str = "Saturday 09:00") -> WorkflowState:
    return WorkflowState(
        run_id="run-1",
        user_id="user-1",
        phase=WorkflowPhase.AWAITING_APPROVAL,
        request=ServiceRequest(service="plumber", area="Indiranagar", time_window="this week"),
        selected_provider_id="provider-1",
        selected_slot=slot,
        quoted_price=price,
    )


def approve(ledger: ApprovalLedger, state: WorkflowState) -> str:
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


def test_approved_matching_quote_moves_state_to_holding() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = approved_state()
    outcome = HoldCoordinator(api, ledger).hold_after_approval(state, approve(ledger, state))

    assert outcome.workflow.phase is WorkflowPhase.HOLDING
    assert outcome.workflow.hold_id == "hold-1"
    assert outcome.workflow.hold_expires_at is not None
    assert outcome.workflow.quoted_price == 1200
    assert outcome.requires_new_approval is False


def test_quote_change_invalidates_old_approval_and_requests_new_confirmation() -> None:
    api = MockProviderAPI(
        scenario=MockScenario(quote_overrides={("provider-1", "Saturday 09:00"): 1499})
    )
    ledger = ApprovalLedger()
    state = approved_state()
    approval_id = approve(ledger, state)

    outcome = HoldCoordinator(api, ledger).hold_after_approval(state, approval_id)

    assert outcome.workflow.phase is WorkflowPhase.AWAITING_APPROVAL
    assert outcome.workflow.quoted_price == 1499
    assert outcome.workflow.hold_id is None
    assert outcome.requires_new_approval is True

    with pytest.raises(ApprovalError, match="no longer matches"):
        HoldCoordinator(api, ledger).hold_after_approval(outcome.workflow, approval_id)


def test_unavailable_slot_does_not_create_a_hold() -> None:
    api = MockProviderAPI(
        scenario=MockScenario(unavailable_slots={("provider-1", "Saturday 09:00")})
    )
    ledger = ApprovalLedger()
    state = approved_state()

    outcome = HoldCoordinator(api, ledger).hold_after_approval(state, approve(ledger, state))

    assert outcome.workflow.phase is WorkflowPhase.AWAITING_APPROVAL
    assert outcome.workflow.hold_id is None
    assert outcome.requires_new_approval is True


def test_changed_slot_cannot_reuse_approval() -> None:
    api = MockProviderAPI()
    ledger = ApprovalLedger()
    state = approved_state()
    approval_id = approve(ledger, state)
    changed = state.model_copy(update={"selected_slot": "Saturday 11:00"})

    with pytest.raises(ApprovalError, match="no longer matches"):
        HoldCoordinator(api, ledger).hold_after_approval(changed, approval_id)

