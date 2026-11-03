"""Approval-gated slot holding and quote reconciliation."""

from dataclasses import dataclass

from .approval import ActionDescriptor, ActionType, ApprovalLedger
from .domain import WorkflowPhase, WorkflowState, transition
from .mocks.provider_api import Hold, MockProviderAPI, ProviderToolError


@dataclass(frozen=True)
class HoldOutcome:
    workflow: WorkflowState
    message: str
    requires_new_approval: bool = False


class HoldCoordinator:
    """Turn an approved search option into a provider hold safely."""

    def __init__(self, provider_api: MockProviderAPI, approvals: ApprovalLedger) -> None:
        self.provider_api = provider_api
        self.approvals = approvals

    def hold_after_approval(self, state: WorkflowState, approval_id: str | None) -> HoldOutcome:
        if state.phase is not WorkflowPhase.AWAITING_APPROVAL:
            raise ValueError(f"Cannot hold a slot from phase {state.phase}")
        if state.selected_provider_id is None or state.selected_slot is None:
            raise ValueError("Provider and slot must be selected before holding")
        if state.quoted_price is None:
            raise ValueError("Search price is required before holding")

        action = ActionDescriptor(
            operation=ActionType.BOOK,
            provider_id=state.selected_provider_id,
            slot=state.selected_slot,
            price=state.quoted_price,
        )
        self.approvals.require_valid(state.run_id, action, approval_id)

        try:
            hold = self.provider_api.hold_slot(state.selected_provider_id, state.selected_slot)
        except ProviderToolError as error:
            if error.code != "slot_unavailable":
                raise
            return HoldOutcome(
                workflow=state,
                message="That slot is no longer available. Please choose another slot.",
                requires_new_approval=True,
            )

        if hold.quoted_price != state.quoted_price:
            updated = state.model_copy(
                update={
                    "quoted_price": hold.quoted_price,
                    "hold_id": None,
                    "hold_expires_at": None,
                }
            )
            return HoldOutcome(
                workflow=updated,
                message=(
                    f"The provider's current quote is ₹{hold.quoted_price}, not "
                    f"₹{state.quoted_price}. Please approve the new price."
                ),
                requires_new_approval=True,
            )

        return HoldOutcome(
            workflow=transition(state.model_copy(
                update={
                    "hold_id": hold.hold_id,
                    "hold_expires_at": hold.expires_at,
                }
            ), WorkflowPhase.HOLDING),
            message=f"Slot held until {hold.expires_at.isoformat()}.",
        )

