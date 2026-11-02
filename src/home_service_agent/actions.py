"""Approval-gated cancellation and rescheduling recovery."""

from dataclasses import dataclass

from .approval import ActionDescriptor, ActionType, ApprovalLedger
from .domain import BookingStatus, WorkflowState
from .mocks.provider_api import MockProviderAPI, ProviderToolError


@dataclass(frozen=True)
class ActionOutcome:
    workflow: WorkflowState
    message: str
    succeeded: bool
    attempts: int


class CancellationCoordinator:
    def __init__(
        self,
        provider_api: MockProviderAPI,
        approvals: ApprovalLedger,
        *,
        max_attempts: int = 3,
    ) -> None:
        self.provider_api = provider_api
        self.approvals = approvals
        self.max_attempts = max_attempts

    def cancel(self, state: WorkflowState, approval_id: str | None) -> ActionOutcome:
        if state.booking_id is None:
            raise ValueError("A booking is required before cancellation")
        if state.booking_status is BookingStatus.CANCELLED:
            return ActionOutcome(state, "Booking is already cancelled.", True, 0)

        action = ActionDescriptor(operation=ActionType.CANCEL, booking_id=state.booking_id)
        self.approvals.require_valid(state.run_id, action, approval_id)
        idempotency_key = f"cancel:{state.run_id}:{state.booking_id}"

        for attempt in range(1, self.max_attempts + 1):
            try:
                self.provider_api.cancel(state.booking_id, idempotency_key)
                updated = state.model_copy(update={"booking_status": BookingStatus.CANCELLED})
                return ActionOutcome(updated, "Booking cancelled.", True, attempt)
            except ProviderToolError as error:
                if error.code == "503" and attempt < self.max_attempts:
                    continue
                return ActionOutcome(
                    state,
                    "Cancellation could not be completed after retries.",
                    False,
                    attempt,
                )

        return ActionOutcome(state, "Cancellation attempts exhausted.", False, self.max_attempts)


class RescheduleCoordinator:
    def __init__(self, provider_api: MockProviderAPI, approvals: ApprovalLedger) -> None:
        self.provider_api = provider_api
        self.approvals = approvals

    def reschedule(
        self,
        state: WorkflowState,
        new_slot: str,
        approval_id: str | None,
    ) -> ActionOutcome:
        if state.booking_id is None:
            raise ValueError("A booking is required before rescheduling")
        if not new_slot:
            raise ValueError("A new slot is required")
        if state.selected_slot == new_slot:
            return ActionOutcome(state, "Booking is already scheduled for that slot.", True, 0)

        action = ActionDescriptor(
            operation=ActionType.RESCHEDULE,
            booking_id=state.booking_id,
            slot=new_slot,
        )
        self.approvals.require_valid(state.run_id, action, approval_id)
        idempotency_key = f"reschedule:{state.run_id}:{state.booking_id}:{new_slot}"

        try:
            updated_booking = self.provider_api.reschedule(
                state.booking_id,
                new_slot,
                idempotency_key,
            )
        except ProviderToolError as error:
            if error.code == "slot_conflict":
                return ActionOutcome(
                    state,
                    "That slot is unavailable. Your existing booking is unchanged.",
                    False,
                    1,
                )
            return ActionOutcome(state, f"Rescheduling failed: {error.message}", False, 1)

        return ActionOutcome(
            state.model_copy(update={"selected_slot": updated_booking.slot}),
            f"Booking rescheduled to {updated_booking.slot}.",
            True,
            1,
        )

