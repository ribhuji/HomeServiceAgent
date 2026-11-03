"""Idempotent booking commit and authoritative recovery."""

from dataclasses import dataclass

from .approval import ActionDescriptor, ActionType, ApprovalLedger
from .domain import BookingStatus, WorkflowPhase, WorkflowState, transition
from .mocks.provider_api import MockProviderAPI, ProviderToolError


@dataclass(frozen=True)
class BookingOutcome:
    workflow: WorkflowState
    message: str
    attempts: int


class BookingCoordinator:
    """Commit a held slot without duplicating work across retries or restarts."""

    def __init__(
        self,
        provider_api: MockProviderAPI,
        approvals: ApprovalLedger,
        *,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.provider_api = provider_api
        self.approvals = approvals
        self.max_attempts = max_attempts

    def book_after_hold(self, state: WorkflowState, approval_id: str | None) -> BookingOutcome:
        if state.phase is WorkflowPhase.COMPLETED and state.booking_status is BookingStatus.BOOKED:
            return BookingOutcome(
                workflow=state,
                message=f"Booking {state.booking_id} is already confirmed.",
                attempts=0,
            )
        if state.phase is not WorkflowPhase.HOLDING:
            raise ValueError(f"Cannot book from phase {state.phase}")
        if state.selected_provider_id is None or state.selected_slot is None:
            raise ValueError("Provider and slot must be selected before booking")
        if state.quoted_price is None or state.hold_id is None:
            raise ValueError("A quoted price and active hold are required before booking")

        action = ActionDescriptor(
            operation=ActionType.BOOK,
            provider_id=state.selected_provider_id,
            slot=state.selected_slot,
            price=state.quoted_price,
        )
        self.approvals.require_valid(state.run_id, action, approval_id)

        idempotency_key = state.idempotency_key or f"book:{state.run_id}"
        working = transition(
            state.model_copy(
                update={
                    "idempotency_key": idempotency_key,
                    "booking_status": None,
                }
            ),
            WorkflowPhase.COMMITTING,
        )

        for attempt in range(1, self.max_attempts + 1):
            try:
                returned_id = self.provider_api.book(
                    working.hold_id or "",
                    idempotency_key,
                )
            except ProviderToolError as error:
                if error.code in {"timeout", "unknown"}:
                    status = self.provider_api.get_booking_status(idempotency_key)
                    if status.status is BookingStatus.BOOKED:
                        return self._complete(working.model_copy(update={"booking_id": status.booking_id}), status.status, attempt)
                    if status.status is BookingStatus.FAILED:
                        return self._failed(working, "The provider reported that booking failed", attempt)
                    if attempt < self.max_attempts:
                        continue
                    return self._failed(working, "Booking could not be verified after retries", attempt)
                if error.code == "slot_taken":
                    return BookingOutcome(
                        workflow=transition(working, WorkflowPhase.CONFLICT),
                        message="The slot was taken before booking could complete.",
                        attempts=attempt,
                    )
                if error.code == "hold_expired":
                    return BookingOutcome(
                        workflow=transition(working, WorkflowPhase.EXPIRED),
                        message="The slot hold expired before booking could complete.",
                        attempts=attempt,
                    )
                return self._failed(working, f"Booking failed: {error.message}", attempt)

            status = self.provider_api.get_booking_status(idempotency_key)
            if status.status is BookingStatus.BOOKED and status.booking_id == returned_id:
                return self._complete(working.model_copy(update={"booking_id": status.booking_id}), status.status, attempt)
            if attempt == self.max_attempts:
                return self._failed(working, "Booking returned without authoritative verification", attempt)

        return self._failed(working, "Booking attempts exhausted", self.max_attempts)

    def _complete(
        self,
        state: WorkflowState,
        status: BookingStatus,
        attempts: int,
    ) -> BookingOutcome:
        verified = transition(
            state.model_copy(update={"booking_status": status}),
            WorkflowPhase.VERIFYING,
        )
        completed = transition(verified, WorkflowPhase.COMPLETED)
        return BookingOutcome(
            workflow=completed,
            message=f"Booking {completed.booking_id} is confirmed.",
            attempts=attempts,
        )

    def _failed(self, state: WorkflowState, message: str, attempts: int) -> BookingOutcome:
        return BookingOutcome(
            workflow=transition(state, WorkflowPhase.FAILED),
            message=message,
            attempts=attempts,
        )

