from datetime import timedelta

import pytest
from pydantic import ValidationError

from home_service_agent.approval import (
    ActionDescriptor,
    ActionType,
    ApprovalError,
    ApprovalLedger,
    execute_approved,
)


def booking_action(*, price: int = 1200, slot: str = "Saturday 10:00") -> ActionDescriptor:
    return ActionDescriptor(
        operation=ActionType.BOOK,
        provider_id="provider-1",
        slot=slot,
        price=price,
    )


def test_approval_is_bound_to_exact_action() -> None:
    ledger = ApprovalLedger()
    action = booking_action()
    approval = ledger.record_user_confirmation("run-1", action)
    executed: list[str] = []

    result = execute_approved(
        ledger,
        run_id="run-1",
        action=action,
        approval_id=approval.approval_id,
        effect=lambda: executed.append("booked") or "booking-1",
    )

    assert result == "booking-1"
    assert executed == ["booked"]


@pytest.mark.parametrize(
    "changed_action",
    [
        booking_action(price=1500),
        booking_action(slot="Sunday 10:00"),
        ActionDescriptor(operation=ActionType.BOOK, provider_id="provider-2", slot="Saturday 10:00", price=1200),
        ActionDescriptor(operation=ActionType.CANCEL, booking_id="booking-1"),
    ],
)
def test_any_action_change_invalidates_approval(changed_action: ActionDescriptor) -> None:
    ledger = ApprovalLedger()
    approval = ledger.record_user_confirmation("run-1", booking_action())
    executed: list[str] = []

    with pytest.raises(ApprovalError, match="no longer matches"):
        execute_approved(
            ledger,
            run_id="run-1",
            action=changed_action,
            approval_id=approval.approval_id,
            effect=lambda: executed.append("must not run"),
        )

    assert executed == []


def test_approval_is_default_deny_and_run_bound() -> None:
    ledger = ApprovalLedger()
    action = booking_action()
    approval = ledger.record_user_confirmation("run-1", action)

    with pytest.raises(ApprovalError, match="Explicit approval"):
        execute_approved(ledger, run_id="run-1", action=action, approval_id=None, effect=lambda: None)

    with pytest.raises(ApprovalError, match="different run"):
        execute_approved(
            ledger,
            run_id="run-2",
            action=action,
            approval_id=approval.approval_id,
            effect=lambda: None,
        )


def test_expired_approval_is_rejected() -> None:
    ledger = ApprovalLedger(approval_ttl=timedelta(seconds=-1))
    action = booking_action()
    approval = ledger.record_user_confirmation("run-1", action)

    with pytest.raises(ApprovalError, match="expired"):
        ledger.require_valid("run-1", action, approval.approval_id)


def test_cancel_and_reschedule_are_also_approval_gated() -> None:
    ledger = ApprovalLedger()
    cancel = ActionDescriptor(operation=ActionType.CANCEL, booking_id="booking-1")
    reschedule = ActionDescriptor(
        operation=ActionType.RESCHEDULE,
        booking_id="booking-1",
        slot="Sunday 10:00",
    )

    cancel_approval = ledger.record_user_confirmation("run-1", cancel)
    reschedule_approval = ledger.record_user_confirmation("run-1", reschedule)

    assert execute_approved(
        ledger,
        run_id="run-1",
        action=cancel,
        approval_id=cancel_approval.approval_id,
        effect=lambda: "cancelled",
    ) == "cancelled"
    assert execute_approved(
        ledger,
        run_id="run-1",
        action=reschedule,
        approval_id=reschedule_approval.approval_id,
        effect=lambda: "rescheduled",
    ) == "rescheduled"


@pytest.mark.parametrize(
    "action",
    [
        {"operation": ActionType.BOOK, "provider_id": "provider-1", "slot": "Saturday 10:00"},
        {"operation": ActionType.CANCEL},
        {"operation": ActionType.RESCHEDULE, "booking_id": "booking-1"},
    ],
)
def test_incomplete_side_effect_actions_are_rejected(action: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="requires"):
        ActionDescriptor(**action)

