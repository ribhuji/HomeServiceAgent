"""Code-enforced approval records for side-effecting actions."""

from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import json
from secrets import token_urlsafe
from typing import Callable, TypeVar

from pydantic import Field, model_validator

from .domain import DomainModel


class ApprovalError(PermissionError):
    """Raised when a side effect has no matching valid approval."""


class ActionType(StrEnum):
    BOOK = "book"
    CANCEL = "cancel"
    RESCHEDULE = "reschedule"


class ActionDescriptor(DomainModel):
    """The exact side effect a user is being asked to approve."""

    operation: ActionType
    provider_id: str | None = None
    booking_id: str | None = None
    slot: str | None = None
    price: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_required_action_fields(self) -> "ActionDescriptor":
        required = {
            ActionType.BOOK: ("provider_id", "slot", "price"),
            ActionType.CANCEL: ("booking_id",),
            ActionType.RESCHEDULE: ("booking_id", "slot"),
        }[self.operation]
        missing = [field for field in required if getattr(self, field) is None]
        if missing:
            raise ValueError(
                f"{self.operation} action requires: {', '.join(missing)}"
            )
        return self

    def canonical(self) -> str:
        """Return a stable representation suitable for hashing and display."""

        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


class ApprovalRecord(DomainModel):
    approval_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    action_digest: str = Field(min_length=64, max_length=64)
    approved_at: datetime
    expires_at: datetime


class ApprovalLedger:
    """Small in-memory ledger; the persistence boundary can move to SQLite later."""

    def __init__(self, *, approval_ttl: timedelta = timedelta(minutes=30)) -> None:
        self.approval_ttl = approval_ttl
        self._records: dict[str, ApprovalRecord] = {}

    def record_user_confirmation(self, run_id: str, action: ActionDescriptor) -> ApprovalRecord:
        """Record confirmation for the exact action shown to the user."""

        now = datetime.now(timezone.utc)
        record = ApprovalRecord(
            approval_id=token_urlsafe(18),
            run_id=run_id,
            action_digest=action.digest(),
            approved_at=now,
            expires_at=now + self.approval_ttl,
        )
        self._records[record.approval_id] = record
        return record

    def require_valid(
        self,
        run_id: str,
        action: ActionDescriptor,
        approval_id: str | None,
    ) -> ApprovalRecord:
        """Return a matching approval or raise before any effect can run."""

        if approval_id is None:
            raise ApprovalError("Explicit approval is required")
        record = self._records.get(approval_id)
        if record is None:
            raise ApprovalError("Approval record was not found")
        if record.run_id != run_id:
            raise ApprovalError("Approval belongs to a different run")
        if record.expires_at <= datetime.now(timezone.utc):
            raise ApprovalError("Approval has expired")
        if record.action_digest != action.digest():
            raise ApprovalError("Approved action no longer matches the requested action")
        return record


ResultT = TypeVar("ResultT")


def execute_approved(
    ledger: ApprovalLedger,
    *,
    run_id: str,
    action: ActionDescriptor,
    approval_id: str | None,
    effect: Callable[[], ResultT],
) -> ResultT:
    """Validate approval in code, then execute the side effect."""

    ledger.require_valid(run_id, action, approval_id)
    return effect()

