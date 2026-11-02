"""Typed domain contracts for the Home-Service Agent.

These models are deliberately independent of FastAPI and LangGraph so they can
be used by the mock tools, graph nodes, and offline tests alike.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainModel(BaseModel):
    """Base model that rejects accidental fields at the domain boundary."""

    model_config = ConfigDict(extra="forbid")


class WorkflowPhase(StrEnum):
    COLLECTING = "COLLECTING"
    SEARCHING = "SEARCHING"
    OPTIONS_READY = "OPTIONS_READY"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    HOLDING = "HOLDING"
    COMMITTING = "COMMITTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    NEEDS_CORRECTION = "NEEDS_CORRECTION"
    CONFLICT = "CONFLICT"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"
    ABORTED_LIMIT = "ABORTED_LIMIT"


class BookingStatus(StrEnum):
    UPCOMING = "upcoming"
    BOOKED = "booked"
    CANCELLED = "cancelled"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class ServiceRequest(DomainModel):
    """Structured user intent; fields remain optional during collection."""

    service: str | None = None
    area: str | None = None
    time_window: str | None = None
    budget: int | None = Field(default=None, ge=0)
    preferences: dict[str, str] = Field(default_factory=dict)

    def missing_fields(self) -> list[str]:
        """Return required request fields that still need user input."""

        required = ("service", "area", "time_window")
        return [name for name in required if getattr(self, name) is None]

    def apply_update(self, **updates: Any) -> "ServiceRequest":
        """Apply only supplied fields, preserving all other known facts."""

        unknown = set(updates) - set(type(self).model_fields)
        if unknown:
            raise ValueError(f"Unknown request fields: {sorted(unknown)}")
        if "preferences" in updates and updates["preferences"] is not None:
            updates["preferences"] = {
                **self.preferences,
                **updates["preferences"],
            }
        return self.model_copy(update=updates)


class RequestUpdate(DomainModel):
    """Structured fields extracted from one user message.

    Unset fields are intentionally distinct from fields supplied in a correction
    so a message never clears unrelated facts by accident.
    """

    service: str | None = None
    area: str | None = None
    time_window: str | None = None
    budget: int | None = Field(default=None, ge=0)
    preferences: dict[str, str] | None = None

    def supplied_fields(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class Provider(DomainModel):
    id: str = Field(min_length=1)
    price: int = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    free_slots: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)


class ProviderDetails(Provider):
    reviews: list[str] = Field(default_factory=list)


class Booking(DomainModel):
    id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    service: str = Field(min_length=1)
    slot: str = Field(min_length=1)
    status: BookingStatus


class ToolError(DomainModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class TerminalOutcome(DomainModel):
    phase: WorkflowPhase
    message: str = Field(min_length=1)
    error: ToolError | None = None

    @field_validator("phase")
    @classmethod
    def phase_must_be_terminal(cls, value: WorkflowPhase) -> WorkflowPhase:
        if value not in TERMINAL_PHASES:
            raise ValueError(f"{value} is not a terminal phase")
        return value


class WorkflowState(DomainModel):
    run_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    phase: WorkflowPhase = WorkflowPhase.COLLECTING
    request: ServiceRequest = Field(default_factory=ServiceRequest)
    selected_provider_id: str | None = None
    selected_slot: str | None = None
    quoted_price: int | None = Field(default=None, ge=0)
    hold_id: str | None = None
    hold_expires_at: datetime | None = None
    booking_id: str | None = None
    idempotency_key: str | None = None
    booking_status: BookingStatus | None = None


TERMINAL_PHASES = frozenset(
    {
        WorkflowPhase.COMPLETED,
        WorkflowPhase.FAILED,
        WorkflowPhase.ABORTED_LIMIT,
    }
)


ALLOWED_TRANSITIONS: dict[WorkflowPhase, frozenset[WorkflowPhase]] = {
    WorkflowPhase.COLLECTING: frozenset(
        {WorkflowPhase.SEARCHING, WorkflowPhase.NEEDS_CORRECTION, WorkflowPhase.ABORTED_LIMIT}
    ),
    WorkflowPhase.SEARCHING: frozenset(
        {WorkflowPhase.OPTIONS_READY, WorkflowPhase.FAILED, WorkflowPhase.ABORTED_LIMIT}
    ),
    WorkflowPhase.OPTIONS_READY: frozenset(
        {WorkflowPhase.AWAITING_APPROVAL, WorkflowPhase.COLLECTING, WorkflowPhase.ABORTED_LIMIT}
    ),
    WorkflowPhase.AWAITING_APPROVAL: frozenset(
        {WorkflowPhase.HOLDING, WorkflowPhase.COLLECTING, WorkflowPhase.ABORTED_LIMIT}
    ),
    WorkflowPhase.HOLDING: frozenset(
        {
            WorkflowPhase.COMMITTING,
            WorkflowPhase.AWAITING_APPROVAL,
            WorkflowPhase.EXPIRED,
            WorkflowPhase.CONFLICT,
            WorkflowPhase.FAILED,
        }
    ),
    WorkflowPhase.COMMITTING: frozenset(
        {WorkflowPhase.VERIFYING, WorkflowPhase.FAILED, WorkflowPhase.ABORTED_LIMIT}
    ),
    WorkflowPhase.VERIFYING: frozenset(
        {WorkflowPhase.COMPLETED, WorkflowPhase.COMMITTING, WorkflowPhase.FAILED}
    ),
    WorkflowPhase.NEEDS_CORRECTION: frozenset({WorkflowPhase.COLLECTING}),
    WorkflowPhase.CONFLICT: frozenset({WorkflowPhase.AWAITING_APPROVAL, WorkflowPhase.FAILED}),
    WorkflowPhase.EXPIRED: frozenset({WorkflowPhase.AWAITING_APPROVAL, WorkflowPhase.FAILED}),
    WorkflowPhase.FAILED: frozenset(),
    WorkflowPhase.COMPLETED: frozenset(),
    WorkflowPhase.ABORTED_LIMIT: frozenset(),
}


def transition(state: WorkflowState, target: WorkflowPhase) -> WorkflowState:
    """Return a new state if the requested workflow transition is allowed."""

    allowed = ALLOWED_TRANSITIONS[state.phase]
    if target not in allowed:
        raise ValueError(f"Invalid workflow transition: {state.phase} -> {target}")
    return state.model_copy(update={"phase": target})

