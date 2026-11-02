from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from home_service_agent.domain import (
    BookingStatus,
    RequestUpdate,
    ServiceRequest,
    TerminalOutcome,
    ToolError,
    WorkflowPhase,
    WorkflowState,
    transition,
)


def test_partial_request_reports_only_missing_required_fields() -> None:
    request = ServiceRequest(service="plumber", area="Indiranagar")

    assert request.missing_fields() == ["time_window"]


def test_request_correction_preserves_unrelated_known_facts() -> None:
    request = ServiceRequest(
        service="plumber",
        area="Indiranagar",
        time_window="weekday mornings",
        budget=1500,
        preferences={"technician": "same as last time"},
    )

    corrected = request.apply_update(area="Koramangala")

    assert corrected.area == "Koramangala"
    assert corrected.service == request.service
    assert corrected.time_window == request.time_window
    assert corrected.budget == request.budget
    assert corrected.preferences == request.preferences


def test_request_update_tracks_only_fields_supplied_by_user() -> None:
    update = RequestUpdate(area="Koramangala", preferences={"technician": "preferred"})

    assert update.supplied_fields() == {
        "area": "Koramangala",
        "preferences": {"technician": "preferred"},
    }


def test_invalid_values_and_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ServiceRequest(budget=-1)

    with pytest.raises(ValidationError):
        ServiceRequest(service="plumber", unexpected="value")


def test_invalid_workflow_transition_is_rejected() -> None:
    state = WorkflowState(run_id="run-1", user_id="user-1")

    with pytest.raises(ValueError, match="Invalid workflow transition"):
        transition(state, WorkflowPhase.COMMITTING)


def test_valid_transition_returns_new_state_without_mutating_input() -> None:
    state = WorkflowState(run_id="run-1", user_id="user-1")

    searching = transition(state, WorkflowPhase.SEARCHING)

    assert state.phase is WorkflowPhase.COLLECTING
    assert searching.phase is WorkflowPhase.SEARCHING
    assert searching.run_id == state.run_id


def test_terminal_outcome_requires_terminal_phase() -> None:
    outcome = TerminalOutcome(
        phase=WorkflowPhase.FAILED,
        message="Provider API failed",
        error=ToolError(code="provider_error", message="unavailable", retryable=True),
    )

    assert outcome.error is not None
    assert outcome.error.retryable is True

    with pytest.raises(ValidationError):
        TerminalOutcome(phase=WorkflowPhase.SEARCHING, message="not terminal")


def test_workflow_state_accepts_booking_metadata() -> None:
    state = WorkflowState(
        run_id="run-1",
        user_id="user-1",
        phase=WorkflowPhase.HOLDING,
        selected_provider_id="provider-1",
        selected_slot="Saturday 10:00",
        quoted_price=1200,
        hold_id="hold-1",
        hold_expires_at=datetime(2026, 11, 2, 10, 10, tzinfo=timezone.utc),
        idempotency_key="book-run-1",
        booking_status=BookingStatus.UPCOMING,
    )

    assert state.quoted_price == 1200
    assert state.booking_status is BookingStatus.UPCOMING

