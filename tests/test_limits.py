import pytest

from home_service_agent.domain import ServiceRequest, WorkflowPhase, WorkflowState
from home_service_agent.limits import (
    ExecutionBudget,
    ExecutionLimits,
    LimitExceeded,
    run_tool_with_budget,
    terminal_for_exception,
)


def test_model_tool_and_retry_limits_are_enforced() -> None:
    budget = ExecutionBudget(ExecutionLimits(max_model_turns=1, max_tool_calls=1, max_retries=1))

    budget.begin_model_turn()
    with pytest.raises(LimitExceeded, match="model turns"):
        budget.begin_model_turn()

    budget.begin_tool_call()
    with pytest.raises(LimitExceeded, match="tool calls"):
        budget.begin_tool_call()

    budget.begin_retry()
    with pytest.raises(LimitExceeded, match="retries"):
        budget.begin_retry()


def test_tool_wrapper_counts_attempts_before_execution() -> None:
    budget = ExecutionBudget(ExecutionLimits(max_tool_calls=1))
    calls: list[str] = []

    assert run_tool_with_budget(budget, lambda: calls.append("called") or "result") == "result"
    with pytest.raises(LimitExceeded):
        run_tool_with_budget(budget, lambda: calls.append("must not run"))

    assert calls == ["called"]


def test_wall_clock_limit_is_enforced_with_injected_clock() -> None:
    now = [100.0]
    budget = ExecutionBudget(
        ExecutionLimits(max_wall_seconds=5),
        clock=lambda: now[0],
    )
    budget.check()
    now[0] = 105.0

    with pytest.raises(LimitExceeded, match="wall-clock"):
        budget.check()


def test_limit_exhaustion_has_clear_terminal_user_message() -> None:
    state = WorkflowState(
        run_id="run-1",
        user_id="user-1",
        request=ServiceRequest(service="plumber"),
    )

    terminal_state, outcome = terminal_for_exception(
        state,
        LimitExceeded("tool calls", 30),
    )

    assert terminal_state.phase is WorkflowPhase.ABORTED_LIMIT
    assert outcome.phase is WorkflowPhase.ABORTED_LIMIT
    assert "tool calls" in outcome.message
    assert "30" not in outcome.message


def test_unknown_errors_have_clear_non_secret_terminal_message() -> None:
    state = WorkflowState(run_id="run-1", user_id="user-1")

    terminal_state, outcome = terminal_for_exception(
        state,
        RuntimeError("database password=super-secret"),
    )

    assert terminal_state.phase is WorkflowPhase.FAILED
    assert outcome.phase is WorkflowPhase.FAILED
    assert "unexpected error" in outcome.message
    assert "super-secret" not in outcome.message
    assert outcome.error.message == "unexpected execution error"

