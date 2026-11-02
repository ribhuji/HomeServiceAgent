"""Bounded execution controls and safe terminal outcomes."""

from dataclasses import dataclass
import time
from typing import Callable, TypeVar

from .domain import TerminalOutcome, ToolError, WorkflowPhase, WorkflowState


class LimitExceeded(RuntimeError):
    def __init__(self, resource: str, limit: int | float) -> None:
        self.resource = resource
        self.limit = limit
        super().__init__(f"{resource} limit exceeded")


@dataclass(frozen=True)
class ExecutionLimits:
    max_model_turns: int = 12
    max_tool_calls: int = 30
    max_retries: int = 3
    max_wall_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_model_turns < 1 or self.max_tool_calls < 1:
            raise ValueError("model and tool limits must be positive")
        if self.max_retries < 0 or self.max_wall_seconds <= 0:
            raise ValueError("retry and wall-clock limits must be non-negative/positive")


@dataclass
class ExecutionBudget:
    limits: ExecutionLimits
    clock: Callable[[], float] = time.monotonic
    started_at: float | None = None
    model_turns: int = 0
    tool_calls: int = 0
    retries: int = 0

    def __post_init__(self) -> None:
        if self.started_at is None:
            self.started_at = self.clock()

    def begin_model_turn(self) -> None:
        self._check_wall_clock()
        if self.model_turns >= self.limits.max_model_turns:
            raise LimitExceeded("model turns", self.limits.max_model_turns)
        self.model_turns += 1

    def begin_tool_call(self) -> None:
        self._check_wall_clock()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise LimitExceeded("tool calls", self.limits.max_tool_calls)
        self.tool_calls += 1

    def begin_retry(self) -> None:
        self._check_wall_clock()
        if self.retries >= self.limits.max_retries:
            raise LimitExceeded("retries", self.limits.max_retries)
        self.retries += 1

    def check(self) -> None:
        self._check_wall_clock()

    def _check_wall_clock(self) -> None:
        assert self.started_at is not None
        if self.clock() - self.started_at >= self.limits.max_wall_seconds:
            raise LimitExceeded("wall-clock time", self.limits.max_wall_seconds)


def terminal_for_exception(
    state: WorkflowState,
    error: Exception,
) -> tuple[WorkflowState, TerminalOutcome]:
    """Convert any exhausted/unknown run error into a clear terminal result."""

    if isinstance(error, LimitExceeded):
        phase = WorkflowPhase.ABORTED_LIMIT
        message = f"I stopped because the {error.resource} limit was reached. Please try again."
        tool_error = ToolError(code="execution_limit", message=str(error), retryable=False)
    else:
        phase = WorkflowPhase.FAILED
        message = "I couldn't complete that request because an unexpected error occurred. Please try again."
        tool_error = ToolError(code="unknown_error", message="unexpected execution error", retryable=False)

    terminal_state = state.model_copy(update={"phase": phase})
    return terminal_state, TerminalOutcome(phase=phase, message=message, error=tool_error)


ResultT = TypeVar("ResultT")


def run_tool_with_budget(
    budget: ExecutionBudget,
    tool: Callable[..., ResultT],
    *args: object,
    **kwargs: object,
) -> ResultT:
    budget.begin_tool_call()
    return tool(*args, **kwargs)

