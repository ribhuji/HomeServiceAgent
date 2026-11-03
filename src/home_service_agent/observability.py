"""Sanitized per-run trace writing."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


_SECRET_KEYS = {
    "api_key", "apikey", "authorization", "password", "secret", "token",
    "access_token", "refresh_token", "openrouter_api_key",
}


def sanitize(value: Any, *, max_string_chars: int = 2000) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).lower().replace("-", "_") in _SECRET_KEYS else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return value[:max_string_chars]
    if hasattr(value, "model_dump"):
        return sanitize(value.model_dump(mode="json"))
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


class TraceRecorder:
    def __init__(self, run_id: str, directory: str | Path) -> None:
        self.run_id = run_id
        self.path = Path(directory) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **sanitize(fields),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")

    def run_started(self, *, user_id: str, phase: str, message: str) -> None:
        self.record("run_started", user_id=user_id, phase=phase, message=message)

    def run_finished(self, *, phase: str, message: str) -> None:
        self.record("run_finished", phase=phase, message=message)

    def transition(self, from_phase: str, to_phase: str, reason: str) -> None:
        self.record("state_transition", from_phase=from_phase, to_phase=to_phase, reason=reason)

    def model_call(
        self,
        model: str,
        *,
        latency_ms: float,
        tokens: int | None = None,
        cost: float | None = None,
        usage: Any = None,
    ) -> None:
        self.record(
            "model_call",
            model=model,
            latency_ms=latency_ms,
            tokens=tokens,
            cost=cost,
            usage=usage,
        )

    def tool_call(self, name: str, args: Any, result: Any, *, latency_ms: float) -> None:
        self.record("tool_call", tool=name, args=args, result=result, latency_ms=latency_ms)

