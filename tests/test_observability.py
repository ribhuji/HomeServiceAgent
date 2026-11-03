import json
from pathlib import Path

from home_service_agent.chat import ChatAgent
from home_service_agent.observability import sanitize


class TraceModel:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "model": "scripted-model",
                "usage": {"total_tokens": 17, "cost": 0.0},
                "choices": [{"message": {"tool_calls": [{
                    "id": "trace-call",
                    "type": "function",
                    "function": {
                        "name": "search_providers",
                        "arguments": '{"service":"plumber","area":"Indiranagar","time_window":"this week"}',
                    },
                }]}}],
            }
        return {"choices": [{"message": {"content": "Options found."}}]}


def test_live_trace_contains_run_state_model_and_tool_events(tmp_path: Path) -> None:
    result = ChatAgent(TraceModel(), trace_directory=tmp_path).handle(
        user_id="user-1",
        thread_id="trace-run",
        message="Find a plumber in Indiranagar this week",
    )

    events = [json.loads(line) for line in (tmp_path / "trace-run.jsonl").read_text().splitlines()]
    names = [event["event"] for event in events]
    assert names[0] == "run_started"
    assert "model_call" in names
    assert "tool_call" in names
    assert "state_transition" in names
    assert names[-1] == "run_finished"
    model_event = next(event for event in events if event["event"] == "model_call")
    assert model_event["tokens"] == 17
    assert model_event["cost"] == 0.0
    assert result["phase"] == "OPTIONS_READY"


def test_trace_sanitizes_secret_fields_and_caps_strings() -> None:
    sanitized = sanitize({"api-key": "secret", "nested": {"token": "value"}, "text": "x" * 3000})

    assert sanitized["api-key"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
    assert len(sanitized["text"]) == 2000
