from pathlib import Path

from home_service_agent.chat import (
    ChatAgent,
    REQUEST_EXTRACTION_SYSTEM_PROMPT,
    extract_request_update,
)
from home_service_agent.memory import PreferenceRepository


class CapturingModel:
    def __init__(self, content: str | None = None) -> None:
        self.messages: list[dict] = []
        self.content = content or (
            '{"service":"plumber","area":"Indiranagar",'
            '"time_window":"this week"}'
        )

    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        self.messages = messages
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content
                    }
                }
            ]
        }


def test_request_extraction_uses_system_prompt_and_separate_user_message() -> None:
    model = CapturingModel()

    ChatAgent(model).handle(
        user_id="user-1",
        message="Find a plumber in Indiranagar this week",
    )

    assert model.messages[0] == {
        "role": "system",
        "content": REQUEST_EXTRACTION_SYSTEM_PROMPT,
    }
    assert model.messages[1]["role"] == "user"
    assert "Find a plumber in Indiranagar this week" in model.messages[1]["content"]


def test_model_budget_text_is_normalized_conservatively() -> None:
    update = extract_request_update(
        '{"service":"plumber","budget":"under 1500 rupees","preferences":""}'
    )

    assert update.budget == 1500
    assert update.preferences is None


def test_live_chat_saves_explicit_area_and_preferences(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    model = CapturingModel(
        '{"service":"plumber","area":"Koramangala",'
        '"time_window":"this week","preferences":{"language":"English"}}'
    )

    ChatAgent(model, preference_repository=repository).handle(
        user_id="user-1",
        message="Find a plumber in Koramangala this week; English is preferred",
    )

    assert repository.get("user-1", "area").value == "Koramangala"
    assert repository.get("user-1", "preferences").value == {"language": "English"}
    repository.close()


def test_live_chat_reuses_confirmed_memory_for_a_new_thread(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    repository.set("user-1", "area", "Koramangala")
    model = CapturingModel('{"service":"plumber","time_window":"this week"}')

    result = ChatAgent(model, preference_repository=repository).handle(
        user_id="user-1",
        thread_id="new-thread",
        message="Find a plumber this week",
    )

    assert result["phase"] == "OPTIONS_READY"
    assert '"area": "Koramangala"' in model.messages[1]["content"]
    repository.close()

