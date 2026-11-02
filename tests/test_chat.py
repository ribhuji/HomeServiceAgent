from fastapi.testclient import TestClient

from home_service_agent.app import app
from home_service_agent.chat import ChatAgent, extract_request_update


class FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        return {
            "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}]
        }


def test_chat_endpoint_compatibility_smoke(monkeypatch) -> None:
    monkeypatch.setattr(
        "home_service_agent.app.chat_agent",
        ChatAgent(FakeModel('{"service":"plumber","area":"Indiranagar","time_window":"this week","budget":1500}')),
    )
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"user_id": "user-1", "message": "Find a plumber in Indiranagar this week under 1500"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["phase"] == "OPTIONS_READY"
    assert len(body["providers"]) == 3
    assert body["thread_id"]


def test_model_budget_text_is_normalized_conservatively() -> None:
    update = extract_request_update(
        '{"service":"plumber","budget":"under 1500 rupees","preferences":""}'
    )

    assert update.budget == 1500
    assert update.preferences is None

