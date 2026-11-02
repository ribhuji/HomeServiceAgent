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


def test_extracts_request_and_returns_provider_options(monkeypatch) -> None:
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


def test_chat_endpoint_resumes_thread_and_asks_only_for_remaining_fields(monkeypatch) -> None:
    agent = ChatAgent(FakeModel('{"service":"plumber"}'))
    monkeypatch.setattr("home_service_agent.app.chat_agent", agent)
    client = TestClient(app)

    first = client.post(
        "/chat",
        json={"user_id": "user-1", "thread_id": "thread-1", "message": "I need a plumber"},
    )
    assert first.json()["phase"] == "NEEDS_CORRECTION"

    agent.model = FakeModel('{"area":"Indiranagar","time_window":"mornings"}')
    second = client.post(
        "/chat",
        json={"user_id": "user-1", "thread_id": "thread-1", "message": "Indiranagar, mornings"},
    )

    assert second.status_code == 200
    assert second.json()["phase"] == "OPTIONS_READY"


def test_invalid_model_json_does_not_create_inferred_fields(monkeypatch) -> None:
    monkeypatch.setattr("home_service_agent.app.chat_agent", ChatAgent(FakeModel("not json")))
    client = TestClient(app)

    response = client.post("/chat", json={"user_id": "user-1", "message": "hello"})

    assert response.status_code == 200
    assert response.json()["phase"] == "NEEDS_CORRECTION"


def test_model_budget_text_is_normalized_conservatively() -> None:
    update = extract_request_update(
        '{"service":"plumber","budget":"under 1500 rupees","preferences":""}'
    )

    assert update.budget == 1500
    assert update.preferences is None

