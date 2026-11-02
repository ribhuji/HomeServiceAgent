from langchain_core.messages import HumanMessage

from home_service_agent.memory import PreferenceRepository
from home_service_agent.server_graph import HOME_SERVICE_USER_ID, build_server_graph


class FakeModel:
    def chat(self, messages, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"service":"plumber","area":"Indiranagar","time_window":"this week"}'
                    }
                }
            ]
        }


def test_server_graph_exposes_messages_channel_and_returns_ai_message() -> None:
    graph = build_server_graph(FakeModel())

    result = graph.invoke(
        {"messages": [HumanMessage(content="Find a plumber in Indiranagar this week")]},
        {"configurable": {"thread_id": "server-test-thread", "user_id": "user-1"}},
    )

    assert "messages" in result
    assert result["messages"][-1].type == "ai"
    assert "provider options" in result["messages"][-1].content.lower()


def test_server_graph_works_when_ui_sends_empty_user_id() -> None:
    graph = build_server_graph(FakeModel())

    result = graph.invoke(
        {"messages": [HumanMessage(content="Find a plumber in Indiranagar this week")]},
        {"configurable": {"thread_id": "empty-user-id-thread", "user_id": ""}},
    )

    assert result["messages"][-1].type == "ai"
    assert "provider options" in result["messages"][-1].content.lower()


def test_server_graph_uses_one_fixed_user_for_all_threads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME_SERVICE_AGENT_MEMORY_DB", str(tmp_path / "agent.db"))
    graph = build_server_graph(FakeModel())

    first = graph.invoke(
        {"messages": [HumanMessage(content="Find a plumber in Indiranagar this week")]},
        {"configurable": {"thread_id": "thread-one", "user_id": "someone-else"}},
    )

    assert first["messages"][-1].type == "ai"
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    assert repository.get(HOME_SERVICE_USER_ID, "area") is not None
    assert repository.get("someone-else", "area") is None
    repository.close()

