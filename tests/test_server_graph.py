from langchain_core.messages import HumanMessage

from home_service_agent.server_graph import build_server_graph


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


def test_server_graph_uses_thread_id_when_ui_sends_empty_user_id() -> None:
    graph = build_server_graph(FakeModel())

    result = graph.invoke(
        {"messages": [HumanMessage(content="Find a plumber in Indiranagar this week")]},
        {"configurable": {"thread_id": "empty-user-id-thread", "user_id": ""}},
    )

    assert result["messages"][-1].type == "ai"
    assert "provider options" in result["messages"][-1].content.lower()

