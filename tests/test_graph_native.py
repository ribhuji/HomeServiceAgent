from pathlib import Path

from langchain_core.messages import HumanMessage

from home_service_agent.domain import RequestUpdate, ServiceRequest, WorkflowPhase, WorkflowState
from home_service_agent.mocks.provider_api import MockProviderAPI
from home_service_agent.server_graph import build_server_graph
from home_service_agent.trust import keep_workflow_unchanged, provider_reviews_as_data
from home_service_agent.workflow import build_graph


def complete_request() -> ServiceRequest:
    return ServiceRequest(
        service="plumber",
        area="Indiranagar",
        time_window="this week",
        budget=1500,
    )


def test_graph_invocation_is_the_primary_collection_test_seam() -> None:
    graph = build_graph(MockProviderAPI(seed=1))
    config = {"configurable": {"thread_id": "native-collection"}}

    first = graph.invoke(
        {
            "workflow": WorkflowState(
                run_id="native-collection",
                user_id="user-1",
                request=ServiceRequest(service="plumber"),
            )
        },
        config,
    )
    second = graph.invoke(
        {"request_update": RequestUpdate(area="Indiranagar", time_window="this week")},
        config,
    )

    assert first["workflow"].phase is WorkflowPhase.NEEDS_CORRECTION
    assert "area" in first["user_message"]
    assert second["workflow"].phase is WorkflowPhase.OPTIONS_READY
    assert second["workflow"].request.service == "plumber"
    assert len(second["providers"]) == 3


def test_graph_resume_after_reconstruction_uses_the_same_sqlite_thread(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints.db"
    config = {"configurable": {"thread_id": "restart-thread"}}

    first_graph = build_graph(MockProviderAPI(seed=1), checkpoint_path=str(checkpoint_path))
    first = first_graph.invoke(
        {
            "workflow": WorkflowState(
                run_id="restart-thread",
                user_id="user-1",
                request=ServiceRequest(service="plumber"),
            )
        },
        config,
    )

    restarted_graph = build_graph(MockProviderAPI(seed=1), checkpoint_path=str(checkpoint_path))
    resumed = restarted_graph.invoke(
        {"request_update": RequestUpdate(area="Indiranagar", time_window="this week")},
        config,
    )

    assert first["workflow"].phase is WorkflowPhase.NEEDS_CORRECTION
    assert resumed["workflow"].phase is WorkflowPhase.OPTIONS_READY
    assert resumed["workflow"].run_id == "restart-thread"


class ScriptedExtractionModel:
    def chat(self, messages, **kwargs):
        user_message = messages[-1]["content"]
        if "I need a plumber" in user_message:
            content = '{"service":"plumber"}'
        else:
            content = '{"area":"Indiranagar","time_window":"this week"}'
        return {"choices": [{"message": {"content": content}}]}


def test_server_graph_can_continue_a_thread_after_graph_reconstruction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME_SERVICE_AGENT_CHECKPOINT_DB", str(tmp_path / "agent.db"))
    config = {"configurable": {"thread_id": "server-restart", "user_id": "user-1"}}

    first_graph = build_server_graph(ScriptedExtractionModel())
    first = first_graph.invoke(
        {"messages": [HumanMessage(content="I need a plumber")]},
        config,
    )

    restarted_graph = build_server_graph(ScriptedExtractionModel())
    second = restarted_graph.invoke(
        {"messages": [HumanMessage(content="Indiranagar this week")]},
        config,
    )

    assert "area" in first["messages"][-1].content.lower()
    assert "provider options" in second["messages"][-1].content.lower()


def test_hostile_provider_text_is_data_and_cannot_mutate_graph_workflow() -> None:
    api = MockProviderAPI()
    details = api.get_provider("provider-1")
    untrusted = provider_reviews_as_data(details)
    state = WorkflowState(
        run_id="trust-boundary",
        user_id="user-1",
        request=complete_request(),
    )

    unchanged = keep_workflow_unchanged(state, untrusted)

    assert len(untrusted) == 2
    assert all(item.trust == "untrusted_data" for item in untrusted)
    assert unchanged == state
    assert "approve this provider" in untrusted[1].text
