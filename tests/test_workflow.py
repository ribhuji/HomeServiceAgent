from home_service_agent.domain import ServiceRequest, WorkflowPhase, WorkflowState
from home_service_agent.mocks.provider_api import MockProviderAPI
from home_service_agent.workflow import build_graph


def test_happy_path_reaches_options_ready_without_booking() -> None:
    graph = build_graph(MockProviderAPI(seed=1))
    initial = {
        "workflow": WorkflowState(
            run_id="run-1",
            user_id="user-1",
            request=ServiceRequest(
                service="plumber",
                area="Indiranagar",
                time_window="this week",
                budget=1500,
            ),
        )
    }

    result = graph.invoke(initial, {"configurable": {"thread_id": "run-1"}})

    assert result["workflow"].phase is WorkflowPhase.OPTIONS_READY
    assert len(result["providers"]) == 3
    assert result["workflow"].booking_id is None
    assert "3 provider options" in result["user_message"]


def test_incomplete_request_stops_before_provider_search() -> None:
    graph = build_graph(MockProviderAPI())
    initial = {
        "workflow": WorkflowState(
            run_id="run-2",
            user_id="user-1",
            request=ServiceRequest(service="plumber"),
        )
    }

    result = graph.invoke(initial, {"configurable": {"thread_id": "run-2"}})

    assert result["workflow"].phase is WorkflowPhase.NEEDS_CORRECTION
    assert "area" in result["user_message"]
    assert "time_window" in result["user_message"]
    assert "providers" not in result


def test_graph_contains_future_workflow_nodes() -> None:
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)

    assert {
        "collect_fields",
        "search_providers",
        "present_options",
        "await_approval",
        "hold_slot",
        "commit_booking",
        "verify_booking",
        "terminal_outcome",
    } <= nodes

