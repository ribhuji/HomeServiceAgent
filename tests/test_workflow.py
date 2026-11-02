from home_service_agent.domain import RequestUpdate, ServiceRequest, WorkflowPhase, WorkflowState
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


def test_correction_resumes_same_thread_and_asks_only_for_remaining_fields() -> None:
    graph = build_graph(MockProviderAPI())
    config = {"configurable": {"thread_id": "run-3"}}
    initial = {
        "workflow": WorkflowState(
            run_id="run-3",
            user_id="user-1",
            request=ServiceRequest(service="plumber"),
        )
    }

    first = graph.invoke(initial, config)
    second = graph.invoke(
        {"request_update": RequestUpdate(area="Indiranagar")},
        config,
    )

    assert first["workflow"].phase is WorkflowPhase.NEEDS_CORRECTION
    assert second["workflow"].phase is WorkflowPhase.NEEDS_CORRECTION
    assert "time_window" in second["user_message"]
    assert "area" not in second["user_message"]
    assert second["workflow"].request.service == "plumber"
    assert second["workflow"].request.area == "Indiranagar"


def test_complete_correction_reaches_options_without_reasking_known_facts() -> None:
    graph = build_graph(MockProviderAPI())
    config = {"configurable": {"thread_id": "run-4"}}
    graph.invoke(
        {
            "workflow": WorkflowState(
                run_id="run-4",
                user_id="user-1",
                request=ServiceRequest(service="plumber"),
            )
        },
        config,
    )

    result = graph.invoke(
        {
            "request_update": RequestUpdate(
                area="Indiranagar",
                time_window="this week",
                budget=1500,
            )
        },
        config,
    )

    assert result["workflow"].phase is WorkflowPhase.OPTIONS_READY
    assert result["workflow"].request.service == "plumber"
    assert result["workflow"].request.area == "Indiranagar"
    assert result["workflow"].request.budget == 1500
    assert len(result["providers"]) == 3


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

