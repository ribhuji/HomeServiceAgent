"""LangGraph workflow skeleton for the Home-Service Agent."""

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .domain import Provider, RequestUpdate, WorkflowPhase, WorkflowState, transition
from .mocks.provider_api import MockProviderAPI


class GraphState(TypedDict):
    workflow: WorkflowState
    providers: NotRequired[list[Provider]]
    user_message: NotRequired[str]
    request_update: NotRequired[RequestUpdate | None]


def collect_fields(state: GraphState) -> dict[str, object]:
    workflow = state["workflow"]
    if workflow.phase is WorkflowPhase.NEEDS_CORRECTION:
        workflow = transition(workflow, WorkflowPhase.COLLECTING)

    request_update = state.get("request_update")
    if request_update is not None:
        workflow = workflow.model_copy(
            update={"request": workflow.request.apply_update(**request_update.supplied_fields())}
        )

    missing = workflow.request.missing_fields()
    if missing:
        updated = transition(workflow, WorkflowPhase.NEEDS_CORRECTION)
        return {
            "workflow": updated,
            "user_message": f"Please provide: {', '.join(missing)}.",
            "request_update": None,
        }

    return {
        "workflow": transition(workflow, WorkflowPhase.SEARCHING),
        "request_update": None,
    }


def route_after_collection(state: GraphState) -> str:
    if state["workflow"].phase is WorkflowPhase.SEARCHING:
        return "search_providers"
    return END


def search_providers(
    state: GraphState,
    *,
    provider_api: MockProviderAPI,
) -> dict[str, object]:
    workflow = state["workflow"]
    request = workflow.request
    providers = provider_api.search_providers(
        service=request.service or "",
        area=request.area or "",
        date_range=request.time_window or "",
    )
    return {
        "workflow": transition(workflow, WorkflowPhase.OPTIONS_READY),
        "providers": providers,
    }


def present_options(state: GraphState) -> dict[str, str]:
    providers = state.get("providers", [])
    return {
        "user_message": f"Found {len(providers)} provider options. Choose one to continue."
    }


def await_approval(state: GraphState) -> GraphState:
    """Placeholder node for milestone 6 approval enforcement."""

    return state


def hold_slot(state: GraphState) -> GraphState:
    """Placeholder node for milestone 7 slot holding."""

    return state


def commit_booking(state: GraphState) -> GraphState:
    """Placeholder node for milestone 8 booking commit/reconciliation."""

    return state


def verify_booking(state: GraphState) -> GraphState:
    """Placeholder node for milestone 8 authoritative verification."""

    return state


def terminal_outcome(state: GraphState) -> GraphState:
    """Placeholder node for bounded terminal outcomes."""

    return state


def build_graph(
    provider_api: MockProviderAPI | None = None,
    *,
    checkpointer: MemorySaver | None = None,
):
    """Build the workflow graph with injectable dependencies for tests."""

    provider_api = provider_api or MockProviderAPI()
    graph = StateGraph(GraphState)
    graph.add_node("collect_fields", collect_fields)
    graph.add_node(
        "search_providers",
        lambda state: search_providers(state, provider_api=provider_api),
    )
    graph.add_node("present_options", present_options)
    graph.add_node("await_approval", await_approval)
    graph.add_node("hold_slot", hold_slot)
    graph.add_node("commit_booking", commit_booking)
    graph.add_node("verify_booking", verify_booking)
    graph.add_node("terminal_outcome", terminal_outcome)

    graph.add_edge(START, "collect_fields")
    graph.add_conditional_edges("collect_fields", route_after_collection)
    graph.add_edge("search_providers", "present_options")
    graph.add_edge("present_options", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())

