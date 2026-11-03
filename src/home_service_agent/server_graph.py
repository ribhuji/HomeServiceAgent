"""LangGraph Server entrypoint for Agent Chat UI.

The exported ``graph`` uses the standard ``messages`` channel expected by the
official Agent Chat UI. Domain workflow state remains owned by ChatAgent and
the existing safety coordinators.
"""

from typing import Annotated, Any, NotRequired, TypedDict
import os
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from home_service_agent.chat import ChatAgent, ChatModel
from home_service_agent.domain import Provider, WorkflowState
from home_service_agent.limits import ExecutionLimits
from home_service_agent.memory import PreferenceRepository


# This deployment is intentionally single-user. Threads are conversations; this
# stable ID is the owner of long-term preferences across all conversations.
HOME_SERVICE_USER_ID = "home-service-user"


class ServerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # The LangGraph Server thread is the durable conversation boundary for the
    # live UI. Keeping the domain snapshot here avoids nesting a second SQLite
    # checkpointer inside the server worker.
    domain_state: NotRequired[dict[str, Any] | None]


def build_server_graph(
    model: ChatModel | None = None,
    preference_repository: PreferenceRepository | None = None,
):
    if preference_repository is None:
        memory_path = os.getenv("HOME_SERVICE_AGENT_MEMORY_DB", "data/agent.db")
        if memory_path != ":memory:":
            Path(memory_path).parent.mkdir(parents=True, exist_ok=True)
        preference_repository = PreferenceRepository(memory_path)
    agent = ChatAgent(
        model=model,
        # The server runtime already checkpoints ServerState. A nested
        # SqliteSaver is incompatible with the local dev server's serializer.
        checkpoint_path=None,
        preference_repository=preference_repository,
        trace_directory=os.getenv("HOME_SERVICE_AGENT_TRACE_DIR", "data/traces"),
        approval_database=os.getenv("HOME_SERVICE_AGENT_APPROVAL_DB", os.getenv("HOME_SERVICE_AGENT_MEMORY_DB", "data/agent.db")),
        execution_limits=ExecutionLimits(
            max_model_turns=int(os.getenv("HOME_SERVICE_AGENT_MAX_MODEL_TURNS", "3")),
            max_tool_calls=int(os.getenv("HOME_SERVICE_AGENT_MAX_TOOL_CALLS", "12")),
            max_wall_seconds=float(os.getenv("HOME_SERVICE_AGENT_MAX_WALL_SECONDS", "300")),
        ),
    )

    def respond(state: ServerState, config: RunnableConfig) -> dict[str, list[AIMessage]]:
        last_message = state["messages"][-1]
        thread_id = config.get("configurable", {}).get("thread_id", "local-ui-thread")
        agent_config = {"configurable": {"thread_id": thread_id}}
        snapshot = state.get("domain_state")
        if snapshot:
            hydrated: dict[str, Any] = {
                "workflow": WorkflowState.model_validate(snapshot["workflow"]),
                "request_update": None,
            }
            if snapshot.get("providers"):
                hydrated["providers"] = [Provider.model_validate(item) for item in snapshot["providers"]]
            if snapshot.get("user_message"):
                hydrated["user_message"] = snapshot["user_message"]
            agent.graph.update_state(agent_config, hydrated)
        result = agent.handle(
            user_id=HOME_SERVICE_USER_ID,
            thread_id=thread_id,
            message=str(last_message.content),
        )
        values = agent.graph.get_state(agent_config).values
        domain_state = None
        if values:
            domain_state = {
                "workflow": values["workflow"].model_dump(mode="json"),
                "providers": [provider.model_dump(mode="json") for provider in values.get("providers", [])],
                "user_message": values.get("user_message"),
            }
        return {
            "messages": [AIMessage(content=result["message"])],
            "domain_state": domain_state,
        }

    builder = StateGraph(ServerState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


graph = build_server_graph()

