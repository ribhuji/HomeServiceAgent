"""LangGraph Server entrypoint for Agent Chat UI.

The exported ``graph`` uses the standard ``messages`` channel expected by the
official Agent Chat UI. Domain workflow state remains owned by ChatAgent and
the existing safety coordinators.
"""

from typing import Annotated, Any, TypedDict
import os
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from home_service_agent.chat import ChatAgent, ChatModel
from home_service_agent.limits import ExecutionLimits
from home_service_agent.memory import PreferenceRepository


# This deployment is intentionally single-user. Threads are conversations; this
# stable ID is the owner of long-term preferences across all conversations.
HOME_SERVICE_USER_ID = "home-service-user"


class ServerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


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
        checkpoint_path=os.getenv("HOME_SERVICE_AGENT_CHECKPOINT_DB"),
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
        result = agent.handle(
            user_id=HOME_SERVICE_USER_ID,
            thread_id=thread_id,
            message=str(last_message.content),
        )
        return {"messages": [AIMessage(content=result["message"])]}

    builder = StateGraph(ServerState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


graph = build_server_graph()

