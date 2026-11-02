"""LangGraph Server entrypoint for Agent Chat UI.

The exported ``graph`` uses the standard ``messages`` channel expected by the
official Agent Chat UI. Domain workflow state remains owned by ChatAgent and
the existing safety coordinators.
"""

from typing import Annotated, Any, TypedDict
import os

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from home_service_agent.chat import ChatAgent, ChatModel


class ServerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_server_graph(model: ChatModel | None = None):
    agent = ChatAgent(model=model, checkpoint_path=os.getenv("HOME_SERVICE_AGENT_CHECKPOINT_DB"))

    def respond(state: ServerState, config: RunnableConfig) -> dict[str, list[AIMessage]]:
        last_message = state["messages"][-1]
        thread_id = config.get("configurable", {}).get("thread_id", "local-ui-thread")
        user_id = config.get("configurable", {}).get("user_id") or thread_id
        result = agent.handle(
            user_id=user_id,
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

