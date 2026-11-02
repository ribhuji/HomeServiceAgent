"""Chat entrypoint connecting the live model adapter to LangGraph."""

import json
import re
from typing import Any, Protocol
from uuid import uuid4

from .domain import Provider, RequestUpdate, ServiceRequest, WorkflowPhase, WorkflowState
from .openrouter import OpenRouterClient
from .workflow import build_graph


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...


def extract_request_update(content: str) -> RequestUpdate:
    """Parse model JSON conservatively; invalid output becomes an empty update."""

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return RequestUpdate()
    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            budget = payload.get("budget")
            if isinstance(budget, str):
                amount = re.search(r"\d+(?:\.\d+)?", budget.replace(",", ""))
                payload["budget"] = int(float(amount.group(0))) if amount else None
            if payload.get("preferences") in ("", None):
                payload["preferences"] = None
        return RequestUpdate.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError):
        return RequestUpdate()


class ChatAgent:
    def __init__(self, model: ChatModel | None = None) -> None:
        self.model = model or OpenRouterClient()
        self.graph = build_graph()

    def handle(self, *, user_id: str, message: str, thread_id: str | None = None) -> dict[str, Any]:
        thread_id = thread_id or str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        current = self.graph.get_state(config)
        known_request = None
        if current.values:
            restored_workflow = self._normalize_workflow(current.values["workflow"])
            known_request = restored_workflow.request.model_dump(exclude_none=True)

        prompt = {
            "role": "user",
            "content": (
                "Extract only explicit home-service request fields from this user message. "
                "Return JSON only with optional keys service, area, time_window, budget, "
                "preferences. Do not infer missing values. Known fields are included only "
                f"to avoid asking for them again: {json.dumps(known_request or {}, sort_keys=True)}.\n"
                f"User message: {message}"
            ),
        }
        response = self.model.chat([prompt])
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        update = extract_request_update(content)

        if current.values:
            graph_input: dict[str, Any] = {"request_update": update}
        else:
            graph_input = {
                "workflow": WorkflowState(
                    run_id=thread_id,
                    user_id=user_id,
                    request=ServiceRequest(),
                ),
                "request_update": update,
            }
        result = self.graph.invoke(graph_input, config)
        workflow = self._normalize_workflow(result["workflow"])
        providers = result.get("providers", [])
        return {
            "thread_id": thread_id,
            "phase": workflow.phase.value,
            "message": result.get("user_message") or self._default_message(workflow.phase),
            "providers": [provider.model_dump(mode="json") for provider in providers],
        }

    @staticmethod
    def _normalize_workflow(value: Any) -> WorkflowState:
        raw = value.model_dump(mode="python") if hasattr(value, "model_dump") else value
        return WorkflowState.model_validate(raw)

    @staticmethod
    def _default_message(phase: WorkflowPhase) -> str:
        if phase is WorkflowPhase.OPTIONS_READY:
            return "I found provider options."
        if phase is WorkflowPhase.NEEDS_CORRECTION:
            return "I need a little more information before searching."
        return "I’m processing your request."

