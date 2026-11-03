"""Boundaries for untrusted provider/tool text.

Free text is represented as data, never as instructions or a state update.
"""

from typing import Literal
import json
import re

from .domain import DomainModel, ProviderDetails, WorkflowState


class UntrustedToolText(DomainModel):
    source: str
    text: str
    trust: Literal["untrusted_data"] = "untrusted_data"

    def as_model_data(self) -> dict[str, str]:
        """Return data for a model message; this is never a system instruction."""

        return {"source": self.source, "trust": self.trust, "text": self.text}


def sanitize_tool_text(text: str, *, max_chars: int = 4000) -> str:
    """Remove dangerous control characters and cap untrusted text size."""

    cleaned = "".join(character for character in text if character in "\n\t" or ord(character) >= 32)
    return cleaned[:max_chars]


def provider_reviews_as_data(details: ProviderDetails) -> list[UntrustedToolText]:
    return [
        UntrustedToolText(
            source=f"provider:{details.id}:review:{index}",
            text=sanitize_tool_text(review),
        )
        for index, review in enumerate(details.reviews)
    ]


_INJECTION_HINTS = (
    r"ignore\s+(?:all\s+)?previous instructions",
    r"reveal\s+(?:the|your)\s+(?:system\s+)?prompt",
    r"override\s+(?:the|your)\s+safety",
    r"approve\s+(?:this|immediately)",
    r"call\s+(?:this|the)\s+tool",
)


def injection_hints(items: list[UntrustedToolText]) -> list[str]:
    """Return non-blocking flags for suspicious free text.

    Detection is observability only. It never blocks a provider or changes the
    workflow, because structured permissions and approvals are the controls.
    """

    flags: list[str] = []
    for item in items:
        if any(re.search(pattern, item.text, flags=re.IGNORECASE) for pattern in _INJECTION_HINTS):
            flags.append(item.source)
    return flags


def render_untrusted_context(items: list[UntrustedToolText]) -> str:
    """Render a visibly delimited data block for model context."""

    payload = [item.as_model_data() for item in items]
    return (
        "UNTRUSTED_TOOL_DATA_BEGIN\n"
        + json.dumps(payload, ensure_ascii=True, sort_keys=True)
        + "\nUNTRUSTED_TOOL_DATA_END"
    )


def keep_workflow_unchanged(state: WorkflowState, _items: list[UntrustedToolText]) -> WorkflowState:
    """Make the trust rule explicit: free text cannot mutate workflow state."""

    return state

