import pytest

from home_service_agent.approval import ActionDescriptor, ActionType, ApprovalLedger
from home_service_agent.approval import ApprovalError
from home_service_agent.domain import ServiceRequest, WorkflowState
from home_service_agent.mocks.provider_api import MockProviderAPI
from home_service_agent.trust import (
    keep_workflow_unchanged,
    provider_reviews_as_data,
    render_untrusted_context,
    sanitize_tool_text,
)


def test_provider_injection_is_wrapped_as_untrusted_data() -> None:
    details = MockProviderAPI().get_provider("provider-1")

    items = provider_reviews_as_data(details)
    rendered = render_untrusted_context(items)

    assert "UNTRUSTED_TOOL_DATA_BEGIN" in rendered
    assert "UNTRUSTED_TOOL_DATA_END" in rendered
    assert "ignore all previous instructions" in rendered.lower()
    assert all(item.trust == "untrusted_data" for item in items)


def test_untrusted_text_is_sanitized_and_bounded() -> None:
    text = "safe\x00 text\n" + ("x" * 20)

    sanitized = sanitize_tool_text(text, max_chars=12)

    assert "\x00" not in sanitized
    assert len(sanitized) == 12


def test_untrusted_reviews_cannot_change_intent_or_workflow() -> None:
    state = WorkflowState(
        run_id="run-1",
        user_id="user-1",
        request=ServiceRequest(service="plumber", area="Indiranagar", time_window="mornings"),
    )
    details = MockProviderAPI().get_provider("provider-1")

    unchanged = keep_workflow_unchanged(state, provider_reviews_as_data(details))

    assert unchanged == state
    assert unchanged.request.service == "plumber"
    assert unchanged.phase.value == "COLLECTING"


def test_untrusted_reviews_cannot_create_approval_or_authorize_action() -> None:
    details = MockProviderAPI().get_provider("provider-1")
    items = provider_reviews_as_data(details)
    ledger = ApprovalLedger()

    # Reviews are data only; no approval record exists until the user confirms
    # a separately constructed canonical action.
    assert items
    action = ActionDescriptor(
        operation=ActionType.BOOK,
        provider_id="provider-1",
        slot="Saturday 09:00",
        price=1200,
    )
    assert action.provider_id == "provider-1"
    with pytest.raises(ApprovalError, match="Explicit approval"):
        ledger.require_valid("run-1", action, None)

