from pathlib import Path

from home_service_agent.chat import ChatAgent, extract_request_update
from home_service_agent.memory import PreferenceRepository
from home_service_agent.provider_tools import PROVIDER_TOOL_DEFINITIONS
from home_service_agent.limits import ExecutionLimits


class ToolCallingModel:
    def __init__(self, *, area="Indiranagar") -> None:
        self.calls: list[tuple[list[dict], dict]] = []
        self.area = area

    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append((messages, kwargs))
        if len(self.calls) == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "search_providers",
                    "arguments": (
                        '{"service":"plumber","area":"%s","time_window":"this week",'
                        '"budget":1500,"preferences":{"time":"mornings"}}'
                    ) % self.area,
                },
            }]}}]}
        return {"choices": [{"message": {"content": "I found provider options."}}]}


class ClarifyingModel:
    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        return {"choices": [{"message": {"content": "Which neighborhood should I search?"}}]}


class AlwaysSearchModel(ToolCallingModel):
    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append((messages, kwargs))
        return {
            "choices": [{"message": {"tool_calls": [{
                "id": f"call-{len(self.calls)}",
                "type": "function",
                "function": {
                    "name": "search_providers",
                    "arguments": '{"service":"plumber","area":"Indiranagar","time_window":"this week"}',
                },
            }]}}]
        }


class HoldRequestModel(ToolCallingModel):
    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append((messages, kwargs))
        if len(self.calls) == 1:
            return {"choices": [{"message": {"tool_calls": [{
                "id": "hold-1",
                "type": "function",
                "function": {
                    "name": "search_providers",
                    "arguments": '{"service":"plumber","area":"Indiranagar","time_window":"this week"}',
                },
            }]}}]}
        if len(self.calls) == 2:
            return {"choices": [{"message": {"tool_calls": [{
                "id": "hold-2",
                "type": "function",
                "function": {
                    "name": "request_slot_hold",
                    "arguments": '{"provider_id":"provider-1","slot":"Saturday 09:00","price":1200}',
                },
            }]}}]}
        return {"choices": [{"message": {"content": "Please confirm the exact slot and price."}}]}


class SundayCorrectionModel(ToolCallingModel):
    def chat(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append((messages, kwargs))
        if len(self.calls) == 1:
            return {"choices": [{"message": {"tool_calls": [{
                "id": "search-1", "type": "function", "function": {
                    "name": "search_providers",
                    "arguments": '{"service":"plumber","area":"Indiranagar","time_window":"Saturday","budget":1500}',
                },
            }]}}]}
        if len(self.calls) == 2:
            return {"choices": [{"message": {"content": "Saturday options found."}}]}
        if len(self.calls) == 3:
            return {"choices": [{"message": {"tool_calls": [{
                "id": "search-2", "type": "function", "function": {
                    "name": "search_providers",
                    "arguments": '{"service":"plumber","area":"Indiranagar","time_window":"Sunday","budget":1500}',
                },
            }]}}]}
        return {"choices": [{"message": {"content": "Sunday options found."}}]}


def test_agent_lets_model_choose_a_tool_and_executes_it() -> None:
    model = ToolCallingModel()

    result = ChatAgent(model).handle(
        user_id="user-1",
        message="Find a plumber in Indiranagar this week",
    )

    assert result["phase"] == "OPTIONS_READY"
    assert len(result["providers"]) == 3
    assert {tool["function"]["name"] for tool in model.calls[0][1]["tools"]} == {
        "search_providers", "get_provider", "list_bookings",
        "get_memory", "set_memory", "forget_memory",
    }
    assert model.calls[1][0][-1]["role"] == "tool"


def test_agent_asks_for_missing_information_without_calling_a_tool() -> None:
    result = ChatAgent(ClarifyingModel()).handle(
        user_id="user-1",
        message="Find a plumber this week",
    )

    assert result["phase"] == "NEEDS_CORRECTION"
    assert result["message"] == "Which neighborhood should I search?"


def test_live_agent_stops_at_model_turn_limit_with_terminal_message() -> None:
    agent = ChatAgent(
        AlwaysSearchModel(),
        execution_limits=ExecutionLimits(max_model_turns=1, max_tool_calls=12, max_wall_seconds=300),
    )

    result = agent.handle(
        user_id="user-1",
        thread_id="bounded-chat",
        message="Find a plumber in Indiranagar this week",
    )

    assert result["phase"] == "ABORTED_LIMIT"
    assert "limit" in result["message"]


def test_live_provider_reviews_are_marked_untrusted_data() -> None:
    class ReviewModel:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []

        def chat(self, messages: list[dict], **kwargs: object) -> dict:
            self.calls.append(messages)
            if len(self.calls) == 1:
                return {"choices": [{"message": {"tool_calls": [{
                    "id": "provider-details",
                    "type": "function",
                    "function": {"name": "get_provider", "arguments": '{"provider_id":"provider-1"}'},
                }]}}]}
            return {"choices": [{"message": {"content": "The provider details are available."}}]}

    model = ReviewModel()
    ChatAgent(model).handle(user_id="user-1", message="Tell me about provider-1")

    tool_content = model.calls[1][-1]["content"]
    assert '"trust": "untrusted_data"' in tool_content
    assert '"trust_boundary"' in tool_content


def test_hold_requires_explicit_confirmation_before_provider_call() -> None:
    from home_service_agent.mocks.provider_api import MockProviderAPI

    api = MockProviderAPI()
    agent = ChatAgent(HoldRequestModel(), provider_api=api)
    thread_id = "approval-thread"

    pending = agent.handle(user_id="user-1", thread_id=thread_id, message="Find a plumber in Indiranagar this week")
    assert pending["phase"] == "AWAITING_APPROVAL"
    assert api.book_call_count == 0

    held = agent.handle(user_id="user-1", thread_id=thread_id, message="yes")
    assert held["phase"] == "HOLDING"
    assert "Shall I book it?" in held["message"]
    assert api.book_call_count == 0

    booked = agent.handle(user_id="user-1", thread_id=thread_id, message="yes")
    assert booked["phase"] == "COMPLETED"
    assert api.book_call_count == 1


def test_search_tool_is_not_reused_after_options_are_ready() -> None:
    model = AlwaysSearchModel()
    agent = ChatAgent(model)

    agent.handle(user_id="user-1", thread_id="hold-thread", message="Find a plumber in Indiranagar this week")
    result = agent.handle(user_id="user-1", thread_id="hold-thread", message="Hold Saturday morning")

    assert result["message"] == "I already have provider options for this request. Please choose a provider and slot, or ask for a different day or time."


def test_slot_selection_uses_existing_options_instead_of_searching_again() -> None:
    model = ToolCallingModel()
    agent = ChatAgent(model)
    thread_id = "slot-selection-thread"
    first = agent.handle(user_id="user-1", thread_id=thread_id, message="Find a plumber in Indiranagar this week")
    assert first["phase"] == "OPTIONS_READY"

    selected = agent.handle(user_id="user-1", thread_id=thread_id, message="Saturday 11 works")

    assert selected["phase"] == "AWAITING_APPROVAL"
    assert "Saturday 11:00" in selected["message"]
    assert len(model.calls) == 2


def test_new_day_request_researches_from_options_ready() -> None:
    model = SundayCorrectionModel()
    agent = ChatAgent(model)
    thread_id = "search-correction-thread"

    first = agent.handle(user_id="user-1", thread_id=thread_id, message="Find a plumber on Saturday")
    assert first["phase"] == "OPTIONS_READY"

    second = agent.handle(user_id="user-1", thread_id=thread_id, message="No, check Sunday instead")

    assert second["phase"] == "OPTIONS_READY"
    assert len(second["providers"]) == 3
    assert model.calls[2][1]["tools"]


def test_model_budget_text_is_normalized_conservatively() -> None:
    update = extract_request_update(
        '{"service":"plumber","budget":"under 1500 rupees","preferences":""}'
    )

    assert update.budget == 1500
    assert update.preferences is None


def test_live_chat_saves_explicit_area_and_preferences(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    model = ToolCallingModel(area="Koramangala")

    ChatAgent(model, preference_repository=repository).handle(
        user_id="user-1",
        message="Find a plumber in Koramangala this week; mornings only",
    )

    assert repository.get("user-1", "area").value == "Koramangala"
    assert repository.get("user-1", "preferences").value == {"time": "mornings"}
    repository.close()


def test_live_chat_reuses_confirmed_memory_for_a_new_thread(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    repository.set("user-1", "area", "Koramangala")
    model = ToolCallingModel(area="Koramangala")

    result = ChatAgent(model, preference_repository=repository).handle(
        user_id="user-1",
        thread_id="new-thread",
        message="Find a plumber this week",
    )

    assert result["phase"] == "OPTIONS_READY"
    assert repository.get("user-1", "area").value == "Koramangala"
    repository.close()


def test_live_chat_loads_all_stable_preferences_across_sessions(tmp_path: Path) -> None:
    class CaptureModel:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def chat(self, messages: list[dict], **kwargs: object) -> dict:
            self.messages = messages
            return {"choices": [{"message": {"content": "I need more information."}}]}

    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    repository.set("user-1", "area", "Koramangala")
    repository.set("user-1", "earliest_time", "10:00")
    repository.set("user-1", "preferred_technician", "tech-7")
    model = CaptureModel()

    ChatAgent(model, preference_repository=repository).handle(
        user_id="user-1",
        thread_id="memory-thread",
        message="Find a plumber this week",
    )

    context = model.messages[1]["content"]
    assert '"area": "Koramangala"' in context
    assert '"earliest_time": "10:00"' in context
    assert '"preferred_technician": "tech-7"' in context
    repository.close()


def test_memory_operations_are_model_tool_calls(tmp_path: Path) -> None:
    class MemoryModel:
        def __init__(self, call: dict) -> None:
            self.call = call
            self.messages: list[dict] = []

        def chat(self, messages: list[dict], **kwargs: object) -> dict:
            self.messages = messages
            if len(messages) == 2:
                return {"choices": [{"message": {"tool_calls": [self.call]}}]}
            return {"choices": [{"message": {"content": "Memory operation complete."}}]}

    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    repository.set("user-1", "area", "Indiranagar")

    get_model = MemoryModel({
        "id": "memory-get", "type": "function",
        "function": {"name": "get_memory", "arguments": "{}"},
    })
    ChatAgent(get_model, preference_repository=repository).handle(
        user_id="user-1", message="What do you remember about me?"
    )
    assert '"key": "area"' in get_model.messages[-1]["content"]
    assert '"value": "Indiranagar"' in get_model.messages[-1]["content"]

    set_model = MemoryModel({
        "id": "memory-set", "type": "function",
        "function": {"name": "set_memory", "arguments": '{"key":"earliest_time","value":"10:00","source":"stated"}'},
    })
    ChatAgent(set_model, preference_repository=repository).handle(
        user_id="user-1", message="Remember that I prefer appointments after 10:00."
    )
    assert repository.get("user-1", "earliest_time").value == "10:00"

    forget_model = MemoryModel({
        "id": "memory-forget", "type": "function",
        "function": {"name": "forget_memory", "arguments": '{"key":"earliest_time"}'},
    })
    ChatAgent(forget_model, preference_repository=repository).handle(
        user_id="user-1", message="Forget my earliest time preference."
    )
    assert repository.get("user-1", "earliest_time") is None
    repository.close()
