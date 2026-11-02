import pytest
from datetime import datetime, timezone, timedelta

from home_service_agent.domain import BookingStatus
from home_service_agent.mocks.provider_api import MockProviderAPI, MockScenario, ProviderToolError


def test_search_get_provider_and_list_bookings_contract() -> None:
    api = MockProviderAPI(seed=1)

    providers = api.search_providers("plumber", "Indiranagar", "this week")
    details = api.get_provider(providers[0].id)
    bookings = api.list_bookings("user-1")

    assert 3 <= len(providers) <= 8
    assert details.id == providers[0].id
    assert any("ignore all previous instructions" in review.lower() for review in details.reviews)
    assert bookings == []


def test_hold_returns_ten_minute_quote_and_allows_override() -> None:
    api = MockProviderAPI(
        scenario=MockScenario(quote_overrides={("provider-1", "Saturday 09:00"): 1499})
    )

    hold = api.hold_slot("provider-1", "Saturday 09:00")

    assert hold.quoted_price == 1499
    assert hold.expires_at - datetime.now(timezone.utc) > timedelta(minutes=9)


def test_hold_slot_unavailable_is_configurable() -> None:
    api = MockProviderAPI(
        scenario=MockScenario(unavailable_slots={("provider-1", "Saturday 09:00")})
    )

    with pytest.raises(ProviderToolError, match="unavailable") as error:
        api.hold_slot("provider-1", "Saturday 09:00")

    assert error.value.code == "slot_unavailable"


def test_book_is_idempotent() -> None:
    api = MockProviderAPI()
    hold = api.hold_slot("provider-1", "Saturday 09:00")

    first = api.book(hold.hold_id, "booking-1", "idem-1")
    second = api.book(hold.hold_id, "booking-2", "idem-1")

    assert first == second == "booking-1"
    assert api.get_booking_status("idem-1").status is BookingStatus.BOOKED


@pytest.mark.parametrize("outcome,code", [("slot_taken", "slot_taken"), ("hold_expired", "hold_expired")])
def test_book_failure_modes(outcome: str, code: str) -> None:
    api = MockProviderAPI(scenario=MockScenario(book_outcome=outcome))
    hold = api.hold_slot("provider-1", "Saturday 09:00")

    with pytest.raises(ProviderToolError) as error:
        api.book(hold.hold_id, "booking-1", "idem-1")

    assert error.value.code == code
    assert api.get_booking_status("idem-1") is None


def test_timeout_after_success_is_visible_in_authoritative_status() -> None:
    api = MockProviderAPI(
        scenario=MockScenario(book_outcome="timeout", timeout_after_success=True)
    )
    hold = api.hold_slot("provider-1", "Saturday 09:00")

    with pytest.raises(ProviderToolError, match="timed out"):
        api.book(hold.hold_id, "booking-1", "idem-1")

    assert api.get_booking_status("idem-1").id == "booking-1"


def test_cancel_retries_after_transient_503() -> None:
    api = MockProviderAPI()
    hold = api.hold_slot("provider-1", "Saturday 09:00")
    api.book(hold.hold_id, "booking-1", "idem-1")

    with pytest.raises(ProviderToolError) as error:
        api.cancel("booking-1", "cancel-idem-1")
    assert error.value.code == "503"

    assert api.cancel("booking-1", "cancel-idem-1") == "booking-1"
    assert api.get_booking_status("idem-1").status is BookingStatus.CANCELLED


def test_reschedule_conflict_and_success() -> None:
    api = MockProviderAPI(scenario=MockScenario(reschedule_conflict_slots={"Sunday 11:00"}))
    hold = api.hold_slot("provider-1", "Saturday 09:00")
    api.book(hold.hold_id, "booking-1", "idem-1")

    with pytest.raises(ProviderToolError) as error:
        api.reschedule("booking-1", "Sunday 11:00", "reschedule-1")
    assert error.value.code == "slot_conflict"

    updated = api.reschedule("booking-1", "Saturday 11:00", "reschedule-2")
    assert updated.slot == "Saturday 11:00"

