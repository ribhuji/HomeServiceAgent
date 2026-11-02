from pathlib import Path

import pytest

from home_service_agent.memory import PreferenceNotFound, PreferenceRepository, PreferenceSource


def test_stated_preference_persists_across_repository_instances(tmp_path: Path) -> None:
    database = tmp_path / "agent.db"
    first = PreferenceRepository(str(database))
    first.set("user-1", "area", "Indiranagar", source=PreferenceSource.STATED)
    first.close()

    second = PreferenceRepository(str(database))
    preference = second.get("user-1", "area")

    assert preference is not None
    assert preference.value == "Indiranagar"
    assert preference.source == PreferenceSource.STATED
    assert preference.confirmed is True
    second.close()


def test_correction_updates_only_the_selected_preference(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    repository.set("user-1", "area", "Indiranagar")
    repository.set("user-1", "earliest_time", "10:00")

    repository.set("user-1", "area", "Koramangala")

    values = {item.key: item.value for item in repository.list("user-1")}
    assert values == {"area": "Koramangala", "earliest_time": "10:00"}
    repository.close()


def test_inferred_preference_is_not_reused_until_confirmed(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    repository.set(
        "user-1",
        "preferred_technician",
        "tech-7",
        source=PreferenceSource.INFERRED,
    )

    assert repository.get("user-1", "preferred_technician") is None
    assert len(repository.list("user-1")) == 0
    assert repository.get("user-1", "preferred_technician", include_unconfirmed=True) is not None

    confirmed = repository.confirm("user-1", "preferred_technician")

    assert confirmed.confirmed is True
    assert repository.get("user-1", "preferred_technician").value == "tech-7"
    repository.close()


def test_forget_tombstones_preference_and_prevents_reuse(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))
    repository.set("user-1", "area", "Indiranagar")

    repository.forget("user-1", "area")

    assert repository.get("user-1", "area") is None
    assert repository.list("user-1") == []
    forgotten = repository.get("user-1", "area", include_forgotten=True)
    assert forgotten is not None
    assert forgotten.forgotten_at is not None
    repository.close()


def test_forget_and_confirm_require_an_active_preference(tmp_path: Path) -> None:
    repository = PreferenceRepository(str(tmp_path / "agent.db"))

    with pytest.raises(PreferenceNotFound):
        repository.forget("user-1", "area")
    with pytest.raises(PreferenceNotFound):
        repository.confirm("user-1", "area")
    repository.close()

