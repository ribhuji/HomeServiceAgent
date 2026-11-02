from pathlib import Path

from start import build_commands, prepare_environment
from home_service_agent.storage import StoragePaths


def test_start_commands_use_the_simplified_ui_workspaces() -> None:
    commands = build_commands("npm")

    assert commands["langgraph"] == ["npm", "run", "dev", "--workspace", "langgraph-server"]
    assert commands["ui"] == ["npm", "run", "dev", "--workspace", "web"]
    assert commands["api"][0].endswith("python")


def test_child_environment_contains_only_local_runtime_configuration() -> None:
    paths = StoragePaths(Path("data"), Path("data/agent.db"), Path("data/checkpoints.db"))

    environment = prepare_environment(paths)

    assert environment["HOME_SERVICE_AGENT_CHECKPOINT_DB"] == "data/checkpoints.db"
    assert environment["NEXT_PUBLIC_API_URL"] == "http://127.0.0.1:2024"
    assert environment["NEXT_PUBLIC_ASSISTANT_ID"] == "home_service_agent"
