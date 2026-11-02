"""Runtime storage bootstrap for local development."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from .memory import PreferenceRepository


@dataclass(frozen=True)
class StoragePaths:
    data_dir: Path
    agent_db: Path
    checkpoints_db: Path


def initialize_storage(data_dir: str | Path = "data") -> StoragePaths:
    """Create local database files and initialize their schemas."""

    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = StoragePaths(
        data_dir=directory,
        agent_db=directory / "agent.db",
        checkpoints_db=directory / "checkpoints.db",
    )

    preferences = PreferenceRepository(str(paths.agent_db))
    preferences.close()

    connection = sqlite3.connect(str(paths.checkpoints_db), check_same_thread=False)
    try:
        SqliteSaver(connection).setup()
    finally:
        connection.close()
    return paths

