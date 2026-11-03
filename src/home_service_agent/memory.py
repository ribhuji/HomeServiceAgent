"""SQLite-backed, provenance-aware user preference memory."""

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from pydantic import Field

from .domain import DomainModel


class PreferenceSource(str):
    STATED = "stated"
    INFERRED = "inferred"


class MemoryPreference(DomainModel):
    user_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: Any
    source: str
    confirmed: bool
    forgotten_at: datetime | None = None
    updated_at: datetime


class PreferenceNotFound(KeyError):
    pass


class PreferenceRepository:
    """Structured preference store with explicit view, correct, and forget APIs."""

    def __init__(self, database: str = ":memory:") -> None:
        # LangGraph executes nodes in worker threads; this repository is owned
        # by one local agent instance and may therefore be used by those threads.
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('stated', 'inferred')),
                confirmed INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
                forgotten_at TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, preference_key)
            )
            """
        )
        self._connection.commit()

    def set(
        self,
        user_id: str,
        key: str,
        value: Any,
        *,
        source: str = PreferenceSource.STATED,
        confirmed: bool | None = None,
    ) -> MemoryPreference:
        if source not in {PreferenceSource.STATED, PreferenceSource.INFERRED}:
            raise ValueError("source must be 'stated' or 'inferred'")
        if confirmed is None:
            confirmed = source == PreferenceSource.STATED
        now = datetime.now(timezone.utc)
        self._connection.execute(
            """
            INSERT INTO user_preferences
                (user_id, preference_key, value_json, source, confirmed, forgotten_at, updated_at)
            VALUES (?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(user_id, preference_key) DO UPDATE SET
                value_json = excluded.value_json,
                source = excluded.source,
                confirmed = excluded.confirmed,
                forgotten_at = NULL,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                key,
                json.dumps(value, sort_keys=True),
                source,
                int(confirmed),
                now.isoformat(),
            ),
        )
        self._connection.commit()
        return self.get(user_id, key, include_unconfirmed=True)  # type: ignore[return-value]

    def confirm(self, user_id: str, key: str) -> MemoryPreference:
        self._require_active_row(user_id, key)
        self._connection.execute(
            "UPDATE user_preferences SET confirmed = 1, updated_at = ? WHERE user_id = ? AND preference_key = ?",
            (datetime.now(timezone.utc).isoformat(), user_id, key),
        )
        self._connection.commit()
        return self.get(user_id, key, include_unconfirmed=True)  # type: ignore[return-value]

    def get(
        self,
        user_id: str,
        key: str,
        *,
        include_unconfirmed: bool = False,
        include_forgotten: bool = False,
    ) -> MemoryPreference | None:
        query = "SELECT * FROM user_preferences WHERE user_id = ? AND preference_key = ?"
        row = self._connection.execute(query, (user_id, key)).fetchone()
        if row is None or (row["forgotten_at"] is not None and not include_forgotten):
            return None
        if not include_unconfirmed and not row["confirmed"]:
            return None
        return self._row_to_preference(row)

    def list(self, user_id: str, *, include_unconfirmed: bool = False) -> list[MemoryPreference]:
        rows = self._connection.execute(
            "SELECT * FROM user_preferences WHERE user_id = ? AND forgotten_at IS NULL ORDER BY preference_key",
            (user_id,),
        ).fetchall()
        return [
            self._row_to_preference(row)
            for row in rows
            if include_unconfirmed or row["confirmed"]
        ]

    def forget(self, user_id: str, key: str) -> None:
        self._require_active_row(user_id, key)
        self._connection.execute(
            "UPDATE user_preferences SET forgotten_at = ?, updated_at = ? WHERE user_id = ? AND preference_key = ?",
            (
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
                user_id,
                key,
            ),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _require_active_row(self, user_id: str, key: str) -> sqlite3.Row:
        row = self._connection.execute(
            "SELECT * FROM user_preferences WHERE user_id = ? AND preference_key = ? AND forgotten_at IS NULL",
            (user_id, key),
        ).fetchone()
        if row is None:
            raise PreferenceNotFound(f"No active preference {key!r} for user {user_id!r}")
        return row

    @staticmethod
    def _row_to_preference(row: sqlite3.Row) -> MemoryPreference:
        return MemoryPreference(
            user_id=row["user_id"],
            key=row["preference_key"],
            value=json.loads(row["value_json"]),
            source=row["source"],
            confirmed=bool(row["confirmed"]),
            forgotten_at=(
                datetime.fromisoformat(row["forgotten_at"])
                if row["forgotten_at"]
                else None
            ),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

