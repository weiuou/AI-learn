from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ACTIVE_RUN_STATUSES = ("created", "running", "waiting_user")
TERMINAL_RUN_STATUSES = ("completed", "failed", "cancelled")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    root_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workspaces_user ON workspaces(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('created','running','waiting_user','completed','failed','cancelled')),
    current_step INTEGER NOT NULL DEFAULT 0,
    final_result TEXT,
    error TEXT,
    model_calls INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    apply_status TEXT NOT NULL DEFAULT 'none' CHECK(apply_status IN ('none','pending','applied','discarded')),
    changed_files_json TEXT NOT NULL DEFAULT '[]',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_workspace ON agent_runs(workspace_id, created_at DESC);

CREATE TABLE IF NOT EXISTS run_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    step INTEGER,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON run_events(run_id, sequence);

CREATE TABLE IF NOT EXISTS file_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK(change_type IN ('created','modified','deleted')),
    diff TEXT NOT NULL DEFAULT '',
    before_sha256 TEXT,
    after_sha256 TEXT,
    UNIQUE(run_id, path)
);

CREATE TABLE IF NOT EXISTS feedback (
    run_id TEXT PRIMARY KEY REFERENCES agent_runs(id) ON DELETE CASCADE,
    rating TEXT NOT NULL CHECK(rating IN ('up','down')),
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._write_lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute(sql, params)
            connection.commit()
            return int(cursor.lastrowid or 0)

    def insert_event(self, run_id: str, event_type: str, payload: dict[str, Any], step: int | None = None) -> int:
        now = utc_now()
        sequence = self.execute(
            "INSERT INTO run_events(run_id,type,step,payload_json,created_at) VALUES(?,?,?,?,?)",
            (run_id, event_type, step, json.dumps(payload, ensure_ascii=False), now),
        )
        if step is not None:
            self.execute(
                "UPDATE agent_runs SET current_step=MAX(current_step,?), updated_at=? WHERE id=?",
                (step, now, run_id),
            )
        return sequence

    def active_run_for_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in ACTIVE_RUN_STATUSES)
        return self.fetchone(
            f"SELECT * FROM agent_runs WHERE workspace_id=? AND status IN ({placeholders}) ORDER BY created_at DESC LIMIT 1",
            (workspace_id, *ACTIVE_RUN_STATUSES),
        )

    def recover_interrupted_runs(self) -> list[str]:
        now = utc_now()
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id FROM agent_runs WHERE status IN ('created','running')"
            ).fetchall()
            run_ids = [row["id"] for row in rows]
            connection.execute(
                "UPDATE agent_runs SET status='failed', error='Server restarted while the run was active.', updated_at=?, finished_at=? WHERE status IN ('created','running')",
                (now, now),
            )
            for run_id in run_ids:
                connection.execute(
                    "INSERT INTO run_events(run_id,type,step,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (run_id, "error", None, json.dumps({"message": "Server restarted while the run was active."}), now),
                )
        return run_ids


def decode_json_columns(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    if "changed_files_json" in result:
        result["changed_files"] = json.loads(result.pop("changed_files_json") or "[]")
    if "cancel_requested" in result:
        result["cancel_requested"] = bool(result["cancel_requested"])
    return result
