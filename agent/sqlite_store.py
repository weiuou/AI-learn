from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable

from .store import TRACE_SCHEMA_VERSION, make_event, timestamp_now


SQLITE_SCHEMA_VERSION = "durable-runtime-v1"


class SQLiteRunStore:
    def __init__(
        self,
        db_path: str | Path = "runs/agent.db",
        failure_injector: Callable[[str], None] | None = None,
    ):
        self.db_path = Path(db_path)
        self.failure_injector = failure_injector
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs(
                  task_id TEXT PRIMARY KEY,
                  user_goal TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  schema_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segments(
                  segment_id TEXT PRIMARY KEY,
                  task_id TEXT NOT NULL REFERENCES runs(task_id) ON DELETE CASCADE,
                  kind TEXT NOT NULL,
                  started_at TEXT NOT NULL,
                  finished_at TEXT,
                  exit_reason TEXT
                );
                CREATE TABLE IF NOT EXISTS events(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  task_id TEXT NOT NULL REFERENCES runs(task_id) ON DELETE CASCADE,
                  segment_id TEXT REFERENCES segments(segment_id) ON DELETE SET NULL,
                  sequence_no INTEGER NOT NULL,
                  event_type TEXT NOT NULL,
                  step INTEGER,
                  timestamp TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  UNIQUE(task_id, sequence_no)
                );
                CREATE TABLE IF NOT EXISTS checkpoints(
                  task_id TEXT PRIMARY KEY REFERENCES runs(task_id) ON DELETE CASCADE,
                  step INTEGER NOT NULL,
                  state_json TEXT NOT NULL,
                  context_pack TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_segments_task_started
                  ON segments(task_id, started_at);
                CREATE INDEX IF NOT EXISTS idx_events_task_sequence
                  ON events(task_id, sequence_no);
                """
            )

    @staticmethod
    def _active_segment(connection: sqlite3.Connection, task_id: str) -> str | None:
        row = connection.execute(
            """SELECT segment_id FROM segments
               WHERE task_id=? AND finished_at IS NULL
               ORDER BY started_at DESC LIMIT 1""",
            (task_id,),
        ).fetchone()
        return row["segment_id"] if row else None

    @staticmethod
    def _insert_event(connection: sqlite3.Connection, task_id: str, event: dict) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_no FROM events WHERE task_id=?",
            (task_id,),
        ).fetchone()
        sequence_no = int(row["next_no"])
        connection.execute(
            """INSERT INTO events(
                 task_id, segment_id, sequence_no, event_type, step, timestamp, payload_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                SQLiteRunStore._active_segment(connection, task_id),
                sequence_no,
                event.get("event_type") or event.get("type"),
                event.get("step"),
                event.get("timestamp") or timestamp_now(),
                json.dumps(event, ensure_ascii=False),
            ),
        )
        return sequence_no

    def _inject(self, stage: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(stage)

    def create_run(self, task_id: str, user_goal: str) -> None:
        created_at = timestamp_now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO runs(task_id, user_goal, status, created_at, updated_at, schema_version)
                       VALUES (?, ?, 'running', ?, ?, ?)""",
                    (task_id, user_goal, created_at, created_at, SQLITE_SCHEMA_VERSION),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Run already exists: {task_id}") from error

    def start_segment(self, task_id: str, segment_id: str, kind: str) -> None:
        if kind not in {"task", "resume", "recovery"}:
            raise ValueError(f"Unsupported segment kind: {kind}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._active_segment(connection, task_id):
                raise ValueError(f"Run already has an open segment: {task_id}")
            try:
                connection.execute(
                    """INSERT INTO segments(segment_id, task_id, kind, started_at)
                       VALUES (?, ?, ?, ?)""",
                    (segment_id, task_id, kind, timestamp_now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"Cannot start segment {segment_id} for run {task_id}") from error
            connection.execute(
                "UPDATE runs SET status='running', updated_at=? WHERE task_id=?",
                (timestamp_now(), task_id),
            )

    def append_event(self, task_id: str, event: dict) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_event(connection, task_id, event)
            connection.execute("UPDATE runs SET updated_at=? WHERE task_id=?", (timestamp_now(), task_id))

    def save_checkpoint(self, task_id: str, state: dict, context_pack: str, step: int) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_event(connection, task_id, make_event("checkpoint_started", {"task_id": task_id}, step))
            self._inject("after_checkpoint_started")
            updated_at = timestamp_now()
            connection.execute(
                """INSERT INTO checkpoints(task_id, step, state_json, context_pack, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(task_id) DO UPDATE SET
                     step=excluded.step,
                     state_json=excluded.state_json,
                     context_pack=excluded.context_pack,
                     updated_at=excluded.updated_at""",
                (task_id, step, json.dumps(state, ensure_ascii=False), context_pack, updated_at),
            )
            self._inject("after_checkpoint_snapshot")
            connection.execute(
                "UPDATE runs SET status='running', updated_at=? WHERE task_id=?",
                (updated_at, task_id),
            )
            self._inject("after_run_update")
            self._insert_event(
                connection,
                task_id,
                make_event(
                    "checkpoint_saved",
                    {
                        "task_id": task_id,
                        "token_estimate": max(1, len(context_pack) // 4),
                    },
                    step,
                ),
            )
            self._inject("after_checkpoint_saved")

    def begin_recovery(
        self,
        task_id: str,
        previous_segment_id: str,
        recovery_segment_id: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._active_segment(connection, task_id) != previous_segment_id:
                raise ValueError(f"Segment is not the active segment for run {task_id}: {previous_segment_id}")
            checkpoint = connection.execute(
                "SELECT step FROM checkpoints WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if checkpoint is None:
                raise FileNotFoundError(f"Missing checkpoint for run: {task_id}")

            changed_at = timestamp_now()
            cursor = connection.execute(
                """UPDATE segments SET finished_at=?, exit_reason='crashed'
                   WHERE task_id=? AND segment_id=? AND finished_at IS NULL""",
                (changed_at, task_id, previous_segment_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Unknown or finished segment: {previous_segment_id}")
            self._inject("after_previous_segment_crashed")

            try:
                connection.execute(
                    """INSERT INTO segments(segment_id, task_id, kind, started_at)
                       VALUES (?, ?, 'recovery', ?)""",
                    (recovery_segment_id, task_id, changed_at),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"Cannot start recovery segment: {recovery_segment_id}") from error
            self._inject("after_recovery_segment_created")

            connection.execute(
                "UPDATE runs SET status='running', updated_at=? WHERE task_id=?",
                (changed_at, task_id),
            )
            self._inject("after_recovery_run_updated")

            self._insert_event(
                connection,
                task_id,
                make_event(
                    "recovery_started",
                    {
                        "task_id": task_id,
                        "previous_segment_id": previous_segment_id,
                        "segment_id": recovery_segment_id,
                        "checkpoint_step": checkpoint["step"],
                    },
                ),
            )
            self._inject("after_recovery_started_event")

    def load_run(self, task_id: str) -> dict:
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE task_id=?", (task_id,)).fetchone()
            if run is None:
                raise FileNotFoundError(f"Unknown run: {task_id}")
            segments = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM segments WHERE task_id=? ORDER BY started_at, rowid",
                    (task_id,),
                )
            ]
            event_rows = list(
                connection.execute(
                    "SELECT sequence_no, payload_json FROM events WHERE task_id=? ORDER BY sequence_no",
                    (task_id,),
                )
            )
            events = [json.loads(row["payload_json"]) for row in event_rows]
            checkpoint_row = connection.execute(
                "SELECT step, state_json, context_pack FROM checkpoints WHERE task_id=?",
                (task_id,),
            ).fetchone()
        checkpoint = None
        if checkpoint_row is not None:
            checkpoint = {
                "state": json.loads(checkpoint_row["state_json"]),
                "context_pack": checkpoint_row["context_pack"],
                "step": checkpoint_row["step"],
            }
        finished_at = segments[-1]["finished_at"] if segments and not any(
            item["finished_at"] is None for item in segments
        ) else None
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "task_id": task_id,
            "task": run["user_goal"],
            "user_goal": run["user_goal"],
            "started_at": run["created_at"],
            "finished_at": finished_at,
            "events": events,
        }
        return {
            "task_id": task_id,
            "user_goal": run["user_goal"],
            "status": run["status"],
            "schema_version": run["schema_version"],
            "segments": segments,
            "events": events,
            "checkpoint": checkpoint,
            "trace": trace,
        }

    def finish_segment(self, task_id: str, segment_id: str, exit_reason: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            finished_at = timestamp_now()
            cursor = connection.execute(
                """UPDATE segments SET finished_at=?, exit_reason=?
                   WHERE task_id=? AND segment_id=? AND finished_at IS NULL""",
                (finished_at, exit_reason, task_id, segment_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Unknown or finished segment: {segment_id}")
            status = "crashed" if exit_reason == "crashed" else "finished"
            connection.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE task_id=?",
                (status, finished_at, task_id),
            )
