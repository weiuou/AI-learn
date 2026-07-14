from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agent.budget import RunBudget
from agent.core import add_event, new_trace, run_agent, save_trace, summarize_usage

from .config import Settings
from .db import Database, utc_now
from .docker_sandbox import DockerSandbox, DockerSandboxFactory
from .filesystem import WorkspaceFilesystem
from .tools import WORKSPACE_OPENAI_TOOLS, WorkspaceToolExecutor


EVENT_TYPE_MAP = {
    "task_started": "run_started",
    "llm_called": "model_call",
    "llm_result": "agent_message",
    "tool_called": "tool_call",
    "tool_result": "tool_result",
    "tool_output": "tool_output",
    "file_changed": "file_changed",
    "protocol_error": "error",
    "error": "error",
    "run_cancelled": "run_cancelled",
}


class RunManager:
    def __init__(
        self,
        database: Database,
        filesystem: WorkspaceFilesystem,
        settings: Settings,
        sandbox_factory: DockerSandboxFactory | None = None,
        model_client_factory=None,
    ):
        self.database = database
        self.filesystem = filesystem
        self.settings = settings
        self.sandbox_factory = sandbox_factory or DockerSandboxFactory(settings)
        self.model_client_factory = model_client_factory
        self.executor = ThreadPoolExecutor(max_workers=settings.max_parallel_runs, thread_name_prefix="workspace-run")
        self._cancel_events: dict[str, threading.Event] = {}
        self._sandboxes: dict[str, DockerSandbox] = {}
        self._lock = threading.RLock()

    def startup(self) -> None:
        self.sandbox_factory.cleanup_orphans()
        self.database.recover_interrupted_runs()

    def shutdown(self) -> None:
        with self._lock:
            sandboxes = list(self._sandboxes.values())
        for sandbox in sandboxes:
            sandbox.destroy()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def start(self, run_id: str) -> None:
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[run_id] = cancel_event
        self.executor.submit(self._execute_run, run_id, cancel_event)

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            cancel_event = self._cancel_events.get(run_id)
            sandbox = self._sandboxes.get(run_id)
        if cancel_event:
            cancel_event.set()
        if sandbox:
            sandbox.destroy()
        return bool(cancel_event or sandbox)

    def _execute_run(self, run_id: str, cancel_event: threading.Event) -> None:
        run = self.database.fetchone("SELECT * FROM agent_runs WHERE id=?", (run_id,))
        if not run:
            return
        workspace = self.database.fetchone("SELECT * FROM workspaces WHERE id=?", (run["workspace_id"],))
        if not workspace:
            return
        if cancel_event.is_set() or run.get("cancel_requested"):
            now = utc_now()
            self.database.execute(
                "UPDATE agent_runs SET status='cancelled', error='Cancelled by user.', updated_at=?, finished_at=? WHERE id=?",
                (now, now, run_id),
            )
            self.database.insert_event(run_id, "run_cancelled", {"status": "cancelled", "message": "Cancelled by user."})
            with self._lock:
                self._cancel_events.pop(run_id, None)
            return
        started = utc_now()
        started_monotonic = time.monotonic()
        self.database.execute(
            "UPDATE agent_runs SET status='running', started_at=?, updated_at=? WHERE id=? AND status='created'",
            (started, started, run_id),
        )
        trace = new_trace(run["task"], task_id=run_id)
        sandbox: DockerSandbox | None = None
        run_root: Path | None = None

        def event_sink(event: dict[str, Any]) -> None:
            raw_type = event.get("event_type") or event.get("type") or "event"
            attrs = event.get("attributes") or event.get("data") or {}
            mapped = EVENT_TYPE_MAP.get(raw_type, raw_type)
            if raw_type == "final_answer":
                mapped = "agent_message"
            payload = {"traceEventType": raw_type, **attrs}
            self.database.insert_event(run_id, mapped, payload, event.get("step"))

        trace["_event_sink"] = event_sink
        for existing in trace.get("events", []):
            event_sink(existing)

        try:
            base, staging = self.filesystem.prepare_run(workspace["id"], run_id)
            run_root = self.filesystem.paths(workspace["id"]).run_artifacts(run_id)
            self._prepare_staging_permissions(staging)
            if cancel_event.is_set():
                raise RuntimeError("Cancelled by user.")

            def emit_output(stream: str, data: str) -> None:
                add_event(trace, "tool_output", {"stream": stream, "data": data, "step": self._trace_step(trace)}, step=self._trace_step(trace))

            sandbox = self.sandbox_factory.create(
                run_id,
                self.filesystem.host_staging_path(workspace["id"], run_id),
                output_callback=emit_output,
            )
            with self._lock:
                self._sandboxes[run_id] = sandbox
            sandbox.start()

            def emit_file_changed(payload: dict[str, Any]) -> None:
                step = self._trace_step(trace)
                add_event(trace, "file_changed", {**payload, "step": step}, step=step)

            tool_executor = WorkspaceToolExecutor(
                self.filesystem,
                staging,
                sandbox,
                file_changed_callback=emit_file_changed,
            )
            model_client = self.model_client_factory() if self.model_client_factory else None
            answer = run_agent(
                run["task"],
                trace,
                max_steps=20,
                run_dir=str(run_root),
                budget=RunBudget(max_steps=20, max_wall_time_sec=self.settings.run_timeout_seconds),
                model_client=model_client,
                tool_executor=tool_executor,
                openai_tools=WORKSPACE_OPENAI_TOOLS,
                cancel_check=cancel_event.is_set,
                model_timeout=60,
            )
            if trace.get("_event_sink_error"):
                raise RuntimeError(f"Trace persistence failed: {trace['_event_sink_error']}")
            trace["finished_at"] = utc_now()
            usage = summarize_usage(trace)
            exit_reason = self._exit_reason(trace)
            changes = self.filesystem.diff(base, staging)
            self._save_changes(run_id, changes)
            duration_ms = int((time.monotonic() - started_monotonic) * 1000)
            if cancel_event.is_set() or exit_reason == "cancelled":
                status, apply_status, error = "cancelled", "none", "Cancelled by user."
            elif exit_reason == "completed":
                status = "waiting_user" if changes else "completed"
                apply_status = "pending" if changes else "applied"
                error = None
            else:
                status, apply_status, error = "failed", "none", f"Run stopped with exit reason: {exit_reason}"
            finished = utc_now()
            self.database.execute(
                """UPDATE agent_runs SET status=?, final_result=?, error=?, model_calls=?, tool_calls=?, duration_ms=?,
                   apply_status=?, changed_files_json=?, updated_at=?, finished_at=? WHERE id=?""",
                (
                    status,
                    answer,
                    error,
                    usage.get("model_calls", 0),
                    usage.get("tool_calls", 0),
                    duration_ms,
                    apply_status,
                    json.dumps([item["path"] for item in changes], ensure_ascii=False),
                    finished,
                    finished,
                    run_id,
                ),
            )
            terminal_payload = {
                "status": status,
                "applyStatus": apply_status,
                "changedFiles": [item["path"] for item in changes],
                "step": self._trace_step(trace),
            }
            if status == "failed":
                add_event(trace, "error", {"message": error, **terminal_payload}, step=self._trace_step(trace))
            add_event(
                trace,
                "run_cancelled" if status == "cancelled" else "run_completed",
                terminal_payload,
                step=self._trace_step(trace),
            )
            save_trace(trace, str(run_root / "trace.jsonl"))
        except Exception as exc:
            finished = utc_now()
            status = "cancelled" if cancel_event.is_set() else "failed"
            message = "Cancelled by user." if cancel_event.is_set() else str(exc)
            self.database.execute(
                "UPDATE agent_runs SET status=?, error=?, updated_at=?, finished_at=?, duration_ms=? WHERE id=?",
                (status, message, finished, finished, int((time.monotonic() - started_monotonic) * 1000), run_id),
            )
            trace["finished_at"] = finished
            terminal_step = self._trace_step(trace)
            if status == "cancelled":
                add_event(trace, "run_cancelled", {"message": message, "status": status, "step": terminal_step}, step=terminal_step)
            else:
                add_event(trace, "error", {"message": message, "status": status, "step": terminal_step}, step=terminal_step)
                add_event(trace, "run_completed", {"status": status, "step": terminal_step}, step=terminal_step)
            if run_root:
                save_trace(trace, str(run_root / "trace.jsonl"))
        finally:
            if sandbox:
                sandbox.destroy()
            with self._lock:
                self._sandboxes.pop(run_id, None)
                self._cancel_events.pop(run_id, None)

    def _prepare_staging_permissions(self, root: Path) -> None:
        for path in [root, *root.rglob("*")]:
            try:
                if os.geteuid() == 0:
                    os.chown(path, self.settings.sandbox_uid, self.settings.sandbox_gid)
                path.chmod(0o770 if path.is_dir() else 0o660)
            except (AttributeError, PermissionError):
                continue

    @staticmethod
    def _trace_step(trace: dict[str, Any]) -> int:
        steps = [event.get("step") for event in trace.get("events", []) if isinstance(event.get("step"), int)]
        return max(steps, default=0)

    @staticmethod
    def _exit_reason(trace: dict[str, Any]) -> str:
        for event in reversed(trace.get("events", [])):
            if event.get("event_type") == "final_answer":
                return (event.get("attributes") or {}).get("exit_reason", "completed")
        return "runtime_error"

    def _save_changes(self, run_id: str, changes: list[dict[str, Any]]) -> None:
        with self.database.transaction() as connection:
            connection.execute("DELETE FROM file_changes WHERE run_id=?", (run_id,))
            for change in changes:
                connection.execute(
                    """INSERT INTO file_changes(run_id,path,change_type,diff,before_sha256,after_sha256)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        run_id,
                        change["path"],
                        change["changeType"],
                        change["diff"],
                        change["beforeSha256"],
                        change["afterSha256"],
                    ),
                )
