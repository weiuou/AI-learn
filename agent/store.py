from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol


TRACE_SCHEMA_VERSION = "agent-harness-trace-v1"


def timestamp_now() -> str:
    return datetime.now().isoformat()


def make_event(event_type: str, attributes: dict | None = None, step: int | None = None) -> dict:
    attrs = dict(attributes or {})
    if step is None:
        step = attrs.get("step")
    if step is not None:
        attrs["step"] = step
    return {
        "event_type": event_type,
        "type": event_type,
        "step": step,
        "timestamp": timestamp_now(),
        "attributes": attrs,
        "data": attrs,
    }


class RunStore(Protocol):
    def create_run(self, task_id: str, user_goal: str) -> None: ...

    def start_segment(self, task_id: str, segment_id: str, kind: str) -> None: ...

    def append_event(self, task_id: str, event: dict) -> None: ...

    def save_checkpoint(self, task_id: str, state: dict, context_pack: str, step: int) -> None: ...

    def load_run(self, task_id: str) -> dict: ...

    def finish_segment(self, task_id: str, segment_id: str, exit_reason: str) -> None: ...


class FileRunStore:
    """Legacy-compatible run storage backed by trace/state/context files."""

    def __init__(self, base_dir: str | Path = "runs"):
        self.base_dir = Path(base_dir)

    def _run_dir(self, task_id: str) -> Path:
        return self.base_dir / task_id

    def _paths(self, task_id: str) -> dict[str, Path]:
        run_dir = self._run_dir(task_id)
        return {
            "run_dir": run_dir,
            "trace": run_dir / "trace.jsonl",
            "state": run_dir / "state.json",
            "context_pack": run_dir / "context_pack.md",
        }

    @staticmethod
    def _write_trace(trace: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {key: value for key, value in trace.items() if key != "events" and not key.startswith("_")}
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"record_type": "trace", **metadata}, ensure_ascii=False) + "\n")
            for event in trace.get("events", []):
                handle.write(json.dumps({"record_type": "event", **event}, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_trace(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Unknown run: {path.parent.name}")
        trace = {"schema_version": TRACE_SCHEMA_VERSION, "events": []}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.pop("record_type", "event")
                if record_type == "trace":
                    trace.update(record)
                else:
                    trace["events"].append(record)
        return trace

    def create_run(self, task_id: str, user_goal: str) -> None:
        paths = self._paths(task_id)
        if paths["trace"].exists():
            raise ValueError(f"Run already exists: {task_id}")
        created_at = timestamp_now()
        self._write_trace(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "task_id": task_id,
                "task": user_goal,
                "user_goal": user_goal,
                "started_at": created_at,
                "finished_at": None,
                "status": "running",
                "segments": [],
                "events": [],
            },
            paths["trace"],
        )

    def start_segment(self, task_id: str, segment_id: str, kind: str) -> None:
        if kind not in {"task", "resume", "recovery"}:
            raise ValueError(f"Unsupported segment kind: {kind}")
        paths = self._paths(task_id)
        trace = self._read_trace(paths["trace"])
        if any(item.get("finished_at") is None for item in trace.get("segments", [])):
            raise ValueError(f"Run already has an open segment: {task_id}")
        if any(item.get("segment_id") == segment_id for item in trace.get("segments", [])):
            raise ValueError(f"Segment already exists: {segment_id}")
        trace.setdefault("segments", []).append(
            {
                "segment_id": segment_id,
                "task_id": task_id,
                "kind": kind,
                "started_at": timestamp_now(),
                "finished_at": None,
                "exit_reason": None,
            }
        )
        trace["status"] = "running"
        trace["finished_at"] = None
        self._write_trace(trace, paths["trace"])

    def append_event(self, task_id: str, event: dict) -> None:
        paths = self._paths(task_id)
        trace = self._read_trace(paths["trace"])
        trace.setdefault("events", []).append(event)
        self._write_trace(trace, paths["trace"])

    def save_checkpoint(self, task_id: str, state: dict, context_pack: str, step: int) -> None:
        paths = self._paths(task_id)
        trace = self._read_trace(paths["trace"])
        trace["events"].append(make_event("checkpoint_started", {"task_id": task_id}, step=step))
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        with paths["state"].open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        paths["context_pack"].write_text(context_pack, encoding="utf-8")
        trace["events"].append(
            make_event(
                "checkpoint_saved",
                {
                    "task_id": task_id,
                    "state_path": str(paths["state"]),
                    "trace_path": str(paths["trace"]),
                    "context_pack_path": str(paths["context_pack"]),
                    "token_estimate": max(1, len(context_pack) // 4),
                },
                step=step,
            )
        )
        self._write_trace(trace, paths["trace"])

    def load_run(self, task_id: str) -> dict:
        paths = self._paths(task_id)
        trace = self._read_trace(paths["trace"])
        checkpoint = None
        if paths["state"].exists() and paths["context_pack"].exists():
            with paths["state"].open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            checkpoint_events = [
                event for event in trace.get("events", []) if event.get("event_type") == "checkpoint_saved"
            ]
            step = checkpoint_events[-1].get("step") if checkpoint_events else 0
            checkpoint = {
                "state": state,
                "context_pack": paths["context_pack"].read_text(encoding="utf-8"),
                "step": step or 0,
            }
        return {
            "task_id": task_id,
            "user_goal": trace.get("user_goal") or trace.get("task"),
            "status": trace.get("status", "finished" if trace.get("finished_at") else "running"),
            "segments": list(trace.get("segments", [])),
            "events": list(trace.get("events", [])),
            "checkpoint": checkpoint,
            "trace": trace,
        }

    def finish_segment(self, task_id: str, segment_id: str, exit_reason: str) -> None:
        paths = self._paths(task_id)
        trace = self._read_trace(paths["trace"])
        segment = next(
            (item for item in trace.get("segments", []) if item.get("segment_id") == segment_id),
            None,
        )
        if segment is None:
            raise ValueError(f"Unknown segment: {segment_id}")
        if segment.get("finished_at") is not None:
            raise ValueError(f"Segment already finished: {segment_id}")
        finished_at = timestamp_now()
        segment["finished_at"] = finished_at
        segment["exit_reason"] = exit_reason
        trace["finished_at"] = finished_at
        trace["status"] = "crashed" if exit_reason == "crashed" else "finished"
        self._write_trace(trace, paths["trace"])
