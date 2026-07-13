import json
import sqlite3
from types import SimpleNamespace

import pytest

from agent import export_run, new_trace, recover_task, resume_task, run_agent
from agent.core import (
    _persist_interrupted_run,
    parse_export_args,
    parse_recover_args,
    parse_resume_args,
    parse_run_args,
)
from agent.replay import validate_trace
from agent.sqlite_store import SQLITE_SCHEMA_VERSION, SQLiteRunStore
from agent.state import new_task_state
from agent.store import FileRunStore, make_event


class FakeClient:
    def __init__(self, content="recovered"):
        self.content = content
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_kwargs):
        self.calls += 1
        message = SimpleNamespace(tool_calls=[], content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _checkpoint_state(task_id="durable", goal="goal"):
    return new_task_state(task_id, goal).model_dump()


def _seed_open_run(store, task_id="durable", include_tool=False):
    store.create_run(task_id, "goal")
    store.start_segment(task_id, "segment-1", "task")
    store.append_event(task_id, make_event("task_started", {"task_id": task_id, "user_goal": "goal"}))
    if include_tool:
        store.append_event(
            task_id,
            make_event(
                "tool_called",
                {
                    "tool_call.id": "historical-call",
                    "tool_call.name": "write_file",
                    "tool_call.arguments": {"path": "old.txt", "content": "old"},
                },
                step=1,
            ),
        )
        store.append_event(
            task_id,
            make_event(
                "tool_result",
                {
                    "tool_call.id": "historical-call",
                    "tool_call.name": "write_file",
                    "tool_call.arguments": {"path": "old.txt", "content": "old"},
                    "observation": {"ok": True, "result": "written"},
                },
                step=1,
            ),
        )
    store.save_checkpoint(task_id, _checkpoint_state(task_id), "saved context", 1)


def test_sqlite_schema_init_is_idempotent(tmp_path):
    db_path = tmp_path / "agent.db"
    first = SQLiteRunStore(db_path)
    first.create_run("kept", "goal")
    SQLiteRunStore(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        version = connection.execute(
            "SELECT schema_version FROM runs WHERE task_id='kept'"
        ).fetchone()[0]
    assert {"runs", "segments", "events", "checkpoints"} <= tables
    assert version == SQLITE_SCHEMA_VERSION


def test_event_sequence_is_monotonic(tmp_path):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store)
    store.append_event("durable", make_event("custom", {"value": 1}, step=2))

    with sqlite3.connect(store.db_path) as connection:
        sequence = [
            row[0]
            for row in connection.execute(
                "SELECT sequence_no FROM events WHERE task_id='durable' ORDER BY sequence_no"
            )
        ]
    assert sequence == list(range(1, len(sequence) + 1))


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_checkpoint_started",
        "after_checkpoint_snapshot",
        "after_run_update",
        "after_checkpoint_saved",
    ],
)
def test_checkpoint_transaction_rolls_back(tmp_path, failure_stage):
    def fail(stage):
        if stage == failure_stage:
            raise RuntimeError("injected checkpoint failure")

    store = SQLiteRunStore(tmp_path / "agent.db", failure_injector=fail)
    store.create_run("rollback", "goal")
    store.start_segment("rollback", "segment-1", "task")
    store.append_event("rollback", make_event("task_started", {"task_id": "rollback"}))

    with pytest.raises(RuntimeError, match="injected"):
        store.save_checkpoint("rollback", _checkpoint_state("rollback"), "context", 3)

    loaded = store.load_run("rollback")
    assert loaded["checkpoint"] is None
    assert [event["event_type"] for event in loaded["events"]] == ["task_started"]
    assert loaded["status"] == "running"


def test_crashed_segment_can_recover(tmp_path):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store)

    answer, _ = recover_task("durable", store=store, model_client=FakeClient("done"))
    loaded = store.load_run("durable")

    assert answer == "done"
    assert [segment["kind"] for segment in loaded["segments"]] == ["task", "recovery"]
    assert loaded["segments"][0]["exit_reason"] == "crashed"
    assert loaded["segments"][1]["exit_reason"] == "completed"
    recovery_events = [event for event in loaded["events"] if event["event_type"] == "recovery_started"]
    assert len(recovery_events) == 1
    assert recovery_events[0]["attributes"]["previous_segment_id"] == "segment-1"
    assert recovery_events[0]["attributes"]["segment_id"] == loaded["segments"][1]["segment_id"]


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_previous_segment_crashed",
        "after_recovery_segment_created",
        "after_recovery_run_updated",
        "after_recovery_started_event",
    ],
)
def test_recovery_handoff_is_atomic(tmp_path, failure_stage):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store)

    def fail(stage):
        if stage == failure_stage:
            raise RuntimeError("injected recovery handoff failure")

    store.failure_injector = fail
    with pytest.raises(RuntimeError, match="injected recovery"):
        store.begin_recovery("durable", "segment-1", "recovery-1")

    loaded = store.load_run("durable")
    assert loaded["status"] == "running"
    assert len(loaded["segments"]) == 1
    assert loaded["segments"][0]["finished_at"] is None
    assert loaded["segments"][0]["exit_reason"] is None
    assert not any(event["event_type"] == "recovery_started" for event in loaded["events"])


def test_recovery_handoff_success_is_consistent(tmp_path):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store)
    store.begin_recovery("durable", "segment-1", "recovery-1")

    loaded = store.load_run("durable")
    assert loaded["status"] == "running"
    assert loaded["segments"][0]["exit_reason"] == "crashed"
    assert loaded["segments"][0]["finished_at"] is not None
    assert loaded["segments"][1]["kind"] == "recovery"
    assert loaded["segments"][1]["finished_at"] is None
    recovery_events = [event for event in loaded["events"] if event["event_type"] == "recovery_started"]
    assert len(recovery_events) == 1
    assert recovery_events[0]["attributes"] == {
        "task_id": "durable",
        "previous_segment_id": "segment-1",
        "segment_id": "recovery-1",
        "checkpoint_step": 1,
    }
    with sqlite3.connect(store.db_path) as connection:
        event_segment = connection.execute(
            "SELECT segment_id FROM events WHERE task_id=? AND event_type='recovery_started'",
            ("durable",),
        ).fetchone()[0]
    assert event_segment == "recovery-1"


def test_recovery_does_not_reexecute_tools(tmp_path):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store, include_tool=True)
    executed = []

    def executor(call):
        executed.append(call)
        raise AssertionError("historical tool must not be executed")

    recover_task(
        "durable",
        store=store,
        model_client=FakeClient("continued without tools"),
        tool_executor=executor,
    )
    assert executed == []


def test_sqlite_export_passes_replay(tmp_path):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store)
    store.append_event(
        "durable",
        make_event("final_answer", {"answer": "done", "exit_reason": "completed"}, step=2),
    )
    store.save_checkpoint("durable", _checkpoint_state(), "final context", 2)
    store.finish_segment("durable", "segment-1", "completed")
    out_path = tmp_path / "exported.jsonl"

    export_run("durable", out_path, store=store)
    records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    trace = {**records[0], "events": records[1:]}
    trace.pop("record_type", None)
    for event in trace["events"]:
        event.pop("record_type", None)
    assert all(result.passed for result in validate_trace(trace))


def test_file_store_and_sqlite_store_contract_match(tmp_path):
    stores = [
        FileRunStore(tmp_path / "files"),
        SQLiteRunStore(tmp_path / "agent.db"),
    ]
    loaded_runs = []
    for store in stores:
        store.create_run("contract", "same goal")
        store.start_segment("contract", "segment-1", "task")
        store.append_event("contract", make_event("task_started", {"task_id": "contract"}))
        store.append_event("contract", make_event("custom", {"same": True}, step=1))
        store.save_checkpoint("contract", _checkpoint_state("contract", "same goal"), "context", 1)
        store.finish_segment("contract", "segment-1", "completed")
        loaded_runs.append(store.load_run("contract"))

    file_run, sqlite_run = loaded_runs
    assert file_run["user_goal"] == sqlite_run["user_goal"] == "same goal"
    assert file_run["status"] == sqlite_run["status"] == "finished"
    assert [item["kind"] for item in file_run["segments"]] == ["task"]
    assert [item["kind"] for item in sqlite_run["segments"]] == ["task"]
    assert [event["event_type"] for event in file_run["events"]] == [
        event["event_type"] for event in sqlite_run["events"]
    ]
    assert file_run["checkpoint"] == sqlite_run["checkpoint"]


def test_recovered_trace_passes_replay(tmp_path):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store)
    recover_task("durable", store=store, model_client=FakeClient("done"))
    assert all(result.passed for result in validate_trace(store.load_run("durable")["trace"]))


def test_store_cli_parsing_and_compatibility():
    default_args = parse_run_args(["goal", "--task-id", "demo"])
    sqlite_args = parse_run_args(["goal", "--task-id", "demo", "--store", "sqlite"])
    assert default_args["store_type"] == "file"
    assert sqlite_args["store_type"] == "sqlite"
    assert parse_resume_args(["demo", "--store", "sqlite", "--max-steps", "3"]) == (
        "demo",
        3,
        "sqlite",
    )
    assert parse_export_args(["demo", "--format", "jsonl", "--out", "out.jsonl"]) == (
        "demo",
        "out.jsonl",
    )
    assert parse_recover_args(["demo", "--max-steps", "4"]) == ("demo", 4)
    with pytest.raises(ValueError, match="--trace"):
        parse_run_args(["goal", "--store", "sqlite", "--trace", "runs/demo/trace.jsonl"])


@pytest.mark.parametrize("store_factory", [FileRunStore, SQLiteRunStore])
def test_duplicate_task_id_is_rejected(tmp_path, store_factory):
    path = tmp_path / ("files" if store_factory is FileRunStore else "agent.db")
    store = store_factory(path)
    store.create_run("duplicate", "goal")
    with pytest.raises(ValueError, match="already exists"):
        store.create_run("duplicate", "goal")


def test_run_agent_file_store_end_to_end(tmp_path):
    trace = new_trace("goal", task_id="file-e2e")
    answer = run_agent(
        "goal",
        trace,
        run_dir=str(tmp_path / "file-e2e"),
        model_client=FakeClient("done"),
    )
    loaded = FileRunStore(tmp_path).load_run("file-e2e")
    assert answer == "done"
    assert loaded["status"] == "finished"
    assert loaded["checkpoint"] is not None
    assert loaded["segments"][0]["exit_reason"] == "completed"


def test_sqlite_resume_requires_closed_segment(tmp_path):
    store = SQLiteRunStore(tmp_path / "agent.db")
    _seed_open_run(store)
    with pytest.raises(ValueError, match="open segment"):
        resume_task("durable", store=store, model_client=FakeClient())

    store.append_event(
        "durable",
        make_event("final_answer", {"answer": "first", "exit_reason": "completed"}, step=2),
    )
    store.save_checkpoint("durable", _checkpoint_state(), "closed context", 2)
    store.finish_segment("durable", "segment-1", "completed")
    answer, _ = resume_task("durable", store=store, model_client=FakeClient("resumed"))
    loaded = store.load_run("durable")
    assert answer == "resumed"
    assert [segment["kind"] for segment in loaded["segments"]] == ["task", "resume"]


def test_file_interrupted_run_can_resume(tmp_path):
    store = FileRunStore(tmp_path)
    store.create_run("file-interrupted", "goal")
    store.start_segment("file-interrupted", "task-1", "task")
    trace = new_trace("goal", task_id="file-interrupted")
    trace["_store"] = store
    trace["_segment_id"] = "task-1"
    for event in trace["events"]:
        store.append_event("file-interrupted", event)
    task_state = new_task_state("file-interrupted", "goal")

    _persist_interrupted_run(
        trace,
        task_state,
        str(tmp_path / "file-interrupted"),
        store,
        "task-1",
    )
    interrupted = store.load_run("file-interrupted")
    assert interrupted["segments"][0]["exit_reason"] == "interrupted"
    assert interrupted["segments"][0]["finished_at"] is not None
    assert any(event["event_type"] == "segment_interrupted" for event in interrupted["events"])

    answer, _ = resume_task(
        "file-interrupted",
        store=store,
        model_client=FakeClient("resumed"),
    )
    loaded = store.load_run("file-interrupted")
    assert answer == "resumed"
    assert [segment["kind"] for segment in loaded["segments"]] == ["task", "resume"]
    assert loaded["segments"][1]["exit_reason"] == "completed"
    assert all(result.passed for result in validate_trace(loaded["trace"]))
