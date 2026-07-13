from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from agent.replay import validate_trace


def _event(event_type, second, step=None, call_id=None, ok=True, exit_reason=None):
    attrs = {}
    if call_id:
        attrs["tool_call.id"] = call_id
    if event_type == "tool_result":
        attrs["observation"] = {"ok": ok, "error_type": None if ok else "FAILED"}
    if exit_reason:
        attrs["exit_reason"] = exit_reason
    return {
        "event_type": event_type,
        "type": event_type,
        "step": step,
        "timestamp": (datetime(2026, 7, 13) + timedelta(seconds=second)).isoformat(),
        "attributes": attrs,
        "data": attrs,
    }


def _happy_trace():
    events = [
        _event("task_started", 0),
        _event("llm_called", 1, 1),
        _event("tool_called", 2, 1, "a"),
        _event("tool_result", 3, 1, "a"),
        _event("final_answer", 4, 1, exit_reason="completed"),
    ]
    return {
        "schema_version": "agent-harness-trace-v1",
        "events": events,
        "usage_summary": {"model_calls": 1, "tool_calls": 1, "context_compressions": 0},
        "budget_summary": {
            "limits": {},
            "consumed": {
                "steps": 1,
                "model_calls": 1,
                "tool_calls": 1,
                "prompt_chars": 0,
                "consecutive_failures": 0,
            },
        },
    }


def _result_map(trace):
    return {item.name: item for item in validate_trace(trace)}


def test_offline_replay_happy_path():
    assert all(item.passed for item in validate_trace(_happy_trace()))


def test_resume_trace_validates_each_segment():
    trace = _happy_trace()
    trace["events"].extend([
        _event("resume_started", 5),
        _event("llm_called", 6, 2),
        _event("final_answer", 7, 2, exit_reason="completed"),
    ])
    trace["usage_summary"]["model_calls"] = 2
    assert _result_map(trace)["segment_termination"].passed


def test_interrupted_trace_passes_replay_before_resume():
    trace = {
        "schema_version": "agent-harness-trace-v1",
        "events": [
            _event("task_started", 0),
            _event("segment_interrupted", 1, step=1, exit_reason="interrupted"),
        ],
    }
    assert all(item.passed for item in validate_trace(trace))


@pytest.mark.parametrize(
    "events",
    [
        [
            _event("task_started", 0),
            _event("segment_interrupted", 1, step=1, exit_reason="interrupted"),
            _event("segment_interrupted", 2, step=1, exit_reason="interrupted"),
        ],
        [
            _event("task_started", 0),
            _event("segment_interrupted", 1, step=1, exit_reason="interrupted"),
            _event("error", 2, step=1),
        ],
    ],
)
def test_interrupted_trace_requires_one_terminal_marker(events):
    trace = {"schema_version": "agent-harness-trace-v1", "events": events}
    assert not _result_map(trace)["segment_termination"].passed


@pytest.mark.parametrize(
    "mutate,invariant",
    [
        (lambda trace: trace["events"].pop(), "segment_termination"),
        (lambda trace: trace["events"].insert(3, deepcopy(trace["events"][2])), "tool_call_pairing"),
        (lambda trace: trace["events"].__setitem__(3, _event("tool_result", 3, 1, "orphan")), "tool_call_pairing"),
        (lambda trace: trace["events"][3].__setitem__("timestamp", datetime(2025, 1, 1).isoformat()), "timeline_order"),
        (lambda trace: trace["usage_summary"].__setitem__("tool_calls", 99), "usage_summary"),
        (lambda trace: trace["budget_summary"]["consumed"].__setitem__("tool_calls", 99), "budget_summary"),
    ],
)
def test_replay_reports_specific_failures(mutate, invariant):
    trace = _happy_trace()
    mutate(trace)
    assert not _result_map(trace)[invariant].passed
