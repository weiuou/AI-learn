from __future__ import annotations

from collections import Counter
from datetime import datetime

from pydantic import BaseModel, Field


class InvariantResult(BaseModel):
    name: str
    passed: bool
    detail: str
    steps: list[int] = Field(default_factory=list)


def _event_type(event):
    return event.get("event_type") or event.get("type")


def _attrs(event):
    return event.get("attributes") or event.get("data") or {}


def _step(event):
    value = event.get("step") or _attrs(event).get("step")
    return value if isinstance(value, int) else None


def recompute_usage_summary(trace):
    summary = {
        "model_calls": 0,
        "usage_calls": 0,
        "missing_usage_calls": 0,
        "tool_calls": 0,
        "context_compressions": 0,
        "api_usage_available": False,
        "cache_usage_available": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
        "cache_hit_rate": None,
    }
    for event in trace.get("events", []):
        event_type = _event_type(event)
        if event_type == "llm_called":
            summary["model_calls"] += 1
        elif event_type == "llm_result":
            usage = _attrs(event).get("usage")
            if usage:
                summary["api_usage_available"] = True
                summary["usage_calls"] += 1
                for key in ["prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"]:
                    summary[key] += usage.get(key) or 0
                if usage.get("cached_tokens") is not None:
                    summary["cache_usage_available"] = True
                    summary["cached_tokens"] += usage.get("cached_tokens") or 0
                if usage.get("cache_creation_tokens") is not None:
                    summary["cache_usage_available"] = True
                    summary["cache_creation_tokens"] += usage.get("cache_creation_tokens") or 0
        elif event_type == "tool_called":
            summary["tool_calls"] += 1
        elif event_type == "context_compressed":
            summary["context_compressions"] += 1
    summary["missing_usage_calls"] = summary["model_calls"] - summary["usage_calls"]
    if summary["cache_usage_available"] and summary["prompt_tokens"]:
        summary["cache_hit_rate"] = summary["cached_tokens"] / summary["prompt_tokens"]
    return summary


def recompute_budget_consumed(trace):
    failures = 0
    prompt_chars = 0
    steps = set()
    for event in trace.get("events", []):
        event_type = _event_type(event)
        attrs = _attrs(event)
        if event_type == "llm_called":
            if _step(event) is not None:
                steps.add(_step(event))
            prompt_chars += attrs.get("prompt_chars") or 0
        elif event_type == "tool_result":
            observation = attrs.get("observation") or attrs.get("result") or {}
            if isinstance(observation, dict) and observation.get("ok"):
                failures = 0
            else:
                failures += 1
    usage = recompute_usage_summary(trace)
    return {
        "steps": len(steps),
        "model_calls": usage["model_calls"],
        "tool_calls": usage["tool_calls"],
        "prompt_chars": prompt_chars,
        "consecutive_failures": failures,
    }


def _validate_schema(trace):
    required = {"schema_version", "events"}
    missing = sorted(required - set(trace))
    valid_events = isinstance(trace.get("events"), list) and all(isinstance(item, dict) for item in trace.get("events", []))
    passed = not missing and valid_events and str(trace.get("schema_version", "")).startswith("agent-harness-trace-v")
    detail = "valid trace envelope" if passed else f"missing={missing}, events_is_list={valid_events}"
    return InvariantResult(name="trace_schema", passed=passed, detail=detail)


def _validate_timeline(trace):
    previous = None
    bad_steps = []
    for event in trace.get("events", []):
        raw = event.get("timestamp")
        try:
            current = datetime.fromisoformat(raw) if raw else None
        except (TypeError, ValueError):
            current = None
        if current is None or (previous is not None and current < previous):
            bad_steps.append(_step(event) or 0)
        if current is not None:
            previous = current
    return InvariantResult(
        name="timeline_order",
        passed=not bad_steps,
        detail="timestamps are non-decreasing" if not bad_steps else "missing or decreasing timestamp",
        steps=bad_steps,
    )


def _validate_tool_pairs(trace):
    called = Counter()
    results = Counter()
    steps = []
    for event in trace.get("events", []):
        event_type = _event_type(event)
        if event_type not in {"tool_called", "tool_result"}:
            continue
        call_id = _attrs(event).get("tool_call.id")
        if not call_id:
            steps.append(_step(event) or 0)
            continue
        (called if event_type == "tool_called" else results)[call_id] += 1
    bad_ids = sorted(set(called) | set(results))
    bad_ids = [item for item in bad_ids if called[item] != 1 or results[item] != 1]
    return InvariantResult(
        name="tool_call_pairing",
        passed=not bad_ids and not steps,
        detail="all tool calls have exactly one result" if not bad_ids and not steps else f"invalid_call_ids={bad_ids}",
        steps=steps,
    )


def _validate_terminations(trace):
    segments = []
    current = []
    for event in trace.get("events", []):
        if _event_type(event) in {"task_started", "resume_started"}:
            if current:
                segments.append(current)
            current = [event]
        elif current:
            current.append(event)
    if current:
        segments.append(current)
    counts = [sum(1 for event in segment if _event_type(event) == "final_answer") for segment in segments]
    passed = bool(segments) and all(count == 1 for count in counts)
    steps = [_step(segment[0]) or 0 for index, segment in enumerate(segments) if counts[index] != 1]
    return InvariantResult(
        name="segment_termination",
        passed=passed,
        detail=f"terminal_counts={counts}",
        steps=steps,
    )


def _validate_usage(trace):
    stored = trace.get("usage_summary")
    if not stored:
        return InvariantResult(name="usage_summary", passed=True, detail="legacy trace has no stored usage summary")
    computed = recompute_usage_summary(trace)
    mismatches = {
        key: (stored.get(key), value)
        for key, value in computed.items()
        if key in stored and stored.get(key) != value
    }
    return InvariantResult(
        name="usage_summary",
        passed=not mismatches,
        detail="usage summary matches events" if not mismatches else f"mismatches={mismatches}",
    )


def _validate_budget(trace):
    stored = trace.get("budget_summary")
    if not stored:
        return InvariantResult(name="budget_summary", passed=True, detail="legacy trace has no budget summary")
    computed = recompute_budget_consumed(trace)
    consumed = stored.get("consumed") or {}
    mismatches = {key: (consumed.get(key), value) for key, value in computed.items() if consumed.get(key) != value}
    return InvariantResult(
        name="budget_summary",
        passed=not mismatches,
        detail="budget summary matches events" if not mismatches else f"mismatches={mismatches}",
    )


def validate_trace(trace) -> list[InvariantResult]:
    return [
        _validate_schema(trace),
        _validate_timeline(trace),
        _validate_tool_pairs(trace),
        _validate_terminations(trace),
        _validate_usage(trace),
        _validate_budget(trace),
    ]


def print_replay_results(results: list[InvariantResult]) -> bool:
    passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        passed = passed and result.passed
    print(f"Replay regression: {'PASS' if passed else 'FAIL'}")
    return passed
