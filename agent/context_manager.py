import json
import re
from pathlib import Path


SHELL_STREAM_LIMIT = 4000
FILE_CONTENT_LIMIT = 8000
TOOL_SUMMARY_LIMIT = 2400
CONTEXT_PACK_STRATEGY = "task_state_context_pack_v1"


def _shorten(value, limit=500):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _head_tail(text, head=1200, tail=1200):
    if len(text) <= head + tail + 80:
        return text
    return (
        text[:head]
        + "\n...[truncated middle]...\n"
        + text[-tail:]
    )


def _line_range_snippet(text, head_lines=40, tail_lines=30):
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return text, f"1-{len(lines)}"

    head = "\n".join(lines[:head_lines])
    tail_start = max(len(lines) - tail_lines + 1, 1)
    tail = "\n".join(lines[-tail_lines:])
    snippet = f"{head}\n...[truncated middle]...\n{tail}"
    return snippet, f"1-{head_lines}, {tail_start}-{len(lines)}"


def _summarize_shell_payload(result):
    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    compact = dict(result)
    compressed = False
    notes = []

    for stream_name, stream_value in [("stdout", stdout), ("stderr", stderr)]:
        if len(stream_value) > SHELL_STREAM_LIMIT:
            compact[stream_name] = _head_tail(stream_value)
            compressed = True
            notes.append(
                f"{stream_name} truncated from {len(stream_value)} chars; kept head and tail."
            )

    output_lines = (stdout or stderr).splitlines()
    if len(output_lines) > 5 and _looks_like_search_output(output_lines):
        compact["search_results_summary"] = [
            _shorten(line, 180) for line in output_lines[:5]
        ]
        compact["search_results_omitted"] = max(0, len(output_lines) - 5)
        compressed = True
        notes.append("search output summarized to the first 5 result lines.")

    if notes:
        compact["summary"] = " ".join(notes)
        compact["truncated"] = True
    return compact, compressed


def _looks_like_search_output(lines):
    if len(lines) <= 5:
        return False
    hits = 0
    for line in lines[:20]:
        if re.search(r"(^[^:\n]+:\d+:)|(^\./)|(^/)|(\.py:)|(\.md:)|(\.json:)", line):
            hits += 1
    return hits >= 3


def compress_tool_result(tool_name, tool_args, tool_result):
    if not isinstance(tool_result, dict):
        text = _shorten(tool_result, TOOL_SUMMARY_LIMIT)
        return {
            "ok": True,
            "result": text,
            "compressed": True,
            "summary": "Non-dict tool result converted to compact text.",
            "tool_name": tool_name,
            "tool_arguments": tool_args or {},
            "compression_strategy": CONTEXT_PACK_STRATEGY,
        }, True

    compact = {
        "ok": tool_result.get("ok"),
        "result": tool_result.get("result"),
        "error_type": tool_result.get("error_type"),
        "message": tool_result.get("message"),
        "recoverable": tool_result.get("recoverable"),
        "suggestion": tool_result.get("suggestion"),
        "tool_name": tool_name,
        "tool_arguments": tool_args or {},
        "compression_strategy": CONTEXT_PACK_STRATEGY,
    }
    compressed = False

    result = tool_result.get("result")
    if tool_name == "run_shell" and isinstance(result, dict):
        compact["result"], compressed = _summarize_shell_payload(result)
    elif tool_name == "read_file" and isinstance(result, str) and len(result) > FILE_CONTENT_LIMIT:
        snippet, line_range = _line_range_snippet(result)
        compact["result"] = {
            "path": (tool_args or {}).get("path"),
            "line_ranges": line_range,
            "snippet": _shorten(snippet, TOOL_SUMMARY_LIMIT),
            "original_chars": len(result),
            "summary": "file content truncated; kept beginning and ending line ranges.",
        }
        compressed = True
    elif isinstance(result, str) and len(result) > TOOL_SUMMARY_LIMIT:
        compact["result"] = _head_tail(result, head=900, tail=900)
        compressed = True
    elif not isinstance(result, (str, type(None), dict, list, int, float, bool)):
        compact["result"] = _shorten(result, TOOL_SUMMARY_LIMIT)
        compressed = True

    compact["compressed"] = compressed
    if compressed and not compact.get("summary"):
        compact["summary"] = "tool result truncated for model context; full observation remains in trace."
        compact["truncated"] = True
    return compact, compressed


def summarize_tool_event(event):
    attrs = event.get("attributes") or event.get("data") or {}
    tool_name = attrs.get("tool_call.name") or attrs.get("tool") or attrs.get("tool_name") or "unknown_tool"
    args = attrs.get("tool_call.arguments") or attrs.get("args") or {}
    observation = attrs.get("observation") or attrs.get("result")
    compact, compressed = compress_tool_result(tool_name, args if isinstance(args, dict) else {}, observation)
    return {
        "step": event.get("step") or attrs.get("step"),
        "tool": tool_name,
        "arguments": args,
        "summary": compact,
        "compressed": compressed,
    }


def collect_recent_tool_summaries(trace, limit=5):
    summaries = []
    for event in reversed(trace.get("events", [])):
        event_type = event.get("event_type") or event.get("type")
        if event_type != "tool_result":
            continue
        summaries.append(summarize_tool_event(event))
        if len(summaries) >= limit:
            break
    return list(reversed(summaries))


def _section(title, items):
    lines = [f"## {title}"]
    if not items:
        lines.append("- None")
    elif isinstance(items, str):
        lines.append(items)
    else:
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def build_context_pack(task_state, recent_trace, tool_summaries, max_chars=12000):
    lines = [
        "# Context Pack",
        "",
        "This is the compact runtime memory for the next model call. Full audit data stays in trace.",
        "If Completed Steps and Recent Tool Calls already satisfy the user goal, stop tool use and answer now.",
        "",
        _section("User Goal", task_state.user_goal),
        "",
        _section("Current Plan", task_state.current_plan),
        "",
        _section("Completed Steps", task_state.completed_steps[-12:]),
        "",
        _section("Open Questions", task_state.open_questions[-8:]),
        "",
        _section("Important Facts", task_state.important_facts[-12:]),
        "",
        _section("Files Touched", task_state.files_touched[-20:]),
        "",
        _section("Commands Run", task_state.commands_run[-12:]),
        "",
        _section("Error History", task_state.error_history[-12:]),
        "",
        _section("Resolved Errors", task_state.resolved_errors[-12:]),
        "",
        _section("Recent Tool Calls", [_format_tool_summary(item) for item in tool_summaries[-5:]]),
        "",
        _section("Recent Error", task_state.last_error or "None"),
        "",
        _section("Next Action", task_state.next_action_hint or "Choose the next smallest useful action."),
    ]

    recent_event_lines = _format_recent_events(recent_trace[-8:])
    if recent_event_lines:
        lines.extend(["", _section("Recent Trace Events", recent_event_lines)])

    pack = "\n".join(lines).strip() + "\n"
    if len(pack) <= max_chars:
        return pack

    trimmed_lines = [
        "# Context Pack",
        "",
        "This context pack was truncated to fit the model input budget.",
        "If Completed Steps and Recent Tool Calls already satisfy the user goal, stop tool use and answer now.",
        "",
        _section("User Goal", task_state.user_goal),
        "",
        _section("Current Plan", task_state.current_plan[:6]),
        "",
        _section("Completed Steps", task_state.completed_steps[-6:]),
        "",
        _section("Important Facts", task_state.important_facts[-8:]),
        "",
        _section("Commands Run", task_state.commands_run[-6:]),
        "",
        _section("Error History", task_state.error_history[-6:]),
        "",
        _section("Resolved Errors", task_state.resolved_errors[-6:]),
        "",
        _section("Recent Tool Calls", [_format_tool_summary(item, limit=700) for item in tool_summaries[-3:]]),
        "",
        _section("Recent Error", task_state.last_error or "None"),
        "",
        _section("Next Action", task_state.next_action_hint or "Choose the next smallest useful action."),
    ]
    pack = "\n".join(trimmed_lines).strip() + "\n"
    if len(pack) > max_chars:
        pack = pack[:max_chars] + "\n...[context_pack_truncated]\n"
    return pack


def _format_tool_summary(item, limit=1200):
    text = json.dumps(item.get("summary"), ensure_ascii=False, sort_keys=True)
    return (
        f"step {item.get('step')}: {item.get('tool')} "
        f"compressed={item.get('compressed')} summary={_shorten(text, limit)}"
    )


def _format_recent_events(events):
    lines = []
    for event in events:
        event_type = event.get("event_type") or event.get("type")
        attrs = event.get("attributes") or event.get("data") or {}
        step = event.get("step") or attrs.get("step")
        if event_type in {"llm_called", "context_pack_built", "checkpoint_saved", "task_state_updated"}:
            continue
        if event_type == "tool_result":
            tool_name = attrs.get("tool_call.name") or attrs.get("tool")
            error = attrs.get("error")
            status = "ok=false" if error else "ok=true"
            lines.append(f"step {step}: tool_result {tool_name} {status}")
        elif event_type == "llm_result":
            calls = attrs.get("tool_calls") or []
            names = [call.get("name") for call in calls]
            lines.append(f"step {step}: llm_result tool_calls={names}")
        elif event_type == "final_answer":
            lines.append(f"step {step}: final_answer exit={attrs.get('exit_reason')}")
        elif event_type:
            lines.append(f"step {step}: {event_type}")
    return lines


def save_context_pack(context_pack, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(context_pack)
