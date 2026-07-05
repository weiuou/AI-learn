import json
import re


COMPRESSION_THRESHOLD = 12000
DEFAULT_TOOL_RESULT_LIMIT = 2000
AGGRESSIVE_TOOL_RESULT_LIMIT = 900
DEFAULT_HEAD_CHARS = 600
DEFAULT_TAIL_CHARS = 600
DEFAULT_STRUCTURE_LINE_LIMIT = 80
AGGRESSIVE_HEAD_CHARS = 300
AGGRESSIVE_TAIL_CHARS = 300
AGGRESSIVE_STRUCTURE_LINE_LIMIT = 35
COMPRESSION_STRATEGY = "generic_text_structure_v1"


def estimate_messages_size(messages):
    return len(json.dumps(messages, ensure_ascii=False))


def _event_type(event):
    return event.get("event_type") or event.get("type")


def _attributes(event):
    return event.get("attributes") or event.get("data") or {}


def _shorten(value, limit=300):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _parse_json_object(value):
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_tool_arguments(raw_arguments):
    if isinstance(raw_arguments, dict):
        return raw_arguments
    return _parse_json_object(raw_arguments) or {}


def _first_non_empty_lines(lines, limit=20):
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            result.append(stripped)
        if len(result) >= limit:
            break
    return result


def _extract_structure_lines(lines, limit=DEFAULT_STRUCTURE_LINE_LIMIT):
    patterns = [
        r"^\s*#{1,6}\s+\S",
        r"^\s*(import|from|package|using|namespace)\b",
        r"^\s*#\s*include\b",
        r"^\s*(export\s+)?(default\s+)?(class|struct|interface|enum)\b",
        r"^\s*(async\s+)?(def|function|func|fn)\b",
        r"^\s*(public|private|protected|static|inline|virtual|constexpr|const|let|var|type)\b",
        r"^\s*if\s+__name__\s*==\s*[\"']__main__[\"']",
        r"\bmain\s*\(",
        r"^\s*[A-Za-z_][\w:<>~*&\s]+\s+[A-Za-z_]\w*(::[A-Za-z_]\w*)?\s*\([^;{}]*\)\s*(const)?\s*[{;]?\s*$",
    ]
    combined = re.compile("|".join(f"(?:{pattern})" for pattern in patterns))
    result = []
    seen = set()

    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or len(stripped) > 220:
            continue
        if not combined.search(stripped):
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"{number}: {stripped}")
        if len(result) >= limit:
            break

    return result


def _summarize_text_result(
    text,
    path=None,
    head_chars=DEFAULT_HEAD_CHARS,
    tail_chars=DEFAULT_TAIL_CHARS,
    structure_line_limit=DEFAULT_STRUCTURE_LINE_LIMIT,
    max_chars=DEFAULT_TOOL_RESULT_LIMIT,
):
    lines = text.splitlines()
    structure_lines = _extract_structure_lines(lines, limit=structure_line_limit)
    first_lines = _first_non_empty_lines(lines, limit=12)
    head = text[:head_chars]
    tail = text[-tail_chars:] if len(text) > tail_chars else ""

    parts = [
        "Compressed text result.",
        f"Path: {path or '<unknown>'}",
        f"Original chars: {len(text)}",
        f"Original lines: {len(lines)}",
    ]

    if structure_lines:
        parts.append("Structure lines:")
        parts.extend(f"- {line}" for line in structure_lines)

    if first_lines:
        parts.append("First non-empty lines:")
        parts.extend(f"- {line}" for line in first_lines)

    parts.append("Head snippet:")
    parts.append(head)

    if tail and tail != head:
        parts.append("Tail snippet:")
        parts.append(tail)

    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "...[compressed_truncated]"
    return summary


def _compact_non_text_result(result, max_chars):
    summary = json.dumps(result, ensure_ascii=False)
    if len(summary) <= max_chars:
        return result, False, len(summary)
    return summary[:max_chars] + "...[compressed_truncated]", True, len(summary)


def _compact_tool_result_payload(
    payload,
    tool_name=None,
    args=None,
    max_chars=DEFAULT_TOOL_RESULT_LIMIT,
    head_chars=DEFAULT_HEAD_CHARS,
    tail_chars=DEFAULT_TAIL_CHARS,
    structure_line_limit=DEFAULT_STRUCTURE_LINE_LIMIT,
):
    if not isinstance(payload, dict):
        text = _shorten(payload, max_chars)
        return {
            "ok": True,
            "result": text,
            "error_type": None,
            "message": None,
            "recoverable": None,
            "suggestion": None,
            "compressed": True,
            "original_chars": len(str(payload)),
            "compressed_chars": len(text),
            "compression_strategy": COMPRESSION_STRATEGY,
            "tool_name": tool_name,
            "tool_arguments": args or {},
        }, True

    compact = {
        "ok": payload.get("ok"),
        "result": payload.get("result"),
        "error_type": payload.get("error_type"),
        "message": payload.get("message"),
        "recoverable": payload.get("recoverable"),
        "suggestion": payload.get("suggestion"),
    }

    result = payload.get("result")
    compressed = False
    original_chars = len(result) if isinstance(result, str) else len(json.dumps(result, ensure_ascii=False))
    path = (args or {}).get("path") if isinstance(args, dict) else None

    if isinstance(result, str) and len(result) > max_chars:
        compact["result"] = _summarize_text_result(
            result,
            path=path,
            head_chars=head_chars,
            tail_chars=tail_chars,
            structure_line_limit=structure_line_limit,
            max_chars=max_chars,
        )
        compressed = True
    elif not isinstance(result, (str, type(None))):
        compact["result"], compressed, original_chars = _compact_non_text_result(result, max_chars)

    compact.update(
        {
            "compressed": compressed,
            "original_chars": original_chars,
            "compressed_chars": len(json.dumps(compact.get("result"), ensure_ascii=False)),
            "compression_strategy": COMPRESSION_STRATEGY,
            "tool_name": tool_name,
            "tool_arguments": args or {},
        }
    )
    return compact, compressed


def _compact_tool_message(
    message,
    tool_call_meta=None,
    max_chars=DEFAULT_TOOL_RESULT_LIMIT,
    head_chars=DEFAULT_HEAD_CHARS,
    tail_chars=DEFAULT_TAIL_CHARS,
    structure_line_limit=DEFAULT_STRUCTURE_LINE_LIMIT,
):
    compacted = dict(message)
    tool_call_meta = tool_call_meta or {}
    tool_name = tool_call_meta.get("name")
    args = _parse_tool_arguments(tool_call_meta.get("arguments"))
    payload = _parse_json_object(message.get("content"))

    if payload is None:
        original = message.get("content") or ""
        compact_payload = {
            "ok": True,
            "result": _shorten(original, max_chars),
            "error_type": None,
            "message": None,
            "recoverable": None,
            "suggestion": None,
            "compressed": len(original) > max_chars,
            "original_chars": len(original),
            "compressed_chars": min(len(original), max_chars),
            "compression_strategy": COMPRESSION_STRATEGY,
            "tool_name": tool_name,
            "tool_arguments": args,
        }
        compacted["content"] = json.dumps(compact_payload, ensure_ascii=False)
        return compacted, 1 if compact_payload["compressed"] else 0

    compact_payload, compressed = _compact_tool_result_payload(
        payload,
        tool_name=tool_name,
        args=args,
        max_chars=max_chars,
        head_chars=head_chars,
        tail_chars=tail_chars,
        structure_line_limit=structure_line_limit,
    )
    compacted["content"] = json.dumps(compact_payload, ensure_ascii=False)
    return compacted, 1 if compressed else 0


def _summarize_trace(trace):
    user_goal = trace.get("user_goal") or trace.get("task") or ""
    completed_steps = []
    successful_files = []
    failed_tools = []

    for event in trace.get("events", []):
        event_type = _event_type(event)
        attrs = _attributes(event)
        step = attrs.get("step") or event.get("step")

        if event_type == "tool_result":
            tool_name = attrs.get("tool_call.name") or attrs.get("tool") or attrs.get("tool_name")
            observation = attrs.get("observation") or attrs.get("result")
            error = attrs.get("error")

            if isinstance(observation, dict) and observation.get("ok"):
                result = observation.get("result")
                if tool_name == "read_file":
                    args = attrs.get("tool_call.arguments") or attrs.get("args") or {}
                    path = args.get("path") if isinstance(args, dict) else None
                    if path:
                        successful_files.append(path)
                completed_steps.append(f"step {step}: {tool_name} succeeded")
            elif error or (isinstance(observation, dict) and not observation.get("ok")):
                err = error or observation
                error_type = err.get("error_type") if isinstance(err, dict) else "UNKNOWN"
                message = err.get("message") if isinstance(err, dict) else _shorten(err)
                failed_tools.append(f"step {step}: {tool_name} failed with {error_type}: {message}")

        elif event_type == "final_answer":
            completed_steps.append(f"step {step}: final answer produced")

    lines = [
        "Context compression summary for the ongoing agent run.",
        f"User goal: {user_goal}",
        "Current task status: continue from the latest preserved messages and avoid repeating failed tool calls.",
    ]

    if completed_steps:
        lines.append("Completed steps:")
        lines.extend(f"- {item}" for item in completed_steps[-8:])

    if successful_files:
        lines.append("Discovered key files/info:")
        for path in sorted(set(successful_files)):
            lines.append(f"- Read file: {path}")

    if failed_tools:
        lines.append("Failed tool calls and reasons:")
        lines.extend(f"- {item}" for item in failed_tools[-8:])

    return "\n".join(lines)


def _split_turns(messages):
    turns = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            turn = [message]
            index += 1
            expected = {
                call.get("id")
                for call in message.get("tool_calls", [])
                if call.get("id")
            }
            while index < len(messages) and messages[index].get("role") == "tool":
                turn.append(messages[index])
                expected.discard(messages[index].get("tool_call_id"))
                index += 1
                if not expected:
                    break
            turns.append(turn)
        else:
            turns.append([message])
            index += 1
    return turns


def _tool_call_meta_by_id(turn):
    result = {}
    for message in turn:
        for call in message.get("tool_calls", []) or []:
            function = call.get("function") or {}
            call_id = call.get("id")
            if call_id:
                result[call_id] = {
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
    return result


def _compact_turn(
    turn,
    max_chars=DEFAULT_TOOL_RESULT_LIMIT,
    head_chars=DEFAULT_HEAD_CHARS,
    tail_chars=DEFAULT_TAIL_CHARS,
    structure_line_limit=DEFAULT_STRUCTURE_LINE_LIMIT,
):
    tool_meta = _tool_call_meta_by_id(turn)
    compacted = []
    compressed_tool_results = 0

    for message in turn:
        if message.get("role") == "tool":
            compacted_message, compressed = _compact_tool_message(
                message,
                tool_call_meta=tool_meta.get(message.get("tool_call_id")),
                max_chars=max_chars,
                head_chars=head_chars,
                tail_chars=tail_chars,
                structure_line_limit=structure_line_limit,
            )
            compacted.append(compacted_message)
            compressed_tool_results += compressed
        else:
            compacted.append(dict(message))

    return compacted, compressed_tool_results


def _compression_options(aggressive=False):
    if aggressive:
        return {
            "max_chars": AGGRESSIVE_TOOL_RESULT_LIMIT,
            "head_chars": AGGRESSIVE_HEAD_CHARS,
            "tail_chars": AGGRESSIVE_TAIL_CHARS,
            "structure_line_limit": AGGRESSIVE_STRUCTURE_LINE_LIMIT,
        }
    return {
        "max_chars": DEFAULT_TOOL_RESULT_LIMIT,
        "head_chars": DEFAULT_HEAD_CHARS,
        "tail_chars": DEFAULT_TAIL_CHARS,
        "structure_line_limit": DEFAULT_STRUCTURE_LINE_LIMIT,
    }


def _build_compressed_messages(messages, trace, aggressive=False):
    if len(messages) <= 4:
        return messages, {
            "compressed_tool_results": 0,
            "compression_strategy": COMPRESSION_STRATEGY,
            "aggressive": aggressive,
        }

    system_message = messages[0]
    user_message = messages[1]
    history = messages[2:]
    turns = _split_turns(history)
    preserved_turns = turns[-2:] if len(turns) > 2 else turns
    options = _compression_options(aggressive=aggressive)
    preserved_messages = []
    compressed_tool_results = 0

    for turn in preserved_turns:
        compacted_turn, compressed = _compact_turn(turn, **options)
        preserved_messages.extend(compacted_turn)
        compressed_tool_results += compressed

    summary_message = {
        "role": "system",
        "content": _summarize_trace(trace),
    }

    diagnostics = {
        "compressed_tool_results": compressed_tool_results,
        "compression_strategy": COMPRESSION_STRATEGY,
        "aggressive": aggressive,
    }
    return [system_message, user_message, summary_message] + preserved_messages, diagnostics


def compress_messages(messages, trace):
    compressed, diagnostics = _build_compressed_messages(messages, trace, aggressive=False)
    if estimate_messages_size(compressed) > COMPRESSION_THRESHOLD:
        compressed, diagnostics = _build_compressed_messages(messages, trace, aggressive=True)
    trace["_last_compression_diagnostics"] = diagnostics
    return compressed


def compression_diagnostics(messages):
    compressed_tool_results = 0

    for message in messages:
        if message.get("role") != "tool":
            continue
        payload = _parse_json_object(message.get("content"))
        if not payload:
            continue
        if payload.get("compressed"):
            compressed_tool_results += 1

    return {
        "compressed_tool_results": compressed_tool_results,
        "compression_strategy": COMPRESSION_STRATEGY,
        "aggressive": False,
    }
