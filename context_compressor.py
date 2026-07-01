import json


COMPRESSION_THRESHOLD = 12000


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


def compress_messages(messages, trace):
    if len(messages) <= 4:
        return messages

    system_message = messages[0]
    user_message = messages[1]
    history = messages[2:]
    turns = _split_turns(history)
    preserved_turns = turns[-2:] if len(turns) > 2 else turns
    preserved_messages = [message for turn in preserved_turns for message in turn]

    summary_message = {
        "role": "system",
        "content": _summarize_trace(trace),
    }

    return [system_message, user_message, summary_message] + preserved_messages
