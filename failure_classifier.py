def _event_type(event):
    return event.get("event_type") or event.get("type")


def _attributes(event):
    return event.get("attributes") or event.get("data") or {}


def _tool_error_types(trace):
    error_types = []

    for event in trace.get("events", []):
        attrs = _attributes(event)
        observation = attrs.get("observation") or attrs.get("result")
        error = attrs.get("error")

        if isinstance(error, dict) and error.get("error_type"):
            error_types.append(error.get("error_type"))
        elif isinstance(observation, dict) and observation.get("error_type"):
            error_types.append(observation.get("error_type"))

    return error_types


def _final_answer_event(trace):
    for event in reversed(trace.get("events", [])):
        if _event_type(event) == "final_answer":
            return event
    return None


def _has_event(trace, event_type):
    return any(_event_type(event) == event_type for event in trace.get("events", []))


def _runtime_error_types(trace):
    error_types = []
    for event in trace.get("events", []):
        if _event_type(event) != "error":
            continue
        attrs = _attributes(event)
        if attrs.get("error_type"):
            error_types.append(attrs.get("error_type"))
    return error_types


def classify_failure(trace, checks, task=None):
    final_event = _final_answer_event(trace)
    if final_event is None:
        return "MAX_STEPS_EXCEEDED"

    final_attrs = _attributes(final_event)
    if final_attrs.get("exit_reason") == "max_steps":
        return "MAX_STEPS_EXCEEDED"

    error_types = _tool_error_types(trace)
    runtime_error_types = _runtime_error_types(trace)
    contains_failed = checks.get("expected_contains") is False

    if "REQUEST_TIMEOUT" in runtime_error_types:
        return "COMMAND_TIMEOUT"

    if runtime_error_types:
        return "UNKNOWN"

    if "COMMAND_TIMEOUT" in error_types:
        return "COMMAND_TIMEOUT"

    if "INVALID_ARGUMENTS" in error_types:
        return "INVALID_ARGUMENTS"

    if "FILE_NOT_FOUND" in error_types and contains_failed:
        return "FILE_NOT_FOUND_UNRECOVERED"

    if _has_event(trace, "context_compressed") and contains_failed:
        return "CONTEXT_LOSS"

    if contains_failed:
        return "FINAL_ANSWER_INCOMPLETE"

    if checks.get("expected_error_types") is False:
        return "TOOL_SELECTION_ERROR"

    if checks.get("expected_tool_policy") is False:
        return "TOOL_POLICY_ERROR"

    if checks.get("forbidden_tool_policy") is False:
        return "TOOL_POLICY_ERROR"

    if checks.get("max_steps") is False:
        return "MAX_STEPS_EXCEEDED"

    return "UNKNOWN"
