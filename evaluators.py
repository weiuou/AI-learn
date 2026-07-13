def _event_type(event):
    return event.get("event_type") or event.get("type")


def _attributes(event):
    return event.get("attributes") or event.get("data") or {}


def _casefold(value):
    return (value or "").casefold()


def check_expected_contains(final_answer, keywords, extra_text=""):
    if not keywords:
        return True

    answer = _casefold(final_answer + "\n" + (extra_text or ""))
    return all(_casefold(keyword) in answer for keyword in keywords)


def check_expected_error_types(trace, error_types):
    if not error_types:
        return True

    expected = set(error_types)
    seen = set()

    for event in trace.get("events", []):
        attrs = _attributes(event)
        observation = attrs.get("observation") or attrs.get("result")
        error = attrs.get("error")

        if isinstance(error, dict) and error.get("error_type"):
            seen.add(error.get("error_type"))

        if isinstance(observation, dict) and observation.get("error_type"):
            seen.add(observation.get("error_type"))

    return expected.issubset(seen)


def check_max_steps(trace, max_steps):
    if max_steps is None:
        return True

    final_event = None
    max_seen_step = 0

    for event in trace.get("events", []):
        attrs = _attributes(event)
        step = event.get("step") or attrs.get("step") or 0
        if isinstance(step, int):
            max_seen_step = max(max_seen_step, step)
        if _event_type(event) == "final_answer":
            final_event = event

    if final_event is None:
        return False

    final_attrs = _attributes(final_event)
    if final_attrs.get("exit_reason") == "max_steps":
        return False

    return max_seen_step <= max_steps


def _matches_policy_expectation(event, expected):
    attrs = _attributes(event)
    tool_name = attrs.get("tool_call.name") or attrs.get("tool")
    if expected.get("tool") and tool_name != expected.get("tool"):
        return False

    args = attrs.get("tool_call.arguments") or attrs.get("args") or {}
    args_text = args if isinstance(args, str) else str(args)
    if expected.get("contains") and expected.get("contains") not in args_text:
        return False

    for key in ["risk_level", "policy_decision", "approval_required", "approved"]:
        if key in expected and attrs.get(key) != expected.get(key):
            return False

    if "error_type" in expected:
        observation = attrs.get("observation") or attrs.get("result")
        error = attrs.get("error")
        seen_error_type = None
        if isinstance(error, dict):
            seen_error_type = error.get("error_type")
        if seen_error_type is None and isinstance(observation, dict):
            seen_error_type = observation.get("error_type")
        if seen_error_type != expected.get("error_type"):
            return False

    return True


def check_expected_tool_policy(trace, expectations):
    if not expectations:
        return True

    events = [
        event
        for event in trace.get("events", [])
        if _event_type(event) in {"tool_called", "tool_result"}
    ]
    return all(
        any(_matches_policy_expectation(event, expected) for event in events)
        for expected in expectations
    )


def check_forbidden_tool_policy(trace, expectations):
    if not expectations:
        return True

    events = [
        event
        for event in trace.get("events", [])
        if _event_type(event) in {"tool_called", "tool_result"}
    ]
    return not any(
        _matches_policy_expectation(event, expected)
        for expected in expectations
        for event in events
    )


def evaluate_task(trace, final_answer, task):
    checks = {}

    if "expected_contains" in task:
        extra_text = ""
        if task.get("category") == "context":
            extra_text = str(trace)
        checks["expected_contains"] = check_expected_contains(
            final_answer,
            task.get("expected_contains") or [],
            extra_text=extra_text,
        )

    if "expected_error_types" in task:
        checks["expected_error_types"] = check_expected_error_types(
            trace,
            task.get("expected_error_types") or [],
        )

    if "max_steps" in task:
        checks["max_steps"] = check_max_steps(trace, task.get("max_steps"))

    if "expected_tool_policy" in task:
        checks["expected_tool_policy"] = check_expected_tool_policy(
            trace,
            task.get("expected_tool_policy") or [],
        )

    if "forbidden_tool_policy" in task:
        checks["forbidden_tool_policy"] = check_forbidden_tool_policy(
            trace,
            task.get("forbidden_tool_policy") or [],
        )

    return checks
