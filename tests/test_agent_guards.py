from types import SimpleNamespace

from agent import RunBudget, new_trace, run_agent


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _completion(tool_calls=None, content=None):
    message = SimpleNamespace(tool_calls=tool_calls or [], content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class FakeClient:
    def __init__(self, completions):
        self.responses = list(completions)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_kwargs):
        self.calls += 1
        return self.responses.pop(0)


def _error_executor(calls):
    def execute(call):
        calls.append(call)
        return {
            "ok": False,
            "result": None,
            "error_type": "FILE_NOT_FOUND",
            "message": "missing",
            "recoverable": True,
            "suggestion": "choose another path",
        }, {}

    return execute


def _success_executor(calls):
    def execute(call):
        calls.append(call)
        return {"ok": True, "result": "ok", "error_type": None}, {}

    return execute


def test_repeated_failure_replans_once_then_stops():
    client = FakeClient([
        _completion([_tool_call(f"call-{index}", "read_file", '{"path":"missing"}')])
        for index in range(4)
    ])
    tool_calls = []
    trace = new_trace("read missing", task_id="loop-test")
    answer = run_agent(
        "read missing",
        trace,
        model_client=client,
        tool_executor=_error_executor(tool_calls),
        budget=RunBudget(max_steps=10, max_model_calls=10),
    )
    loop_events = [event for event in trace["events"] if event["event_type"] == "loop_detected"]
    final = [event for event in trace["events"] if event["event_type"] == "final_answer"][-1]
    assert "循环" in answer
    assert [event["attributes"]["action"] for event in loop_events] == ["replan", "stop"]
    assert final["attributes"]["exit_reason"] == "loop_detected"
    assert client.calls == 4
    assert len(tool_calls) == 4


def test_tool_budget_prevents_second_execution_and_result():
    client = FakeClient([
        _completion([
            _tool_call("one", "read_file", '{"path":"one"}'),
            _tool_call("two", "read_file", '{"path":"two"}'),
        ])
    ])
    executed = []
    trace = new_trace("two calls", task_id="budget-test")
    run_agent(
        "two calls",
        trace,
        model_client=client,
        tool_executor=_success_executor(executed),
        budget=RunBudget(max_steps=5, max_model_calls=5, max_tool_calls=1),
    )
    event_types = [event["event_type"] for event in trace["events"]]
    assert len(executed) == 1
    assert event_types.count("tool_called") == 1
    assert event_types.count("tool_result") == 1
    assert "budget_exceeded" in event_types
    assert trace["budget_summary"]["exceeded_limit"]["limit"] == "max_tool_calls"
