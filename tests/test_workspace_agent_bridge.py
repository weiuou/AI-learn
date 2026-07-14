from agent.core import new_trace, run_agent


class ModelMustNotRun:
    class Chat:
        class Completions:
            def create(self, **kwargs):
                raise AssertionError("Model should not be called after cancellation.")

        completions = Completions()

    chat = Chat()


def test_cancel_check_stops_before_model_call(tmp_path):
    trace = new_trace("cancel this", task_id="cancel-test")
    answer = run_agent(
        "cancel this",
        trace,
        run_dir=str(tmp_path),
        model_client=ModelMustNotRun(),
        cancel_check=lambda: True,
    )
    assert answer == "任务已由用户取消。"
    event_types = [event["event_type"] for event in trace["events"]]
    assert "run_cancelled" in event_types
    final = next(event for event in trace["events"] if event["event_type"] == "final_answer")
    assert final["attributes"]["exit_reason"] == "cancelled"
