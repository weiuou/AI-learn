from agent.loop_detector import LoopDetector, action_fingerprint, normalize_arguments


def test_argument_normalization_is_stable():
    assert normalize_arguments({"b": 2, "a": {"z": 1, "x": 0}}) == '{"a":{"x":0,"z":1},"b":2}'
    assert action_fingerprint("read_file", {"a": 1}, "FILE_NOT_FOUND") == action_fingerprint(
        "read_file", '{"a": 1}', "FILE_NOT_FOUND"
    )


def test_repeat_same_missing_file_allows_one_replan_then_stops():
    detector = LoopDetector()
    decisions = [
        detector.observe("read_file", {"path": "missing"}, "FILE_NOT_FOUND", False, step)
        for step in [1, 2, 3]
    ]
    assert decisions[-1].pattern == "repeat_same_action"
    assert decisions[-1].should_stop is False
    detector.activate_recovery()
    stopped = detector.observe("read_file", {"path": "missing"}, "FILE_NOT_FOUND", False, 4)
    assert stopped.pattern == "recovery_failed"
    assert stopped.should_stop is True


def test_alternating_failed_actions():
    detector = LoopDetector()
    decision = None
    for step, tool in enumerate(["read_file", "run_shell", "read_file", "run_shell"], start=1):
        decision = detector.observe(tool, {"value": tool}, "FAILED", False, step)
    assert decision.pattern == "alternating_actions"
    assert decision.steps == (1, 2, 3, 4)


def test_success_resets_consecutive_failures():
    detector = LoopDetector()
    detector.observe("a", {}, "FAILED", False, 1)
    detector.observe("b", {}, "FAILED", False, 2)
    detector.observe("c", {}, None, True, 3)
    decision = detector.observe("d", {}, "FAILED", False, 4)
    assert not decision.detected


def test_three_distinct_failures_detect_no_progress():
    detector = LoopDetector()
    detector.observe("a", {}, "FAILED", False, 1)
    detector.observe("b", {}, "FAILED", False, 2)
    decision = detector.observe("c", {}, "FAILED", False, 3)
    assert decision.pattern == "consecutive_failures"
    assert decision.should_stop is False
