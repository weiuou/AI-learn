import pytest
from pydantic import ValidationError

from agent.budget import BudgetGuard, RunBudget


def test_run_budget_requires_positive_limits():
    with pytest.raises(ValidationError):
        RunBudget(max_steps=0)


def test_prompt_and_tool_budgets_stop_before_work():
    guard = BudgetGuard(RunBudget(max_steps=2, max_model_calls=2, max_tool_calls=1, max_prompt_chars=10))
    assert guard.check_before_model(6) is None
    guard.record_model_call(6)
    exceeded = guard.check_before_model(5)
    assert exceeded.limit == "max_prompt_chars"

    tool_guard = BudgetGuard(RunBudget(max_tool_calls=1))
    assert tool_guard.check_before_tool() is None
    tool_guard.record_tool_call()
    assert tool_guard.check_before_tool().limit == "max_tool_calls"


def test_consecutive_failure_recovery_grace():
    guard = BudgetGuard(RunBudget(max_consecutive_failures=3))
    for _ in range(3):
        guard.record_tool_result(False)
    assert guard.check_before_model(1).limit == "max_consecutive_failures"
    guard.exceeded = None
    guard.grant_failure_recovery()
    assert guard.check_before_model(1) is None
    guard.record_tool_result(True)
    assert guard.consecutive_failures == 0


def test_wall_time_budget_uses_injected_monotonic_clock():
    ticks = iter([10.0, 16.0])
    guard = BudgetGuard(RunBudget(max_wall_time_sec=5), clock=lambda: next(ticks))
    assert guard.check_before_tool().limit == "max_wall_time_sec"
