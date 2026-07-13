from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel, Field


class RunBudget(BaseModel):
    max_steps: int = Field(default=20, gt=0)
    max_model_calls: int = Field(default=20, gt=0)
    max_tool_calls: int = Field(default=30, gt=0)
    max_prompt_chars: int = Field(default=120_000, gt=0)
    max_wall_time_sec: int = Field(default=300, gt=0)
    max_consecutive_failures: int = Field(default=3, gt=0)


@dataclass(frozen=True)
class BudgetExceeded:
    limit: str
    limit_value: int
    consumed: int | float
    attempted: int | float


class BudgetGuard:
    """Stateful, independently testable guard around one run segment."""

    def __init__(
        self,
        budget: RunBudget | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.budget = budget or RunBudget()
        self.clock = clock
        self.started_at = clock()
        self.steps = 0
        self.model_calls = 0
        self.tool_calls = 0
        self.prompt_chars = 0
        self.consecutive_failures = 0
        self.exceeded: Optional[BudgetExceeded] = None
        self.failure_recovery_granted = False

    def elapsed_sec(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    def _exceed(self, name, limit, consumed, attempted) -> BudgetExceeded:
        self.exceeded = BudgetExceeded(name, limit, consumed, attempted)
        return self.exceeded

    def _check_wall_time(self) -> Optional[BudgetExceeded]:
        elapsed = self.elapsed_sec()
        if elapsed >= self.budget.max_wall_time_sec:
            return self._exceed(
                "max_wall_time_sec",
                self.budget.max_wall_time_sec,
                elapsed,
                elapsed,
            )
        return None

    def check_before_model(self, prompt_chars: int) -> Optional[BudgetExceeded]:
        wall = self._check_wall_time()
        if wall:
            return wall
        if self.steps >= self.budget.max_steps:
            return self._exceed("max_steps", self.budget.max_steps, self.steps, self.steps + 1)
        if self.model_calls >= self.budget.max_model_calls:
            return self._exceed(
                "max_model_calls",
                self.budget.max_model_calls,
                self.model_calls,
                self.model_calls + 1,
            )
        attempted_prompt_chars = self.prompt_chars + max(0, prompt_chars)
        if attempted_prompt_chars > self.budget.max_prompt_chars:
            return self._exceed(
                "max_prompt_chars",
                self.budget.max_prompt_chars,
                self.prompt_chars,
                attempted_prompt_chars,
            )
        if (
            self.consecutive_failures >= self.budget.max_consecutive_failures
            and not self.failure_recovery_granted
        ):
            return self._exceed(
                "max_consecutive_failures",
                self.budget.max_consecutive_failures,
                self.consecutive_failures,
                self.consecutive_failures,
            )
        return None

    def record_model_call(self, prompt_chars: int) -> None:
        self.steps += 1
        self.model_calls += 1
        self.prompt_chars += max(0, prompt_chars)

    def check_before_tool(self) -> Optional[BudgetExceeded]:
        wall = self._check_wall_time()
        if wall:
            return wall
        if self.tool_calls >= self.budget.max_tool_calls:
            return self._exceed(
                "max_tool_calls",
                self.budget.max_tool_calls,
                self.tool_calls,
                self.tool_calls + 1,
            )
        return None

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_tool_result(self, ok: bool) -> None:
        if ok:
            self.consecutive_failures = 0
            self.failure_recovery_granted = False
        else:
            self.consecutive_failures += 1

    def grant_failure_recovery(self) -> None:
        self.failure_recovery_granted = True

    def summary(self) -> dict:
        exceeded = None
        if self.exceeded:
            exceeded = {
                "limit": self.exceeded.limit,
                "limit_value": self.exceeded.limit_value,
                "consumed": self.exceeded.consumed,
                "attempted": self.exceeded.attempted,
            }
        return {
            "limits": self.budget.model_dump(),
            "consumed": {
                "steps": self.steps,
                "model_calls": self.model_calls,
                "tool_calls": self.tool_calls,
                "prompt_chars": self.prompt_chars,
                "consecutive_failures": self.consecutive_failures,
            },
            "elapsed_sec": round(self.elapsed_sec(), 6),
            "exceeded_limit": exceeded,
        }
