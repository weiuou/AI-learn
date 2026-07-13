from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


RECOVERY_HINT = {
    "type": "loop_recovery",
    "instruction": "The recent tool actions made no progress. Re-plan once, choose a materially different action, and do not repeat the detected pattern.",
}


def normalize_arguments(arguments) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            return arguments.strip()
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_fingerprint(tool_name: str, arguments, error_type: str | None) -> str:
    return "\x1f".join([tool_name or "unknown_tool", normalize_arguments(arguments), error_type or ""])


def fingerprint_summary(fingerprint: str) -> str:
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class LoopDecision:
    detected: bool = False
    pattern: str | None = None
    steps: tuple[int, ...] = ()
    fingerprints: tuple[str, ...] = ()
    should_stop: bool = False
    recovery_hint: dict | None = None


class LoopDetector:
    def __init__(self):
        self.actions: list[tuple[str, int, bool]] = []
        self.failure_steps: list[int] = []
        self.recovery_used = False
        self.recovery_pending = False
        self.awaiting_recovery_outcome = False

    def activate_recovery(self) -> None:
        if self.recovery_pending:
            self.recovery_pending = False
            self.awaiting_recovery_outcome = True

    def _pattern(self):
        if len(self.actions) >= 3:
            last = self.actions[-3:]
            if last[0][0] == last[1][0] == last[2][0]:
                return "repeat_same_action", tuple(item[1] for item in last), tuple(item[0] for item in last)
        if len(self.actions) >= 4:
            last = self.actions[-4:]
            if last[0][0] == last[2][0] and last[1][0] == last[3][0] and last[0][0] != last[1][0]:
                return "alternating_actions", tuple(item[1] for item in last), tuple(item[0] for item in last)
        if len(self.failure_steps) >= 3:
            last = self.actions[-3:]
            if last[0][0] == last[2][0] and last[0][0] != last[1][0]:
                # A/B/A is a strong alternating candidate; wait one action so
                # A/B/A/B is reported with the more specific pattern.
                return None
            return "consecutive_failures", tuple(self.failure_steps[-3:]), ()
        return None

    def observe(self, tool_name: str, arguments, error_type: str | None, ok: bool, step: int) -> LoopDecision:
        fingerprint = action_fingerprint(tool_name, arguments, error_type if not ok else None)
        self.actions.append((fingerprint, step, ok))

        if ok:
            self.failure_steps = []
            self.awaiting_recovery_outcome = False
        else:
            self.failure_steps.append(step)
            if self.awaiting_recovery_outcome:
                return LoopDecision(
                    detected=True,
                    pattern="recovery_failed",
                    steps=(step,),
                    fingerprints=(fingerprint_summary(fingerprint),),
                    should_stop=True,
                )

        if self.recovery_pending:
            return LoopDecision()

        matched = self._pattern()
        if not matched:
            return LoopDecision()

        pattern, steps, fingerprints = matched
        summaries = tuple(fingerprint_summary(item) for item in fingerprints)
        if self.recovery_used:
            return LoopDecision(True, pattern, steps, summaries, True)

        self.recovery_used = True
        self.recovery_pending = True
        return LoopDecision(True, pattern, steps, summaries, False, dict(RECOVERY_HINT))
