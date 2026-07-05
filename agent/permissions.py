import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyResult:
    decision: PolicyDecision
    risk_level: ToolRiskLevel
    risk_reason: str
    error_type: Optional[str] = None
    suggestion: Optional[str] = None

    @property
    def approval_required(self):
        return self.decision == PolicyDecision.REQUIRE_APPROVAL


ALLOWED_COMMANDS = [
    "pwd",
    "ls",
    "cat",
    "grep",
    "find",
    "sed",
    "python",
    "python3",
    "pytest",
    "git diff",
    "git status",
]

DENIED_PATTERNS = [
    (r"\brm\s+-[^\n;|&]*r[^\n;|&]*f|\brm\s+-[^\n;|&]*f[^\n;|&]*r", "rm -rf is not allowed."),
    (r"(^|[;&|]\s*)sudo\b", "sudo is not allowed."),
    (r"(^|[;&|]\s*)curl\b", "curl is not allowed."),
    (r"(^|[;&|]\s*)wget\b", "wget is not allowed."),
    (r"(^|[;&|]\s*)ssh\b", "ssh is not allowed."),
    (r"(^|[;&|]\s*)scp\b", "scp is not allowed."),
    (r"\bchmod\s+777\b", "chmod 777 is not allowed."),
    (r"(^|[;&|]\s*)mkfs\b", "mkfs is not allowed."),
    (re.escape(":(){ :|:& };:"), "fork bomb pattern is not allowed."),
    (r">+\s*/dev/", "redirecting output into /dev is not allowed."),
    (r">+\s*/etc/", "redirecting output into /etc is not allowed."),
    (r">+\s*(~|/users/[^/]+)/\.ssh/", "redirecting output into ~/.ssh is not allowed."),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _first_command_token(command):
    stripped = command.strip()
    if not stripped:
        return ""

    first_segment = re.split(r"\s*(?:;|&&|\|\|)\s*", stripped, maxsplit=1)[0]
    parts = first_segment.split()
    if not parts:
        return ""

    if len(parts) >= 2 and f"{parts[0]} {parts[1]}" in ALLOWED_COMMANDS:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def _external_absolute_path(command):
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    for part in parts[1:]:
        candidate = part.strip()
        if not candidate.startswith("/"):
            continue
        resolved = Path(candidate).resolve()
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            return candidate
    return None


def evaluate_shell_command(command):
    lowered = command.strip().lower()
    if not lowered:
        return PolicyResult(
            decision=PolicyDecision.DENY,
            risk_level=ToolRiskLevel.HIGH,
            risk_reason="empty shell command",
            error_type="INVALID_ARGUMENTS",
            suggestion="Provide a non-empty command.",
        )

    for pattern, reason in DENIED_PATTERNS:
        if re.search(pattern, lowered):
            return PolicyResult(
                decision=PolicyDecision.DENY,
                risk_level=ToolRiskLevel.HIGH,
                risk_reason=reason,
                error_type="PERMISSION_DENIED",
                suggestion="Use a safer read-only command such as ls, git status, or git diff.",
            )

    external_path = _external_absolute_path(command)
    if external_path:
        return PolicyResult(
            decision=PolicyDecision.DENY,
            risk_level=ToolRiskLevel.HIGH,
            risk_reason=f"absolute path outside project sandbox is not allowed: {external_path}",
            error_type="PERMISSION_DENIED",
            suggestion="Use a relative path inside the project directory.",
        )

    token = _first_command_token(lowered)
    if token not in ALLOWED_COMMANDS:
        return PolicyResult(
            decision=PolicyDecision.REQUIRE_APPROVAL,
            risk_level=ToolRiskLevel.HIGH,
            risk_reason=f"command '{token or lowered}' is not in the allow list",
            error_type="PERMISSION_DENIED",
            suggestion="Use an allowed project command such as ls, find, pytest, git status, or git diff.",
        )

    if token in {"pytest", "python", "python3"}:
        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            risk_level=ToolRiskLevel.MEDIUM,
            risk_reason=f"allowed project execution command: {token}",
        )

    return PolicyResult(
        decision=PolicyDecision.ALLOW,
        risk_level=ToolRiskLevel.LOW,
        risk_reason=f"allowed read-only inspection command: {token}",
    )


def tool_error(error_type, message, recoverable=True, suggestion=None):
    return {
        "ok": False,
        "result": None,
        "error_type": error_type,
        "message": message,
        "recoverable": recoverable,
        "suggestion": suggestion,
    }


def permission_denied(message, suggestion=None):
    return tool_error(
        "PERMISSION_DENIED",
        message,
        True,
        suggestion or "Ask for approval or choose a safer project-scoped action.",
    )
