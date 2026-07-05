import os
import subprocess
from pathlib import Path

from .permissions import permission_denied, tool_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_OUTPUT_CHARS = 8000
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL")


def tool_success(result):
    return {
        "ok": True,
        "result": result,
        "error_type": None,
        "message": None,
        "recoverable": None,
        "suggestion": None,
    }


def shorten(value, limit=500):
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def resolve_project_path(path):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return None, permission_denied(
            f"Path is outside the project sandbox: {path}",
            "Choose a path inside the project directory.",
        )
    return resolved, None


def clean_shell_env():
    clean = {}
    allow = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PWD",
        "SHELL",
        "TERM",
        "TMPDIR",
        "USER",
    }
    for key, value in os.environ.items():
        if key in allow and not any(marker in key.upper() for marker in SECRET_MARKERS):
            clean[key] = value
    return clean


def truncate_output(text, max_chars):
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "...[truncated]", True


def run_shell_command(command, cwd=None, timeout_sec=DEFAULT_TIMEOUT_SECONDS, max_output_chars=DEFAULT_MAX_OUTPUT_CHARS):
    cwd_path, error = resolve_project_path(cwd or ".")
    if error:
        return error, {"truncated": False, "timeout_sec": timeout_sec}
    if not cwd_path.is_dir():
        return tool_error(
            "INVALID_ARGUMENTS",
            f"cwd is not a directory inside the project sandbox: {cwd or '.'}",
            True,
            "Use an existing project directory as cwd.",
        ), {"truncated": False, "timeout_sec": timeout_sec}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=str(cwd_path),
            env=clean_shell_env(),
        )
    except subprocess.TimeoutExpired as e:
        stdout, stdout_truncated = truncate_output(e.stdout or "", max_output_chars)
        stderr, stderr_truncated = truncate_output(e.stderr or "", max_output_chars)
        return tool_error(
            "COMMAND_TIMEOUT",
            f"Command timed out after {timeout_sec} seconds.",
            True,
            "Narrow the command with filters, limit output, or inspect a smaller path.",
        ), {
            "timeout_sec": timeout_sec,
            "stdout_preview": stdout,
            "stderr_preview": stderr,
            "truncated": stdout_truncated or stderr_truncated,
        }
    except Exception as e:
        return tool_error(
            "COMMAND_FAILED",
            f"Command could not be executed: {e}",
            True,
            "Try a simpler project-scoped command.",
        ), {"truncated": False, "timeout_sec": timeout_sec}

    stdout, stdout_truncated = truncate_output(result.stdout, max_output_chars)
    stderr, stderr_truncated = truncate_output(result.stderr, max_output_chars)
    metadata = {
        "timeout_sec": timeout_sec,
        "exit_code": result.returncode,
        "stdout_preview": stdout,
        "stderr_preview": stderr,
        "truncated": stdout_truncated or stderr_truncated,
    }
    payload = {
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": metadata["truncated"],
    }
    if result.returncode != 0:
        output = stderr.strip() or stdout.strip()
        message = f"Command exited with return code {result.returncode}."
        if output:
            message = f"{message} Output: {shorten(output, 300)}"
        return tool_error(
            "COMMAND_FAILED",
            message,
            True,
            "Read stderr and retry with a narrower or corrected command.",
        ), metadata
    return tool_success(payload), metadata
