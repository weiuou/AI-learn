from dataclasses import dataclass

from .approval import request_cli_approval
from .permissions import PolicyDecision, ToolRiskLevel, evaluate_shell_command, permission_denied, tool_error
from .sandbox import DEFAULT_MAX_OUTPUT_CHARS, DEFAULT_TIMEOUT_SECONDS, resolve_project_path, run_shell_command, tool_success


@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: object
    risk_level: ToolRiskLevel
    approval_required: bool


def read_file(path):
    resolved, error = resolve_project_path(path)
    if error:
        return error, {
            "risk_level": ToolRiskLevel.LOW.value,
            "approval_required": False,
            "approved": None,
            "policy_decision": PolicyDecision.DENY.value,
            "risk_reason": "path escapes project sandbox",
            "truncated": False,
        }

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            return tool_success(f.read()), {
                "risk_level": ToolRiskLevel.LOW.value,
                "approval_required": False,
                "approved": None,
                "policy_decision": PolicyDecision.ALLOW.value,
                "risk_reason": "read-only project file access",
                "truncated": False,
            }
    except FileNotFoundError:
        return tool_error(
            "FILE_NOT_FOUND",
            f"{path} does not exist",
            True,
            "Use run_shell to list files, or search with find . -iname '*readme*' / find . -name '*.py'.",
        ), {
            "risk_level": ToolRiskLevel.LOW.value,
            "approval_required": False,
            "approved": None,
            "policy_decision": PolicyDecision.ALLOW.value,
            "risk_reason": "read-only project file access",
            "truncated": False,
        }
    except PermissionError:
        return permission_denied(
            f"Permission denied while reading {path}",
            "Ask the user for permission or choose a readable project file.",
        ), {
            "risk_level": ToolRiskLevel.LOW.value,
            "approval_required": False,
            "approved": None,
            "policy_decision": PolicyDecision.DENY.value,
            "risk_reason": "filesystem permission denied",
            "truncated": False,
        }
    except Exception as e:
        return tool_error(
            "READ_ERROR",
            f"Could not read {path}: {e}",
            True,
            "Check that the path is a regular UTF-8 text file, or list nearby files first.",
        ), {
            "risk_level": ToolRiskLevel.LOW.value,
            "approval_required": False,
            "approved": None,
            "policy_decision": PolicyDecision.ALLOW.value,
            "risk_reason": "read-only project file access",
            "truncated": False,
        }


def write_file(path, content):
    resolved, error = resolve_project_path(path)
    metadata = {
        "risk_level": ToolRiskLevel.MEDIUM.value,
        "approval_required": True,
        "approved": False,
        "policy_decision": PolicyDecision.REQUIRE_APPROVAL.value,
        "risk_reason": "writes project file content",
        "truncated": False,
    }
    if error:
        metadata["policy_decision"] = PolicyDecision.DENY.value
        metadata["risk_reason"] = "path escapes project sandbox"
        return error, metadata

    approved = request_cli_approval("write_file", {"path": path}, ToolRiskLevel.MEDIUM.value, metadata["risk_reason"])
    metadata["approved"] = approved
    if not approved:
        return permission_denied(
            f"write_file requires approval for path: {path}",
            "Approve the write in an interactive run, or choose a read-only action.",
        ), metadata

    try:
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        metadata["policy_decision"] = PolicyDecision.ALLOW.value
        return tool_success(f"Wrote {len(content)} characters to {path}"), metadata
    except PermissionError:
        metadata["policy_decision"] = PolicyDecision.DENY.value
        metadata["risk_reason"] = "filesystem permission denied"
        return permission_denied(
            f"Permission denied while writing {path}",
            "Ask the user for permission or choose a writable project path.",
        ), metadata
    except Exception as e:
        return tool_error(
            "WRITE_ERROR",
            f"Could not write {path}: {e}",
            True,
            "Check the directory exists and retry with a writable path.",
        ), metadata


def run_shell(command, cwd=".", timeout_sec=DEFAULT_TIMEOUT_SECONDS, max_output_chars=DEFAULT_MAX_OUTPUT_CHARS):
    policy = evaluate_shell_command(command)
    metadata = {
        "risk_level": policy.risk_level.value,
        "approval_required": policy.approval_required,
        "approved": None,
        "policy_decision": policy.decision.value,
        "risk_reason": policy.risk_reason,
        "timeout_sec": timeout_sec,
        "truncated": False,
    }

    if policy.decision == PolicyDecision.DENY:
        return permission_denied(
            f"Command blocked by policy: {policy.risk_reason}",
            policy.suggestion,
        ), metadata

    if policy.decision == PolicyDecision.REQUIRE_APPROVAL:
        approved = request_cli_approval(
            "run_shell",
            {"command": command, "cwd": cwd},
            policy.risk_level.value,
            policy.risk_reason,
        )
        metadata["approved"] = approved
        if not approved:
            return permission_denied(
                f"Command requires approval and was not approved: {command}",
                policy.suggestion,
            ), metadata

    result, run_metadata = run_shell_command(
        command,
        cwd=cwd,
        timeout_sec=timeout_sec,
        max_output_chars=max_output_chars,
    )
    metadata.update(run_metadata)
    if policy.decision == PolicyDecision.ALLOW:
        metadata["approved"] = True
    return result, metadata


TOOLS = {
    "read_file": ToolSpec("read_file", read_file, ToolRiskLevel.LOW, False),
    "write_file": ToolSpec("write_file", write_file, ToolRiskLevel.MEDIUM, True),
    "run_shell": ToolSpec("run_shell", run_shell, ToolRiskLevel.HIGH, True),
}


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a UTF-8 text file inside the project sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The project path to the file to read."}
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text content to a project file. Requires approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The project path to the file to write."},
                    "content": {"type": "string", "description": "The content to write."},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "执行项目沙箱内允许的 shell 命令；危险命令会返回 PERMISSION_DENIED。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 shell 命令"},
                    "cwd": {"type": "string", "description": "项目内工作目录，默认当前项目根目录"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def validate_tool_args(tool_name, args):
    if tool_name not in TOOLS:
        return tool_error(
            "TOOL_NOT_FOUND",
            f"Tool '{tool_name}' is not available.",
            True,
            f"Use one of these tools: {', '.join(sorted(TOOLS))}.",
        )

    if not isinstance(args, dict):
        return tool_error(
            "INVALID_ARGUMENTS",
            "Tool arguments must be a JSON object.",
            True,
            "Call the same tool again with a JSON object matching its schema.",
        )

    required = {
        "read_file": {"path": str},
        "write_file": {"path": str, "content": str},
        "run_shell": {"command": str},
    }[tool_name]

    optional = {
        "run_shell": {"cwd": str},
    }.get(tool_name, {})

    for name, expected_type in required.items():
        if name not in args:
            return tool_error(
                "INVALID_ARGUMENTS",
                f"Missing required argument: {name}",
                True,
                f"Call {tool_name} again and include '{name}'.",
            )
        if not isinstance(args[name], expected_type):
            return tool_error(
                "INVALID_ARGUMENTS",
                f"Argument '{name}' must be a {expected_type.__name__}.",
                True,
                f"Call {tool_name} again with '{name}' as a {expected_type.__name__}.",
            )

    for name, expected_type in optional.items():
        if name in args and not isinstance(args[name], expected_type):
            return tool_error(
                "INVALID_ARGUMENTS",
                f"Argument '{name}' must be a {expected_type.__name__}.",
                True,
                f"Call {tool_name} again with '{name}' as a {expected_type.__name__}.",
            )

    return None


def execute_tool(action):
    tool_name = action.get("tool")
    args = action.get("args", {})
    validation_error = validate_tool_args(tool_name, args)
    if validation_error:
        return validation_error, {
            "risk_level": TOOLS.get(tool_name, ToolSpec(tool_name, None, ToolRiskLevel.HIGH, True)).risk_level.value,
            "approval_required": TOOLS.get(tool_name, ToolSpec(tool_name, None, ToolRiskLevel.HIGH, True)).approval_required,
            "approved": False,
            "policy_decision": PolicyDecision.DENY.value,
            "risk_reason": "tool arguments failed validation",
            "truncated": False,
        }

    spec = TOOLS[tool_name]
    return spec.function(**args)
