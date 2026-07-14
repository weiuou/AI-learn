from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .docker_sandbox import DockerSandbox, SandboxUnavailable
from .filesystem import WorkspaceFilesystem, WorkspaceLimitError, WorkspacePathError


def tool_success(result: Any) -> dict[str, Any]:
    return {"ok": True, "result": result, "error_type": None, "message": None, "recoverable": None, "suggestion": None}


def tool_error(error_type: str, message: str, suggestion: str | None = None, recoverable: bool = True) -> dict[str, Any]:
    return {
        "ok": False,
        "result": None,
        "error_type": error_type,
        "message": message,
        "recoverable": recoverable,
        "suggestion": suggestion,
    }


WORKSPACE_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories inside the current Workspace staging area.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative directory path; defaults to the Workspace root."}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the current Workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file in the current Workspace staging area.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for literal text in UTF-8 Workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string", "description": "Optional file glob such as *.py."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a Bash command inside the isolated, offline Docker sandbox. The working directory starts at /workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


class WorkspaceToolExecutor:
    def __init__(
        self,
        filesystem: WorkspaceFilesystem,
        staging_root: Path,
        sandbox: DockerSandbox,
        file_changed_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.filesystem = filesystem
        self.staging_root = staging_root
        self.sandbox = sandbox
        self.file_changed_callback = file_changed_callback

    @staticmethod
    def _metadata(tool_name: str) -> dict[str, Any]:
        shell = tool_name == "run_shell"
        return {
            "risk_level": "high" if shell else ("medium" if tool_name == "write_file" else "low"),
            "approval_required": False,
            "approved": True,
            "policy_decision": "allow",
            "risk_reason": "isolated Docker sandbox" if shell else "bound Workspace staging path",
            "truncated": False,
        }

    def __call__(self, action: dict[str, Any]):
        tool_name = action.get("tool")
        args = action.get("args")
        metadata = self._metadata(str(tool_name))
        if not isinstance(args, dict):
            return tool_error("INVALID_ARGUMENTS", "Tool arguments must be an object."), metadata
        before = self.filesystem.manifest(self.staging_root) if tool_name in {"write_file", "run_shell"} else None
        try:
            if tool_name == "list_files":
                result = tool_success(self.filesystem.list_files(self.staging_root, args.get("path", ".")))
            elif tool_name == "read_file":
                self._require(args, "path", str)
                result = tool_success(self.filesystem.read_file(self.staging_root, args["path"]))
            elif tool_name == "write_file":
                self._require(args, "path", str)
                self._require(args, "content", str)
                self.filesystem.write_file(self.staging_root, args["path"], args["content"])
                result = tool_success(f"Wrote {len(args['content'])} characters to {args['path']}")
            elif tool_name == "search_files":
                self._require(args, "query", str)
                result = tool_success(self.filesystem.search_files(
                    self.staging_root,
                    args["query"],
                    args.get("path", "."),
                    args.get("glob"),
                ))
            elif tool_name == "run_shell":
                self._require(args, "command", str)
                shell_result = self.sandbox.exec(args["command"])
                metadata["truncated"] = bool(shell_result.get("truncated"))
                if shell_result["returncode"] == 0:
                    result = tool_success(shell_result)
                else:
                    error_type = "COMMAND_TIMEOUT" if shell_result.get("timedOut") else "COMMAND_FAILED"
                    result = tool_error(error_type, f"Command exited with code {shell_result['returncode']}.", "Inspect stdout/stderr and retry.")
                    result["result"] = shell_result
            else:
                return tool_error("TOOL_NOT_FOUND", f"Unknown tool: {tool_name}"), metadata
            if before is not None:
                self._emit_changes(before, self.filesystem.manifest(self.staging_root), str(tool_name))
            return result, metadata
        except KeyError as exc:
            return tool_error("INVALID_ARGUMENTS", str(exc)), metadata
        except FileNotFoundError as exc:
            return tool_error("FILE_NOT_FOUND", f"File does not exist: {Path(str(exc)).name}"), metadata
        except WorkspaceLimitError as exc:
            return tool_error("WORKSPACE_LIMIT", str(exc), recoverable=False), metadata
        except WorkspacePathError as exc:
            metadata["policy_decision"] = "deny"
            metadata["approved"] = False
            return tool_error("PERMISSION_DENIED", str(exc), "Use a relative path inside the Workspace."), metadata
        except SandboxUnavailable as exc:
            metadata["policy_decision"] = "deny"
            metadata["approved"] = False
            return tool_error("SANDBOX_UNAVAILABLE", str(exc), recoverable=False), metadata
        except Exception as exc:
            return tool_error("TOOL_ERROR", f"Tool failed: {exc}"), metadata

    @staticmethod
    def _require(args: dict[str, Any], name: str, expected_type: type) -> None:
        if name not in args or not isinstance(args[name], expected_type):
            raise KeyError(f"Argument '{name}' must be a {expected_type.__name__}.")

    def _emit_changes(self, before: dict[str, str], after: dict[str, str], tool_name: str) -> None:
        if not self.file_changed_callback:
            return
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            change_type = "created" if path not in before else "deleted" if path not in after else "modified"
            self.file_changed_callback({"path": path, "changeType": change_type, "tool": tool_name})
