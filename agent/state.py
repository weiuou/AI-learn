from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TaskState(BaseModel):
    task_id: str
    user_goal: str
    current_plan: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    important_facts: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    error_history: list[str] = Field(default_factory=list)
    resolved_errors: list[str] = Field(default_factory=list)
    last_error: Optional[str] = None
    next_action_hint: Optional[str] = None


DEFAULT_PLAN = [
    "Clarify the task goal from the user request.",
    "Use project-scoped tools to gather or change only the needed information.",
    "Track completed work, important facts, errors, and the next action after each step.",
    "Stop tool use when enough information is available and answer the user directly.",
]


def safe_task_id(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("_")
    return cleaned or "task"


def new_task_state(task_id, user_goal):
    return TaskState(
        task_id=safe_task_id(task_id),
        user_goal=user_goal,
        current_plan=list(DEFAULT_PLAN),
        next_action_hint="Start with the first useful tool call or answer directly if no tool is needed.",
    )


def load_task_state(path):
    with open(path, "r", encoding="utf-8") as f:
        return TaskState.model_validate(json.load(f))


def save_task_state(task_state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(task_state.model_dump(), f, ensure_ascii=False, indent=2)


def _event_type(event):
    return event.get("event_type") or event.get("type")


def _attributes(event):
    return event.get("attributes") or event.get("data") or {}


def _shorten(value, limit=240):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _append_unique(items, value, limit=30):
    if not value or value in items:
        return
    items.append(value)
    if len(items) > limit:
        del items[:-limit]


def _format_list(values):
    return "、".join(values)


def _normalize_path(path):
    if not path:
        return ""
    candidate = Path(str(path)).expanduser()
    try:
        if candidate.is_absolute():
            candidate = candidate.resolve()
            try:
                return candidate.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return candidate.as_posix().lstrip("/")
    except OSError:
        pass
    return str(path).replace("\\", "/").lstrip("./")


def _path_category(path):
    normalized = _normalize_path(path)
    if normalized in {"agent.py", "agent/cli.py", "agent/core.py"}:
        return "已阅读入口与核心循环"
    if normalized in {
        "agent/tools.py",
        "agent/permissions.py",
        "agent/sandbox.py",
        "agent/approval.py",
    }:
        return "已阅读工具与沙箱模块"
    if normalized in {
        "agent/state.py",
        "agent/context_manager.py",
        "context_compressor.py",
        "context_manager_notes.md",
    }:
        return "已阅读上下文模块"
    if normalized in {"eval_runner.py", "evaluators.py"} or normalized.startswith("evals/"):
        return "已阅读评测模块"
    if normalized == "readme.md" or normalized.startswith("docs/") or normalized.endswith(".md"):
        return "已阅读文档"
    return "已阅读其他文件"


def _aggregate_read_steps(paths):
    grouped = {}
    for path in paths:
        normalized = _normalize_path(path)
        if not normalized:
            continue
        grouped.setdefault(_path_category(normalized), [])
        if normalized not in grouped[_path_category(normalized)]:
            grouped[_path_category(normalized)].append(normalized)

    order = [
        "已阅读入口与核心循环",
        "已阅读工具与沙箱模块",
        "已阅读上下文模块",
        "已阅读评测模块",
        "已阅读文档",
        "已阅读其他文件",
    ]
    return [
        f"{category}：{_format_list(grouped[category])}"
        for category in order
        if grouped.get(category)
    ]


def _aggregate_write_steps(paths):
    normalized_paths = []
    for path in paths:
        normalized = _normalize_path(path)
        if normalized and normalized not in normalized_paths:
            normalized_paths.append(normalized)
    if not normalized_paths:
        return []
    return [f"已写入文件：{_format_list(normalized_paths)}"]


def _tool_args(attrs):
    args = attrs.get("tool_call.arguments") or attrs.get("args") or {}
    return args if isinstance(args, dict) else {}


def _tool_name(attrs):
    return attrs.get("tool_call.name") or attrs.get("tool") or attrs.get("tool_name") or "unknown_tool"


def update_task_state_from_trace(task_state, trace):
    task_state.completed_steps = []
    task_state.open_questions = []
    task_state.important_facts = []
    task_state.files_touched = []
    task_state.commands_run = []
    task_state.error_history = []
    task_state.resolved_errors = []
    task_state.last_error = None
    task_state.next_action_hint = "Continue from the latest completed step."

    read_paths = []
    written_paths = []
    successful_tools = []
    pending_errors = []
    saw_resume = False
    saw_checkpoint = False
    saw_final_success = False

    for event in trace.get("events", []):
        event_type = _event_type(event)
        attrs = _attributes(event)

        if event_type == "tool_called":
            tool_name = _tool_name(attrs)
            args = _tool_args(attrs)
            if tool_name in {"read_file", "write_file"} and args.get("path"):
                _append_unique(task_state.files_touched, _normalize_path(args.get("path")), limit=50)
            if tool_name == "run_shell" and args.get("command"):
                _append_unique(task_state.commands_run, _shorten(args.get("command"), 180), limit=50)

        elif event_type == "tool_result":
            tool_name = _tool_name(attrs)
            observation = attrs.get("observation") or attrs.get("result")
            error = attrs.get("error")
            args = _tool_args(attrs)

            if isinstance(observation, dict) and observation.get("ok"):
                if tool_name in {"read_file", "write_file"} and args.get("path"):
                    _append_unique(task_state.files_touched, _normalize_path(args.get("path")), limit=50)
                    if tool_name == "read_file":
                        _append_unique(read_paths, args.get("path"), limit=100)
                    else:
                        _append_unique(written_paths, args.get("path"), limit=100)
                elif tool_name == "run_shell":
                    result = observation.get("result")
                    if isinstance(result, dict) and "returncode" in result:
                        _append_unique(
                            successful_tools,
                            f"已运行 shell 命令，returncode={result.get('returncode')}",
                            limit=20,
                        )
                else:
                    _append_unique(successful_tools, f"已执行工具：{tool_name}", limit=20)
                for pending in pending_errors:
                    _append_unique(task_state.resolved_errors, pending, limit=50)
                pending_errors = []
                task_state.last_error = None
                task_state.next_action_hint = (
                    "If the user goal now has enough evidence, stop tool use and answer directly; "
                    "otherwise choose the next smallest useful action."
                )
            else:
                err = error or observation or {}
                error_type = err.get("error_type") if isinstance(err, dict) else "UNKNOWN"
                message = err.get("message") if isinstance(err, dict) else _shorten(err)
                suggestion = err.get("suggestion") if isinstance(err, dict) else None
                task_state.last_error = f"{error_type}: {_shorten(message)}"
                error_summary = f"{tool_name} {task_state.last_error}"
                _append_unique(task_state.error_history, error_summary, limit=50)
                _append_unique(pending_errors, error_summary, limit=50)
                if suggestion:
                    task_state.next_action_hint = suggestion
                else:
                    task_state.next_action_hint = "Inspect the error and retry with corrected arguments or a narrower action."

        elif event_type == "protocol_error":
            message = attrs.get("error") or "Protocol error."
            task_state.last_error = _shorten(message)
            error_summary = f"protocol_error: {task_state.last_error}"
            _append_unique(task_state.error_history, error_summary, limit=50)
            _append_unique(pending_errors, error_summary, limit=50)
            task_state.next_action_hint = "Retry with a valid tool call or provide a final answer if enough information exists."

        elif event_type == "error":
            message = attrs.get("message") or attrs.get("error") or "Runtime error."
            task_state.last_error = _shorten(message)
            error_summary = f"runtime_error: {task_state.last_error}"
            _append_unique(task_state.error_history, error_summary, limit=50)
            _append_unique(pending_errors, error_summary, limit=50)
            task_state.next_action_hint = "Resume from the checkpoint and avoid repeating the failing action unchanged."

        elif event_type in {"resume_started", "recovery_started"}:
            saw_resume = True

        elif event_type == "checkpoint_saved":
            saw_checkpoint = True

        elif event_type == "final_answer":
            exit_reason = attrs.get("exit_reason")
            if exit_reason == "max_steps":
                task_state.last_error = attrs.get("answer") or "Reached max steps."
                error_summary = f"max_steps: {task_state.last_error}"
                _append_unique(task_state.error_history, error_summary, limit=50)
                _append_unique(pending_errors, error_summary, limit=50)
                task_state.next_action_hint = "Resume the task and continue from completed_steps without repeating finished work."
            elif exit_reason:
                saw_final_success = True
                for pending in pending_errors:
                    _append_unique(task_state.resolved_errors, pending, limit=50)
                pending_errors = []
                task_state.last_error = None
                task_state.next_action_hint = "Task appears complete; only resume if the user asks for more work."

    task_state.completed_steps.extend(_aggregate_read_steps(read_paths))
    task_state.completed_steps.extend(_aggregate_write_steps(written_paths))
    for item in successful_tools:
        _append_unique(task_state.completed_steps, item, limit=50)
    if saw_resume:
        _append_unique(task_state.completed_steps, "已从 checkpoint 恢复执行", limit=50)
        _append_unique(task_state.important_facts, "任务发生过 resume，已从 checkpoint 继续执行", limit=30)
    if saw_checkpoint:
        _append_unique(task_state.important_facts, "任务已保存 checkpoint，可从 state.json 和 trace.jsonl 恢复", limit=30)
    if written_paths:
        _append_unique(task_state.important_facts, f"已写入文件：{_format_list([_normalize_path(path) for path in written_paths])}", limit=30)
    if saw_final_success:
        _append_unique(task_state.completed_steps, "已产出最终回答", limit=50)
        _append_unique(task_state.important_facts, "任务已产出最终回答", limit=30)
    if task_state.last_error is None and pending_errors:
        task_state.last_error = pending_errors[-1]

    return task_state
