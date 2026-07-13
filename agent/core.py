from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from openai import OpenAI

from context_compressor import (
    COMPRESSION_THRESHOLD,
    compress_messages,
    compression_diagnostics,
    estimate_messages_size,
)

from .context_manager import (
    build_context_pack,
    collect_recent_tool_summaries,
    compress_tool_result,
    save_context_pack,
)
from .budget import BudgetGuard, RunBudget
from .loop_detector import LoopDetector
from .replay import recompute_budget_consumed
from .sqlite_store import SQLiteRunStore
from .state import (
    TaskState,
    load_task_state,
    new_task_state,
    safe_task_id,
    save_task_state,
    update_task_state_from_trace,
)
from .store import FileRunStore, RunStore, make_event


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "MiniMax-M3")
DEFAULT_TRACE_DIR = "runs"
DEFAULT_API_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
TRACE_JSONL_NAME = "trace.jsonl"
STATE_JSON_NAME = "state.json"
CONTEXT_PACK_NAME = "context_pack.md"

client = None


def get_client():
    global client
    if client is None:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            timeout=DEFAULT_API_TIMEOUT_SECONDS,
        )
    return client


def now():
    return datetime.now().isoformat()


def estimate_text_tokens(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, len(text) // 4)


def to_plain_data(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump())
    if hasattr(value, "to_dict"):
        return to_plain_data(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            key: to_plain_data(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def first_number(*values):
    for value in values:
        if isinstance(value, (int, float)):
            return value
    return None


def normalize_usage(raw_usage):
    usage = to_plain_data(raw_usage) or {}
    if not isinstance(usage, dict):
        return None

    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}

    prompt_tokens = first_number(
        usage.get("prompt_tokens"),
        usage.get("input_tokens"),
        usage.get("input_token_count"),
    )
    completion_tokens = first_number(
        usage.get("completion_tokens"),
        usage.get("output_tokens"),
        usage.get("output_token_count"),
    )
    total_tokens = first_number(usage.get("total_tokens"), usage.get("total_token_count"))
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    cached_tokens = first_number(
        prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None,
        prompt_details.get("cache_read_tokens") if isinstance(prompt_details, dict) else None,
        prompt_details.get("cached_input_tokens") if isinstance(prompt_details, dict) else None,
        usage.get("cached_tokens"),
        usage.get("cache_hit_tokens"),
        usage.get("prompt_cache_hit_tokens"),
        usage.get("input_cached_tokens"),
    )
    cache_creation_tokens = first_number(
        prompt_details.get("cache_creation_tokens") if isinstance(prompt_details, dict) else None,
        prompt_details.get("cache_write_tokens") if isinstance(prompt_details, dict) else None,
        usage.get("cache_creation_tokens"),
        usage.get("prompt_cache_miss_tokens"),
    )
    reasoning_tokens = first_number(
        completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None,
        usage.get("reasoning_tokens"),
    )

    cache_hit_rate = None
    if cached_tokens is not None and prompt_tokens:
        cache_hit_rate = cached_tokens / prompt_tokens

    return {
        "raw": usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_hit_rate": cache_hit_rate,
        "reasoning_tokens": reasoning_tokens,
    }


def summarize_usage(trace):
    summary = {
        "model_calls": 0,
        "usage_calls": 0,
        "missing_usage_calls": 0,
        "tool_calls": 0,
        "context_compressions": 0,
        "api_usage_available": False,
        "cache_usage_available": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
        "cache_hit_rate": None,
    }

    for event in trace.get("events", []):
        event_type = event.get("event_type") or event.get("type")
        attrs = event.get("attributes") or event.get("data") or {}

        if event_type == "llm_called":
            summary["model_calls"] += 1
        elif event_type == "llm_result":
            usage = attrs.get("usage")
            if usage:
                summary["api_usage_available"] = True
                summary["usage_calls"] += 1
                for key in [
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "reasoning_tokens",
                ]:
                    summary[key] += usage.get(key) or 0

                if usage.get("cached_tokens") is not None:
                    summary["cache_usage_available"] = True
                    summary["cached_tokens"] += usage.get("cached_tokens") or 0
                if usage.get("cache_creation_tokens") is not None:
                    summary["cache_usage_available"] = True
                    summary["cache_creation_tokens"] += usage.get("cache_creation_tokens") or 0
        elif event_type == "tool_called":
            summary["tool_calls"] += 1
        elif event_type == "context_compressed":
            summary["context_compressions"] += 1

    summary["missing_usage_calls"] = summary["model_calls"] - summary["usage_calls"]

    if summary["cache_usage_available"] and summary["prompt_tokens"]:
        summary["cache_hit_rate"] = summary["cached_tokens"] / summary["prompt_tokens"]

    return summary


def update_budget_summary(trace, budget_guard):
    summary = budget_guard.summary()
    summary["segment_consumed"] = summary.get("consumed") or {}
    summary["consumed"] = recompute_budget_consumed(trace)
    trace["budget_summary"] = summary
    return summary


def format_percent(value):
    if value is None:
        return "unavailable"
    return f"{value * 100:.2f}%"


def tool_success(result):
    return {
        "ok": True,
        "result": result,
        "error_type": None,
        "message": None,
        "recoverable": None,
        "suggestion": None,
    }


def tool_error(error_type, message, recoverable=True, suggestion=None):
    return {
        "ok": False,
        "result": None,
        "error_type": error_type,
        "message": message,
        "recoverable": recoverable,
        "suggestion": suggestion,
    }


def shorten(value, limit=500):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def message_summary(messages):
    parts = []
    for message in messages[-5:]:
        role = message.get("role")
        content = message.get("content")
        if message.get("tool_calls"):
            names = [
                call.get("function", {}).get("name", "unknown")
                for call in message.get("tool_calls", [])
            ]
            parts.append(f"{role}: tool_calls={names}")
        elif content:
            parts.append(f"{role}: {shorten(content, 160)}")
        else:
            parts.append(f"{role}: <empty>")
    return " | ".join(parts)


def add_event(trace, event_type, attributes=None, step=None):
    event = make_event(event_type, attributes, step)
    trace["events"].append(event)
    store = trace.get("_store")
    if store is not None:
        store.append_event(trace["task_id"], event)
    return event


def make_task_id(value=None):
    return safe_task_id(value or datetime.now().strftime("%Y%m%d_%H%M%S"))


def run_dir_for_task(task_id, base_dir=DEFAULT_TRACE_DIR):
    return os.path.join(base_dir, safe_task_id(task_id))


def artifact_paths(task_id, base_dir=DEFAULT_TRACE_DIR, run_dir=None):
    directory = run_dir or run_dir_for_task(task_id, base_dir=base_dir)
    return {
        "run_dir": directory,
        "trace": os.path.join(directory, TRACE_JSONL_NAME),
        "state": os.path.join(directory, STATE_JSON_NAME),
        "context_pack": os.path.join(directory, CONTEXT_PACK_NAME),
    }


def save_trace(trace, trace_path):
    directory = os.path.dirname(trace_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if trace_path.endswith(".jsonl"):
        metadata = {
            key: value
            for key, value in trace.items()
            if key not in {"events"} and not key.startswith("_")
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "trace", **metadata}, ensure_ascii=False) + "\n")
            for event in trace.get("events", []):
                f.write(json.dumps({"record_type": "event", **event}, ensure_ascii=False) + "\n")
        return
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


def load_trace(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            events = json.load(f)
            return {
                "schema_version": "agent-harness-trace-v1",
                "events": events,
            }

        if trace_path.endswith(".jsonl"):
            trace = {"schema_version": "agent-harness-trace-v1", "events": []}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.pop("record_type", "event")
                if record_type == "trace":
                    trace.update(record)
                elif record_type == "event":
                    trace["events"].append(record)
            return trace

        return json.load(f)


def new_trace(user_task, task_id=None):
    resolved_task_id = make_task_id(task_id)
    trace = {
        "schema_version": "agent-harness-trace-v1",
        "task_id": resolved_task_id,
        "task": user_task,
        "user_goal": user_task,
        "started_at": now(),
        "finished_at": None,
        "events": [],
    }
    add_event(
        trace,
        "task_started",
        {
            "task_id": resolved_task_id,
            "user_goal": user_task,
            "token_estimate": estimate_text_tokens(user_task),
        },
    )
    return trace


from .tools import OPENAI_TOOLS, execute_tool


def clean_model_content(content):
    if not content:
        return ""

    content = content.strip()
    while "<think>" in content and "</think>" in content:
        start = content.find("<think>")
        end = content.find("</think>") + len("</think>")
        content = content[:start] + content[end:]
    return content.strip()


def maybe_compress_context(messages, trace, step):
    before_size = estimate_messages_size(messages)
    if before_size <= COMPRESSION_THRESHOLD:
        return messages

    compressed = compress_messages(messages, trace)
    after_size = estimate_messages_size(compressed)
    diagnostics = trace.pop("_last_compression_diagnostics", compression_diagnostics(compressed))
    add_event(
        trace,
        "context_compressed",
        {
            "step": step,
            "user_goal": trace.get("user_goal"),
            "model_input_summary": message_summary(compressed),
            "token_estimate": estimate_text_tokens(compressed),
            "before_chars": before_size,
            "after_chars": after_size,
            "kept_recent_turns": 2,
            "compressed_tool_results": diagnostics.get("compressed_tool_results"),
            "compression_strategy": diagnostics.get("compression_strategy"),
            "compression_aggressive": diagnostics.get("aggressive"),
        },
        step=step,
    )
    return compressed


def base_system_message():
    return {
        "role": "system",
        "content": (
            "你是一个最小 Agent Harness。"
            "你可以通过工具读取文件、写文件、运行低风险 shell 命令。"
            "工具返回的是统一 JSON：ok/result/error_type/message/recoverable/suggestion。"
            "你每轮会收到 Context Pack，它是任务状态和历史摘要，不是完整聊天历史。"
            "遇到 recoverable=true 的错误时，优先根据 suggestion 自己恢复，例如列目录、搜索文件、修正参数。"
            "不要重复 Context Pack 中已经完成的步骤；从 next_action_hint 和最近工具结果继续。"
            "当你已经获得足够信息后，不要再调用工具，直接用中文回答用户。"
        ),
    }


def build_model_messages(user_task, context_pack, recent_messages):
    return [
        base_system_message(),
        {"role": "user", "content": user_task},
        {
            "role": "system",
            "content": context_pack,
        },
    ] + list(recent_messages or [])


def trim_recent_messages(messages, max_turns=2):
    turns = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            turn = [message]
            index += 1
            expected = {
                call.get("id")
                for call in message.get("tool_calls", [])
                if call.get("id")
            }
            while index < len(messages) and messages[index].get("role") == "tool":
                turn.append(messages[index])
                expected.discard(messages[index].get("tool_call_id"))
                index += 1
                if not expected:
                    break
            turns.append(turn)
        else:
            turns.append([message])
            index += 1

    kept = []
    for turn in turns[-max_turns:]:
        kept.extend(turn)
    return kept


def _max_trace_step(trace):
    max_step = 0
    for event in trace.get("events", []):
        attrs = event.get("attributes") or event.get("data") or {}
        step = event.get("step") or attrs.get("step")
        if isinstance(step, int):
            max_step = max(max_step, step)
    return max_step


def _recent_events(trace, limit=30):
    return trace.get("events", [])[-limit:]


def _sync_trace_from_store(trace, store):
    loaded = store.load_run(trace["task_id"])
    stored_trace = loaded["trace"]
    trace["events"] = list(stored_trace.get("events", []))
    for key in ["segments", "status", "started_at", "finished_at"]:
        if key in stored_trace:
            trace[key] = stored_trace[key]
        elif key in loaded:
            trace[key] = loaded[key]


def _latest_exit_reason(trace):
    for event in reversed(trace.get("events", [])):
        if (event.get("event_type") or event.get("type")) == "final_answer":
            attrs = event.get("attributes") or event.get("data") or {}
            return attrs.get("exit_reason") or "completed"
        if (event.get("event_type") or event.get("type")) in {
            "task_started",
            "resume_started",
            "recovery_started",
        }:
            break
    return None


def persist_checkpoint(trace, task_state, run_dir, context_pack, step=None):
    store = trace.get("_store")
    if not run_dir and store is None:
        return
    update_task_state_from_trace(task_state, trace)
    add_event(
        trace,
        "task_state_updated",
        {
            "task_id": task_state.task_id,
            "completed_steps": list(task_state.completed_steps),
            "last_error": task_state.last_error,
            "next_action_hint": task_state.next_action_hint,
            "token_estimate": estimate_text_tokens(task_state.model_dump()),
        },
        step=step,
    )
    if store is not None:
        store.save_checkpoint(
            task_state.task_id,
            task_state.model_dump(),
            context_pack,
            step or 0,
        )
        _sync_trace_from_store(trace, store)
        exit_reason = _latest_exit_reason(trace)
        segment_id = trace.get("_segment_id")
        if exit_reason and segment_id and not trace.get("_segment_finished"):
            store.finish_segment(task_state.task_id, segment_id, exit_reason)
            trace["_segment_finished"] = True
            _sync_trace_from_store(trace, store)
        return

    paths = artifact_paths(task_state.task_id, run_dir=run_dir)
    save_task_state(task_state, paths["state"])
    save_context_pack(context_pack, paths["context_pack"])
    add_event(trace, "checkpoint_saved", {"task_id": task_state.task_id}, step=step)
    save_trace(trace, paths["trace"])


def prepare_context_pack(trace, task_state, step, run_dir=None):
    update_task_state_from_trace(task_state, trace)
    tool_summaries = collect_recent_tool_summaries(trace, limit=5)
    context_pack = build_context_pack(
        task_state,
        _recent_events(trace, limit=24),
        tool_summaries,
    )
    add_event(
        trace,
        "context_pack_built",
        {
            "step": step,
            "task_id": task_state.task_id,
            "chars": len(context_pack),
            "tool_summaries": len(tool_summaries),
            "token_estimate": estimate_text_tokens(context_pack),
        },
        step=step,
    )
    if run_dir and trace.get("_store") is None:
        save_context_pack(context_pack, artifact_paths(task_state.task_id, run_dir=run_dir)["context_pack"])
    return context_pack


def reconstruct_recent_messages(trace, max_turns=2):
    turns = []
    events = trace.get("events", [])
    for event in events:
        event_type = event.get("event_type") or event.get("type")
        attrs = event.get("attributes") or event.get("data") or {}
        if event_type != "llm_result":
            continue
        tool_calls = attrs.get("tool_calls") or []
        if not tool_calls:
            continue
        assistant_message = {
            "role": "assistant",
            "content": attrs.get("content"),
            "tool_calls": [
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": call.get("name"),
                        "arguments": call.get("arguments"),
                    },
                }
                for call in tool_calls
            ],
        }
        turn = [assistant_message]
        wanted = {call.get("id") for call in tool_calls}
        for result_event in events:
            result_type = result_event.get("event_type") or result_event.get("type")
            result_attrs = result_event.get("attributes") or result_event.get("data") or {}
            if result_type != "tool_result" or result_attrs.get("tool_call.id") not in wanted:
                continue
            tool_name = result_attrs.get("tool_call.name") or result_attrs.get("tool")
            tool_args = result_attrs.get("tool_call.arguments") or result_attrs.get("args") or {}
            observation = result_attrs.get("observation") or result_attrs.get("result")
            compact, _ = compress_tool_result(tool_name, tool_args if isinstance(tool_args, dict) else {}, observation)
            turn.append(
                {
                    "role": "tool",
                    "tool_call_id": result_attrs.get("tool_call.id"),
                    "content": json.dumps(compact, ensure_ascii=False),
                }
            )
        turns.append(turn)

    messages = []
    for turn in turns[-max_turns:]:
        messages.extend(turn)
    return messages


def run_agent(
    user_task,
    trace,
    max_steps=50,
    task_state=None,
    run_dir=None,
    recent_messages=None,
    start_step=None,
    budget=None,
    budget_guard=None,
    loop_detector=None,
    model_client=None,
    tool_executor=None,
    clock=None,
    store: RunStore | None = None,
    segment_id: str | None = None,
):
    if store is None and run_dir:
        store = FileRunStore(Path(run_dir).parent)
        try:
            loaded = store.load_run(trace.get("task_id"))
        except FileNotFoundError:
            store.create_run(trace.get("task_id"), user_task)
            segment_id = segment_id or f"task-{uuid4().hex}"
            store.start_segment(trace.get("task_id"), segment_id, "task")
            for existing_event in trace.get("events", []):
                store.append_event(trace.get("task_id"), existing_event)
        else:
            open_segments = [item for item in loaded.get("segments", []) if item.get("finished_at") is None]
            if open_segments:
                segment_id = segment_id or open_segments[-1]["segment_id"]
            elif segment_id is None:
                segment_id = f"task-{uuid4().hex}"
                store.start_segment(trace.get("task_id"), segment_id, "task")
    if store is not None:
        trace["_store"] = store
        trace["_segment_id"] = segment_id

    if task_state is None:
        task_state = new_task_state(trace.get("task_id") or make_task_id(), user_task)
    recent_messages = list(recent_messages or [])
    if start_step is None:
        start_step = _max_trace_step(trace) + 1
    if budget is None:
        budget = RunBudget(max_steps=max_steps)
    if budget_guard is None:
        guard_kwargs = {"budget": budget}
        if clock is not None:
            guard_kwargs["clock"] = clock
        budget_guard = BudgetGuard(**guard_kwargs)
    update_budget_summary(trace, budget_guard)
    loop_detector = loop_detector or LoopDetector()
    model_client = model_client or get_client()
    tool_executor = tool_executor or execute_tool
    context_pack = prepare_context_pack(trace, task_state, start_step, run_dir=run_dir)

    step = start_step
    while True:
        context_pack = prepare_context_pack(trace, task_state, step, run_dir=run_dir)
        messages = build_model_messages(user_task, context_pack, recent_messages)
        messages = maybe_compress_context(messages, trace, step)
        input_summary = message_summary(messages)
        token_estimate = estimate_text_tokens(messages)
        prompt_chars = estimate_messages_size(messages)

        exceeded = budget_guard.check_before_model(prompt_chars)
        if exceeded:
            answer = f"运行预算已超限：{exceeded.limit}。"
            add_event(
                trace,
                "budget_exceeded",
                {
                    "step": step,
                    "limit": exceeded.limit,
                    "limit_value": exceeded.limit_value,
                    "consumed": exceeded.consumed,
                    "attempted": exceeded.attempted,
                },
                step=step,
            )
            add_event(trace, "final_answer", {"step": step, "user_goal": user_task, "answer": answer, "exit_reason": "budget_exceeded", "token_estimate": estimate_text_tokens(answer)}, step=step)
            update_budget_summary(trace, budget_guard)
            context_pack = prepare_context_pack(trace, task_state, step, run_dir=run_dir)
            persist_checkpoint(trace, task_state, run_dir, context_pack, step=step)
            return answer

        budget_guard.record_model_call(prompt_chars)
        update_budget_summary(trace, budget_guard)

        add_event(
            trace,
            "llm_called",
            {
                "step": step,
                "user_goal": user_task,
                "model": MODEL,
                "model_input_summary": input_summary,
                "token_estimate": token_estimate,
                "prompt_chars": prompt_chars,
            },
            step=step,
        )

        completion = model_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )

        message = completion.choices[0].message
        usage = normalize_usage(getattr(completion, "usage", None))
        tool_calls = message.tool_calls or []
        tool_call_summaries = [
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
            for tool_call in tool_calls
        ]

        add_event(
            trace,
            "llm_result",
            {
                "step": step,
                "user_goal": user_task,
                "model_input_summary": input_summary,
                "content": message.content,
                "tool_calls": tool_call_summaries,
                "usage": usage,
                "token_estimate": estimate_text_tokens((message.content or "") + json.dumps(tool_call_summaries, ensure_ascii=False)),
            },
            step=step,
        )

        if not tool_calls:
            raw_content = message.content or ""
            answer = clean_model_content(raw_content)

            if not answer:
                add_event(
                    trace,
                    "protocol_error",
                    {
                        "step": step,
                        "user_goal": user_task,
                        "model_input_summary": input_summary,
                        "error": "Model returned no tool_calls and empty content.",
                        "token_estimate": estimate_text_tokens(raw_content),
                    },
                    step=step,
                )
                answer = "任务结束，但模型没有给出最终答案。"
                exit_reason = "runtime_error"
            else:
                exit_reason = "completed"

            add_event(
                trace,
                "final_answer",
                {
                    "step": step,
                    "user_goal": user_task,
                    "model_input_summary": input_summary,
                    "answer": answer,
                    "exit_reason": exit_reason,
                    "token_estimate": estimate_text_tokens(answer),
                },
                step=step,
            )
            update_budget_summary(trace, budget_guard)
            context_pack = prepare_context_pack(trace, task_state, step, run_dir=run_dir)
            persist_checkpoint(trace, task_state, run_dir, context_pack, step=step)
            return answer

        assistant_recent_message = (
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )
        recent_messages.append(assistant_recent_message)
        pending_recovery_hint = None

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments
            tool_metadata = {
                "risk_level": "high",
                "approval_required": False,
                "approved": False,
                "policy_decision": "deny",
                "risk_reason": "invalid tool call arguments",
                "truncated": False,
            }

            exceeded = budget_guard.check_before_tool()
            if exceeded:
                answer = f"运行预算已超限：{exceeded.limit}。"
                add_event(trace, "budget_exceeded", {"step": step, "limit": exceeded.limit, "limit_value": exceeded.limit_value, "consumed": exceeded.consumed, "attempted": exceeded.attempted}, step=step)
                add_event(trace, "final_answer", {"step": step, "user_goal": user_task, "answer": answer, "exit_reason": "budget_exceeded", "token_estimate": estimate_text_tokens(answer)}, step=step)
                update_budget_summary(trace, budget_guard)
                context_pack = prepare_context_pack(trace, task_state, step, run_dir=run_dir)
                persist_checkpoint(trace, task_state, run_dir, context_pack, step=step)
                return answer
            budget_guard.record_tool_call()
            update_budget_summary(trace, budget_guard)

            try:
                tool_args = json.loads(raw_args)
            except json.JSONDecodeError as e:
                tool_args = raw_args
                tool_result = tool_error(
                    "INVALID_ARGUMENTS",
                    f"Invalid tool arguments JSON: {e}",
                    True,
                    "Call the tool again with valid JSON arguments.",
                )
            else:
                tool_result, tool_metadata = tool_executor({"tool": tool_name, "args": tool_args})

            add_event(
                trace,
                "tool_called",
                {
                    "step": step,
                    "user_goal": user_task,
                    "model_input_summary": input_summary,
                    "tool_call.id": tool_call.id,
                    "tool_call.name": tool_name,
                    "tool_call.arguments": tool_args,
                    **tool_metadata,
                    "token_estimate": estimate_text_tokens(tool_args),
                },
                step=step,
            )

            error = None if tool_result.get("ok") else {
                "error_type": tool_result.get("error_type"),
                "message": tool_result.get("message"),
                "recoverable": tool_result.get("recoverable"),
                "suggestion": tool_result.get("suggestion"),
            }

            add_event(
                trace,
                "tool_result",
                {
                    "step": step,
                    "user_goal": user_task,
                    "model_input_summary": input_summary,
                    "tool_call.id": tool_call.id,
                    "tool_call.name": tool_name,
                    "tool_call.arguments": tool_args,
                    "observation": tool_result,
                    "error": error,
                    **tool_metadata,
                    "token_estimate": estimate_text_tokens(tool_result),
                },
                step=step,
            )

            budget_guard.record_tool_result(bool(tool_result.get("ok")))
            update_budget_summary(trace, budget_guard)
            loop_decision = loop_detector.observe(
                tool_name,
                tool_args,
                tool_result.get("error_type"),
                bool(tool_result.get("ok")),
                step,
            )
            if loop_decision.detected:
                add_event(
                    trace,
                    "loop_detected",
                    {
                        "step": step,
                        "pattern": loop_decision.pattern,
                        "related_steps": list(loop_decision.steps),
                        "fingerprints": list(loop_decision.fingerprints),
                        "action": "stop" if loop_decision.should_stop else "replan",
                    },
                    step=step,
                )
                if loop_decision.should_stop:
                    answer = "检测到重复且无进展的工具循环，任务已停止。"
                    add_event(trace, "final_answer", {"step": step, "user_goal": user_task, "answer": answer, "exit_reason": "loop_detected", "token_estimate": estimate_text_tokens(answer)}, step=step)
                    update_budget_summary(trace, budget_guard)
                    context_pack = prepare_context_pack(trace, task_state, step, run_dir=run_dir)
                    persist_checkpoint(trace, task_state, run_dir, context_pack, step=step)
                    return answer
                pending_recovery_hint = loop_decision.recovery_hint
                budget_guard.grant_failure_recovery()

            compact_tool_result, compressed = compress_tool_result(
                tool_name,
                tool_args if isinstance(tool_args, dict) else {},
                tool_result,
            )
            if compressed:
                add_event(
                    trace,
                    "tool_result_compressed",
                    {
                        "step": step,
                        "user_goal": user_task,
                        "tool_call.id": tool_call.id,
                        "tool_call.name": tool_name,
                        "original_chars": len(json.dumps(tool_result, ensure_ascii=False)),
                        "compressed_chars": len(json.dumps(compact_tool_result, ensure_ascii=False)),
                        "summary": compact_tool_result.get("summary"),
                        "token_estimate": estimate_text_tokens(compact_tool_result),
                    },
                    step=step,
                )

            recent_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(compact_tool_result, ensure_ascii=False),
                }
            )
            recent_messages = trim_recent_messages(recent_messages, max_turns=2)
            context_pack = prepare_context_pack(trace, task_state, step, run_dir=run_dir)
            persist_checkpoint(trace, task_state, run_dir, context_pack, step=step)
        if pending_recovery_hint:
            loop_detector.activate_recovery()
            recent_messages.append({"role": "system", "content": json.dumps(pending_recovery_hint, ensure_ascii=False)})
            recent_messages = trim_recent_messages(recent_messages, max_turns=2)
        step += 1


def print_trace(trace_path):
    trace = load_trace(trace_path)
    usage_summary = trace.get("usage_summary") or summarize_usage(trace)
    print(f"Task: {trace.get('user_goal') or trace.get('task')}")
    print(f"Started: {trace.get('started_at')}")
    print(f"Finished: {trace.get('finished_at')}")
    print("Usage:")
    print(f"  Model calls: {usage_summary.get('model_calls')}")
    print(f"  Tool calls: {usage_summary.get('tool_calls')}")
    print(f"  Context compressions: {usage_summary.get('context_compressions')}")
    if usage_summary.get("api_usage_available"):
        print(f"  Usage calls: {usage_summary.get('usage_calls')}")
        print(f"  Missing usage calls: {usage_summary.get('missing_usage_calls')}")
        print(f"  Prompt tokens: {usage_summary.get('prompt_tokens')}")
        print(f"  Completion tokens: {usage_summary.get('completion_tokens')}")
        print(f"  Total tokens: {usage_summary.get('total_tokens')}")
        print(f"  Reasoning tokens: {usage_summary.get('reasoning_tokens')}")
        if usage_summary.get("cache_usage_available"):
            print(f"  Cached tokens: {usage_summary.get('cached_tokens')}")
            print(f"  Cache creation tokens: {usage_summary.get('cache_creation_tokens')}")
            print(f"  Cache hit rate: {format_percent(usage_summary.get('cache_hit_rate'))}")
        else:
            print("  Cache usage: unavailable")
    else:
        print("  API usage: unavailable")
    print()

    for event in trace.get("events", []):
        event_type = event.get("event_type") or event.get("type")
        attrs = event.get("attributes") or event.get("data") or {}
        step = event.get("step") or attrs.get("step") or "-"

        if event_type == "llm_called":
            print(f"[{step}] LLM called: model={attrs.get('model')} tokens~{attrs.get('token_estimate')}")
        elif event_type == "llm_result":
            tool_calls = attrs.get("tool_calls") or []
            names = [call.get("name") for call in tool_calls]
            usage = attrs.get("usage") or {}
            usage_text = ""
            if usage:
                usage_text = f" usage={usage.get('total_tokens')} tokens"
            if names:
                print(f"[{step}] LLM requested tools: {', '.join(names)}{usage_text}")
            else:
                print(f"[{step}] LLM returned content{usage_text}: {shorten(attrs.get('content') or '', 500)}")
        elif event_type == "tool_called":
            name = attrs.get("tool_call.name") or attrs.get("tool")
            args = attrs.get("tool_call.arguments") or attrs.get("args")
            policy = attrs.get("policy_decision")
            risk = attrs.get("risk_level")
            approved = attrs.get("approved")
            print(
                f"[{step}] Tool called: {name} {json.dumps(args, ensure_ascii=False)} "
                f"risk={risk} policy={policy} approved={approved}"
            )
        elif event_type == "tool_result":
            name = attrs.get("tool_call.name") or attrs.get("tool")
            observation = attrs.get("observation") or attrs.get("result")
            if isinstance(observation, dict) and observation.get("ok"):
                result = observation.get("result")
                print(
                    f"[{step}] Tool result: {name} ok=true "
                    f"risk={attrs.get('risk_level')} policy={attrs.get('policy_decision')} "
                    f"len={len(shorten(result, 100000))}"
                )
                print(f"      observation: {shorten(result)}")
            else:
                error = attrs.get("error") or observation or {}
                print(
                    f"[{step}] Tool result: {name} ok=false "
                    f"error_type={error.get('error_type')} recoverable={error.get('recoverable')} "
                    f"risk={attrs.get('risk_level')} policy={attrs.get('policy_decision')}"
                )
                print(f"      message: {error.get('message')}")
                print(f"      suggestion: {error.get('suggestion')}")
        elif event_type == "context_compressed":
            print(
                f"[{step}] Context compressed: "
                f"{attrs.get('before_chars')} chars -> {attrs.get('after_chars')} chars"
            )
        elif event_type == "final_answer":
            print(f"[{step}] Final answer ({attrs.get('exit_reason')}): {shorten(attrs.get('answer') or '', 500)}")
        elif event_type == "protocol_error":
            print(f"[{step}] Protocol error: {attrs.get('error')}")
        elif event_type == "error":
            print(f"[{step}] Error: {attrs.get('message')}")
        else:
            print(f"[{step}] {event_type}: {shorten(attrs)}")


def parse_run_args(argv):
    if not argv:
        return None

    trace_path = None
    task_id = None
    max_steps = 50
    store_type = "file"
    args = list(argv)
    trace_requested = "--trace" in args
    if "--store" in args:
        index = args.index("--store")
        if index + 1 >= len(args):
            raise ValueError("--store requires file or sqlite.")
        store_type = args[index + 1]
        if store_type not in {"file", "sqlite"}:
            raise ValueError("--store must be file or sqlite.")
        del args[index : index + 2]
    if "--trace" in args:
        index = args.index("--trace")
        if index + 1 >= len(args):
            raise ValueError("--trace requires a path.")
        trace_path = args[index + 1]
        del args[index : index + 2]

    if "--task-id" in args:
        index = args.index("--task-id")
        if index + 1 >= len(args):
            raise ValueError("--task-id requires a value.")
        task_id = make_task_id(args[index + 1])
        del args[index : index + 2]

    if "--max-steps" in args:
        index = args.index("--max-steps")
        if index + 1 >= len(args):
            raise ValueError("--max-steps requires a number.")
        try:
            max_steps = int(args[index + 1])
        except ValueError as e:
            raise ValueError("--max-steps must be an integer.") from e
        del args[index : index + 2]

    user_task = " ".join(args).strip()
    if not user_task:
        raise ValueError("No task specified. Please provide a task as a command-line argument.")

    task_id = task_id or make_task_id()
    run_dir = run_dir_for_task(task_id)
    paths = artifact_paths(task_id, run_dir=run_dir)
    trace_path = trace_path or paths["trace"]
    if store_type == "sqlite" and trace_requested:
        raise ValueError("--trace cannot be combined with --store sqlite; use export after the run.")

    return {
        "user_task": user_task,
        "task_id": task_id,
        "run_dir": run_dir,
        "trace_path": trace_path,
        "max_steps": max_steps,
        "store_type": store_type,
    }


def parse_resume_args(argv):
    if not argv:
        raise ValueError("Usage: python3 agent.py resume <task_id> [--max-steps N]")

    args = list(argv)
    task_id = make_task_id(args.pop(0))
    max_steps = 50
    store_type = "file"

    if "--store" in args:
        index = args.index("--store")
        if index + 1 >= len(args):
            raise ValueError("--store requires file or sqlite.")
        store_type = args[index + 1]
        if store_type not in {"file", "sqlite"}:
            raise ValueError("--store must be file or sqlite.")
        del args[index : index + 2]

    if "--max-steps" in args:
        index = args.index("--max-steps")
        if index + 1 >= len(args):
            raise ValueError("--max-steps requires a number.")
        try:
            max_steps = int(args[index + 1])
        except ValueError as e:
            raise ValueError("--max-steps must be an integer.") from e
        del args[index : index + 2]

    if args:
        raise ValueError(f"Unknown resume arguments: {' '.join(args)}")

    return task_id, max_steps, store_type


def _resolve_store(store_type="file", base_dir=DEFAULT_TRACE_DIR, store=None):
    if store is not None:
        return store
    if store_type == "sqlite":
        return SQLiteRunStore(Path(base_dir) / "agent.db")
    return FileRunStore(base_dir)


def _persist_trace_metadata_if_file(store, trace, task_id, base_dir):
    if isinstance(store, FileRunStore):
        save_trace(trace, str(Path(store.base_dir) / task_id / TRACE_JSONL_NAME))


def resume_task(
    task_id,
    max_steps=50,
    base_dir=DEFAULT_TRACE_DIR,
    store_type="file",
    store=None,
    model_client=None,
    tool_executor=None,
):
    active_store = _resolve_store(store_type, base_dir, store)
    loaded = active_store.load_run(task_id)
    if any(item.get("finished_at") is None for item in loaded.get("segments", [])):
        raise ValueError("Cannot resume a run with an open segment; use recover for a crashed SQLite run.")
    checkpoint = loaded.get("checkpoint")
    if checkpoint is None:
        raise FileNotFoundError(f"Missing checkpoint for run: {task_id}")

    task_state = TaskState.model_validate(checkpoint["state"])
    trace = loaded["trace"]
    segment_id = f"resume-{uuid4().hex}"
    active_store.start_segment(task_id, segment_id, "resume")
    trace["_store"] = active_store
    trace["_segment_id"] = segment_id
    add_event(
        trace,
        "resume_started",
        {
            "task_id": task_state.task_id,
            "segment_id": segment_id,
            "token_estimate": estimate_text_tokens(task_state.model_dump()),
        },
    )
    recent_messages = [{"role": "system", "content": checkpoint["context_pack"]}]
    recent_messages.extend(reconstruct_recent_messages(trace, max_turns=2))
    answer = run_agent(
        task_state.user_goal,
        trace,
        max_steps=max_steps,
        task_state=task_state,
        run_dir=run_dir_for_task(task_id, base_dir=base_dir) if store_type == "file" else None,
        recent_messages=recent_messages,
        start_step=_max_trace_step(trace) + 1,
        model_client=model_client,
        tool_executor=tool_executor,
        store=active_store,
        segment_id=segment_id,
    )
    trace["usage_summary"] = summarize_usage(trace)
    _persist_trace_metadata_if_file(active_store, trace, task_id, base_dir)
    effective_base_dir = str(active_store.base_dir) if isinstance(active_store, FileRunStore) else base_dir
    paths = artifact_paths(task_id, base_dir=effective_base_dir)
    if store_type == "sqlite" or isinstance(active_store, SQLiteRunStore):
        paths = {**paths, "database": str(active_store.db_path)}
    return answer, paths


def recover_task(
    task_id,
    max_steps=50,
    base_dir=DEFAULT_TRACE_DIR,
    store=None,
    model_client=None,
    tool_executor=None,
):
    active_store = _resolve_store("sqlite", base_dir, store)
    if not isinstance(active_store, SQLiteRunStore):
        raise ValueError("Crash recovery requires SQLiteRunStore.")
    loaded = active_store.load_run(task_id)
    checkpoint = loaded.get("checkpoint")
    if checkpoint is None:
        raise FileNotFoundError(f"Missing checkpoint for run: {task_id}")
    open_segments = [item for item in loaded.get("segments", []) if item.get("finished_at") is None]
    if not open_segments:
        raise ValueError(f"Run has no crashed/open segment to recover: {task_id}")
    previous_segment_id = open_segments[-1]["segment_id"]
    segment_id = f"recovery-{uuid4().hex}"
    active_store.begin_recovery(task_id, previous_segment_id, segment_id)

    trace = active_store.load_run(task_id)["trace"]
    trace["_store"] = active_store
    trace["_segment_id"] = segment_id
    task_state = TaskState.model_validate(checkpoint["state"])
    recent_messages = [{"role": "system", "content": checkpoint["context_pack"]}]
    recent_messages.extend(reconstruct_recent_messages(trace, max_turns=2))
    answer = run_agent(
        task_state.user_goal,
        trace,
        max_steps=max_steps,
        task_state=task_state,
        recent_messages=recent_messages,
        start_step=max(checkpoint["step"] + 1, _max_trace_step(trace) + 1),
        model_client=model_client,
        tool_executor=tool_executor,
        store=active_store,
        segment_id=segment_id,
    )
    trace["usage_summary"] = summarize_usage(trace)
    return answer, {"database": str(active_store.db_path), "task_id": task_id}


def export_run(task_id, out_path, base_dir=DEFAULT_TRACE_DIR, store=None):
    active_store = _resolve_store("sqlite", base_dir, store)
    if not isinstance(active_store, SQLiteRunStore):
        raise ValueError("SQLite export requires SQLiteRunStore.")
    trace = active_store.load_run(task_id)["trace"]
    trace["usage_summary"] = summarize_usage(trace)
    save_trace(trace, str(out_path))
    return str(out_path)


def parse_export_args(argv):
    if not argv:
        raise ValueError("Usage: python3 agent.py export <task_id> --format jsonl --out <path>")
    args = list(argv)
    task_id = make_task_id(args.pop(0))
    export_format = None
    out_path = None
    if "--format" in args:
        index = args.index("--format")
        if index + 1 >= len(args):
            raise ValueError("--format requires jsonl.")
        export_format = args[index + 1]
        del args[index : index + 2]
    if "--out" in args:
        index = args.index("--out")
        if index + 1 >= len(args):
            raise ValueError("--out requires a path.")
        out_path = args[index + 1]
        del args[index : index + 2]
    if args:
        raise ValueError(f"Unknown export arguments: {' '.join(args)}")
    if export_format != "jsonl":
        raise ValueError("Only --format jsonl is supported.")
    if not out_path:
        raise ValueError("--out is required.")
    return task_id, out_path


def parse_recover_args(argv):
    if not argv:
        raise ValueError("Usage: python3 agent.py recover <task_id> [--max-steps N]")
    task_id, max_steps, store_type = parse_resume_args(argv)
    if store_type != "file":
        raise ValueError("recover always uses the default SQLite store; omit --store.")
    return task_id, max_steps


def _save_legacy_trace_if_needed(trace, trace_path, canonical_trace_path):
    if not trace_path or trace_path == canonical_trace_path:
        return
    save_trace(trace, trace_path)


def _persist_interrupted_run(trace, task_state, run_dir, active_store, segment_id):
    add_event(trace, "error", {"message": "Interrupted by user.", "token_estimate": 1})
    context_pack = prepare_context_pack(trace, task_state, _max_trace_step(trace), run_dir=run_dir)
    persist_checkpoint(trace, task_state, run_dir, context_pack, step=_max_trace_step(trace))
    if not isinstance(active_store, FileRunStore):
        return

    add_event(
        trace,
        "segment_interrupted",
        {
            "task_id": task_state.task_id,
            "segment_id": segment_id,
            "exit_reason": "interrupted",
        },
        step=_max_trace_step(trace),
    )
    active_store.finish_segment(task_state.task_id, segment_id, "interrupted")
    trace["_segment_finished"] = True
    _sync_trace_from_store(trace, active_store)


def main():
    if len(sys.argv) < 2:
        print("No task specified. Please provide a task as a command-line argument.")
        sys.exit(1)

    if sys.argv[1] == "trace":
        if len(sys.argv) < 3:
            print("Usage: python3 agent.py trace <trace-file>")
            sys.exit(1)
        print_trace(sys.argv[2])
        return

    if sys.argv[1] == "eval":
        from eval_runner import parse_eval_args, run_eval

        try:
            tasks_path, out_path = parse_eval_args(sys.argv[2:])
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        run_eval(tasks_path, out_path)
        return

    if sys.argv[1] == "replay":
        if len(sys.argv) != 3:
            print("Usage: python3 agent.py replay <trace-file>")
            sys.exit(1)
        from .replay import print_replay_results, validate_trace

        try:
            replay_trace = load_trace(sys.argv[2])
            replay_passed = print_replay_results(validate_trace(replay_trace))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"Replay failed: {e}")
            sys.exit(1)
        if not replay_passed:
            sys.exit(1)
        return

    if sys.argv[1] == "resume":
        try:
            task_id, max_steps, store_type = parse_resume_args(sys.argv[2:])
            answer, paths = resume_task(task_id, max_steps=max_steps, store_type=store_type)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"Resumed task: {task_id}")
        print("\nFinal Answer:")
        print(answer)
        print(f"Checkpoint saved to {paths.get('database') or paths['run_dir']}")
        return

    if sys.argv[1] == "recover":
        try:
            task_id, max_steps = parse_recover_args(sys.argv[2:])
            answer, paths = recover_task(task_id, max_steps=max_steps)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"Recovered task: {task_id}")
        print("\nFinal Answer:")
        print(answer)
        print(f"Checkpoint saved to {paths['database']}")
        return

    if sys.argv[1] == "export":
        try:
            task_id, out_path = parse_export_args(sys.argv[2:])
            export_run(task_id, out_path)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"Exported task {task_id} to {out_path}")
        return

    try:
        run_args = parse_run_args(sys.argv[1:])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    user_task = run_args["user_task"]
    task_id = run_args["task_id"]
    store_type = run_args["store_type"]
    run_dir = run_args["run_dir"]
    trace_path = run_args["trace_path"]
    paths = artifact_paths(task_id, run_dir=run_dir)
    print(f"Executing user task: {user_task}")
    print(f"Task id: {task_id}")

    trace = None
    task_state = None
    try:
        active_store = _resolve_store(store_type)
        active_store.create_run(task_id, user_task)
        segment_id = f"task-{uuid4().hex}"
        active_store.start_segment(task_id, segment_id, "task")
        trace = new_trace(user_task, task_id=task_id)
        trace["_store"] = active_store
        trace["_segment_id"] = segment_id
        for event in trace["events"]:
            active_store.append_event(task_id, event)
        task_state = new_task_state(task_id, user_task)
        answer = run_agent(
            user_task,
            trace,
            max_steps=run_args["max_steps"],
            task_state=task_state,
            run_dir=run_dir if store_type == "file" else None,
            store=active_store,
            segment_id=segment_id,
        )
        print("\nFinal Answer:")
        print(answer)
    except KeyboardInterrupt:
        if trace is not None and task_state is not None:
            effective_run_dir = run_dir if store_type == "file" else None
            _persist_interrupted_run(
                trace,
                task_state,
                effective_run_dir,
                active_store,
                segment_id,
            )
        print("\nInterrupted. Checkpoint saved.")
        follow_up = f"python3 agent.py resume {task_id}" if store_type == "file" else f"python3 agent.py recover {task_id}"
        print(f"Continue with: {follow_up}")
    except Exception as e:
        if trace is not None and task_state is not None:
            add_event(trace, "error", {"message": str(e), "token_estimate": estimate_text_tokens(str(e))})
            effective_run_dir = run_dir if store_type == "file" else None
            context_pack = prepare_context_pack(trace, task_state, _max_trace_step(trace), run_dir=effective_run_dir)
            persist_checkpoint(trace, task_state, effective_run_dir, context_pack, step=_max_trace_step(trace))
        print(f"Error: {e}")
    finally:
        if trace is not None:
            trace["usage_summary"] = summarize_usage(trace)
            if store_type == "file":
                save_trace(trace, paths["trace"])
                _save_legacy_trace_if_needed(trace, trace_path, paths["trace"])
                print(f"Checkpoint saved to {paths['run_dir']}")
                print(f"Trace saved to {paths['trace']}")
            else:
                print(f"Checkpoint saved to {Path(DEFAULT_TRACE_DIR) / 'agent.db'}")


if __name__ == "__main__":
    main()
