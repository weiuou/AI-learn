import json
import os
import sys
from datetime import datetime

from openai import OpenAI

from context_compressor import (
    COMPRESSION_THRESHOLD,
    compress_messages,
    compression_diagnostics,
    estimate_messages_size,
)


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
    attrs = attributes or {}
    if step is None:
        step = attrs.get("step")
    if step is not None:
        attrs["step"] = step

    trace["events"].append(
        {
            "event_type": event_type,
            "type": event_type,
            "step": step,
            "timestamp": now(),
            "attributes": attrs,
            "data": attrs,
        }
    )


def save_trace(trace, trace_path):
    directory = os.path.dirname(trace_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


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


def run_agent(user_task, trace, max_steps=50):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个最小 Agent Harness。"
                "你可以通过工具读取文件、写文件、运行低风险 shell 命令。"
                "工具返回的是统一 JSON：ok/result/error_type/message/recoverable/suggestion。"
                "遇到 recoverable=true 的错误时，优先根据 suggestion 自己恢复，例如列目录、搜索文件、修正参数。"
                "当你已经获得足够信息后，不要再调用工具，直接用中文回答用户。"
            ),
        },
        {"role": "user", "content": user_task},
    ]

    for step in range(1, max_steps + 1):
        messages = maybe_compress_context(messages, trace, step)
        input_summary = message_summary(messages)
        token_estimate = estimate_text_tokens(messages)

        add_event(
            trace,
            "llm_called",
            {
                "step": step,
                "user_goal": user_task,
                "model": MODEL,
                "model_input_summary": input_summary,
                "token_estimate": token_estimate,
            },
            step=step,
        )

        completion = get_client().chat.completions.create(
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
                exit_reason = "empty_content"
            else:
                exit_reason = "no_tool_calls"

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
            return answer

        messages.append(
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
                tool_result, tool_metadata = execute_tool({"tool": tool_name, "args": tool_args})

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

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

    answer = f"达到最大循环次数 {max_steps}，任务未完成。"
    add_event(
        trace,
        "final_answer",
        {
            "step": max_steps,
            "user_goal": user_task,
            "answer": answer,
            "exit_reason": "max_steps",
            "token_estimate": estimate_text_tokens(answer),
        },
        step=max_steps,
    )
    return answer


def load_trace(trace_path):
    with open(trace_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        return None, None

    trace_path = None
    args = list(argv)
    if "--trace" in args:
        index = args.index("--trace")
        if index + 1 >= len(args):
            raise ValueError("--trace requires a path.")
        trace_path = args[index + 1]
        del args[index : index + 2]

    user_task = " ".join(args).strip()
    if not user_task:
        raise ValueError("No task specified. Please provide a task as a command-line argument.")

    if trace_path is None:
        trace_filename = datetime.now().strftime("%Y%m%d_%H%M%S.json")
        trace_path = os.path.join(DEFAULT_TRACE_DIR, trace_filename)

    return user_task, trace_path


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

    try:
        user_task, trace_path = parse_run_args(sys.argv[1:])
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Executing user task: {user_task}")

    trace = {
        "schema_version": "agent-harness-trace-v1",
        "task": user_task,
        "user_goal": user_task,
        "started_at": now(),
        "finished_at": None,
        "events": [],
    }
    add_event(trace, "task_started", {"user_goal": user_task, "token_estimate": estimate_text_tokens(user_task)})

    try:
        answer = run_agent(user_task, trace, max_steps=50)
        print("\nFinal Answer:")
        print(answer)
    except Exception as e:
        add_event(trace, "error", {"message": str(e), "token_estimate": estimate_text_tokens(str(e))})
        print(f"Error: {e}")
    finally:
        trace["finished_at"] = now()
        trace["usage_summary"] = summarize_usage(trace)
        save_trace(trace, trace_path)
        print(f"Trace saved to {trace_path}")


if __name__ == "__main__":
    main()
