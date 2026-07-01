import json
import os
import re
import subprocess
import sys
from datetime import datetime

from openai import OpenAI

from context_compressor import COMPRESSION_THRESHOLD, compress_messages, estimate_messages_size


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

client = None


def get_client():
    global client
    if client is None:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL"),
        )
    return client


def now():
    return datetime.now().isoformat()


def estimate_text_tokens(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return max(1, len(text) // 4)


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


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return tool_success(f.read())
    except FileNotFoundError:
        return tool_error(
            "FILE_NOT_FOUND",
            f"{path} does not exist",
            True,
            "Use run_shell to list files, or search with find . -iname '*readme*' / find . -name '*.py'.",
        )
    except PermissionError:
        return tool_error(
            "PERMISSION_DENIED",
            f"Permission denied while reading {path}",
            False,
            "Ask the user for permission or choose a readable project file.",
        )
    except Exception as e:
        return tool_error(
            "READ_ERROR",
            f"Could not read {path}: {e}",
            True,
            "Check that the path is a regular UTF-8 text file, or list nearby files first.",
        )


def write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return tool_success(f"Wrote {len(content)} characters to {path}")
    except PermissionError:
        return tool_error(
            "PERMISSION_DENIED",
            f"Permission denied while writing {path}",
            False,
            "Ask the user for permission or choose a writable project path.",
        )
    except Exception as e:
        return tool_error(
            "WRITE_ERROR",
            f"Could not write {path}: {e}",
            True,
            "Check the directory exists and retry with a writable path.",
        )


def is_safe_shell_command(command):
    lowered = command.strip().lower()
    blocked_patterns = [
        (r"\brm\s+-[^\n;|&]*r[^\n;|&]*f|\brm\s+-[^\n;|&]*f[^\n;|&]*r", "rm -rf is blocked."),
        (r"(^|[;&|]\s*)sudo\b", "sudo is blocked."),
        (r"(^|[;&|]\s*)curl\b", "curl is blocked."),
        (r"(^|[;&|]\s*)wget\b", "wget is blocked."),
        (r"(^|[;&|]\s*)ssh\b", "ssh is blocked."),
        (r"(^|[;&|]\s*)scp\b", "scp is blocked."),
        (r"\bchmod\s+777\b", "chmod 777 is blocked."),
        (r"(^|[;&|]\s*)mkfs\b", "mkfs is blocked."),
        (re.escape(":(){ :|:& };:"), "fork bomb pattern is blocked."),
        (r">+\s*/etc/", "redirecting output into /etc/ is blocked."),
        (r">+\s*(~|/users/[^/]+)/\.ssh/", "redirecting output into ~/.ssh/ is blocked."),
    ]

    for pattern, reason in blocked_patterns:
        if re.search(pattern, lowered):
            return False, reason
    return True, None


def run_shell(command):
    is_safe, reason = is_safe_shell_command(command)
    if not is_safe:
        return tool_error(
            "COMMAND_BLOCKED",
            reason,
            False,
            "Use a read-only inspection command such as ls, find, pwd, cat, sed, or python3 -m py_compile.",
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return tool_error(
            "COMMAND_TIMEOUT",
            "Command timed out after 10 seconds.",
            True,
            "Narrow the command with filters, limit output, or inspect a smaller path.",
        )
    except Exception as e:
        return tool_error(
            "COMMAND_FAILED",
            f"Command could not be executed: {e}",
            True,
            "Try a simpler read-only command.",
        )

    payload = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        message = f"Command exited with return code {result.returncode}."
        if output:
            message = f"{message} Output: {shorten(output, 300)}"
        return tool_error(
            "COMMAND_FAILED",
            message,
            True,
            "Read stderr and retry with a narrower or corrected command.",
        )
    return tool_success(payload)


TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
}

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file to read."}
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
            "description": "Write UTF-8 text content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file to write."},
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
            "description": "执行只读或低风险 shell 命令，主要用于 ls、pwd、cat、find、sed、python3 -m py_compile 等项目检查命令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 shell 命令"}
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

    return None


def execute_tool(action):
    tool_name = action.get("tool")
    args = action.get("args", {})
    validation_error = validate_tool_args(tool_name, args)
    if validation_error:
        return validation_error
    return TOOLS[tool_name](**args)


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
                        "token_estimate": estimate_text_tokens(tool_args),
                    },
                    step=step,
                )
                tool_result = execute_tool({"tool": tool_name, "args": tool_args})

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
    print(f"Task: {trace.get('user_goal') or trace.get('task')}")
    print(f"Started: {trace.get('started_at')}")
    print(f"Finished: {trace.get('finished_at')}")
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
            if names:
                print(f"[{step}] LLM requested tools: {', '.join(names)}")
            else:
                print(f"[{step}] LLM returned content: {shorten(attrs.get('content') or '', 500)}")
        elif event_type == "tool_called":
            name = attrs.get("tool_call.name") or attrs.get("tool")
            args = attrs.get("tool_call.arguments") or attrs.get("args")
            print(f"[{step}] Tool called: {name} {json.dumps(args, ensure_ascii=False)}")
        elif event_type == "tool_result":
            name = attrs.get("tool_call.name") or attrs.get("tool")
            observation = attrs.get("observation") or attrs.get("result")
            if isinstance(observation, dict) and observation.get("ok"):
                result = observation.get("result")
                print(f"[{step}] Tool result: {name} ok=true len={len(shorten(result, 100000))}")
                print(f"      observation: {shorten(result)}")
            else:
                error = attrs.get("error") or observation or {}
                print(
                    f"[{step}] Tool result: {name} ok=false "
                    f"error_type={error.get('error_type')} recoverable={error.get('recoverable')}"
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
        save_trace(trace, trace_path)
        print(f"Trace saved to {trace_path}")


if __name__ == "__main__":
    main()
