import sys
import os
import json
import subprocess
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE_URL")
)
MODEL = os.getenv("OPENAI_MODEL","MiniMax-M3")

def add_event(trace, event_type, data):
    trace["events"].append({
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    })

def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return f"Error reading file {path}: {e}"

def write_file(path,content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}"
    except Exception as e:
        print(f"Error writing to file {path}: {e}")
        return f"Error writing to file {path}: {e}"

def run_shell(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "Command timed out after 10 seconds."
        }

def now():
    return datetime.now().isoformat()

def save_trace(trace, trace_path):
    try:
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving trace to {trace_path}: {e}")


def clean_model_content(content):
    if not content:
        return ""

    content = content.strip()

    while "<think>" in content and "</think>" in content:
        start = content.find("<think>")
        end = content.find("</think>") + len("</think>")
        content = content[:start] + content[end:]

    return content.strip()


def validate_tool_args(tool_name, args):
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    if tool_name == "read_file":
        if "path" not in args:
            raise ValueError("Missing required argument: path")
        if not isinstance(args["path"], str):
            raise ValueError("Argument 'path' must be a string.")
    elif tool_name == "write_file":
        if "path" not in args:
            raise ValueError("Missing required argument: path")
        if "content" not in args:
            raise ValueError("Missing required argument: content")
        if not isinstance(args["path"], str):
            raise ValueError("Argument 'path' must be a string.")
        if not isinstance(args["content"], str):
            raise ValueError("Argument 'content' must be a string.")
    elif tool_name == "run_shell":
        if "command" not in args:
            raise ValueError("Missing required argument: command")
        if not isinstance(args["command"], str):
            raise ValueError("Argument 'command' must be a string.")
    else:
        raise ValueError(f"Tool '{tool_name}' is not available.")

TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell
}
OPENAI_TOOLS = [
    {
        "type": "function",
        "function":{
            "name": "read_file",
            "description": "Read the content of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read."
                    }
                },
                "required": ["path"],
                "additionalProperties": False
            }
        }
    },{
        "type": "function",
        "function":{
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": False
            }
        }
    }, {
        "type": "function",
        "function":{
            "name": "run_shell",
            "description": "执行安全的 shell 命令。主要用于 ls、pwd、cat、pytest 等项目检查命令。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令"
                    }
                },
                "required": ["command"],
                "additionalProperties": False
            }
        }
    }
]
def execute_tool(action):
    tool_name = action.get("tool")
    args = action.get("args",{})

    if tool_name not in TOOLS:
        raise ValueError(f"Tool '{tool_name}' is not available.")
    validate_tool_args(tool_name, args)
    return TOOLS[tool_name](**args)

def run_agent(user_task, trace, max_steps=8):
    messages = [
        {
            "role":"system",
            "content":(
                "你是一个最小 Agent Loop。"
                "你可以通过工具读取文件、写文件、运行安全 shell 命令。"
                "当你需要了解项目内容时，优先调用 read_file。"
                "当你已经获得足够信息后，不要再调用工具，直接用中文回答用户。"
            )
        },{
            "role":"user",
            "content": user_task
        }
    ]
    for step in range(1,max_steps + 1):
        add_event(trace,"llm_called", {
            "purpose": "agent_step",
            "step": step,
            "model": MODEL,
        })

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto"
        )

        message = completion.choices[0].message

        add_event(trace,"llm_result",{
            "purpose": "agent_step",
            "step": step,
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
                for tool_call in (message.tool_calls or [])
            ]
        })

        if not message.tool_calls:
            raw_content = message.content or ""
            answer = clean_model_content(raw_content)

            if answer:
                add_event(trace,"final_answer",{
                    "step": step,
                    "answer": answer,
                    "exit_reason": "no_tool_calls"
                })

                return answer

            add_event(trace, "protocol_error", {
                "step": step,
                "content": raw_content,
                "error": "Model returned no tool_calls and empty content."
            })

            fallback_answer = "任务结束，但模型没有给出最终答案。"
            add_event(trace,"final_answer",{
                "step": step,
                "answer": fallback_answer,
                "exit_reason": "empty_content"
            })
            return fallback_answer
        
        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]
        }

        messages.append(assistant_message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                tool_result = {
                    "ok": False,
                    "error": f"Invalid tool arguments JSON: {str(e)}",
                    "raw_arguments": tool_call.function.arguments
                }
            else:
                action = {
                    "tool": tool_name,
                    "args": tool_args
                }

                add_event(trace, "tool_called", {
                    "step": step,
                    "tool_call_id": tool_call.id,
                    "tool": tool_name,
                    "args": tool_args
                })

                try:
                    result = execute_tool(action)
                    tool_result = {
                        "ok": True,
                        "result": result
                    }
                except Exception as e:
                    tool_result = {
                        "ok": False,
                        "error": str(e)
                    }

            add_event(trace, "tool_result", {
                "step": step,
                "tool_call_id": tool_call.id,
                "tool": tool_name,
                "result": tool_result
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

    answer = f"达到最大循环次数 {max_steps}，任务未完成。"

    add_event(trace, "final_answer", {
        "answer": answer,
        "exit_reason": "max_steps"
    })

    return answer


def main():
    if len(sys.argv) < 2:
        print("No task specified. Please provide a task as a command-line argument.")
        sys.exit(1)
        
    user_task = sys.argv[1]
    print(f"Executing user task: {user_task}")

    trace = {
        "task": user_task,
        "started_at": now(),
        "finished_at": None,
        "events": []
    }
    os.makedirs("traces", exist_ok=True)
    trace_filename = datetime.now().strftime("%Y%m%d_%H%M%S.json")
    trace_path = os.path.join("traces", trace_filename)

    add_event(trace, "task_started", {"task": user_task})

    try:
        answer = run_agent(user_task, trace, max_steps=8)
        print("\nFinal Answer:")
        print(answer)

    except Exception as e:
        add_event(trace, "error", {"message": str(e)})
        print(f"Error: {e}")

    finally:
        trace["finished_at"] = now()
        save_trace(trace, trace_path)
        print(f"Trace saved to {trace_path}")

if __name__ == "__main__":
    main()
