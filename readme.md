# Mini Agent Harness

这是一个不用 LangChain / LangGraph、手写实现的最小 Agent Harness。项目从最小 Agent Loop 升级为更容易调试的版本：支持工具调用、结构化 trace、统一工具错误、错误恢复提示、危险 shell 命令拦截，以及初版 context compression。

## 当前能力

- 使用 OpenAI Chat Completions native tools。
- 支持工具：`read_file(path)`、`write_file(path, content)`、`run_shell(command)`。
- 最大执行步数：50 步，防止 Agent 无限循环。
- 自动读取 `.env` 中的模型配置。
- 每次运行保存完整 trace 到 `runs/*.json`。
- 支持回放 trace：`python3 agent.py trace runs/demo.json`。
- 工具返回统一 `ToolResult` JSON，便于模型根据错误类型恢复。
- `run_shell` 有最小安全策略，会拦截高风险命令。
- 当 messages 超过 12000 字符时，会触发初版 context compression。

## 环境配置

复制 `.env.example` 为 `.env`，填入自己的配置：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
```

`.env` 已加入 `.gitignore`，不要提交真实密钥。

## 运行方式

执行一个 Agent 任务：

```bash
python3 agent.py "读取 README，总结这个项目"
```

指定 trace 输出文件：

```bash
python3 agent.py "读取 README，总结这个项目" --trace runs/demo.json
```

回放一次 trace：

```bash
python3 agent.py trace runs/demo.json
```

旧的 `traces/*.json` 文件也可以用同一个回放命令查看。

## ToolResult 格式

所有工具都返回统一结构。

成功：

```json
{
  "ok": true,
  "result": "...",
  "error_type": null,
  "message": null,
  "recoverable": null,
  "suggestion": null
}
```

失败：

```json
{
  "ok": false,
  "result": null,
  "error_type": "FILE_NOT_FOUND",
  "message": "README2.md does not exist",
  "recoverable": true,
  "suggestion": "Use run_shell to list files, or search with find . -iname '*readme*' / find . -name '*.py'."
}
```

目前覆盖的错误类型：

- `FILE_NOT_FOUND`
- `INVALID_ARGUMENTS`
- `TOOL_NOT_FOUND`
- `COMMAND_TIMEOUT`
- `COMMAND_BLOCKED`
- `COMMAND_FAILED`
- `PERMISSION_DENIED`
- `READ_ERROR`
- `WRITE_ERROR`

## Trace 记录

trace 文件使用 JSON 保存，一次 Agent run 会记录这些事件：

- `task_started`
- `llm_called`
- `llm_result`
- `tool_called`
- `tool_result`
- `context_compressed`
- `final_answer`
- `protocol_error`
- `error`

每个事件包含：

- `event_type`
- `step`
- `timestamp`
- `attributes`

常见 attributes 包括：

- `user_goal`
- `model_input_summary`
- `tool_call.name`
- `tool_call.arguments`
- `observation`
- `error`
- `token_estimate`

## Shell 安全策略

`run_shell(command)` 仍然是一个简单工具，不是完整沙箱。当前会拦截这些高风险模式：

- `rm -rf`
- `sudo`
- `curl`
- `wget`
- `ssh`
- `scp`
- `chmod 777`
- `mkfs`
- fork bomb：`:(){ :|:& };:`
- 重定向写入 `/etc/`
- 重定向写入 `~/.ssh/`

被拦截时，工具返回 `COMMAND_BLOCKED`，不会真的执行命令。

## Context Compression

`context_compressor.py` 提供：

- `estimate_messages_size(messages)`
- `compress_messages(messages, trace)`

触发条件：messages 字符数超过 12000。

压缩策略：

- 保留 system message。
- 保留原始 user task。
- 保留最近 2 轮 assistant/tool 原文。
- 把更早的 observation 摘成一个 summary。
- summary 中保留用户目标、已完成步骤、成功读过的文件、失败过的工具调用和失败原因。

触发后 trace 中会出现 `context_compressed` 事件。

## 验收命令

语法检查：

```bash
python3 -m py_compile agent.py context_compressor.py
```

正常文件读取：

```bash
python3 agent.py "读取 agent.py，说明这个 Agent Loop 是怎么工作的" --trace runs/read_agent.json
python3 agent.py trace runs/read_agent.json
```

文件不存在后的恢复：

```bash
python3 agent.py "读取 README2.md，如果不存在，就自己找到正确的 Python 文件并总结。" --trace runs/recovery.json
python3 agent.py trace runs/recovery.json
```

危险命令拦截：

```bash
python3 agent.py "运行 rm -rf /tmp/agent-test" --trace runs/blocked_command.json
python3 agent.py trace runs/blocked_command.json
```

长上下文压缩：

```bash
python3 agent.py "多次读取 agent.py 并总结每个函数的作用、问题和改进建议。" --trace runs/compression.json
python3 agent.py trace runs/compression.json
```

## 今天完成的升级

- 把最小 Agent Loop 升级成 Mini Agent Harness。
- 新增结构化 `ToolResult`。
- 新增 trace 回放命令。
- 新增 `runs/*.json` 作为默认 trace 输出目录。
- 新增危险 shell 命令拦截。
- 新增 `context_compressor.py`。
- 新增 `.env` 自动读取和 `.env.example`。
- 更新 `.gitignore`，忽略 `.env` 和 `runs/`。

## 明天可以继续

下一步可以基于今天的结构化 trace 做 Eval Harness 初版：

- 定义 `tasks.json`。
- 批量运行 Agent。
- 每个任务保存 trace。
- 自动判断 success / fail。
- 根据 `error_type` 和 `exit_reason` 统计失败原因。
