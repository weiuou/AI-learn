# Mini Agent Harness

这是一个不用 LangChain / LangGraph、手写实现的最小 Agent Harness。项目从最小 Agent Loop 升级为更容易调试、可审计的版本：支持工具调用、结构化 trace、统一工具错误、错误恢复提示、Sandbox / Permission、Eval Harness，以及初版 context compression。

## 当前能力

- 使用 OpenAI Chat Completions native tools。
- 支持工具：`read_file(path)`、`write_file(path, content)`、`run_shell(command, cwd=".")`。
- 最大执行步数：50 步，防止 Agent 无限循环。
- 自动读取 `.env` 中的模型配置。
- 每次运行保存完整 trace 到 `runs/*.json`。
- 支持回放 trace：`python3 agent.py trace runs/demo.json`。
- trace 会统计模型调用次数、工具调用次数和 API 返回的 token usage。
- 工具返回统一 `ToolResult` JSON，便于模型根据错误类型恢复。
- 工具有风险等级、命令策略、项目目录边界、超时、输出截断和 human approval 中断点。
- Eval 可以通过 trace 判断具体工具调用是 `allow`、`deny` 还是 `require_approval`。
- 当 messages 超过 12000 字符时，会触发初版 context compression。
- 每个新任务默认保存 checkpoint 到 `runs/{task_id}/`。
- 支持 `TaskState`、`Context Pack`、工具结果摘要和 `agent resume <task_id>`。
- 支持 Run Budget、无进展循环检测和完全离线的 Replay Regression。
- 支持可替换的 File / SQLite Run Store、原子 checkpoint 和崩溃恢复。

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

指定任务 ID，便于恢复：

```bash
python3 agent.py "读取 README，总结这个项目" --task-id demo_readme
```

限制步数，模拟中断或未完成任务：

```bash
python3 agent.py "读取 README，再运行测试" --task-id demo_resume --max-steps 1
python3 agent.py resume demo_resume
```

指定 trace 输出文件：

```bash
python3 agent.py "读取 README，总结这个项目" --trace runs/demo.json
```

回放一次 trace：

```bash
python3 agent.py trace runs/demo.json
```

离线验证 trace invariant（不会调用模型或真实工具）：

```bash
python3 agent.py replay runs/demo_resume/trace.jsonl
```

使用 SQLite Durable Runtime：

```bash
python3 agent.py "读取 README 并总结" --task-id durable_demo --store sqlite
python3 agent.py resume durable_demo --store sqlite
python3 agent.py export durable_demo --format jsonl --out runs/durable_demo.jsonl
python3 agent.py replay runs/durable_demo.jsonl
```

SQLite 默认写入 `runs/agent.db`。普通命令仍默认使用文件模式；`--trace` 只适用于文件模式，SQLite run 需要通过 `export` 生成 JSONL。

旧的 `traces/*.json` 文件也可以用同一个回放命令查看。

## Checkpoint / Resume

新任务默认写入：

```text
runs/{task_id}/trace.jsonl
runs/{task_id}/state.json
runs/{task_id}/context_pack.md
```

- `trace.jsonl` 是完整审计日志，保存每个模型调用、工具调用、工具完整 observation、压缩事件和 checkpoint 事件。
- `state.json` 保存结构化 `TaskState`，包括用户目标、当前计划、已完成步骤、关键事实、触碰文件、最近错误和下一步建议。
- `context_pack.md` 是下一轮模型调用使用的紧凑上下文，不再直接塞完整历史。

恢复任务：

```bash
python3 agent.py resume <task_id>
```

恢复时会读取 `state.json` 和 `trace.jsonl`，重建 Context Pack，并只带上最近 1-2 轮压缩后的原始工具消息继续执行。

## Durable Runtime / Crash Recovery

`RunStore` 把运行循环与持久化实现隔离。`FileRunStore` 保持上述三文件布局；`SQLiteRunStore` 使用 `runs`、`segments`、`events`、`checkpoints` 四张表，并保证 task 内事件序号严格递增。

`resume` 和 `recover` 语义不同：`resume` 主动继续一个已经正常关闭的 segment；`recover` 只处理 SQLite 中最后一个没有关闭的 segment，从最后一次成功 checkpoint 创建 recovery segment：

```bash
python3 agent.py recover <task_id>
```

原 segment 会标记为 `crashed`。恢复只把历史工具 observation 转成上下文，不会重新执行历史工具；恢复之后模型新产生的工具调用仍按正常权限策略执行。SQLite checkpoint 在一个事务内写入 `checkpoint_started`、snapshot、run 状态和 `checkpoint_saved`，失败时整体回滚。

设计说明见 [《Agent Harness 为什么需要 Durable Runtime》](docs/agent-durable-runtime.md)。

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
- `COMMAND_BLOCKED`（旧 trace 兼容）
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
- `budget_exceeded`
- `loop_detected`

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
- `risk_level`
- `approval_required`
- `approved`
- `policy_decision`
- `risk_reason`
- `timeout_sec`
- `truncated`
- `observation`
- `error`
- `usage`
- `token_estimate`

## Usage 统计

新生成的 trace 会在顶层写入 `usage_summary`，并在每个 `llm_result` 事件里保存模型 API 返回的原始 usage 摘要。

`usage_summary` 包括：

- `model_calls`
- `usage_calls`
- `missing_usage_calls`
- `tool_calls`
- `context_compressions`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `reasoning_tokens`
- `cached_tokens`
- `cache_creation_tokens`
- `cache_hit_rate`

cache 命中率的计算方式：

```text
cache_hit_rate = cached_tokens / prompt_tokens
```

如果当前模型或 API provider 没有返回 cache 相关字段，回放时会显示 `Cache usage: unavailable`。如果没有返回任何 usage 字段，会显示 `API usage: unavailable`。

## Run Budget

`agent/budget.py` 将运行限制集中在独立的 `RunBudget` 与 `BudgetGuard` 中。Guard 在模型请求和工具执行之前检查限制，累计 step、模型调用、工具调用、所有 prompt 字符、墙钟耗时和连续工具失败。超限动作不会执行，trace 写入 `budget_exceeded`，终止原因统一为 `budget_exceeded`。

```python
RunBudget(
    max_steps=20,
    max_model_calls=20,
    max_tool_calls=30,
    max_prompt_chars=120_000,
    max_wall_time_sec=300,
    max_consecutive_failures=3,
)
```

CLI 的 `--max-steps` 会覆盖当前运行段的 step 上限；模型调用仍受独立的 `max_model_calls` 限制。Trace 顶层 `budget_summary` 保存 limits、整条 trace 的 consumed、当前运行段的 `segment_consumed`、耗时和触发的限制。

## Loop Detection

`agent/loop_detector.py` 使用 `tool_name + normalized_arguments + error_type` 生成稳定动作指纹，检测同一动作连续三次、最近四个动作满足 `A/B/A/B` 且 `A != B`，以及没有成功 observation 的连续三次工具失败。

第一次命中只注入一次结构化 recovery hint，要求模型重新规划。新的成功 observation 会结束恢复期；恢复后的再次失败或之后再次命中循环会停止运行并使用 `exit_reason=loop_detected`。Trace 只保存模式、相关 step 和指纹摘要，不保存隐式思维链。

## Offline Replay Regression

`agent/replay.py` 提供 `validate_trace(trace) -> list[InvariantResult]`。Replay 仅加载 JSON/JSONL，检查 trace schema、时间顺序、工具调用/结果配对、每个 task/resume 运行段唯一终止事件，以及 usage/budget summary 是否能从事件重算。每项 invariant 输出 PASS/FAIL；任一失败时 CLI 返回非零退出码。

旧 trace 的 `exit_reason=no_tool_calls` 在 Eval 中兼容映射为 `completed`。包含 resume 的 trace 以 `task_started` 和 `resume_started` 分段，每段必须恰好包含一个 `final_answer`。

## Sandbox / Permission

当前不是 Docker 级沙箱，而是一个可评测的最小执行边界：

- 工具层：工具注册时标注 `risk_level` 和 `approval_required`。
- 参数层：工具参数必须通过 schema 校验。
- 权限层：危险命令直接拒绝；需要 approval 的动作在非交互 eval 中默认拒绝。
- 环境层：文件路径和 shell `cwd` 限制在项目根目录内，shell 默认 10 秒超时，输出默认截断到 8000 字符，并清理 API key / token 类环境变量。
- 审计层：trace 记录 `policy_decision`、`risk_reason`、`approved` 和工具结果。

`run_shell(command)` 当前会拒绝这些高风险模式：

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
- 重定向写入 `/dev/`
- 重定向写入 `~/.ssh/`

被拦截时，工具返回 `PERMISSION_DENIED`，不会真的执行命令。允许列表包括 `pwd`、`ls`、`cat`、`grep`、`find`、`sed`、`python`、`python3`、`pytest`、`git diff`、`git status`。

权限控制和错误恢复的区别：

- 权限控制决定工具动作能不能发生，例如拒绝 `rm -rf` 或 `/etc/passwd`。
- 错误恢复发生在动作被允许之后，例如文件不存在、命令参数错、测试失败后让 Agent 改用更合适的检查方式。

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

## Context Manager

`agent/context_manager.py` 提供：

- `build_context_pack(task_state, recent_trace, tool_summaries, max_chars=12000)`
- `compress_tool_result(tool_name, tool_args, tool_result)`
- `collect_recent_tool_summaries(trace, limit=5)`

压缩规则：

- shell `stdout/stderr` 超过 4000 字符时，Context Pack 只保留 head、tail 和 summary。
- `read_file` 文件内容超过 8000 字符时，Context Pack 只保留文件路径、行号范围和片段。
- 搜索型输出超过 5 条时，Context Pack 保留前 5 条和省略计数。
- trace 仍保存工具返回的完整 observation；压缩只影响下一轮模型输入。

Context Manager 和旧 `context_compressor.py` 的关系：

- Context Manager 是默认路径，每轮都会构造 Context Pack。
- `context_compressor.py` 作为兜底，只有短消息窗口仍超过 12000 字符时才触发。

## 验收命令

语法检查：

```bash
python3 -m py_compile agent.py agent/*.py context_compressor.py eval_runner.py evaluators.py failure_classifier.py
```

正常文件读取：

```bash
python3 agent.py "读取 agent.py，说明这个 Agent Loop 是怎么工作的" --trace runs/read_agent.json
python3 agent.py trace runs/read_agent.json
```

Checkpoint / resume：

```bash
python3 agent.py "读取 readme.md，再运行 pytest，总结下一步" --task-id demo_resume --max-steps 1
python3 agent.py resume demo_resume
python3 agent.py trace runs/demo_resume/trace.jsonl
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

越界文件访问：

```bash
python3 agent.py "读取项目目录外的 /etc/passwd" --trace runs/cwd_escape.json
python3 agent.py trace runs/cwd_escape.json
```

批量 eval：

```bash
python3 agent.py eval evals/tasks.jsonl --out runs/eval_report.json
```

报告额外包含 `termination_summary`、`loop_detection` 和 `replay_regression`，用于观察终止分布、循环恢复效果和历史 trace invariant 回归结果。

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
- 新增 `agent/` 包，拆出 permissions、sandbox、tools、approval。
- 新增危险 shell 命令和越界文件访问的 `PERMISSION_DENIED`。
- 新增安全 eval 和 `security_summary`。
- 新增 `context_compressor.py`。
- 新增 `.env` 自动读取和 `.env.example`。
- 更新 `.gitignore`，忽略 `.env` 和 `runs/`。

## 明天可以继续

运行边界设计说明见 [《Agent 为什么需要预算、循环检测和离线回放》](docs/agent-runtime-guardrails.md)。

下一步可以把 shell policy 从字符串规则升级为更可靠的命令解析，并补充网络权限、diff approval、以及基于历史 trace 的 replay regression。
