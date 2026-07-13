# Agent Harness 为什么需要 Durable Runtime

Trace 是一次运行中按时间追加的完整事实记录，保存模型调用、工具请求、工具 observation、错误与终止原因，主要用于审计和离线回放。State 是从这些事实归纳出的当前任务状态，例如已完成步骤、关键发现和下一步。Checkpoint 则是在一个确定时刻，把 State 与下一轮所需的 Context Pack 一起持久化，形成可以继续执行的恢复点。Event store 负责以 append-only 方式长期保存 Trace 事件，不覆盖历史事实。

逐个写 `trace.jsonl`、`state.json` 和 `context_pack.md` 时，进程可能在任意两个文件之间退出：State 已更新但 Trace 没有对应事件，或成功事件已经写入而 Context Pack 仍是旧版本。恢复程序无法判断哪个文件可信，这就是崩溃一致性问题。原子 checkpoint 把开始事件、快照更新、run 状态与成功事件放在同一事务中；失败时全部回滚，数据库里只会出现旧的完整状态或新的完整状态。
这也让故障边界和恢复语义能够被确定性测试稳定验证。

Append-only event 与 mutable snapshot 必须同时存在。事件提供不可变的因果历史，便于追责、调试和重新计算；快照提供快速读取，避免每次恢复都扫描全部事件。Resume 表示用户主动继续一个正常结束的 segment，而 crash recovery 表示系统接管一个没有终止记录的 segment，所以后者必须标记旧 segment 为 crashed，并从最后成功 checkpoint 创建新的 recovery segment。

恢复时不能重放真实工具。历史写文件、执行命令或网络请求可能带有副作用，再执行一次会破坏外部状态；恢复只读取旧 observation 作为上下文，之后仅执行模型新产生的动作。当前 Harness 是单进程、本地学习项目，SQLite 无需独立服务，却提供事务、约束、索引和可靠的单文件存储，比引入 PostgreSQL 更容易部署和测试，也足以支撑这一阶段的并发与数据规模。
