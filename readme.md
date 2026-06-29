# 这是一个最小Agent Loop项目
# 编码任务

- [ ]  手写最小 Agent Loop
    - 不用 LangChain。
    - 支持工具：read_file(path)、write_file(path, content)、run_shell(command)。
    - 最大循环次数：8 次。
    - 处理不存在的工具。
    - 处理参数缺失或类型错误。
    - shell 超时：10 秒。
    - 模型必须通过 final 退出。

# 最终验收

- [ ]  可以运行：`agent "读取 README，总结这个项目是做什么的"`
- [ ]  程序能调用 read_file。
- [ ]  程序能把 README 内容交给模型。
- [ ]  程序能输出总结。
- [ ]  程序能保存完整 trace。