# 评估样本仓库：task_queue

这是 **CodeAgentX 评估用样本仓库**。仓库里的缺陷是**刻意植入的测试数据**，
用于离线验证审查 Agent 的检索与定位效果，**请勿修复**。

被审目标是一个最小可运行的异步任务队列 worker（只用标准库：
`queue` / `pickle` / `logging`），规模刻意做小，便于人工核对
"标注的行号区间里到底是不是那段代码"。

## 目录结构

```
worker/
  __init__.py     包初始化
  settings.py     队列名、日志级别等配置常量（无缺陷）
  queue.py        入队 / 出队 / 任务状态维护
  retry.py        任务失败后的重试策略
  tasks.py        内置任务类型与处理器注册表（无缺陷）
  runner.py       worker 主循环：取任务、执行、回写状态
```

## 请勿"修好"这里的缺陷

本仓库**故意保留了几处典型缺陷**，供评估使用
（不要把它们当成需要修复的 bug，它们是测试数据）：

| id | 位置 | 缺陷 |
| --- | --- | --- |
| `pickle-untrusted-payload` | `worker/queue.py` | 用 `pickle.loads` 反序列化队列里（可能来自外部）的任务数据，可被构造恶意 payload 执行任意代码 |
| `unbounded-retry` | `worker/retry.py` | 任务失败后无限重试，且没有任何退避策略 |
| `broad-except-swallow` | `worker/runner.py` | `except Exception` 捕获后只打印日志，不重新抛出也不标记失败，任务静默丢失 |
| `task-state-race` | `worker/queue.py` | 任务状态"读—改—写"非原子，多 worker 并发时会互相覆盖 |
| `no-task-timeout` | `worker/runner.py` | 任务执行没有超时，卡死的任务会永久占住 worker |
| `sensitive-payload-logging` | `worker/runner.py` | 日志把整个任务 payload 原样打出来（含口令 / token 等敏感字段） |

> 上表的 `id` 与评估标注 `data/evaluation/task_queue/labels.json` 一一对应：
> 标注只允许来自这张表，表里没有的一律不算 ground truth
> （防止把模型事后发现的问题补进标签，那是对评测结果的污染）。

## 运行说明

仓库不依赖任何第三方包，直接 `import worker.runner` 即可。
本仓库**不提供**可执行入口与测试，仅作为静态审查的素材。
