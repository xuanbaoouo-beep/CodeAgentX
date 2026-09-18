"""worker 配置项。

本文件不含缺陷，只集中放置队列名、日志级别等常量，
便于其它模块 import（真实项目里这些值会来自环境变量）。
"""

#: 队列名（真实项目里指向 broker 里的具体队列）
QUEUE_NAME = "default"

#: 日志级别
LOG_LEVEL = "INFO"

#: 工作线程数
WORKER_CONCURRENCY = 4
