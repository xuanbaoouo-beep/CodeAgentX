"""任务重试策略。

缺陷见 README（请勿修复）：失败后无限重试且没有任何退避。
"""


def should_retry(attempts):
    """判断某次失败后是否还应该重试。

    本函数只做展示：真正的重试循环在 :func:`run_with_retry` 里。
    """
    return attempts >= 0


def run_with_retry(task, handler):
    """执行任务，失败就重试，直到成功为止。

    缺陷：重试没有次数上限，也没有任何退避（backoff / 指数退避 / 抖动），
    失败任务会被立刻反复重试，打满 CPU 并持续冲击下游依赖。
    """
    while True:
        try:
            return handler(task)
        except Exception:
            continue
