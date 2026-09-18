"""worker 主循环：取任务、执行、回写状态。

缺陷见 README（请勿修复）：异常被吞、任务执行无超时、payload 原样进日志。
"""

import logging

from worker.queue import dequeue, set_status
from worker.retry import run_with_retry

logger = logging.getLogger(__name__)


def log_task(task):
    """打印任务内容，方便排查问题。

    缺陷：把整个任务 payload 原样写进日志，其中可能包含口令、token 等
    敏感字段，日志一旦被读取就等同于凭据泄露。
    """
    logger.info("开始处理任务：%s", task)


def execute(task):
    """执行单个任务并回写状态。

    缺陷：执行任务时没有设置超时，卡死的任务会永久占住 worker，
    队列后面的任务再也得不到处理。
    """
    set_status(task["id"], "running")
    handler = task["handler"]
    result = run_with_retry(task, handler)
    set_status(task["id"], "done")
    return result


def run_once():
    """从队列取一个任务并执行。

    缺陷：``except Exception`` 捕获后只打印一条日志，既不重新抛出，
    也不把任务标记为 failed，任务会静默丢失，调用方无从感知。
    """
    task = dequeue()
    log_task(task)
    try:
        return execute(task)
    except Exception as exc:
        logger.warning("任务执行失败：%s", exc)
