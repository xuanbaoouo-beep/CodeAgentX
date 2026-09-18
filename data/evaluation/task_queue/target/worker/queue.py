"""任务队列：入队、出队与任务状态维护。

缺陷见 README（请勿修复）：pickle 反序列化不可信数据、状态读改写非原子。
"""

import pickle
import queue

#: 进程内的任务队列，元素是序列化后的任务数据
_QUEUE: "queue.Queue[bytes]" = queue.Queue()

#: 任务状态表：task_id -> 状态字符串
_STATUS: dict[str, str] = {}


def enqueue(task):
    """把任务对象序列化后放进队列。"""
    _QUEUE.put(pickle.dumps(task))


def dequeue():
    """从队列里取出一个任务并还原成对象。

    缺陷：用 pickle.loads 直接反序列化队列里的数据，而队列内容可能来自
    外部（其它进程 / 网络 / 上游生产者），攻击者可构造恶意 payload，
    在反序列化时执行任意代码（RCE）。
    """
    raw = _QUEUE.get()
    return pickle.loads(raw)


def set_status(task_id, status):
    """更新任务状态并返回旧状态。

    缺陷：任务状态是"读—改—写"三步，中间没有加锁，多 worker 并发更新
    同一个任务时会互相覆盖（丢失更新）。
    """
    current = _STATUS.get(task_id, "pending")
    _STATUS[task_id] = status
    return current


def get_status(task_id):
    """读取任务状态，未知任务返回 pending。"""
    return _STATUS.get(task_id, "pending")
