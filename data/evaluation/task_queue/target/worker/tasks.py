"""内置任务类型：把任务名映射到具体的处理函数。

本文件不含缺陷，只提供跑通 worker 所需的最小任务集合。
"""


def add(task):
    """示例任务：把 payload 里的两个数相加。"""
    return task["a"] + task["b"]


def send_mail(task):
    """示例任务：发送邮件（演示用，只打印）。"""
    print(f"[mail] to={task['to']} subject={task['subject']}")
    return True


#: 任务名 -> 处理函数
TASKS = {
    "add": add,
    "send_mail": send_mail,
}


def get_handler(name):
    """按任务名取出处理函数。"""
    return TASKS[name]
