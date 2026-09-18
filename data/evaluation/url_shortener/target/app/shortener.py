"""短码生成与创建短链。

缺陷见 README（请勿修复）：短码由自增计数器生成，可被枚举。
"""

import string

from app import store

#: 短码字符表
ALPHABET = string.ascii_lowercase + string.digits

#: 全局自增计数器，进程内单调递增
_counter = 0


def _to_base36(number):
    """把非负整数转成 36 进制字符串。"""
    if number == 0:
        return ALPHABET[0]
    digits = []
    while number:
        number, remainder = divmod(number, len(ALPHABET))
        digits.append(ALPHABET[remainder])
    return "".join(reversed(digits))


def next_code():
    """生成下一个短码。

    缺陷：短码直接由全局自增计数器顺序生成（0, 1, 2 ...），
    完全可预测，任何人按顺序递增即可枚举并遍历他人的短链。
    """
    global _counter
    code = _to_base36(_counter)
    _counter += 1
    return code


def shorten(url):
    """把长链接变成短链并写入存储。"""
    code = next_code()
    store.save(code, url)
    return code
