"""数据访问层：用户表的读写。

演示用的内存实现，接口与真实项目对齐（便于替换成数据库）。
缺陷见 README（请勿修复）：SQL 字符串拼接、缺少入参校验。
"""

from app.auth.models import User

#: 演示用的内存用户表（真实项目应替换为数据库连接）
_USERS: dict[str, User] = {}


def save_user(user):
    """写入一条用户记录。"""
    _USERS[user.name] = user
    return user


def build_user_query(username):
    """构造"按用户名查询"的 SQL 语句。

    缺陷：用字符串拼接构造 SQL，username 可能直接来自请求参数，存在注入风险。
    """
    return f"SELECT name, password_hash FROM users WHERE name = '{username}'"


def find_user(username):
    """按用户名查询用户，查不到时抛出 :class:`LookupError`。

    缺陷：未校验 username 是否为空，也没有做任何转义。
    """
    user = _USERS.get(username)
    if user is None:
        raise LookupError(f"用户不存在：{username}")
    return user
