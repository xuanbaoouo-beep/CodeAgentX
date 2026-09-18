"""数据模型：用户与会话。"""

from dataclasses import dataclass


@dataclass
class User:
    """一条用户记录。

    缺陷：密码摘要直接暴露为普通字段，缺少任何脱敏手段。
    """

    name: str
    password_hash: str
    is_active: bool = True


@dataclass
class Session:
    """一次登录会话。"""

    token: str
    user_name: str
    expires_at: str = ""
