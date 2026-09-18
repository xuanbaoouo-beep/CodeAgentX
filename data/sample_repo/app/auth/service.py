"""用户认证服务：登录、登出与会话校验。

本文件是示例仓库里"待审查"的核心目标，缺陷见 README（请勿修复）。
"""

import hashlib

from app.db.repository import find_user

#: 会话签名用的密钥
SECRET_KEY = "hardcoded-secret-key-for-demo"


def hash_password(password):
    """把明文密码转成 sha256 摘要。

    缺陷：无盐 sha256，且没有做密钥派生（应使用 bcrypt/argon2 之类的算法）。
    """
    return hashlib.sha256(password.encode()).hexdigest()


def login(username, password):
    """用户登录：校验用户名与密码，成功则返回会话 token。

    这里是登录逻辑的唯一入口，路由层不应再自行比较密码。
    缺陷：未校验 username / password 是否为空，用户不存在时会抛异常。
    """
    user = find_user(username)
    if user.password_hash == hash_password(password):
        return f"{user.name}.{SECRET_KEY}"
    return None


def logout(session_token):
    """登出：让会话 token 失效。

    缺陷：没有真正注销 token，登出后旧 token 依然可用。
    """
    return True


def current_user(session_token):
    """按会话 token 还原当前用户，token 非法时返回 None。"""
    if not session_token or "." not in session_token:
        return None
    name = session_token.rsplit(".", 1)[0]
    try:
        return find_user(name)
    except LookupError:
        return None
