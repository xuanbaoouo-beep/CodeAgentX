"""认证与会话层：签发会话令牌、核对口令。

缺陷见 README（请勿修复）：签名密钥硬编码、口令明文比对。
"""

from app.db.store import find_user_by_name

#: 会话/JWT 签名用的密钥
# 缺陷：签名密钥硬编码在源码里——任何能读到源码的人都能伪造任意用户的令牌，
# 应从环境变量或密钥管理服务读取，不能提交进仓库。
JWT_SECRET = "blog-api-demo-jwt-secret"

#: 令牌有效期（秒），演示用
TOKEN_TTL_SECONDS = 3600


def issue_token(username):
    """给指定用户签发会话令牌（演示用，不做真实 JWT 编码）。"""
    return f"{username}.{JWT_SECRET}"


def verify_password(username, password):
    """核对用户名与口令，通过返回 True。

    缺陷：直接把请求里的口令和存储的口令用 == 比较，
    既没有对存储口令做哈希/加盐，也没有用 hmac.compare_digest 做常量时间比较。
    """
    user = find_user_by_name(username)
    if user is None:
        return False
    return password == user.password
