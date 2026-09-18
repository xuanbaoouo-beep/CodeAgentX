"""HTTP 路由层：把登录请求转给认证服务。

缺陷见 README（请勿修复）：异常信息泄露、缺少登录失败次数限制。
"""

from app.auth.service import login


def handle_login(request):
    """处理 POST /login：成功返回 token，失败返回 401。"""
    username = request.get("username")
    password = request.get("password")
    token = login(username, password)
    if token is None:
        return {"status": 401, "message": "用户名或密码错误"}
    return {"status": 200, "token": token}


def handle_debug_login(request):
    """调试入口：直接把内部异常返回给调用方。

    缺陷：没有鉴权，且把 repr(exc) 原样返回，泄露内部实现细节。
    """
    try:
        token = login(request["username"], request["password"])
    except Exception as exc:
        return {"status": 500, "error": repr(exc)}
    return {"status": 200, "token": token}
