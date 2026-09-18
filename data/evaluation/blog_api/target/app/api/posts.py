"""HTTP 路由层：帖子相关接口。

标准库实现的"Flask 风格"等价简化（见 README「怎么跑起来看」）：
不 import flask，用普通函数模拟视图——``handle_xxx(request)`` 就相当于一个 view，
``request`` 是解析后的请求字典，返回值 ``{"status": ..., "body": ...}`` 对应状态码与响应体。

缺陷见 README（请勿修复）：删帖接口没有任何鉴权、分页大小不设上限。
"""

from app.api.auth import issue_token, verify_password
from app.services import post_service


def handle_login(request):
    """处理 POST /login：核对口令并签发令牌。"""
    username = request.get("username", "")
    password = request.get("password", "")
    if not verify_password(username, password):
        return {"status": 401, "body": {"message": "用户名或口令错误"}}
    return {"status": 200, "body": {"token": issue_token(username)}}


def handle_list_posts(request):
    """处理 GET /posts：按分页返回帖子列表。

    缺陷：page_size 直接取自请求参数且不设上限，
    请求方传个 page_size=100000000 就能让服务一次性构造超大响应（可被用来拖垮服务）。
    """
    page = int(request.get("page", 1))
    page_size = int(request.get("page_size", 20))
    posts = post_service.list_posts(page=page, page_size=page_size)
    return {"status": 200, "body": [post.to_dict() for post in posts]}


def handle_list_posts_by_tag(request):
    """处理 GET /posts?tag=xxx：按标签筛选帖子。"""
    posts = post_service.list_posts_by_tag(request.get("tag", ""))
    return {"status": 200, "body": [post.to_dict() for post in posts]}


def handle_delete_post(request):
    """处理 DELETE /posts/<id>：删除一条帖子。

    缺陷：整个处理函数里没有任何鉴权/权限校验——既没有校验调用方已登录，
    也没有校验调用方是管理员或帖子作者，任何人都能删掉任意帖子。
    """
    post_id = int(request.get("post_id", 0))
    deleted = post_service.delete_post(post_id, requester=request.get("username"))
    if not deleted:
        return {"status": 404, "body": {"message": "帖子不存在"}}
    return {"status": 200, "body": {"deleted": post_id}}
