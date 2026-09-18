"""HTTP 接口层：创建短链、跳转与统计。

缺陷见 README（请勿修复）：跳转未校验域名白名单（开放重定向）、
统计接口没有任何鉴权。
"""

from app import store
from app.shortener import shorten
from app.validators import is_valid_url


def create_link(request):
    """POST /shorten：校验 URL 并返回短码。"""
    url = request.get("url", "")
    if not is_valid_url(url):
        return {"status": 400, "message": "URL 非法"}
    return {"status": 200, "code": shorten(url)}


def redirect(code):
    """GET /<code>：按短码把浏览器跳转到目标地址。

    缺陷：跳转前不校验目标域名白名单，任何站点都能被生成为短链，
    攻击者可拿本站当钓鱼跳板（开放重定向）。
    """
    target = store.get(code)
    store.record_hit(code)
    return {"status": 302, "location": target}


def stats(request):
    """GET /stats：返回全站短链与点击统计。

    缺陷：这个接口没有任何鉴权（不校验登录态 / API key），
    任意调用者都能拿到全站数据。
    """
    return {
        "links": len(store.all_links()),
        "hits": store.total_hits(),
    }
