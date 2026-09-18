"""链接预览：抓取目标页面的标题。

缺陷见 README（请勿修复）：按用户传入的 URL 去抓取，未做内网地址拦截。
"""


def _http_get(url):
    """占位实现：此处等价于一次 HTTP 请求（如 requests.get(url)）。

    评估样本不引入第三方网络库，因此只返回伪造的 HTML 响应体。
    """
    return f"<html><head><title>{url}</title></head></html>"


def _extract_title(html):
    """从 HTML 响应体里取出 <title> 的内容。"""
    start = html.find("<title>")
    end = html.find("</title>")
    if start == -1 or end == -1:
        return ""
    return html[start + len("<title>") : end]


def fetch_title(url):
    """按用户传入的 URL 抓取页面标题，用于生成链接预览。

    缺陷：服务端直接拿着用户传入的 URL 发起请求，既没有校验协议，
    也没有拦截 127.0.0.1 / 169.254.169.254 / 内网网段等地址，
    可被用来探测内网服务或读取云元数据（SSRF）。
    """
    html = _http_get(url)
    return _extract_title(html)


def preview(request):
    """GET /preview?url=...：返回链接预览结果。"""
    return {"status": 200, "title": fetch_title(request.get("url", ""))}
