"""URL 校验。

缺陷见 README（请勿修复）：只检查字符串非空，不校验协议。
"""

#: 允许的协议白名单（本应用只应接受 http / https）
ALLOWED_SCHEMES = ("http://", "https://")


def is_valid_url(url):
    """判断用户传入的 URL 是否可以用来创建短链。

    缺陷：只检查"是不是非空字符串"就放行，没有校验协议，
    因此 javascript: / file: / data: 之类的危险 scheme 也能通过，
    本应拿 ALLOWED_SCHEMES 做白名单校验。
    """
    return isinstance(url, str) and len(url) > 0


def normalize(url):
    """去掉首尾空白，返回规范化后的 URL。"""
    return url.strip()
