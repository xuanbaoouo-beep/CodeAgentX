"""短链存储：内存实现，接口与真实项目对齐（便于换成数据库）。

本文件不含缺陷，只负责短码与目标 URL 的读写和点击计数。
"""

#: 短码 -> 目标 URL
_LINKS: dict[str, str] = {}

#: 短码 -> 点击数
_HITS: dict[str, int] = {}


def save(code, url):
    """保存一条短链。"""
    _LINKS[code] = url
    return code


def get(code):
    """按短码取目标 URL，不存在时返回 None。"""
    return _LINKS.get(code)


def record_hit(code):
    """记录一次点击。"""
    _HITS[code] = _HITS.get(code, 0) + 1


def all_links():
    """返回全部短链的副本。"""
    return dict(_LINKS)


def total_hits():
    """返回全站累计点击数。"""
    return sum(_HITS.values())
