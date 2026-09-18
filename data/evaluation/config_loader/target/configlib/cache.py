"""配置缓存：按名字缓存已经加载好的配置。

缺陷见 README（请勿修复）：缓存键只用文件名，不含来源/环境。
"""

import threading

_CACHE = {}
_LOCK = threading.Lock()


def cache_key(filename):
    """计算配置的缓存键。

    缺陷：键只用文件名，不含来源/环境（dev/prod、不同配置中心），
    prod 进程读到 dev 的同名配置后会一直命中错误缓存（多环境串味）。
    """
    return filename


def get_cached(name, loader):
    """按缓存键取配置，未命中则调用 loader 并写入缓存。"""
    key = cache_key(name)
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = loader(name)
        return _CACHE[key]
