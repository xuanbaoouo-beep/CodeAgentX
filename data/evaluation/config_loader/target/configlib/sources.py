"""配置来源读取：从文件系统读取配置文件。

缺陷见 README（请勿修复）：读取失败时静默回退到默认配置。
"""

import json
import os

CONFIG_DIR = "/etc/myapp/conf"

#: 读取失败时使用的兜底配置
FALLBACK_CONFIG = {"port": 8080, "timeout": 30, "debug": False}


def read_source(name):
    """读取 <CONFIG_DIR>/<name>.json，返回原始键值对字典。

    缺陷：文件不存在或 JSON 解析失败时，只静默返回 FALLBACK_CONFIG，
    既不抛异常也不打日志/告警，配置写错会被当成默认值悄悄跑起来。
    """
    path = os.path.join(CONFIG_DIR, "%s.json" % name)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return dict(FALLBACK_CONFIG)
