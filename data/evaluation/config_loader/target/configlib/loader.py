"""配置加载入口：读取配置源 → 解析值 → 写日志 → 交给调用方。

缺陷见 README（请勿修复）：用 eval() 解析配置值；把整份配置打进日志。
"""

import logging

from configlib import sources

logger = logging.getLogger(__name__)


def parse_value(raw):
    """把配置里的原始字符串解析成 Python 值。

    缺陷：直接对配置内容调用 eval()，只要配置里写
    "__import__('os').system('id')"，加载配置时就会执行任意代码。
    """
    return eval(raw)


def load_config(name):
    """加载指定名字的配置，返回解析后的字典。

    缺陷：加载成功后把整份配置（含 password / secret_key 等敏感项）
    原样写进日志，任何能看到日志的人都能拿到凭据。
    """
    raw = sources.read_source(name)
    config = {key: parse_value(value) for key, value in raw.items()}
    logger.info("已加载配置 %s: %s", name, config)
    return config
