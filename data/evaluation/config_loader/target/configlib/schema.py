"""配置结构定义与取值。

缺陷见 README（请勿修复）：内置默认管理员口令；配置值不做类型校验。
"""

#: 未显式配置管理员口令时使用的兜底值
DEFAULT_ADMIN_PASSWORD = "admin123"

#: 各配置项期望的类型
EXPECTED_TYPES = {
    "port": int,
    "timeout": int,
    "debug": bool,
    "admin_user": str,
    "admin_password": str,
}


def get_admin_password(config):
    """取管理员口令。

    缺陷：配置里没有 admin_password 时静默使用内置默认口令 "admin123"，
    部署方毫不知情，线上等于留了一个人人皆知的后门。
    """
    return config.get("admin_password", DEFAULT_ADMIN_PASSWORD)


def coerce(config):
    """按 EXPECTED_TYPES 整理配置，返回规整后的字典。

    缺陷：只判断配置项是否存在，从不校验类型、也不做转换，
    port="abc" 这类错误会被原样带进后续代码，报错点远离真正原因。
    """
    result = {}
    for key in EXPECTED_TYPES:
        if key in config:
            result[key] = config[key]
    return result
