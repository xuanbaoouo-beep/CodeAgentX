"""路径解析工具：把用户传入的相对路径映射到存储根目录下。

缺陷见 README（请勿修复）：下载路径未做规范化与前缀校验（路径穿越）。
"""

import os

#: 所有用户文件都应当落在这个目录之下
STORAGE_ROOT = "/var/lib/fileservice/data"


def storage_path(relative_path: str) -> str:
    """把用户提供的相对路径拼到存储根目录下，返回绝对路径。

    缺陷：只用 os.path.join 做字符串拼接，既没有 realpath/resolve 规范化，
    也没有校验拼接结果是否仍位于 STORAGE_ROOT 之内。
    传入 "../../etc/passwd" 即可穿越到存储根目录之外读取任意文件。
    """
    return os.path.join(STORAGE_ROOT, relative_path)
