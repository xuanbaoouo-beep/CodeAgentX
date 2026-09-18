"""临时文件工具：为下载/预览生成临时副本。

缺陷见 README（请勿修复）：临时文件用完不清理，文件句柄也不关闭。
"""

import tempfile


def make_temp_copy(content):
    """把内容写进临时文件，返回临时文件路径。

    缺陷：mkstemp 返回的 fd 从不 close()，临时文件用完也不删除、不注册清理，
    每调用一次就泄漏一个文件描述符，并在磁盘上残留一份文件。
    """
    fd, path = tempfile.mkstemp(prefix="fileservice-")
    with open(path, "wb") as handle:
        handle.write(content)
    return path


def preview_upload(content):
    """生成预览用的临时文件，交给调用方读取。"""
    return make_temp_copy(content)
