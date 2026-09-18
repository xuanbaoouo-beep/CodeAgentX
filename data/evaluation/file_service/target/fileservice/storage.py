"""存储层：把上传内容落盘，并调用外部工具处理文件。

缺陷见 README（请勿修复）：用 shell=True 拼接用户文件名（命令注入）。
"""

import os
import subprocess

STORAGE_ROOT = "/var/lib/fileservice/data"


def save_upload(user, filename, content, content_type):
    """把上传内容写入 STORAGE_ROOT/<user>/<filename>，返回存储结果。"""
    user_dir = os.path.join(STORAGE_ROOT, user)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, filename)
    with open(path, "wb") as handle:
        handle.write(content)
    return {"status": 201, "path": path, "content_type": content_type}


def read_file(full_path):
    """读取并按字节返回文件内容。"""
    with open(full_path, "rb") as handle:
        return handle.read()


def convert_to_pdf(filename):
    """调用外部转换工具，把上传的文件转成 PDF。

    缺陷：shell=True，且把用户提供的 filename 直接拼进命令字符串，
    形如 "a.txt; rm -rf /" 的文件名会被 shell 当成两条命令依次执行。
    """
    command = "doc2pdf --input %s --output %s.pdf" % (filename, filename)
    return subprocess.run(command, shell=True, capture_output=True)
