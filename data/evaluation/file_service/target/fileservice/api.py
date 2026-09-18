"""接口层：用普通函数模拟 HTTP 端点（本机无 Web 框架，做等价简化）。

真实项目里这些函数对应 Flask/FastAPI 的路由处理函数，
此处只保留与缺陷相关的核心逻辑，请求/响应都退化成普通字典。

缺陷见 README（请勿修复）：下载接口不校验文件的归属者。
"""

from fileservice import paths, storage

#: 模拟的当前登录用户（真实实现里来自会话中间件）
CURRENT_USER = "alice"


def download(relative_path):
    """处理 GET /files/<path>：读取并返回文件内容。

    缺陷：只按路径取文件，从不校验该文件是否属于 CURRENT_USER，
    任意登录用户都能下载他人文件（越权访问）。
    """
    full_path = paths.storage_path(relative_path)
    return {"status": 200, "content": storage.read_file(full_path)}


def upload(filename, content, content_type):
    """处理 POST /files：保存上传内容并返回存储结果。"""
    return storage.save_upload(CURRENT_USER, filename, content, content_type)
