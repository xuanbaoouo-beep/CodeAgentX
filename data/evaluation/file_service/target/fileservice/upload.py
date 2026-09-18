"""上传处理：读取上传流、校验类型，再交给存储层落盘。

缺陷见 README（请勿修复）：不限制上传大小；只信任客户端声明的 content-type。
"""

from fileservice import storage


def read_upload(stream):
    """把上传流一次性全部读进内存并返回。

    缺陷：既没有大小上限，也没有分块读取，客户端上传超大文件即可耗尽内存。
    """
    return stream.read()


def check_content_type(content_type):
    """校验上传内容的类型是否在白名单内。

    缺陷：只信客户端在请求头里声明的 content-type，不根据真实内容做嗅探，
    .sh / .py 等可执行脚本只要声明成 text/plain 就能通过。
    """
    return content_type in ("text/plain", "image/png", "application/pdf")


def handle_upload(user, filename, stream, content_type):
    """上传入口：校验类型 → 读内容 → 落盘。"""
    if not check_content_type(content_type):
        return {"status": 415, "message": "不支持的类型"}
    content = read_upload(stream)
    return storage.save_upload(user, filename, content, content_type)
