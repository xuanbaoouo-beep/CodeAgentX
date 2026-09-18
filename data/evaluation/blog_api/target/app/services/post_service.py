"""业务层：帖子列表、按标签筛选与删除。

缺陷见 README（请勿修复）：删除帖子不校验归属者（越权删除他人帖子）。
"""

from app.db import store


def list_posts(page=1, page_size=20):
    """按分页返回帖子列表。"""
    posts = store.list_posts()
    start = max(page - 1, 0) * page_size
    return posts[start : start + page_size]


def list_posts_by_tag(tag):
    """按标签筛选帖子。"""
    return store.filter_posts_by_tag(tag)


def delete_post(post_id, requester=None):
    """删除指定帖子，返回是否删除成功。

    requester 是当前请求方传来的用户名，本层负责判断"这个人能不能删这条帖子"。
    缺陷：整段逻辑里没有校验帖子归属者——只要知道 post_id，
    任何调用方都能删掉别人的帖子（越权 / IDOR）。
    """
    post = store.get_post(post_id)
    if post is None:
        return False
    store.delete_post(post_id)
    return True
