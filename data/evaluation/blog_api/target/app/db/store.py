"""数据访问层：用户表与帖子表的读写。

演示用的内存实现（``_USERS`` / ``_POSTS`` 两张"表"）。
真实项目里这一层会换成数据库驱动，所以保留了 ``build_tag_query``
这种"构造 SQL 语句"的函数，方便替换时对齐接口。

缺陷见 README（请勿修复）：按标签查询时用字符串拼接构造 SQL。
"""

from app.models.post import Post, User

#: 演示用的内存用户表（key: 用户名）
_USERS: dict[str, User] = {}

#: 演示用的内存帖子表（key: 帖子 id）
_POSTS: dict[int, Post] = {}

#: 帖子 id 自增序列
_NEXT_ID = 1


def save_user(user):
    """写入一条用户记录。"""
    _USERS[user.name] = user
    return user


def find_user_by_name(name):
    """按用户名查询用户，查不到返回 None。"""
    return _USERS.get(name)


def next_post_id():
    """分配下一个帖子 id。"""
    global _NEXT_ID
    post_id = _NEXT_ID
    _NEXT_ID += 1
    return post_id


def insert_post(post):
    """写入一条帖子记录。"""
    _POSTS[post.id] = post
    return post


def get_post(post_id):
    """按 id 查询帖子，查不到返回 None。"""
    return _POSTS.get(post_id)


def list_posts():
    """返回全部帖子（按 id 升序）。"""
    return sorted(_POSTS.values(), key=lambda item: item.id)


def delete_post(post_id):
    """按 id 删除帖子，返回是否真的删掉了一条。"""
    return _POSTS.pop(post_id, None) is not None


def build_tag_query(tag):
    """构造"按标签查询帖子"的 SQL 语句。

    缺陷：直接用字符串拼接把 tag 拼进 SQL，tag 可能直接来自请求参数，
    存在 SQL 注入风险（应改成参数化查询，例如 WHERE tags LIKE ?）。
    """
    return f"SELECT id, title, author FROM posts WHERE tags LIKE '%{tag}%'"


def filter_posts_by_tag(tag):
    """按标签筛选帖子：真实项目会把 ``build_tag_query`` 的结果交给数据库执行。"""
    build_tag_query(tag)  # 真实实现：拿这条 SQL 去执行
    return [post for post in list_posts() if tag in post.tags]
