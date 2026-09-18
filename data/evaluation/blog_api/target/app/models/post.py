"""帖子与用户的数据模型。

演示用 ``dataclass``，字段与真实项目的表结构对齐（便于换成数据库行对象）。
"""

from dataclasses import dataclass, field


@dataclass
class User:
    """用户：作者身份与角色。"""

    name: str
    password: str  # 演示用的口令字段（真实项目应存哈希值）
    role: str = "reader"


@dataclass
class Post:
    """帖子。"""

    id: int
    title: str
    author: str
    tags: list = field(default_factory=list)
    content: str = ""

    def to_dict(self):
        """转成可 JSON 序列化的字典。"""
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "tags": list(self.tags),
            "content": self.content,
        }
