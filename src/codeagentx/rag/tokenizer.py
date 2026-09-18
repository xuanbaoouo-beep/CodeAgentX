"""代码/自然语言分词（供词法检索与离线 embedding 共用）。

为什么要自己写：
- 代码里常见 ``get_user_name`` / ``getUserName`` 这类标识符，整词匹配会漏。
  拆成 ``get`` / ``user`` / ``name`` 后，自然语言查询"用户名怎么取"才可能命中。
- 中文无法按空格切分，这里用**单字 + 相邻 bigram**：
  "用户登录" -> 用户 / 户登 / 登录，能匹配到注释"处理用户登录"。
"""

from __future__ import annotations

import re

#: 粗切：英文单词/标识符、数字、单个汉字
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[\u4e00-\u9fff]")
#: 驼峰边界：`getUserName` -> `get_User_Name`；`HTTPResponse` -> `HTTP_Response`
_CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

#: 词法检索中区分度极低的词（代码关键字 + 英文虚词）
CODE_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "assert",
        "async",
        "at",
        "await",
        "be",
        "been",
        "break",
        "but",
        "by",
        "class",
        "cls",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "false",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "it",
        "its",
        "lambda",
        "none",
        "nonlocal",
        "not",
        "of",
        "on",
        "or",
        "pass",
        "raise",
        "return",
        "self",
        "than",
        "that",
        "the",
        "then",
        "these",
        "this",
        "those",
        "to",
        "true",
        "try",
        "was",
        "were",
        "while",
        "with",
        "yield",
    }
)


def tokenize_code(
    text: str,
    *,
    drop_stopwords: bool = True,
    cjk_bigrams: bool = True,
) -> list[str]:
    """把代码或自然语言切成用于词法匹配的 token（保序、可重复）。

    Args:
        text: 待切分文本。
        drop_stopwords: 是否过滤 :data:`CODE_STOPWORDS` 中的低区分度词。
        cjk_bigrams: 是否对中文补充相邻二字组合（提升中文查询召回）。
    """
    tokens: list[str] = []
    cjk_buffer: list[str] = []

    for match in _TOKEN_PATTERN.finditer(text or ""):
        value = match.group(0)
        if _CJK_PATTERN.fullmatch(value):
            cjk_buffer.append(value)
            continue
        _flush_cjk(cjk_buffer, tokens, bigrams=cjk_bigrams)
        _append_identifier(value, tokens, drop_stopwords=drop_stopwords)

    _flush_cjk(cjk_buffer, tokens, bigrams=cjk_bigrams)
    return tokens


def _flush_cjk(buffer: list[str], tokens: list[str], *, bigrams: bool) -> None:
    if not buffer:
        return
    tokens.extend(buffer)
    if bigrams:
        tokens.extend(buffer[index] + buffer[index + 1] for index in range(len(buffer) - 1))
    buffer.clear()


def _append_identifier(value: str, tokens: list[str], *, drop_stopwords: bool) -> None:
    lowered = value.lower()
    pieces = _split_identifier(value)
    # 整体标识符额外保留一份，便于用户直接搜索完整名字；
    # 若拆分结果就是它本身（`foo` -> ['foo']），则不要重复添加。
    if pieces != [lowered] and (not drop_stopwords or lowered not in CODE_STOPWORDS):
        tokens.append(lowered)
    for piece in pieces:
        if piece and (not drop_stopwords or piece not in CODE_STOPWORDS):
            tokens.append(piece)


def _split_identifier(value: str) -> list[str]:
    """把标识符拆成小写子词：``getUserName`` -> ``['get', 'user', 'name']``。"""
    expanded = _CAMEL_BOUNDARY_PATTERN.sub("_", value)
    return [piece.lower() for piece in expanded.split("_") if piece]
