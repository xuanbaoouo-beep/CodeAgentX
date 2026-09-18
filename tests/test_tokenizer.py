"""分词测试：标识符拆词、中文 bigram、停用词与确定性。"""

from __future__ import annotations

from codeagentx.rag.tokenizer import CODE_STOPWORDS, tokenize_code


class TestIdentifierSplitting:
    def test_snake_case_is_split(self) -> None:
        tokens = tokenize_code("get_user_name")
        assert {"get", "user", "name"} <= set(tokens)

    def test_camel_case_is_split(self) -> None:
        tokens = tokenize_code("getUserName")
        assert {"get", "user", "name"} <= set(tokens)

    def test_upper_abbreviation_is_split(self) -> None:
        assert {"http", "response"} <= set(tokenize_code("HTTPResponse"))

    def test_upper_snake_case_is_split(self) -> None:
        assert {"max", "size"} <= set(tokenize_code("MAX_SIZE"))

    def test_original_identifier_is_kept_for_exact_match(self) -> None:
        # 原标识符保留，便于用户直接搜索完整名字
        assert "get_user_name" in tokenize_code("get_user_name")

    def test_dunder_is_split(self) -> None:
        assert "init" in tokenize_code("__init__")

    def test_digits_are_kept(self) -> None:
        assert tokenize_code("utf8") == ["utf8"]
        assert tokenize_code("42") == ["42"]


class TestStopwords:
    def test_keywords_are_dropped_by_default(self) -> None:
        tokens = tokenize_code("def login(self, user): return user")
        assert "def" not in tokens
        assert "self" not in tokens
        assert "return" not in tokens
        assert "login" in tokens

    def test_stopwords_can_be_kept(self) -> None:
        assert tokenize_code("def login", drop_stopwords=False) == ["def", "login"]

    def test_stopword_set_covers_code_keywords(self) -> None:
        assert {"def", "class", "self", "import", "return"} <= CODE_STOPWORDS


class TestChineseHandling:
    def test_single_characters_are_produced(self) -> None:
        tokens = tokenize_code("登录")
        assert "登" in tokens
        assert "录" in tokens

    def test_bigrams_are_produced(self) -> None:
        assert "登录" in tokenize_code("用户登录")

    def test_bigrams_can_be_disabled(self) -> None:
        assert "登录" not in tokenize_code("用户登录", cjk_bigrams=False)

    def test_chinese_and_ascii_are_both_tokenized(self) -> None:
        tokens = tokenize_code("处理用户登录的 login 函数")
        assert "登录" in tokens
        assert "login" in tokens

    def test_chinese_overlap_between_query_and_doc(self) -> None:
        query = set(tokenize_code("用户登录逻辑在哪"))
        doc = set(tokenize_code("处理用户登录的服务"))
        assert {"用户", "户登", "登录"} <= query & doc


class TestDeterminism:
    def test_same_input_gives_same_output(self) -> None:
        text = "def get_user(id): return db.query(id)"
        assert tokenize_code(text) == tokenize_code(text)

    def test_empty_input_gives_empty_list(self) -> None:
        assert tokenize_code("") == []
        assert tokenize_code(None) == []  # type: ignore[arg-type]

    def test_punctuation_is_ignored(self) -> None:
        assert tokenize_code("foo.bar;baz") == ["foo", "bar", "baz"]
