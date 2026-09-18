"""BM25 测试：打分行为、IDF 区分度、过滤、边界条件。"""

from __future__ import annotations

import pytest

from codeagentx.rag.bm25 import BM25Index
from codeagentx.rag.vector_store import StoredDocument


def _doc(doc_id: str, text: str, **metadata: object) -> StoredDocument:
    return StoredDocument(vector_id=doc_id, text=text, metadata=dict(metadata))


CORPUS = [
    _doc("login", "def login(user, password):\n    return check_password(user, password)", path="auth.py"),
    _doc("logout", "def logout(user):\n    return clear_session(user)", path="auth.py"),
    _doc("matrix", "def multiply(left, right):\n    return left @ right", path="math.py"),
]


def _index(documents=None, **kwargs) -> BM25Index:
    return BM25Index(CORPUS if documents is None else documents, **kwargs)


class TestConstruction:
    def test_invalid_k1_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="k1"):
            _index(k1=0)

    def test_invalid_b_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="b 必须"):
            _index(b=1.5)

    def test_size_and_stats(self) -> None:
        index = _index()
        assert index.size == 3
        stats = index.stats.as_dict()
        assert stats["documents"] == 3
        assert stats["tokens"] > 0
        assert stats["avg_length"] > 0
        assert stats["vocabulary"] > 0

    def test_empty_corpus_has_zero_length(self) -> None:
        index = _index([])
        assert index.size == 0
        assert index.stats.avg_length == 0.0
        assert index.search("login") == []


class TestSearch:
    def test_exact_identifier_ranks_first(self) -> None:
        hits = _index().search("login password", limit=3)
        assert hits[0].vector_id == "login"

    def test_unrelated_term_returns_nothing(self) -> None:
        assert _index().search("blockchain consensus") == []

    def test_hit_carries_text_and_metadata(self) -> None:
        hit = _index().search("login", limit=1)[0]
        assert "def login(user, password)" in hit.text
        assert hit.metadata == {"path": "auth.py"}
        assert hit.score > 0

    def test_rare_term_outweighs_common_term(self) -> None:
        documents = [
            _doc("a", "def handler(request): pass"),
            _doc("b", "def handler(other): pass"),
            _doc("c", "def handler(third): pass"),
            _doc("d", "def rare_marker(handler): pass"),
        ]
        hits = _index(documents).search("handler rare_marker", limit=4)
        assert hits[0].vector_id == "d"

    def test_scores_are_sorted_descending(self) -> None:
        hits = _index().search("user password session login logout", limit=3)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_limit_is_respected(self) -> None:
        assert len(_index().search("user", limit=1)) == 1

    def test_non_positive_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            _index().search("login", limit=0)

    def test_stopword_only_query_returns_nothing(self) -> None:
        assert _index().search("def return") == []

    def test_empty_query_returns_nothing(self) -> None:
        assert _index().search("") == []

    def test_query_terms_are_deduplicated(self) -> None:
        index = _index()
        assert index.search("login login login", limit=1)[0].score == pytest.approx(
            index.search("login", limit=1)[0].score
        )

    def test_metadata_filters_are_applied(self) -> None:
        documents = [
            _doc("py", "def login(user): pass", language="python"),
            _doc("md", "login 说明文档", language="markdown"),
        ]
        hits = _index(documents).search("login", limit=5, filters={"language": "python"})
        assert [hit.vector_id for hit in hits] == ["py"]

    def test_limit_counts_only_filtered_hits(self) -> None:
        documents = [
            _doc("md", "login 说明文档", language="markdown"),
            _doc("py", "def login(user): pass", language="python"),
        ]
        hits = _index(documents).search("login", limit=1, filters={"language": "python"})
        assert [hit.vector_id for hit in hits] == ["py"]

    def test_chinese_query_matches_chinese_document(self) -> None:
        documents = [
            _doc("cn", "处理用户登录的服务", path="cn.py"),
            _doc("en", "compute matrix product", path="math.py"),
        ]
        hits = _index(documents).search("用户登录逻辑", limit=2)
        assert [hit.vector_id for hit in hits] == ["cn"]

    def test_results_are_deterministic_for_equal_scores(self) -> None:
        documents = [_doc("z", "def login(): pass"), _doc("a", "def login(): pass")]
        hits = _index(documents).search("login", limit=2)
        assert [hit.vector_id for hit in hits] == ["a", "z"]

    def test_longer_document_scores_lower_all_else_equal(self) -> None:
        documents = [
            _doc("short", "login"),
            _doc("long", "login " + "padding " * 50),
        ]
        hits = _index(documents).search("login", limit=2)
        assert hits[0].vector_id == "short"
