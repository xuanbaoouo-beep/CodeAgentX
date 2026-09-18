"""检索器测试：RRF 融合、双路行为、语料重建、端到端可检索性。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from codeagentx.rag.embedder import BaseEmbedder, HashEmbedder
from codeagentx.rag.indexer import RepositoryIndexer
from codeagentx.rag.retriever import HybridRetriever, RetrievedChunk
from codeagentx.rag.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    SearchHit,
    StoredDocument,
    VectorRecord,
)

SAMPLE = '''"""用户服务。"""

import hashlib


def login(user, password):
    """校验用户登录。"""
    return hashlib.sha256(password.encode()).hexdigest() == user.salt


def render_matrix(rows):
    return [[0 for _ in row] for row in rows]
'''


class _StubEmbedder(BaseEmbedder):
    """固定向量，便于精确控制"语义侧"的排序。"""

    def __init__(self, *, dim: int = 4) -> None:
        super().__init__(dim=dim, model_id="stub")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _ScriptedStore(BaseVectorStore):
    """``search`` 返回脚本化的名次，用来精确验证 RRF 融合逻辑。"""

    def __init__(self, *, semantic_order: Sequence[str], documents: Sequence[StoredDocument]) -> None:
        super().__init__(dim=4)
        self.semantic_order = list(semantic_order)
        self.documents = {document.vector_id: document for document in documents}

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        hits = [
            SearchHit(
                vector_id=vector_id,
                score=1.0 - rank * 0.1,
                text=self.documents[vector_id].text,
                metadata=self.documents[vector_id].metadata,
            )
            for rank, vector_id in enumerate(self.semantic_order)
        ]
        return hits[:limit]

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        raise NotImplementedError

    def fetch_all(self, *, limit: int | None = None) -> list[StoredDocument]:
        documents = [self.documents[key] for key in sorted(self.documents)]
        return documents if limit is None else documents[:limit]

    def delete(self, vector_ids: Sequence[str]) -> int:
        raise NotImplementedError

    def count(self) -> int:
        return len(self.documents)

    def clear(self) -> None:
        raise NotImplementedError


def _document(vector_id: str, text: str, **metadata: object) -> StoredDocument:
    return StoredDocument(vector_id=vector_id, text=text, metadata=dict(metadata))


class TestRetrievedChunk:
    def test_location_and_symbol(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c", path="pkg/a.py", content="pass", start_line=3, end_line=9,
            kind="method", name="login", parent="Service",
        )
        assert chunk.location == "pkg/a.py:3-9"
        assert chunk.symbol == "Service.login"

    def test_symbol_is_empty_for_module_blocks(self) -> None:
        assert RetrievedChunk(chunk_id="c", path="a.py", content="pass").symbol == ""

    def test_to_text_contains_position_header(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c", path="pkg/a.py", content="def login(): pass",
            start_line=1, end_line=1, kind="function", name="login",
        )
        assert chunk.to_text() == "# pkg/a.py:1-1 function login\ndef login(): pass"

    def test_to_dict_rounds_scores(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c", path="a.py", content="pass", score=0.032_258_06,
            sources=("semantic",), semantic_score=0.9, lexical_score=None,
        )
        payload = chunk.to_dict()
        assert payload["score"] == 0.032258
        assert payload["sources"] == ["semantic"]
        assert payload["lexical_score"] is None
        # 结构化结果要能直接拿到文件与行号，调用方不必去解析 location 字符串
        assert payload["path"] == "a.py"
        assert payload["location"] == "a.py:0-0"

    def test_from_hit_maps_metadata(self) -> None:
        hit = SearchHit(
            vector_id="c", score=0.5, text="body",
            metadata={"path": "a.py", "start_line": 2, "end_line": 4, "kind": "function", "name": "f"},
        )
        chunk = RetrievedChunk.from_hit(hit, score=0.1, sources=["semantic"], semantic_score=0.5)
        assert (chunk.path, chunk.start_line, chunk.end_line, chunk.kind, chunk.name) == (
            "a.py", 2, 4, "function", "f"
        )
        assert chunk.content == "body"
        assert chunk.sources == ("semantic",)

    def test_from_hit_tolerates_missing_metadata(self) -> None:
        chunk = RetrievedChunk.from_hit(
            SearchHit(vector_id="c", score=0.5, text="body"), score=0.1, sources=["lexical"]
        )
        assert chunk.path == ""
        assert chunk.start_line == 0
        assert chunk.lexical_score is None


class TestHybridRetrieverValidation:
    def test_invalid_candidates_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="candidates"):
            HybridRetriever(embedder=_StubEmbedder(), store=InMemoryVectorStore(dim=4), candidates=0)

    def test_invalid_rrf_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="rrf_k"):
            HybridRetriever(embedder=_StubEmbedder(), store=InMemoryVectorStore(dim=4), rrf_k=0)

    def test_non_positive_top_k_is_rejected(self) -> None:
        retriever = HybridRetriever(embedder=_StubEmbedder(), store=InMemoryVectorStore(dim=4))
        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve("login", top_k=0)


class TestEmptyStore:
    def test_retrieve_returns_nothing(self) -> None:
        retriever = HybridRetriever(embedder=_StubEmbedder(), store=InMemoryVectorStore(dim=4))
        assert retriever.retrieve("login") == []

    def test_semantic_search_skips_embedding_when_empty(self) -> None:
        class _Exploding(_StubEmbedder):
            def embed_query(self, text: str) -> list[float]:  # pragma: no cover - 不应被调用
                raise AssertionError("空库时不应调用向量化")

        retriever = HybridRetriever(embedder=_Exploding(), store=InMemoryVectorStore(dim=4))
        assert retriever.semantic_search("login") == []


class TestRRFFusion:
    """融合逻辑用脚本化向后端验证，不受具体向量/分词实现影响。"""

    DOCUMENTS = [
        _document("both", "login handler", path="a.py"),
        _document("semantic_only", "cache eviction", path="b.py"),
        _document("lexical_only", "login", path="c.py"),
    ]

    def _retriever(self, *, semantic_order=("both", "semantic_only"), **kwargs) -> HybridRetriever:
        store = _ScriptedStore(semantic_order=semantic_order, documents=self.DOCUMENTS)
        return HybridRetriever(embedder=_StubEmbedder(), store=store, **kwargs)

    def test_document_hit_by_both_paths_ranks_first(self) -> None:
        results = self._retriever().retrieve("login", top_k=3)
        assert results[0].chunk_id == "both"
        assert set(results[0].sources) == {"semantic", "lexical"}

    def test_single_path_documents_are_kept(self) -> None:
        results = self._retriever().retrieve("login", top_k=3)
        assert {chunk.chunk_id for chunk in results} == {"both", "semantic_only", "lexical_only"}

    def test_rrf_score_is_reciprocal_rank(self) -> None:
        """只被一路召回时，融合分就是该路名次的倒数（k=60，第 2 名 → 1/62）。"""
        results = {chunk.chunk_id: chunk for chunk in self._retriever(rrf_k=60).retrieve("login", top_k=3)}
        assert results["semantic_only"].score == pytest.approx(1 / 62)
        assert results["both"].score > results["lexical_only"].score

    def test_per_source_scores_are_recorded(self) -> None:
        results = {chunk.chunk_id: chunk for chunk in self._retriever().retrieve("login", top_k=3)}
        assert results["both"].semantic_score == pytest.approx(1.0)
        assert results["both"].lexical_score is not None
        assert results["semantic_only"].lexical_score is None

    def test_lexical_path_can_be_disabled(self) -> None:
        results = self._retriever(enable_lexical=False).retrieve("login", top_k=3)
        assert [chunk.chunk_id for chunk in results] == ["both", "semantic_only"]
        assert all(chunk.sources == ("semantic",) for chunk in results)

    def test_top_k_larger_than_candidates_still_returns_hits(self) -> None:
        results = self._retriever(candidates=1).retrieve("login", top_k=3)
        assert len(results) == 3

    def test_top_k_limits_result_size(self) -> None:
        assert len(self._retriever().retrieve("login", top_k=1)) == 1

    def test_results_are_sorted_by_fused_score(self) -> None:
        scores = [chunk.score for chunk in self._retriever().retrieve("login", top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_describe_reports_corpus_after_retrieval(self) -> None:
        retriever = self._retriever()
        assert retriever.describe()["lexical_corpus"] is None  # 还没检索，未建语料
        retriever.retrieve("login")
        described = retriever.describe()
        assert described["lexical_enabled"] is True
        assert described["lexical_corpus"]["documents"] == 3
        assert described["stored_chunks"] == 3

    def test_refresh_drops_cached_corpus(self) -> None:
        retriever = self._retriever()
        retriever.retrieve("login")
        retriever.refresh()
        assert retriever.describe()["lexical_corpus"] is None


@pytest.fixture
def indexed(tmp_path):
    """把样例仓库真正索引一遍，返回 (retriever, embedder)。"""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "service.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "README.md").write_text("# 说明\n\n用户服务模块。\n", encoding="utf-8")
    embedder = HashEmbedder(dim=512)
    store = InMemoryVectorStore(dim=embedder.dim)
    RepositoryIndexer(embedder=embedder, store=store).index_repository(tmp_path)
    return HybridRetriever(embedder=embedder, store=store), embedder, store


class TestIndexedRepository:
    def test_finds_the_login_function(self, indexed) -> None:
        retriever, _, _ = indexed
        results = retriever.retrieve("login password hashlib", top_k=3)
        assert results[0].name == "login"
        assert results[0].path == "pkg/service.py"
        assert "def login(user, password)" in results[0].content

    def test_results_carry_position_metadata(self, indexed) -> None:
        retriever, _, _ = indexed
        results = retriever.retrieve("matrix rows", top_k=3)
        assert results
        for chunk in results:
            assert chunk.start_line <= chunk.end_line
            assert chunk.location.startswith(chunk.path)
        assert any(chunk.path == "pkg/service.py" and chunk.kind == "function" for chunk in results)

    def test_metadata_filters_restrict_results(self, indexed) -> None:
        retriever, _, _ = indexed
        results = retriever.retrieve("用户服务说明", top_k=5, filters={"language": "markdown"})
        assert [chunk.path for chunk in results] == ["README.md"]

    def test_corpus_is_rebuilt_when_store_grows(self, indexed) -> None:
        retriever, _, store = indexed
        retriever.retrieve("login")
        store.upsert([
            VectorRecord(
                vector_id="pkg/extra.py::function:uniquely_named::1-2#0",
                vector=HashEmbedder(dim=512).embed_query("def uniquely_named_token(): pass"),
                text="def uniquely_named_token():\n    pass",
                metadata={
                    "chunk_id": "pkg/extra.py::function:uniquely_named::1-2#0",
                    "path": "pkg/extra.py",
                    "start_line": 1,
                    "end_line": 2,
                    "kind": "function",
                    "name": "uniquely_named",
                    "parent": "",
                    "part": 0,
                    "language": "python",
                },
            )
        ])
        results = retriever.retrieve("uniquely_named_token", top_k=1)
        assert results[0].path == "pkg/extra.py"

    def test_retrieval_is_deterministic(self, indexed) -> None:
        retriever, _, _ = indexed
        first = [chunk.chunk_id for chunk in retriever.retrieve("login user", top_k=3)]
        second = [chunk.chunk_id for chunk in retriever.retrieve("login user", top_k=3)]
        assert first == second

    def test_symbol_names_are_available_to_callers(self, indexed) -> None:
        """检索结果要能告诉上层"这是哪个符号"，Agent 才能据此定位与引用。"""
        retriever, _, _ = indexed
        results = retriever.retrieve("render_matrix rows", top_k=1)
        assert results[0].symbol == "render_matrix"
        assert results[0].kind == "function"
