"""索引编排测试：批处理、重置、维度校验、统计，以及"索引后能检索到"的闭环。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from codeagentx.config import Config
from codeagentx.core.exceptions import RAGError
from codeagentx.rag.chunker import CodeChunk, chunk_python_source
from codeagentx.rag.embedder import DEFAULT_HASH_DIM, BaseEmbedder, HashEmbedder
from codeagentx.rag.indexer import (
    IndexStats,
    RepositoryIndexer,
    build_indexer,
    build_vector_store,
)
from codeagentx.rag.vector_store import InMemoryVectorStore, QdrantVectorStore

SAMPLE = '''"""用户服务。"""

import hashlib


def login(user, password):
    """校验用户登录。"""
    return hashlib.sha256(password.encode()).hexdigest() == user.salt


def logout(user):
    return None
'''

README = "# 示例仓库\n\n这是用于测试的说明文档。\n"


class _CountingEmbedder(BaseEmbedder):
    """包一层 HashEmbedder，用于记录批量大小并制造异常返回。"""

    def __init__(self, *, dim: int = DEFAULT_HASH_DIM, drop_last: bool = False, extra_dim: int = 0) -> None:
        super().__init__(dim=dim, model_id="counting")
        self._inner = HashEmbedder(dim=dim)
        self.batches: list[int] = []
        self.drop_last = drop_last
        self.extra_dim = extra_dim

    @property
    def is_semantic(self) -> bool:
        return self._inner.is_semantic

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        vectors = self._inner.embed_documents(texts)
        if self.drop_last and vectors:
            vectors = vectors[:-1]
        if self.extra_dim:
            vectors = [vector + [0.0] * self.extra_dim for vector in vectors]
        return vectors


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "service.py").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    return tmp_path


def _chunks() -> list[CodeChunk]:
    return chunk_python_source(SAMPLE, "pkg/service.py")


def _indexer(**kwargs) -> RepositoryIndexer:
    embedder = kwargs.pop("embedder", None) or _CountingEmbedder()
    return RepositoryIndexer(
        embedder=embedder, store=InMemoryVectorStore(dim=embedder.dim), **kwargs
    )


class TestIndexStats:
    def test_as_dict_rounds_duration(self) -> None:
        stats = IndexStats(backend="InMemoryVectorStore", dim=8, semantic=False, files=2, chunks=7)
        assert stats.as_dict() == {
            "backend": "InMemoryVectorStore",
            "dim": 8,
            "semantic": False,
            "files": 2,
            "chunks": 7,
            "duration": 0.0,
        }


class TestIndexChunks:
    def test_invalid_batch_size_is_rejected(self) -> None:
        embedder = _CountingEmbedder()
        with pytest.raises(ValueError, match="batch_size"):
            RepositoryIndexer(embedder=embedder, store=InMemoryVectorStore(dim=embedder.dim), batch_size=0)

    def test_all_chunks_are_written(self) -> None:
        indexer = _indexer()
        chunks = _chunks()
        indexer.index_chunks(chunks)
        assert indexer.store.count() == len(chunks)

    def test_vector_id_is_chunk_id(self) -> None:
        indexer = _indexer()
        first = _chunks()[0]
        indexer.index_chunks([first])
        assert indexer.store.get(first.chunk_id) is not None

    def test_stored_text_is_code_body_and_metadata_keeps_position(self) -> None:
        indexer = _indexer()
        indexer.index_chunks([chunk for chunk in _chunks() if chunk.name == "login"])
        hit = indexer.store.search(_CountingEmbedder().embed_query("login password"), limit=1)[0]
        assert "def login(user, password)" in hit.text
        assert hit.metadata["chunk_id"] == hit.vector_id
        assert hit.metadata["path"] == "pkg/service.py"
        assert hit.metadata["name"] == "login"
        assert hit.metadata["kind"] == "function"

    def test_chunks_are_embedded_in_batches(self) -> None:
        embedder = _CountingEmbedder()
        indexer = RepositoryIndexer(embedder=embedder, store=InMemoryVectorStore(dim=embedder.dim), batch_size=2)
        chunks = _chunks()
        indexer.index_chunks(chunks)
        assert embedder.batches == [2, 1]
        assert sum(embedder.batches) == len(chunks)

    def test_reset_clears_previous_content(self) -> None:
        indexer = _indexer()
        indexer.index_chunks(_chunks())
        before = indexer.store.count()
        indexer.index_chunks(_chunks()[:1], reset=True)
        assert indexer.store.count() == 1 < before

    def test_reindexing_is_idempotent(self) -> None:
        """chunk_id 是确定性的，重复索引只会覆盖，不会产生重复记录。"""
        indexer = _indexer()
        chunks = _chunks()
        indexer.index_chunks(chunks)
        indexer.index_chunks(chunks, reset=False)
        assert indexer.store.count() == len(chunks)

    def test_empty_chunk_list_is_noop(self) -> None:
        indexer = _indexer()
        stats = indexer.index_chunks([])
        assert stats.chunks == 0
        assert indexer.store.count() == 0

    def test_stats_report_files_and_chunks(self) -> None:
        indexer = _indexer()
        stats = indexer.index_chunks(_chunks())
        assert stats.files == 1
        assert stats.chunks == len(_chunks())
        assert stats.dim == DEFAULT_HASH_DIM
        assert stats.semantic is False
        assert stats.backend == "InMemoryVectorStore"

    def test_dimension_mismatch_gives_actionable_error(self) -> None:
        embedder = _CountingEmbedder(extra_dim=3)
        indexer = RepositoryIndexer(embedder=embedder, store=InMemoryVectorStore(dim=embedder.dim))
        with pytest.raises(RAGError, match="EMBEDDING_DIM"):
            indexer.index_chunks(_chunks())
        assert indexer.store.count() == 0  # 校验在写入之前

    def test_embedding_count_mismatch_is_rejected(self) -> None:
        embedder = _CountingEmbedder(drop_last=True)
        indexer = RepositoryIndexer(embedder=embedder, store=InMemoryVectorStore(dim=embedder.dim))
        with pytest.raises(RAGError, match="条数不匹配"):
            indexer.index_chunks(_chunks())

    def test_describe_exposes_pipeline_shape(self) -> None:
        described = _indexer().describe()
        assert described["backend"] == "InMemoryVectorStore"
        assert described["store_dim"] == DEFAULT_HASH_DIM
        assert described["embedder"]["model_id"] == "counting"


class TestIndexRepository:
    def test_indexes_python_and_markdown_files(self, sample_repo) -> None:
        indexer = _indexer()
        stats = indexer.index_repository(sample_repo)
        assert stats.files == 2
        assert stats.chunks >= 3
        assert indexer.store.count() == stats.chunks

    def test_reset_removes_chunks_of_deleted_files(self, sample_repo) -> None:
        """reset 的意义：文件被删掉后，旧块不能残留在向量库里。"""
        indexer = _indexer()
        before = indexer.index_repository(sample_repo).chunks
        extra = sample_repo / "pkg" / "extra.py"
        extra.write_text("def extra():\n    return 1\n", encoding="utf-8")
        assert indexer.index_repository(sample_repo).chunks > before

        extra.unlink()
        assert indexer.index_repository(sample_repo).chunks == before
        assert indexer.store.count() == before

    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="不是目录"):
            _indexer().index_repository(tmp_path / "nope")

    def test_indexed_repository_is_searchable(self, sample_repo) -> None:
        """W4 验收的核心路径：索引完就能按语义/词法相似度找回相关代码。"""
        embedder = _CountingEmbedder()
        indexer = RepositoryIndexer(embedder=embedder, store=InMemoryVectorStore(dim=embedder.dim))
        indexer.index_repository(sample_repo)
        hits = indexer.store.search(embedder.embed_query("login password hashlib"), limit=3)
        assert hits[0].metadata["name"] == "login"
        assert hits[0].location.startswith("pkg/service.py:")

    def test_extra_chunk_options_are_forwarded(self, sample_repo) -> None:
        indexer = _indexer()
        small = indexer.index_repository(sample_repo, max_chunk_lines=2, max_chunk_chars=60)
        default = indexer.index_repository(sample_repo)
        assert small.chunks > default.chunks


class TestBuildVectorStore:
    def test_defaults_to_memory(self) -> None:
        store = build_vector_store(Config(), dim=16)
        assert isinstance(store, InMemoryVectorStore)
        assert store.dim == 16

    def test_qdrant_is_used_when_requested_with_injected_client(self) -> None:
        store = build_vector_store(Config(vector_backend="qdrant"), dim=16, client=object())
        assert isinstance(store, QdrantVectorStore)
        assert store.dim == 16

    def test_auto_falls_back_to_memory_without_qdrant_client(self) -> None:
        store = build_vector_store(Config(vector_backend="auto"), dim=16)
        if store.__class__ is QdrantVectorStore:  # 环境里确实装了 qdrant-client
            pytest.skip("本机已安装 qdrant-client，auto 会选 Qdrant")
        assert isinstance(store, InMemoryVectorStore)

    def test_explicit_backend_overrides_config(self) -> None:
        store = build_vector_store(Config(vector_backend="memory"), dim=8, backend="qdrant", client=object())
        assert isinstance(store, QdrantVectorStore)

    def test_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="未知的向量库后端"):
            build_vector_store(Config(), dim=8, backend="redis")

    def test_invalid_config_value_is_rejected_at_load_time(self) -> None:
        with pytest.raises(ValidationError, match="vector_backend"):
            Config(vector_backend="redis")


class TestBuildIndexer:
    def test_pipeline_is_wired_with_matching_dimensions(self) -> None:
        indexer = build_indexer(Config(embedding_api_key="", llm_api_key=""))
        assert indexer.embedder.degraded is True
        assert indexer.store.dim == indexer.embedder.dim == DEFAULT_HASH_DIM

    def test_injected_components_are_respected(self) -> None:
        embedder = _CountingEmbedder(dim=64)
        store = InMemoryVectorStore(dim=64)
        indexer = build_indexer(Config(), embedder=embedder, store=store, batch_size=4)
        assert indexer.embedder is embedder
        assert indexer.store is store
        assert indexer.batch_size == 4

    def test_no_fallback_without_key_raises(self) -> None:
        with pytest.raises(RAGError, match="EMBEDDING_API_KEY"):
            build_indexer(Config(embedding_api_key="", llm_api_key=""), allow_fallback=False)
