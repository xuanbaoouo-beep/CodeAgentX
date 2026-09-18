"""索引编排：仓库 → 分块 → 向量化 → 入库。

一次索引做三件事：

1. :func:`~codeagentx.rag.chunker.chunk_repository` 遍历仓库并按 AST 切块；
2. 用 :class:`~codeagentx.rag.embedder.BaseEmbedder` 批量向量化
   （输入是 ``CodeChunk.to_text()``，带"路径 + 符号名"头，让函数名/类名也参与相似度）；
3. 写入 :class:`~codeagentx.rag.vector_store.BaseVectorStore`
   （``chunk_id`` 作向量 id，``CodeChunk.to_metadata()`` 作元数据，检索时据此还原位置）。

为什么把编排单独成一个模块：分块、向量化、存储三者各自可单测，
组合逻辑（批大小、维度校验、重置、统计）集中在这里，
检索器和上层 Agent 只需要一句 ``indexer.index_repository(root)``。
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeagentx.core.exceptions import RAGError
from codeagentx.core.logger import get_logger, log_event
from codeagentx.rag.chunker import CodeChunk, chunk_repository
from codeagentx.rag.embedder import BaseEmbedder, build_embedder
from codeagentx.rag.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorRecord,
)

logger = get_logger("rag.indexer")

#: 单批向量化的块数（越大越快，但单次请求体也越大）
DEFAULT_INDEX_BATCH_SIZE = 32
#: 支持的向量库后端
BACKENDS: tuple[str, ...] = ("memory", "qdrant", "auto")


@dataclass
class IndexStats:
    """一次索引的结果统计。"""

    backend: str
    dim: int
    semantic: bool
    files: int = 0
    chunks: int = 0
    duration: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dim": self.dim,
            "semantic": self.semantic,
            "files": self.files,
            "chunks": self.chunks,
            "duration": round(self.duration, 3),
        }


class RepositoryIndexer:
    """把代码仓库写进向量库。

    ``embedder`` 与 ``store`` 由外部注入（便于测试与替换后端），
    二者维度必须一致：``embedder`` 的声明维度只作参考，
    **以实际返回的向量长度为准**，不一致时抛出可操作的 :class:`RAGError`。
    """

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    # ------------------------------------------------------------ 索引
    def index_repository(
        self,
        root: str | Path,
        *,
        reset: bool = True,
        **chunk_kwargs: Any,
    ) -> IndexStats:
        """切分并索引整个仓库；``reset=True`` 时先清空向量库（避免残留旧文件）。"""
        chunks = chunk_repository(root, **chunk_kwargs)
        logger.info("仓库 %s 切分出 %d 个块，开始向量化", root, len(chunks))
        return self.index_chunks(chunks, reset=reset)

    def index_chunks(self, chunks: Sequence[CodeChunk], *, reset: bool = False) -> IndexStats:
        """索引已切好的块（供增量更新与测试直接调用）。"""
        started = time.perf_counter()
        if reset:
            self.store.clear()

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self.embedder.embed_documents([chunk.to_text() for chunk in batch])
            if len(vectors) != len(batch):
                raise RAGError(
                    f"向量化返回条数不匹配：期望 {len(batch)}，得到 {len(vectors)}"
                )
            self._ensure_dim(vectors[0])
            self.store.upsert(
                [
                    VectorRecord(
                        vector_id=chunk.chunk_id,
                        vector=vector,
                        text=chunk.content,
                        metadata=chunk.to_metadata(),
                    )
                    for chunk, vector in zip(batch, vectors, strict=True)
                ]
            )

        stats = IndexStats(
            backend=type(self.store).__name__,
            dim=self.store.dim,
            semantic=self.embedder.is_semantic,
            files=len({chunk.path for chunk in chunks}),
            chunks=len(chunks),
            duration=time.perf_counter() - started,
        )
        log_event(
            logger,
            "rag.index",
            backend=stats.backend,
            files=stats.files,
            chunks=stats.chunks,
            dim=stats.dim,
            semantic=stats.semantic,
            seconds=round(stats.duration, 3),
        )
        return stats

    def describe(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.describe(),
            "backend": type(self.store).__name__,
            "store_dim": self.store.dim,
            "batch_size": self.batch_size,
        }

    # ------------------------------------------------------------ 内部
    def _ensure_dim(self, vector: Sequence[float]) -> None:
        """用真实向量校验维度——比信任配置里的 ``EMBEDDING_DIM`` 可靠。"""
        if len(vector) == self.store.dim:
            return
        raise RAGError(
            f"向量维度与向量库不一致：模型返回 {len(vector)} 维，向量库为 {self.store.dim} 维",
            detail=(
                "若使用真实 embedding 接口，请把 .env 的 EMBEDDING_DIM "
                f"改成模型实际维度（{len(vector)}）后重建索引"
            ),
        )


# ------------------------------------------------------------------ 工厂
def build_vector_store(
    config: Any,
    *,
    dim: int,
    backend: str | None = None,
    client: Any | None = None,
) -> BaseVectorStore:
    """按配置构造向量库。

    Args:
        backend: ``memory`` / ``qdrant`` / ``auto``；``None`` 时取
            ``config.vector_backend``。``auto`` 表示"装了 qdrant-client 就用 Qdrant"。
        client: 注入的 Qdrant 客户端（测试用，省略则按 URL 连接）。
    """
    choice = (backend or getattr(config, "vector_backend", "memory") or "memory").lower()
    if choice == "auto":
        choice = "qdrant" if _qdrant_installed() else "memory"
        if choice == "memory":
            logger.warning("vector_backend=auto 但未安装 qdrant-client，回退内存向量库")
    if choice == "qdrant":
        return QdrantVectorStore(
            url=getattr(config, "qdrant_url", "http://localhost:6333"),
            collection=getattr(config, "qdrant_collection", "codeagentx_code"),
            api_key=getattr(config, "qdrant_api_key", ""),
            dim=dim,
            client=client,
        )
    if choice != "memory":
        raise ValueError(f"未知的向量库后端：{choice!r}，可选 {list(BACKENDS)}")
    return InMemoryVectorStore(dim=dim)


def build_indexer(
    config: Any,
    *,
    embedder: BaseEmbedder | None = None,
    store: BaseVectorStore | None = None,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
    allow_fallback: bool = True,
) -> RepositoryIndexer:
    """按配置组装一条完整的索引流水线。

    ``store`` 的维度取自 ``embedder``（而不是配置），
    保证"降级为 HashEmbedder 时维度自动变 512"这类情况不会撞上维度不匹配。
    """
    embedder = embedder or build_embedder(config, allow_fallback=allow_fallback)
    store = store or build_vector_store(config, dim=embedder.dim)
    return RepositoryIndexer(embedder=embedder, store=store, batch_size=batch_size)


def _qdrant_installed() -> bool:
    return importlib.util.find_spec("qdrant_client") is not None
