"""混合检索：语义召回 + BM25 词法召回 + RRF 融合。

单靠向量检索会漏掉"精确标识符"（``parse_config`` 被语义稀释），
单靠 BM25 又不懂同义与中文表达（"登录逻辑" vs ``authenticate``）。
两路各取 ``candidates`` 条候选，用 **RRF（Reciprocal Rank Fusion）** 融合::

    score(d) = Σ_source  1 / (rrf_k + rank_source(d))

RRF 只看**名次**不看原始分数，因此不需要在余弦相似度与 BM25 分数之间做归一化，
是工程上最省心的融合方式（``rrf_k`` 默认 60，即经典的 Cormack 取值）。

检索结果 :class:`RetrievedChunk` 同时保留两路各自的原始分数，
便于评估（W8）与消融（W9）分析"到底是哪一路召回的"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from codeagentx.core.logger import get_logger, log_event
from codeagentx.rag.bm25 import BM25Index
from codeagentx.rag.chunker import format_chunk_header
from codeagentx.rag.embedder import BaseEmbedder
from codeagentx.rag.vector_store import BaseVectorStore, SearchHit, StoredDocument

logger = get_logger("rag.retriever")

#: 默认返回条数
DEFAULT_TOP_K = 5
#: 每一路召回的候选数（要比 top_k 大，给融合留出余地）
DEFAULT_CANDIDATES = 20
#: RRF 平滑参数
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    """检索到的代码片段（已带位置信息，可直接喂给 LLM 或展示）。"""

    chunk_id: str
    path: str
    content: str
    start_line: int = 0
    end_line: int = 0
    kind: str = ""
    name: str = ""
    parent: str = ""
    language: str = "python"
    score: float = 0.0
    sources: tuple[str, ...] = ()
    semantic_score: float | None = None
    lexical_score: float | None = None

    @property
    def location(self) -> str:
        """``path:start-end``。"""
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def symbol(self) -> str:
        """符号名（带类名前缀），如 ``Service.login``。"""
        if not self.name:
            return ""
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def to_text(self) -> str:
        """带位置头的完整文本（喂 LLM 时用它，模型才知道代码在哪）。"""
        header = format_chunk_header(
            self.path, self.start_line, self.end_line, self.kind, self.name, self.parent
        )
        return f"{header}\n{self.content}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "location": self.location,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind,
            "name": self.name,
            "score": round(self.score, 6),
            "sources": list(self.sources),
            "semantic_score": None if self.semantic_score is None else round(self.semantic_score, 6),
            "lexical_score": None if self.lexical_score is None else round(self.lexical_score, 6),
            "content": self.content,
        }

    @classmethod
    def from_hit(
        cls,
        hit: SearchHit,
        *,
        score: float,
        sources: Sequence[str],
        semantic_score: float | None = None,
        lexical_score: float | None = None,
    ) -> RetrievedChunk:
        metadata = hit.metadata
        return cls(
            chunk_id=hit.vector_id,
            path=str(metadata.get("path", "")),
            content=hit.text,
            start_line=int(metadata.get("start_line") or 0),
            end_line=int(metadata.get("end_line") or 0),
            kind=str(metadata.get("kind", "")),
            name=str(metadata.get("name", "")),
            parent=str(metadata.get("parent", "")),
            language=str(metadata.get("language", "")),
            score=score,
            sources=tuple(sources),
            semantic_score=semantic_score,
            lexical_score=lexical_score,
        )


@dataclass
class _Fused:
    """融合过程中的累加器。"""

    hit: SearchHit
    score: float = 0.0
    sources: list[str] = field(default_factory=list)
    semantic_score: float | None = None
    lexical_score: float | None = None


class HybridRetriever:
    """语义 + 词法双路召回，RRF 融合。

    ``embedder`` 与 ``store`` 必须和索引时用的是**同一套**，
    否则向量空间对不上（典型症状：检索结果毫无相关性）。
    """

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        candidates: int = DEFAULT_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
        enable_lexical: bool = True,
    ) -> None:
        if candidates <= 0:
            raise ValueError("candidates 必须为正整数")
        if rrf_k <= 0:
            raise ValueError("rrf_k 必须为正整数")
        self.embedder = embedder
        self.store = store
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.enable_lexical = enable_lexical
        self._lexical: BM25Index | None = None
        self._lexical_count = -1

    # ------------------------------------------------------------ 检索
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """检索与 ``query`` 最相关的代码片段（按融合分降序）。"""
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正整数，收到 {top_k}")
        limit = max(self.candidates, top_k)

        semantic_hits = self.semantic_search(query, limit=limit, filters=filters)
        lexical_hits = (
            self.lexical_search(query, limit=limit, filters=filters)
            if self.enable_lexical
            else []
        )

        fused: dict[str, _Fused] = {}
        for source, hits in (("semantic", semantic_hits), ("lexical", lexical_hits)):
            for rank, hit in enumerate(hits, start=1):
                entry = fused.get(hit.vector_id)
                if entry is None:
                    entry = _Fused(hit=hit)
                    fused[hit.vector_id] = entry
                entry.score += 1.0 / (self.rrf_k + rank)
                entry.sources.append(source)
                if source == "semantic":
                    entry.semantic_score = hit.score
                else:
                    entry.lexical_score = hit.score

        ordered = sorted(fused.values(), key=lambda item: (-item.score, item.hit.vector_id))
        results = [
            RetrievedChunk.from_hit(
                entry.hit,
                score=entry.score,
                sources=entry.sources,
                semantic_score=entry.semantic_score,
                lexical_score=entry.lexical_score,
            )
            for entry in ordered[:top_k]
        ]
        log_event(
            logger,
            "rag.retrieve",
            query=query,
            top_k=top_k,
            semantic=len(semantic_hits),
            lexical=len(lexical_hits),
            returned=len(results),
        )
        return results

    def semantic_search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_CANDIDATES,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """只走向量一路（供消融实验与排障使用）。"""
        if not self.store.count():
            return []
        vector = self.embedder.embed_query(query)
        return self.store.search(vector, limit=limit, filters=filters)

    def lexical_search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_CANDIDATES,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """只走 BM25 一路（供消融实验与排障使用）。"""
        return self._lexical_index().search(query, limit=limit, filters=filters)

    # ------------------------------------------------------------ 词法索引
    def refresh(self) -> None:
        """索引内容变化后强制重建 BM25 语料（``retrieve`` 也会按文档数自动重建）。"""
        self._lexical = None
        self._lexical_count = -1

    def _lexical_index(self) -> BM25Index:
        count = self.store.count()
        if self._lexical is None or self._lexical_count != count:
            # 词法语料必须与 embedding 输入完全一致（都带位置/符号头），
            # 否则两路"看"到的文本不同，融合就失去意义
            documents = [
                StoredDocument(
                    vector_id=document.vector_id,
                    text=_corpus_text(document),
                    metadata=document.metadata,
                )
                for document in self.store.fetch_all()
            ]
            self._lexical = BM25Index(documents)
            self._lexical_count = count
            logger.debug("BM25 语料已重建：%d 篇文档", self._lexical.size)
        return self._lexical

    def describe(self) -> dict[str, Any]:
        corpus = self._lexical.stats.as_dict() if self._lexical is not None else None
        return {
            "embedder": self.embedder.describe(),
            "backend": type(self.store).__name__,
            "stored_chunks": self.store.count(),
            "candidates": self.candidates,
            "rrf_k": self.rrf_k,
            "lexical_enabled": self.enable_lexical,
            "lexical_corpus": corpus,
        }


def _corpus_text(document: StoredDocument) -> str:
    """把入库文档还原成"和 embedding 输入一致"的文本（位置/符号头 + 正文）。"""
    metadata = document.metadata
    header = format_chunk_header(
        str(metadata.get("path", "")),
        int(metadata.get("start_line") or 0),
        int(metadata.get("end_line") or 0),
        str(metadata.get("kind", "")),
        str(metadata.get("name", "")),
        str(metadata.get("parent", "")),
    )
    return f"{header}\n{document.text}"
