"""BM25 词法检索。

为什么需要它：向量检索靠"语义相近"，但在代码场景里**精确标识符**（``parse_config``、
``_login``）经常被向量模型稀释成"差不多的意思"，导致真正该命中的函数排不进前几名。
BM25 只看词频与逆文档频率，正好补上这块：**两路召回 → RRF 融合**（见 :mod:`codeagentx.rag.retriever`）。

分词复用 :func:`~codeagentx.rag.tokenizer.tokenize_code`（标识符拆词 + 中文 bigram），
与离线向量化用的是同一套规则，保证两条召回路径看到的"词"是一致的。

公式（Okapi BM25）::

    score(q, d) = Σ_t IDF(t) · tf·(k1+1) / (tf + k1·(1 - b + b·|d|/avgdl))
    IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from codeagentx.rag.tokenizer import tokenize_code
from codeagentx.rag.vector_store import SearchHit, StoredDocument, matches_metadata

#: 词频饱和参数（越大越"奖励重复出现"，1.2~2.0 是常见取值）
DEFAULT_K1 = 1.5
#: 文档长度归一化强度（0=不归一，1=完全归一）
DEFAULT_B = 0.75


@dataclass
class BM25Stats:
    """语料规模统计，便于排查"为什么什么都搜不到"。"""

    documents: int = 0
    tokens: int = 0
    avg_length: float = 0.0
    vocabulary: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "tokens": self.tokens,
            "avg_length": round(self.avg_length, 2),
            "vocabulary": self.vocabulary,
        }


@dataclass(frozen=True)
class _Document:
    """已分词的一篇文档。"""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    term_freq: dict[str, int] = field(default_factory=dict)
    length: int = 0


class BM25Index:
    """内存 BM25 索引，面向"单个仓库几千个代码块"的规模。

    构建后会保留全部词频统计，因此**索引内容变化时需要重建**
    （``HybridRetriever`` 会在文档数量变化时自动重建）。
    """

    def __init__(
        self,
        documents: Sequence[StoredDocument],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 必须为正数")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b 必须落在 [0, 1] 区间")
        self.k1 = k1
        self.b = b
        self._documents: list[_Document] = []
        for document in documents:
            tokens = tokenize_code(document.text)
            self._documents.append(
                _Document(
                    doc_id=document.vector_id,
                    text=document.text,
                    metadata=document.metadata,
                    term_freq=dict(Counter(tokens)),
                    length=len(tokens),
                )
            )
        self._avg_length = (
            sum(document.length for document in self._documents) / len(self._documents)
            if self._documents
            else 0.0
        )
        document_freq: Counter[str] = Counter()
        for document in self._documents:
            document_freq.update(document.term_freq.keys())
        total = len(self._documents)
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    # ------------------------------------------------------------ 信息
    @property
    def size(self) -> int:
        return len(self._documents)

    @property
    def stats(self) -> BM25Stats:
        return BM25Stats(
            documents=len(self._documents),
            tokens=sum(document.length for document in self._documents),
            avg_length=self._avg_length,
            vocabulary=len(self._idf),
        )

    # ------------------------------------------------------------ 检索
    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """按 BM25 打分检索；分数只用于排序，**不是**概率或相似度。"""
        if limit <= 0:
            raise ValueError(f"limit 必须为正整数，收到 {limit}")
        if not self._documents or self._avg_length <= 0:
            return []

        terms = [term for term in dict.fromkeys(tokenize_code(query)) if term in self._idf]
        if not terms:
            return []

        scored: list[tuple[float, str, _Document]] = []
        for document in self._documents:
            if filters and not matches_metadata(document.metadata, filters):
                continue
            score = self._score(document, terms)
            if score > 0:
                scored.append((score, document.doc_id, document))
        # 同分时按 doc_id 排序，保证结果稳定可复现
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchHit(
                vector_id=document.doc_id,
                score=score,
                text=document.text,
                metadata=document.metadata,
            )
            for score, _, document in scored[:limit]
        ]

    def _score(self, document: _Document, terms: Sequence[str]) -> float:
        total = 0.0
        normalization = self.k1 * (
            1 - self.b + self.b * document.length / self._avg_length
        )
        for term in terms:
            freq = document.term_freq.get(term)
            if not freq:
                continue
            total += self._idf[term] * freq * (self.k1 + 1) / (freq + normalization)
        return total
