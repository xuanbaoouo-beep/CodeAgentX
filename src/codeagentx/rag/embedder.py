"""向量化（Embedding）封装。

两种实现
--------
``OpenAICompatEmbedder``
    走 OpenAI 兼容的 ``/embeddings`` 接口，DashScope / ModelScope / 本地 vLLM 均可。
    这是**唯一具备真实语义能力**的实现。
``HashEmbedder``
    纯本地确定性实现（token 哈希 + TF 加权 + L2 归一），零网络依赖。
    用于离线测试、CI 与无密钥演示。**它不做语义理解**，
    因此不能用它的检索效果去评估系统的真实能力（见 README 的评估声明）。

:func:`build_embedder` 按配置自动选择：有密钥用真实接口，否则降级为 ``HashEmbedder``
并把 ``degraded`` 标记出来，由上层决定是否提示用户。
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from codeagentx.core.exceptions import RAGError
from codeagentx.core.logger import get_logger, log_event
from codeagentx.rag.tokenizer import tokenize_code

logger = get_logger("rag.embedder")

#: 离线实现的默认向量维度
DEFAULT_HASH_DIM = 512
#: 单次请求的文本条数上限（DashScope text-embedding-v3 上限为 10）
DEFAULT_BATCH_SIZE = 10
#: 单条文本的字符数上限（接口侧也有长度限制，先本地截断避免整批失败）
DEFAULT_MAX_CHARS = 4000


@dataclass
class EmbeddingStats:
    """向量化用量统计。``calls + failed_calls`` 等于实际发起的请求批次数。"""

    calls: int = 0
    failed_calls: int = 0
    texts: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "failed_calls": self.failed_calls, "texts": self.texts}


class BaseEmbedder(ABC):
    """向量化器抽象基类。"""

    def __init__(self, *, dim: int, model_id: str) -> None:
        if dim <= 0:
            raise ValueError("向量维度必须为正整数")
        self._dim = dim
        self.model_id = model_id
        self.stats = EmbeddingStats()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_semantic(self) -> bool:
        """是否具备真实语义能力（离线实现返回 ``False``）。"""
        return True

    @property
    def degraded(self) -> bool:
        """是否处于降级模式。"""
        return not self.is_semantic

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。返回顺序与入参严格一致。"""

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询文本。"""
        vectors = self.embed_documents([text])
        if not vectors:
            raise RAGError("向量化失败：未返回任何向量")
        return vectors[0]

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dim": self.dim,
            "semantic": self.is_semantic,
            "stats": self.stats.as_dict(),
        }


class HashEmbedder(BaseEmbedder):
    """离线确定性向量化：token 哈希到固定维度 + TF 加权 + L2 归一。

    它能反映**词法重叠**（查询与代码有相同 token 时相似度高），
    但**不能反映语义相似**（"登录" 与 "authenticate" 之间没有关联）。
    """

    def __init__(self, *, dim: int = DEFAULT_HASH_DIM, model_id: str = "hash-embedder") -> None:
        super().__init__(dim=dim, model_id=model_id)

    @property
    def is_semantic(self) -> bool:
        return False

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [self._embed_one(text) for text in texts]
        self.stats.calls += 1 if texts else 0
        self.stats.texts += len(texts)
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        counts: dict[int, int] = {}
        for token in tokenize_code(text):
            index = _stable_hash(token) % self._dim
            counts[index] = counts.get(index, 0) + 1
        vector = [0.0] * self._dim
        for index, count in counts.items():
            vector[index] = 1.0 + math.log(count)  # TF 加权，抑制高频 token
        return _l2_normalize(vector)


class OpenAICompatEmbedder(BaseEmbedder):
    """走 OpenAI 兼容 ``/embeddings`` 接口的真实向量化实现。"""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        base_url: str,
        dim: int = 1024,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_chars: int = DEFAULT_MAX_CHARS,
        client: Any | None = None,
    ) -> None:
        super().__init__(dim=dim, model_id=model_id)
        if not api_key.strip():
            raise RAGError("未配置 EMBEDDING_API_KEY，无法使用真实向量化接口")
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.max_chars = max_chars
        self._client = client

    @property
    def client(self) -> Any:
        """惰性创建 OpenAI 客户端（仅在真正调用时才要求安装依赖）。"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - openai 是必需依赖
                raise RAGError("未安装 openai，无法使用真实向量化接口") from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = [text[: self.max_chars] for text in batch]
        try:
            response = self.client.embeddings.create(model=self.model_id, input=payload)
            data = sorted(response.data, key=lambda item: item.index)
            vectors = [list(item.embedding) for item in data]
        except Exception as exc:  # noqa: BLE001 - 统一翻译为 RAGError
            self.stats.failed_calls += 1
            log_event(
                logger,
                "embedding_failed",
                level=30,
                model_id=self.model_id,
                error=type(exc).__name__,
            )
            raise RAGError(
                f"向量化请求失败：{type(exc).__name__}: {exc}",
                detail=f"model={self.model_id} base_url={self.base_url}",
            ) from exc

        if len(vectors) != len(batch):
            self.stats.failed_calls += 1
            raise RAGError(
                f"向量化返回条数不匹配：期望 {len(batch)}，实际 {len(vectors)}",
                detail=f"model={self.model_id}",
            )
        self.stats.calls += 1
        self.stats.texts += len(batch)
        if vectors and len(vectors[0]) != self._dim:
            # 以接口实际返回的维度为准（配置里的 dim 可能过时）
            logger.info(
                "向量维度与配置不一致，已按接口返回的 %d 更新（配置为 %d）",
                len(vectors[0]),
                self._dim,
            )
            self._dim = len(vectors[0])
        return vectors


def build_embedder(
    config: Any,
    *,
    allow_fallback: bool = True,
    dim: int | None = None,
    client: Any | None = None,
) -> BaseEmbedder:
    """按配置构造向量化器。

    有 ``EMBEDDING_API_KEY``（或与 LLM 同网关时的 ``LLM_API_KEY``）时使用真实接口，
    否则降级为 :class:`HashEmbedder`；``allow_fallback=False`` 时直接报错。
    """
    model_id = getattr(config, "embedding_model_id", "text-embedding-v3")
    api_key = (getattr(config, "embedding_api_key", "") or "").strip()
    base_url = getattr(config, "embedding_base_url", "")
    llm_base_url = (getattr(config, "llm_base_url", "") or "").rstrip("/")
    llm_api_key = (getattr(config, "llm_api_key", "") or "").strip()
    declared_dim = dim or int(getattr(config, "embedding_dim", 1024))

    # 只填了一个 key 时，若 embedding 与 LLM 指向同一网关，允许复用该 key
    if not api_key and llm_api_key and base_url.rstrip("/") == llm_base_url:
        api_key = llm_api_key
        logger.info("EMBEDDING_API_KEY 为空，复用同一网关的 LLM_API_KEY")

    if api_key:
        return OpenAICompatEmbedder(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            dim=declared_dim,
            client=client,
        )

    if not allow_fallback:
        raise RAGError(
            "未配置 EMBEDDING_API_KEY，无法构造真实向量化器",
            detail="请复制 .env.example 为 .env 并填写 EMBEDDING_API_KEY / EMBEDDING_BASE_URL",
        )
    logger.warning("未配置 EMBEDDING_API_KEY，向量化降级为离线 HashEmbedder（无语义能力）")
    return HashEmbedder(dim=DEFAULT_HASH_DIM)


def _stable_hash(token: str) -> int:
    """跨进程稳定的哈希（不能用内置 ``hash``，它受 PYTHONHASHSEED 影响）。"""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
