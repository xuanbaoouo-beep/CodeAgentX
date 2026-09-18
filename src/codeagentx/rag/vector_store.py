"""向量存储。

两个后端
--------
``InMemoryVectorStore``
    numpy 余弦相似度，零外部依赖，是**默认实现**。
    面向"单个仓库几百到几千个代码块"的规模，够快也够简单。
``QdrantVectorStore``
    可选的持久化后端（``pip install "codeagentx[rag]"`` + 一个可访问的 Qdrant 服务）。
    自动化测试通过注入假 client 覆盖参数拼装与返回解析；**真机联调**另做过一次：
    本地 Docker 起 Qdrant，索引 22 个代码块后**换新进程**只做检索，
    仍能读回 22 条并命中 ``app/auth/service.py``（跨进程持久化成立）。

两者都实现 :class:`BaseVectorStore`，上层检索器不关心具体后端。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from codeagentx.core.exceptions import RAGError
from codeagentx.core.logger import get_logger

logger = get_logger("rag.vector_store")

#: Qdrant 分页拉取（scroll）的单页条数
DEFAULT_SCROLL_PAGE = 256


@dataclass(frozen=True)
class VectorRecord:
    """一条待入库记录。"""

    vector_id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredDocument:
    """一条已入库的文档（不含向量）。

    用于"把库里所有内容取出来"的场景——目前是 BM25 词法检索构建语料。
    """

    vector_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    """一条检索结果。"""

    vector_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return str(self.metadata.get("path", ""))

    @property
    def location(self) -> str:
        """``path:start-end`` 形式的位置标识。"""
        start = self.metadata.get("start_line")
        end = self.metadata.get("end_line")
        if start is None:
            return self.path
        return f"{self.path}:{start}-{end}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "score": round(self.score, 6),
            "text": self.text,
            "metadata": self.metadata,
        }


class BaseVectorStore(ABC):
    """向量存储抽象基类。"""

    def __init__(self, *, dim: int) -> None:
        if dim <= 0:
            raise ValueError("向量维度必须为正整数")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """写入或覆盖记录，返回写入条数。"""

    @abstractmethod
    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """按余弦相似度检索，结果按分数降序。"""

    @abstractmethod
    def fetch_all(self, *, limit: int | None = None) -> list[StoredDocument]:
        """取回全部文档（不含向量），用于构建词法检索语料或做统计。"""

    @abstractmethod
    def delete(self, vector_ids: Sequence[str]) -> int:
        """删除指定记录，返回实际删除条数。"""

    @abstractmethod
    def count(self) -> int:
        """当前记录总数。"""

    @abstractmethod
    def clear(self) -> None:
        """清空全部记录。"""

    def close(self) -> None:
        """释放资源（默认无操作，子类按需覆盖）。"""
        return None


class InMemoryVectorStore(BaseVectorStore):
    """基于 numpy 的内存向量库。"""

    def __init__(self, *, dim: int) -> None:
        super().__init__(dim=dim)
        self._records: dict[str, VectorRecord] = {}
        self._order: list[str] = []
        self._matrix: np.ndarray | None = None

    # ------------------------------------------------------------ 写入
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        written = 0
        for record in records:
            if len(record.vector) != self._dim:
                raise RAGError(
                    f"向量维度不匹配：记录 {record.vector_id} 为 {len(record.vector)}，"
                    f"期望 {self._dim}"
                )
            self._records[record.vector_id] = record
            written += 1
        if written:
            self._matrix = None  # 失效缓存，下次检索时重建
        return written

    def delete(self, vector_ids: Sequence[str]) -> int:
        removed = sum(1 for vector_id in vector_ids if self._records.pop(vector_id, None))
        if removed:
            self._matrix = None
        return removed

    def fetch_all(self, *, limit: int | None = None) -> list[StoredDocument]:
        documents = [
            StoredDocument(vector_id=key, text=self._records[key].text, metadata=self._records[key].metadata)
            for key in sorted(self._records)
        ]
        return documents if limit is None else documents[:limit]

    def clear(self) -> None:
        self._records.clear()
        self._order = []
        self._matrix = None

    # ------------------------------------------------------------ 检索
    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise RAGError(f"limit 必须为正整数，收到 {limit}")
        if not self._records:
            return []

        query = np.asarray(list(vector), dtype=np.float32)
        if query.shape[0] != self._dim:
            raise RAGError(f"查询向量维度不匹配：{query.shape[0]} != {self._dim}")
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        scores = self._matrix_view() @ (query / norm)

        hits: list[SearchHit] = []
        for index in np.argsort(-scores, kind="stable"):
            record = self._records[self._order[int(index)]]
            if filters and not matches_metadata(record.metadata, filters):
                continue
            score = float(scores[int(index)])
            if score_threshold is not None and score < score_threshold:
                continue
            hits.append(
                SearchHit(
                    vector_id=record.vector_id,
                    score=score,
                    text=record.text,
                    metadata=record.metadata,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def count(self) -> int:
        return len(self._records)

    def get(self, vector_id: str) -> VectorRecord | None:
        """按 id 取回记录（调试与测试用）。"""
        return self._records.get(vector_id)

    def _matrix_view(self) -> np.ndarray:
        """行 L2 归一化后的向量矩阵（惰性构建并缓存）。"""
        if self._matrix is None:
            self._order = sorted(self._records)
            matrix = np.asarray(
                [self._records[key].vector for key in self._order], dtype=np.float32
            )
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.clip(norms, 1e-12, None, out=norms)
            self._matrix = matrix / norms
        return self._matrix


class QdrantVectorStore(BaseVectorStore):
    """Qdrant 后端（可选，依赖 ``qdrant-client>=1.8``）。

    Qdrant 的 point id 只接受**无符号整数或 UUID**，而本项目的 ``chunk_id``
    形如 ``src/a.py::function:login::3-7#0``，因此写入时用 :func:`to_point_id`
    把任意字符串稳定映射成 UUID5；检索时再从 payload 的 ``chunk_id`` 还原，
    使上层的向量 id 始终是原本的 ``chunk_id``（与内存后端一致）。

    .. note::
        启用前请确认 Qdrant 服务可访问（``pip install "codeagentx[rag]"``）。
        集合已存在时 :meth:`_ensure_collection` 会校验向量维度，
        避免"换过 embedding 模型、索引没重建"这类配置漂移被拖到写入时才报错。
    """

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        dim: int,
        api_key: str = "",
        client: Any | None = None,
    ) -> None:
        super().__init__(dim=dim)
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RAGError(
                    "未安装 qdrant-client，无法使用 Qdrant 后端",
                    detail="请执行 pip install qdrant-client，或改用 InMemoryVectorStore",
                ) from exc
            self._client = QdrantClient(
                url=self.url, api_key=self.api_key or None, timeout=30.0
            )
        return self._client

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        self._ensure_collection()
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=to_point_id(record.vector_id),
                vector=list(record.vector),
                payload={"text": record.text, "chunk_id": record.vector_id, **record.metadata},
            )
            for record in records
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if limit <= 0:
            raise RAGError(f"limit 必须为正整数，收到 {limit}")
        self._ensure_collection()
        hits: list[SearchHit] = []
        for item in self._query(vector, limit=limit, score_threshold=score_threshold, filters=filters):
            payload = dict(item.payload or {})
            hits.append(
                SearchHit(
                    # 优先还原业务 id（chunk_id），退回到 Qdrant 的 point id
                    vector_id=str(payload.pop("chunk_id", "") or item.id),
                    score=float(item.score),
                    text=str(payload.pop("text", "")),
                    metadata=payload,
                )
            )
        return hits

    def delete(self, vector_ids: Sequence[str]) -> int:
        ids = list(vector_ids)
        if not ids:
            return 0
        self._ensure_collection()
        self.client.delete(
            collection_name=self.collection,
            points_selector=[to_point_id(vector_id) for vector_id in ids],
            wait=True,
        )
        return len(ids)

    def fetch_all(self, *, limit: int | None = None) -> list[StoredDocument]:
        self._ensure_collection()
        documents: list[StoredDocument] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=DEFAULT_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                documents.append(
                    StoredDocument(
                        vector_id=str(payload.pop("chunk_id", "") or point.id),
                        text=str(payload.pop("text", "")),
                        metadata=payload,
                    )
                )
            if offset is None or (limit is not None and len(documents) >= limit):
                break
        return documents if limit is None else documents[:limit]

    def count(self) -> int:
        self._ensure_collection()
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def clear(self) -> None:
        self.client.delete_collection(collection_name=self.collection)
        logger.info("已删除 Qdrant collection：%s", self.collection)

    def _ensure_collection(self) -> None:
        # 先访问 client：依赖缺失时由 client 属性抛出带安装提示的 RAGError，
        # 而不是让 `from qdrant_client.models import ...` 抛出原始 ImportError。
        client = self.client
        from qdrant_client.models import Distance, VectorParams

        if client.collection_exists(self.collection):
            self._assert_dim(client)
            return
        client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
        )
        logger.info("已创建 Qdrant collection：%s（dim=%d）", self.collection, self._dim)

    def _assert_dim(self, client: Any) -> None:
        """已存在的集合维度必须与当前配置一致。

        换了 embedding 模型（维度随之改变）却没清空旧集合时，Qdrant 会在**写入时**
        抛一条难以定位的底层错误；这里提前拦成带修复提示的 RAGError。
        """
        vectors = client.get_collection(self.collection).config.params.vectors
        # 命名向量（dict）不在此后端的用法内，取不到 size 就跳过校验，不误报
        existing = getattr(vectors, "size", None)
        if existing is None or int(existing) == self._dim:
            return
        raise RAGError(
            f"Qdrant collection {self.collection} 的向量维度是 {int(existing)}，"
            f"与当前配置的 {self._dim} 不一致",
            detail=(
                "换了 embedding 模型（或改过 EMBEDDING_DIM）后需要清空旧集合并重建索引："
                f"删除 Qdrant 里的 collection {self.collection}，"
                f"或把 .env 的 EMBEDDING_DIM 改回 {int(existing)}"
            ),
        )

    def _query(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        score_threshold: float | None,
        filters: dict[str, Any] | None,
    ) -> list[Any]:
        """执行检索，兼容新旧两代 qdrant-client 的查询接口。"""
        query_filter = _to_qdrant_filter(filters)
        if hasattr(self.client, "query_points"):  # qdrant-client >= 1.10
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(vector),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
            return list(response.points)
        return list(  # qdrant-client < 1.10 的老接口
            self.client.search(
                collection_name=self.collection,
                query_vector=list(vector),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
        )


#: UUID5 命名空间（固定值，不可更改，否则历史数据的 point id 会全部变化）
_POINT_ID_NAMESPACE = uuid.UUID("6f1d3b0e-6a2c-4d9f-9c2b-0a5f7e4d1c88")


def to_point_id(vector_id: str) -> str | int:
    """把任意向量 id 映射成 Qdrant 接受的 point id（无符号整数或 UUID）。

    已是整数或 UUID 形式时原样返回，其余（如 ``src/a.py::function:f::1-9#0``）
    用 UUID5 稳定映射——**同一 id 在任何进程、任何机器上都得到同一个 UUID**，
    因此可以据此删除与还原。
    """
    if vector_id.isdigit():
        return int(vector_id)
    try:
        return str(uuid.UUID(vector_id))
    except ValueError:
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, vector_id))


def matches_metadata(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """元数据等值过滤：所有键都必须匹配（语义检索与词法检索共用同一套语义）。"""
    return all(metadata.get(key) == value for key, value in filters.items())


def _to_qdrant_filter(filters: dict[str, Any] | None) -> Any:
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
    )
