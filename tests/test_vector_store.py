"""向量库测试：内存后端的排序/过滤/缓存，Qdrant 后端的参数拼装与结果解析。

这里通过注入假的 ``qdrant_client`` 模块与假 client 验证封装逻辑
（参数是否传对、返回是否解析对），**不验证真实服务行为**——
真机联调单独做过（本地 Docker Qdrant 索引 + 换进程检索，见 ``vector_store`` 模块文档）。
"""

from __future__ import annotations

import sys
import uuid
from types import ModuleType, SimpleNamespace

import pytest

from codeagentx.core.exceptions import RAGError
from codeagentx.rag import vector_store as vector_store_module
from codeagentx.rag.vector_store import (
    BaseVectorStore,
    InMemoryVectorStore,
    QdrantVectorStore,
    SearchHit,
    VectorRecord,
    matches_metadata,
    to_point_id,
)


def _record(vector_id: str, vector: list[float], *, text: str = "", **metadata: object) -> VectorRecord:
    return VectorRecord(
        vector_id=vector_id, vector=vector, text=text or vector_id, metadata=dict(metadata)
    )


# ------------------------------------------------------------------ 数据结构
class TestSearchHit:
    def test_location_uses_path_and_line_span(self) -> None:
        hit = SearchHit(
            vector_id="a", score=0.5, text="x",
            metadata={"path": "src/a.py", "start_line": 1, "end_line": 9},
        )
        assert hit.path == "src/a.py"
        assert hit.location == "src/a.py:1-9"

    def test_location_falls_back_to_path_without_lines(self) -> None:
        hit = SearchHit(vector_id="a", score=0.5, text="x", metadata={"path": "a.py"})
        assert hit.location == "a.py"

    def test_location_is_empty_for_plain_records(self) -> None:
        assert SearchHit(vector_id="a", score=0.0, text="x").location == ""

    def test_to_dict_rounds_score_and_keeps_text(self) -> None:
        hit = SearchHit(vector_id="a", score=0.123456789, text="body", metadata={"path": "a.py"})
        payload = hit.to_dict()
        assert payload["score"] == 0.123457
        assert payload["text"] == "body"
        assert payload["metadata"] == {"path": "a.py"}


class TestMetadataFilter:
    def test_all_keys_must_match(self) -> None:
        metadata = {"language": "python", "path": "a.py"}
        assert matches_metadata(metadata, {"language": "python"}) is True
        assert matches_metadata(metadata, {"language": "python", "path": "a.py"}) is True
        assert matches_metadata(metadata, {"language": "python", "path": "b.py"}) is False

    def test_missing_key_does_not_match(self) -> None:
        assert matches_metadata({"language": "python"}, {"path": "a.py"}) is False


class TestPointIdMapping:
    """Qdrant 只接受 int/UUID 形式的 point id，业务 id 必须先映射。"""

    def test_digit_string_becomes_int(self) -> None:
        assert to_point_id("42") == 42

    def test_uuid_is_kept_as_is(self) -> None:
        value = "6f1d3b0e-6a2c-4d9f-9c2b-0a5f7e4d1c88"
        assert to_point_id(value) == value

    def test_chunk_id_becomes_stable_uuid(self) -> None:
        mapped = to_point_id("pkg/service.py::function:login::5-9#0")
        assert isinstance(mapped, str)
        assert mapped == to_point_id("pkg/service.py::function:login::5-9#0")
        assert mapped != to_point_id("pkg/service.py::function:logout::12-13#0")

    def test_mapped_id_is_a_parseable_uuid(self) -> None:
        """必须是 uuid.UUID 能解析的形式，否则 Qdrant 会拒绝写入。"""
        assert uuid.UUID(str(to_point_id("a.py::function:f::1-2#0")))


# ------------------------------------------------------------------ 内存后端
class TestInMemoryVectorStore:
    def test_invalid_dimension_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="维度"):
            InMemoryVectorStore(dim=0)

    def test_implements_abstract_base(self) -> None:
        assert isinstance(InMemoryVectorStore(dim=4), BaseVectorStore)

    def test_upsert_returns_written_count(self) -> None:
        store = InMemoryVectorStore(dim=2)
        assert store.upsert([_record("a", [1.0, 0.0]), _record("b", [0.0, 1.0])]) == 2
        assert store.count() == 2

    def test_upsert_same_id_overwrites(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0], text="old")])
        store.upsert([_record("a", [0.0, 1.0], text="new")])
        assert store.count() == 1
        assert store.get("a") is not None
        assert store.get("a").text == "new"

    def test_dimension_mismatch_on_upsert_is_reported(self) -> None:
        store = InMemoryVectorStore(dim=3)
        with pytest.raises(RAGError, match="bad"):
            store.upsert([_record("bad", [1.0, 0.0])])
        assert store.count() == 0

    def test_upsert_empty_batch_is_noop(self) -> None:
        store = InMemoryVectorStore(dim=2)
        assert store.upsert([]) == 0
        assert store.count() == 0

    def test_get_returns_none_for_unknown_id(self) -> None:
        assert InMemoryVectorStore(dim=2).get("missing") is None

    # -------------------------------------------------------- 检索
    def test_search_ranks_by_cosine_similarity(self) -> None:
        store = InMemoryVectorStore(dim=3)
        store.upsert([
            _record("near", [1.0, 0.0, 0.0]),
            _record("mid", [0.7, 0.7, 0.0]),
            _record("far", [0.0, 0.0, 1.0]),
        ])
        hits = store.search([1.0, 0.0, 0.0], limit=3)
        assert [hit.vector_id for hit in hits] == ["near", "mid", "far"]

    def test_scores_are_descending_and_normalized(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [2.0, 0.0]), _record("b", [1.0, 1.0])])
        hits = store.search([3.0, 0.0], limit=2)
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].score > hits[1].score
        assert hits[0].score <= 1.0

    def test_limit_is_respected(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record(str(index), [1.0, float(index)]) for index in range(5)])
        assert len(store.search([1.0, 0.0], limit=2)) == 2

    def test_score_threshold_filters_low_hits(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("same", [1.0, 0.0]), _record("orthogonal", [0.0, 1.0])])
        hits = store.search([1.0, 0.0], limit=5, score_threshold=0.5)
        assert [hit.vector_id for hit in hits] == ["same"]

    def test_metadata_filter_selects_matching_records(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([
            _record("py", [1.0, 0.0], language="python", path="a.py"),
            _record("md", [1.0, 0.0], language="markdown", path="b.md"),
        ])
        hits = store.search([1.0, 0.0], limit=5, filters={"language": "python"})
        assert [hit.vector_id for hit in hits] == ["py"]

    def test_limit_counts_only_filtered_hits(self) -> None:
        """先过滤再计入 limit，否则过滤条件会"吃掉"名额。"""
        store = InMemoryVectorStore(dim=2)
        store.upsert([
            _record("md1", [1.0, 0.0], language="markdown"),
            _record("md2", [0.9, 0.1], language="markdown"),
            _record("py", [0.8, 0.2], language="python"),
        ])
        hits = store.search([1.0, 0.0], limit=1, filters={"language": "python"})
        assert [hit.vector_id for hit in hits] == ["py"]

    def test_ties_are_ordered_deterministically(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("c", [1.0, 0.0]), _record("a", [1.0, 0.0]), _record("b", [1.0, 0.0])])
        hits = store.search([1.0, 0.0], limit=3)
        assert [hit.vector_id for hit in hits] == ["a", "b", "c"]

    def test_search_on_empty_store_returns_nothing(self) -> None:
        assert InMemoryVectorStore(dim=2).search([1.0, 0.0]) == []

    def test_non_positive_limit_is_rejected(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0])])
        with pytest.raises(RAGError, match="limit"):
            store.search([1.0, 0.0], limit=0)

    def test_zero_query_vector_returns_nothing(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0])])
        assert store.search([0.0, 0.0]) == []

    def test_query_dimension_mismatch_is_rejected(self) -> None:
        store = InMemoryVectorStore(dim=3)
        store.upsert([_record("a", [1.0, 0.0, 0.0])])
        with pytest.raises(RAGError, match="维度不匹配"):
            store.search([1.0, 0.0])

    def test_hit_carries_text_and_metadata(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0], text="def login(): ...", path="a.py", start_line=3, end_line=6)])
        hit = store.search([1.0, 0.0])[0]
        assert hit.text == "def login(): ..."
        assert hit.location == "a.py:3-6"

    # -------------------------------------------------------- 删除与缓存
    def test_delete_removes_only_existing_ids(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0]), _record("b", [0.0, 1.0])])
        assert store.delete(["a", "missing"]) == 1
        assert store.count() == 1

    def test_delete_empty_is_noop(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0])])
        assert store.delete([]) == 0
        assert store.count() == 1

    def test_clear_removes_everything(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0])])
        store.clear()
        assert store.count() == 0
        assert store.search([1.0, 0.0]) == []

    # -------------------------------------------------------- 全量拉取
    def test_fetch_all_returns_sorted_documents_without_vectors(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([
            _record("b", [0.0, 1.0], text="second", path="b.py"),
            _record("a", [1.0, 0.0], text="first", path="a.py"),
        ])
        documents = store.fetch_all()
        assert [document.vector_id for document in documents] == ["a", "b"]
        assert documents[0].text == "first"
        assert documents[0].metadata == {"path": "a.py"}
        assert not hasattr(documents[0], "vector")

    def test_fetch_all_honours_limit(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record(str(index), [1.0, 0.0]) for index in range(4)])
        assert len(store.fetch_all(limit=2)) == 2

    def test_fetch_all_on_empty_store(self) -> None:
        assert InMemoryVectorStore(dim=2).fetch_all() == []

    def test_matrix_cache_is_rebuilt_after_upsert(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("old", [1.0, 0.0])])
        assert store.search([1.0, 0.0], limit=1)[0].vector_id == "old"
        store.upsert([_record("new", [1.0, 0.0])])
        assert [hit.vector_id for hit in store.search([1.0, 0.0], limit=2)] == ["new", "old"]

    def test_matrix_cache_is_rebuilt_after_delete(self) -> None:
        store = InMemoryVectorStore(dim=2)
        store.upsert([_record("a", [1.0, 0.0]), _record("b", [0.9, 0.1])])
        assert len(store.search([1.0, 0.0], limit=5)) == 2
        store.delete(["a"])
        hits = store.search([1.0, 0.0], limit=5)
        assert [hit.vector_id for hit in hits] == ["b"]

    def test_close_is_noop(self) -> None:
        InMemoryVectorStore(dim=2).close()


# ------------------------------------------------------------------ Qdrant 后端
class _PointStruct:
    def __init__(self, *, id: str, vector: list[float], payload: dict) -> None:  # noqa: A002
        self.id = id
        self.vector = vector
        self.payload = payload


class _ScoredPoint:
    def __init__(self, *, id: str, score: float, payload: dict) -> None:  # noqa: A002
        self.id = id
        self.score = score
        self.payload = payload


class _MatchValue:
    def __init__(self, *, value: object) -> None:
        self.value = value


class _FieldCondition:
    def __init__(self, *, key: str, match: _MatchValue) -> None:
        self.key = key
        self.match = match


class _Filter:
    def __init__(self, *, must: list[_FieldCondition]) -> None:
        self.must = must


class _VectorParams:
    def __init__(self, *, size: int, distance: str) -> None:
        self.size = size
        self.distance = distance


def _install_fake_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    """把假的 ``qdrant_client`` 注入 sys.modules，供 QdrantVectorStore 导入。"""
    models = ModuleType("qdrant_client.models")
    models.PointStruct = _PointStruct
    models.MatchValue = _MatchValue
    models.FieldCondition = _FieldCondition
    models.Filter = _Filter
    models.VectorParams = _VectorParams
    models.Distance = SimpleNamespace(COSINE="Cosine")

    package = ModuleType("qdrant_client")
    package.models = models
    monkeypatch.setitem(sys.modules, "qdrant_client", package)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)


class _FakeQdrantClient:
    """假 client：以内存字典模拟 collection 与 point，用于校验封装逻辑。"""

    def __init__(self) -> None:
        self.collections: dict[str, dict] = {}
        self.created: list[str] = []
        self.deleted_collections: list[str] = []
        self.last_query: dict = {}

    # -------- collection
    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: _VectorParams) -> None:
        self.created.append(collection_name)
        self.collections[collection_name] = {"config": vectors_config, "points": {}}

    def delete_collection(self, *, collection_name: str) -> None:
        self.deleted_collections.append(collection_name)
        self.collections.pop(collection_name, None)

    def get_collection(self, collection_name: str) -> SimpleNamespace:
        vectors = self.collections[collection_name]["config"]
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
        )

    # -------- point
    def upsert(self, *, collection_name: str, points: list[_PointStruct], wait: bool = True) -> None:
        for point in points:
            self.collections[collection_name]["points"][point.id] = (point.vector, point.payload)

    def delete(self, *, collection_name: str, points_selector: list[str], wait: bool = True) -> None:
        for point_id in points_selector:
            self.collections[collection_name]["points"].pop(point_id, None)

    def count(self, *, collection_name: str, exact: bool = True) -> SimpleNamespace:
        return SimpleNamespace(count=len(self.collections[collection_name]["points"]))

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: int | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[SimpleNamespace], int | None]:
        keys = list(self.collections[collection_name]["points"])
        start = int(offset or 0)
        page = keys[start : start + limit]
        points = [
            SimpleNamespace(id=key, payload=dict(self.collections[collection_name]["points"][key][1]))
            for key in page
        ]
        next_offset = start + len(page) if start + len(page) < len(keys) else None
        return points, next_offset

    def _rank(self, collection_name: str, vector: list[float], limit: int, score_threshold):
        scored = []
        for point_id, (stored, payload) in self.collections[collection_name]["points"].items():
            score = sum(a * b for a, b in zip(stored, vector, strict=True))
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(_ScoredPoint(id=point_id, score=score, payload=dict(payload)))
        scored.sort(key=lambda point: -point.score)
        return scored[:limit]


class _ModernFakeClient(_FakeQdrantClient):
    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        score_threshold: float | None,
        query_filter: object,
        with_payload: bool,
    ) -> SimpleNamespace:
        self.last_query = {
            "collection_name": collection_name,
            "vector": query,
            "limit": limit,
            "score_threshold": score_threshold,
            "filter": query_filter,
            "with_payload": with_payload,
        }
        return SimpleNamespace(points=self._rank(collection_name, query, limit, score_threshold))


class _LegacyFakeClient(_FakeQdrantClient):
    """模拟 qdrant-client < 1.10：没有 query_points，只有 search。"""

    def search(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        score_threshold: float | None,
        query_filter: object,
        with_payload: bool,
    ) -> list[_ScoredPoint]:
        self.last_query = {
            "collection_name": collection_name,
            "vector": query_vector,
            "limit": limit,
            "score_threshold": score_threshold,
            "filter": query_filter,
            "with_payload": with_payload,
        }
        return self._rank(collection_name, query_vector, limit, score_threshold)


def _qdrant_store(client: _FakeQdrantClient, *, dim: int = 2) -> QdrantVectorStore:
    return QdrantVectorStore(url="http://localhost:6333", collection="c", dim=dim, client=client)


class TestQdrantVectorStore:
    def test_missing_dependency_gives_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "qdrant_client", None)
        store = QdrantVectorStore(url="http://localhost:6333", collection="c", dim=2)
        with pytest.raises(RAGError, match="qdrant-client"):
            store.count()

    def test_collection_is_created_on_first_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        store = _qdrant_store(client, dim=4)
        assert store.upsert([_record("a", [1.0, 0.0, 0.0, 0.0], path="a.py")]) == 1
        assert client.created == ["c"]
        assert client.collections["c"]["config"].size == 4
        assert client.collections["c"]["config"].distance == "Cosine"

    def test_existing_collection_is_not_recreated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        client.create_collection(collection_name="c", vectors_config=_VectorParams(size=2, distance="Cosine"))
        _qdrant_store(client).upsert([_record("a", [1.0, 0.0])])
        assert client.created == ["c"]  # 仅测试自己建的那一次

    def test_existing_collection_with_wrong_dim_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """换过 embedding 模型却没清空旧集合时，要提前报"维度不符"而不是等 Qdrant 报底层错。"""
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        _qdrant_store(client, dim=2).count()  # 建出 dim=2 的集合
        with pytest.raises(RAGError, match="维度"):
            _qdrant_store(client, dim=3).count()

    def test_text_and_metadata_go_into_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        _qdrant_store(client).upsert([_record("a", [1.0, 0.0], text="body", path="a.py", start_line=1)])
        stored_id, (_, payload) = next(iter(client.collections["c"]["points"].items()))
        assert stored_id == to_point_id("a")  # 业务 id 已映射成 Qdrant 接受的 point id
        assert payload == {"text": "body", "chunk_id": "a", "path": "a.py", "start_line": 1}

    def test_empty_upsert_does_not_touch_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        assert _qdrant_store(client).upsert([]) == 0
        assert client.created == []

    def test_search_maps_response_into_hits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        store = _qdrant_store(client)
        store.upsert([
            _record("near", [1.0, 0.0], text="near body", path="a.py", start_line=1, end_line=2),
            _record("far", [0.0, 1.0], text="far body", path="b.py"),
        ])
        hits = store.search([1.0, 0.0], limit=5)
        assert [hit.vector_id for hit in hits] == ["near", "far"]
        assert hits[0].text == "near body"
        assert hits[0].metadata == {"path": "a.py", "start_line": 1, "end_line": 2}
        assert hits[0].location == "a.py:1-2"

    def test_search_returns_business_id_not_point_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """上层拿到的必须是 chunk_id，否则无法回查源码。"""
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        store = _qdrant_store(client)
        chunk_id = "pkg/service.py::function:login::5-9#0"
        store.upsert([_record(chunk_id, [1.0, 0.0], text="body")])
        assert store.search([1.0, 0.0])[0].vector_id == chunk_id
        assert store.delete([chunk_id]) == 1
        assert store.count() == 0

    def test_search_passes_threshold_and_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        store = _qdrant_store(client)
        store.upsert([_record("a", [1.0, 0.0]), _record("b", [0.0, 1.0])])
        hits = store.search([1.0, 0.0], limit=1, score_threshold=0.5)
        assert [hit.vector_id for hit in hits] == ["a"]
        assert client.last_query["limit"] == 1
        assert client.last_query["score_threshold"] == 0.5

    def test_search_builds_filter_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        store = _qdrant_store(client)
        store.upsert([_record("a", [1.0, 0.0], language="python")])
        store.search([1.0, 0.0], filters={"language": "python"})
        query_filter = client.last_query["filter"]
        assert isinstance(query_filter, _Filter)
        assert query_filter.must[0].key == "language"
        assert query_filter.must[0].match.value == "python"

    def test_search_without_filters_sends_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        store = _qdrant_store(client)
        store.upsert([_record("a", [1.0, 0.0])])
        store.search([1.0, 0.0])
        assert client.last_query["filter"] is None

    def test_legacy_client_uses_search_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _LegacyFakeClient()
        store = _qdrant_store(client)
        store.upsert([_record("a", [1.0, 0.0], text="body")])
        hits = store.search([1.0, 0.0], limit=3)
        assert [hit.vector_id for hit in hits] == ["a"]
        assert hits[0].text == "body"

    def test_non_positive_limit_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        with pytest.raises(RAGError, match="limit"):
            _qdrant_store(_ModernFakeClient()).search([1.0, 0.0], limit=-1)

    def test_count_reads_exact_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        store = _qdrant_store(_ModernFakeClient())
        store.upsert([_record("a", [1.0, 0.0]), _record("b", [0.0, 1.0])])
        assert store.count() == 2

    def test_delete_removes_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        store = _qdrant_store(_ModernFakeClient())
        store.upsert([_record("a", [1.0, 0.0]), _record("b", [0.0, 1.0])])
        assert store.delete(["a"]) == 1
        assert store.count() == 1

    def test_delete_empty_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        assert _qdrant_store(client).delete([]) == 0
        assert client.created == []

    def test_clear_drops_the_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        client = _ModernFakeClient()
        store = _qdrant_store(client)
        store.upsert([_record("a", [1.0, 0.0])])
        store.clear()
        assert client.deleted_collections == ["c"]

    def test_fetch_all_restores_business_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        store = _qdrant_store(_ModernFakeClient())
        store.upsert([
            _record("pkg/a.py::function:f::1-2#0", [1.0, 0.0], text="body", path="pkg/a.py"),
            _record("pkg/b.py::module_header::1-2#0", [0.0, 1.0], text="head", path="pkg/b.py"),
        ])
        documents = store.fetch_all()
        assert [document.vector_id for document in documents] == [
            "pkg/a.py::function:f::1-2#0",
            "pkg/b.py::module_header::1-2#0",
        ]
        assert documents[0].text == "body"
        assert documents[0].metadata == {"path": "pkg/a.py"}

    def test_fetch_all_pages_through_scroll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_qdrant(monkeypatch)
        monkeypatch.setattr(vector_store_module, "DEFAULT_SCROLL_PAGE", 2)
        store = _qdrant_store(_ModernFakeClient())
        store.upsert([_record(f"chunk-{index}", [1.0, 0.0]) for index in range(5)])
        assert len(store.fetch_all()) == 5
        assert len(store.fetch_all(limit=3)) == 3
