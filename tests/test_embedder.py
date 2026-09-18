"""向量化测试：离线确定性、相似度行为、真实接口批量与错误处理。"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from codeagentx.config import Config
from codeagentx.core.exceptions import RAGError
from codeagentx.rag.embedder import (
    BaseEmbedder,
    HashEmbedder,
    OpenAICompatEmbedder,
    build_embedder,
)


def _cosine(left: list[float], right: list[float]) -> float:
    """两个已 L2 归一化的向量的余弦相似度。"""
    return sum(a * b for a, b in zip(left, right, strict=True))


class _FakeEmbeddingsAPI:
    """模拟 OpenAI 的 embeddings.create，并记录调用。"""

    def __init__(self, *, dim: int = 4, shuffle: bool = True, drop_last: bool = False) -> None:
        self.dim = dim
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.calls: list[tuple[str, list[str]]] = []

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append((model, list(input)))
        items = [
            SimpleNamespace(index=index, embedding=[float(index + 1)] * self.dim)
            for index in range(len(input))
        ]
        if self.shuffle:
            items.reverse()
        if self.drop_last and items:
            items.pop()
        return SimpleNamespace(data=items)


class _FakeClient:
    def __init__(self, api: _FakeEmbeddingsAPI) -> None:
        self.embeddings = api


class _BrokenClient:
    class _Broken:
        def create(self, **_kwargs: object) -> None:
            raise RuntimeError("connection reset")

    embeddings = _Broken()


# ------------------------------------------------------------------ 离线实现
class TestHashEmbedder:
    def test_is_marked_as_non_semantic(self) -> None:
        embedder = HashEmbedder()
        assert embedder.is_semantic is False
        assert embedder.degraded is True

    def test_dimension_is_configurable(self) -> None:
        assert len(HashEmbedder(dim=64).embed_query("hello")) == 64

    def test_vectors_are_l2_normalized(self) -> None:
        vector = HashEmbedder(dim=64).embed_query("def login(user): pass")
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-9)

    def test_result_is_deterministic(self) -> None:
        embedder = HashEmbedder(dim=128)
        assert embedder.embed_query("def login(user)") == embedder.embed_query("def login(user)")

    def test_identical_text_similarity_is_one(self) -> None:
        embedder = HashEmbedder()
        text = "def authenticate(user, password): return check(user, password)"
        assert math.isclose(_cosine(embedder.embed_query(text), embedder.embed_query(text)), 1.0)

    def test_overlapping_text_is_closer_than_unrelated(self) -> None:
        embedder = HashEmbedder()
        left = embedder.embed_query("def authenticate(user, password): return True")
        near = embedder.embed_query("def authenticate(user, password): pass")
        far = embedder.embed_query("matrix = numpy.zeros(shape)")
        assert _cosine(left, near) > _cosine(left, far)

    def test_chinese_query_matches_chinese_doc(self) -> None:
        embedder = HashEmbedder()
        query = embedder.embed_query("用户登录逻辑在哪")
        doc = embedder.embed_query("处理用户登录的服务")
        other = embedder.embed_query("计算矩阵乘法")
        assert _cosine(query, doc) > _cosine(query, other)

    def test_empty_text_gives_zero_vector(self) -> None:
        assert HashEmbedder(dim=8).embed_query("") == [0.0] * 8

    def test_stats_are_counted(self) -> None:
        embedder = HashEmbedder()
        embedder.embed_documents(["a", "b", "c"])
        assert embedder.stats.texts == 3
        assert embedder.stats.calls == 1
        assert embedder.stats.failed_calls == 0

    def test_empty_batch_does_not_count_a_call(self) -> None:
        embedder = HashEmbedder()
        assert embedder.embed_documents([]) == []
        assert embedder.stats.calls == 0

    def test_batch_order_matches_input(self) -> None:
        embedder = HashEmbedder(dim=64)
        vectors = embedder.embed_documents(["alpha", "beta"])
        assert vectors[0] == embedder.embed_query("alpha")
        assert vectors[1] == embedder.embed_query("beta")

    def test_invalid_dimension_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="维度"):
            HashEmbedder(dim=0)


# ------------------------------------------------------------------ 真实实现
class TestOpenAICompatEmbedder:
    def test_requires_api_key(self) -> None:
        with pytest.raises(RAGError, match="EMBEDDING_API_KEY"):
            OpenAICompatEmbedder(model_id="m", api_key="   ", base_url="http://x")

    def test_requests_are_batched(self) -> None:
        api = _FakeEmbeddingsAPI()
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4, batch_size=2,
            client=_FakeClient(api),
        )
        embedder.embed_documents(["a", "b", "c"])
        assert [len(payload) for _, payload in api.calls] == [2, 1]
        assert embedder.stats.calls == 2
        assert embedder.stats.texts == 3

    def test_model_id_is_forwarded(self) -> None:
        api = _FakeEmbeddingsAPI()
        embedder = OpenAICompatEmbedder(
            model_id="text-embedding-v3", api_key="k", base_url="http://x", dim=4,
            client=_FakeClient(api),
        )
        embedder.embed_documents(["a"])
        assert api.calls[0][0] == "text-embedding-v3"

    def test_results_follow_input_order_despite_shuffled_response(self) -> None:
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4,
            client=_FakeClient(_FakeEmbeddingsAPI(dim=4, shuffle=True)),
        )
        vectors = embedder.embed_documents(["first", "second"])
        assert vectors[0] == [1.0, 1.0, 1.0, 1.0]
        assert vectors[1] == [2.0, 2.0, 2.0, 2.0]

    def test_dimension_adapts_to_api_response(self) -> None:
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=1024,
            client=_FakeClient(_FakeEmbeddingsAPI(dim=8)),
        )
        embedder.embed_documents(["a"])
        assert embedder.dim == 8

    def test_long_text_is_truncated_before_sending(self) -> None:
        api = _FakeEmbeddingsAPI()
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4, max_chars=10,
            client=_FakeClient(api),
        )
        embedder.embed_documents(["x" * 100])
        assert api.calls[0][1] == ["x" * 10]

    def test_request_failure_is_translated(self) -> None:
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4, client=_BrokenClient()
        )
        with pytest.raises(RAGError, match="向量化请求失败"):
            embedder.embed_documents(["a"])
        assert embedder.stats.failed_calls == 1
        assert embedder.stats.calls == 0

    def test_count_mismatch_is_rejected(self) -> None:
        api = _FakeEmbeddingsAPI(drop_last=True)
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4,
            client=_FakeClient(api),
        )
        with pytest.raises(RAGError, match="条数不匹配"):
            embedder.embed_documents(["a", "b"])
        assert embedder.stats.failed_calls == 1

    def test_empty_input_makes_no_request(self) -> None:
        api = _FakeEmbeddingsAPI()
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4,
            client=_FakeClient(api),
        )
        assert embedder.embed_documents([]) == []
        assert api.calls == []

    def test_is_marked_as_semantic(self) -> None:
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4,
            client=_FakeClient(_FakeEmbeddingsAPI()),
        )
        assert embedder.is_semantic is True
        assert embedder.degraded is False

    def test_embed_query_delegates_to_batch(self) -> None:
        api = _FakeEmbeddingsAPI()
        embedder = OpenAICompatEmbedder(
            model_id="m", api_key="k", base_url="http://x", dim=4,
            client=_FakeClient(api),
        )
        assert embedder.embed_query("only") == [1.0, 1.0, 1.0, 1.0]
        assert len(api.calls) == 1


# ------------------------------------------------------------------ 工厂
class TestBuildEmbedder:
    def test_uses_real_embedder_when_key_present(self) -> None:
        config = Config(embedding_api_key="emb-key", embedding_dim=16)
        embedder = build_embedder(config, client=object())
        assert isinstance(embedder, OpenAICompatEmbedder)
        assert embedder.dim == 16

    def test_falls_back_to_hash_without_key(self) -> None:
        embedder = build_embedder(Config(embedding_api_key="", llm_api_key=""))
        assert isinstance(embedder, HashEmbedder)
        assert embedder.degraded is True

    def test_no_fallback_raises_without_key(self) -> None:
        with pytest.raises(RAGError, match="EMBEDDING_API_KEY"):
            build_embedder(Config(embedding_api_key="", llm_api_key=""), allow_fallback=False)

    def test_reuses_llm_key_when_same_gateway(self) -> None:
        config = Config(
            embedding_api_key="",
            embedding_base_url="http://gateway/v1",
            llm_base_url="http://gateway/v1/",
            llm_api_key="shared-key",
        )
        assert isinstance(build_embedder(config, client=object()), OpenAICompatEmbedder)

    def test_does_not_reuse_llm_key_across_gateways(self) -> None:
        config = Config(
            embedding_api_key="",
            embedding_base_url="http://dashscope/v1",
            llm_base_url="http://modelscope/v1",
            llm_api_key="another-key",
        )
        assert isinstance(build_embedder(config), HashEmbedder)

    def test_returns_base_embedder_type(self) -> None:
        assert isinstance(build_embedder(Config(embedding_api_key="")), BaseEmbedder)
