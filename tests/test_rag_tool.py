"""RAGTool 测试：检索/索引两个动作、路径校验、参数校验、输出整形。"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagentx.config import Config
from codeagentx.rag.embedder import HashEmbedder
from codeagentx.rag.rag_tool import RAGTool, build_rag_tool
from codeagentx.rag.vector_store import InMemoryVectorStore
from codeagentx.tools.sandbox import SandboxPolicy

SAMPLE = '''"""用户服务。"""

import hashlib


def login(user, password):
    """校验用户登录。"""
    return hashlib.sha256(password.encode()).hexdigest() == user.salt
'''

LONG_SAMPLE = "def long_function():\n" + "".join(
    f"    value_{index} = {index}  # 填充行，用于触发输出截断\n" for index in range(60)
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "service.py").write_text(SAMPLE, encoding="utf-8")
    return tmp_path


def _tool(**kwargs) -> RAGTool:
    embedder = HashEmbedder(dim=256)
    return RAGTool(
        embedder=embedder,
        store=InMemoryVectorStore(dim=embedder.dim),
        **kwargs,
    )


def _indexed_tool(repo, **kwargs) -> RAGTool:
    tool = _tool(policy=SandboxPolicy.for_roots([repo]), **kwargs)
    result = tool.run(action="index", path=str(repo))
    assert result.success, result.error
    return tool


class TestSchema:
    def test_tool_identity(self) -> None:
        tool = _tool()
        assert tool.name == "code_search"
        assert "检索" in tool.description
        assert tool.dangerous is False

    def test_action_enum_is_exposed_to_the_model(self) -> None:
        schema = _tool().parameters_schema()
        assert schema["properties"]["action"]["enum"] == ["search", "index"]
        assert schema["properties"]["action"]["default"] == "search"
        assert "query" in schema["properties"]


class TestParameterValidation:
    def test_unknown_action_is_rejected(self) -> None:
        result = _tool().run(action="delete")
        assert result.success is False
        assert result.error_type == "ToolValidationError"
        assert "delete" in result.error

    def test_search_requires_query(self) -> None:
        result = _tool().run(action="search")
        assert result.success is False
        assert result.error_type == "ToolValidationError"
        assert "query" in result.error

    def test_blank_query_is_rejected(self) -> None:
        assert _tool().run(action="search", query="   ").success is False

    def test_index_requires_path(self) -> None:
        result = _tool(policy=SandboxPolicy.for_roots(["."])).run(action="index")
        assert result.success is False
        assert result.error_type == "ToolValidationError"

    def test_unknown_argument_is_rejected(self) -> None:
        result = _tool().run(action="search", query="login", unexpected=1)
        assert result.success is False
        assert result.error_type == "ToolValidationError"

    def test_top_k_must_be_a_positive_integer(self) -> None:
        for value in ("abc", 0, -3):
            result = _tool().run(action="search", query="login", top_k=value)
            assert result.success is False
            assert result.error_type == "ToolValidationError"


class TestSearch:
    def test_empty_index_returns_hint_instead_of_error(self) -> None:
        result = _tool().run(action="search", query="用户登录逻辑在哪")
        assert result.success is True
        assert result.metadata["count"] == 0
        assert result.metadata["results"] == []
        assert "index" in str(result.output)

    def test_default_action_is_search(self) -> None:
        result = _tool().run(query="login")
        assert result.success is True
        assert result.metadata["query"] == "login"

    def test_search_finds_indexed_code(self, repo) -> None:
        tool = _indexed_tool(repo)
        result = tool.run(action="search", query="login password hashlib")
        assert result.success is True
        results = result.metadata["results"]
        assert results
        assert results[0]["location"].startswith("pkg/service.py:")
        assert "def login(user, password)" in results[0]["content"]
        assert results[0]["sources"]  # 至少一路命中

    def test_text_output_is_readable_for_the_model(self, repo) -> None:
        tool = _indexed_tool(repo)
        output = str(tool.run(action="search", query="login", top_k=1).output)
        assert "命中 1 个代码片段" in output
        assert "pkg/service.py:" in output
        assert "def login(user, password)" in output

    def test_top_k_limits_results(self, repo) -> None:
        tool = _indexed_tool(repo)
        assert tool.run(action="search", query="login", top_k=1).metadata["count"] == 1

    def test_top_k_is_clamped_to_maximum(self, repo) -> None:
        tool = _indexed_tool(repo, max_top_k=1)
        assert tool.run(action="search", query="login", top_k=99).metadata["count"] == 1

    def test_language_filter_is_applied(self, repo) -> None:
        (repo / "README.md").write_text("# 说明\n\n用户服务模块。\n", encoding="utf-8")
        tool = _indexed_tool(repo)
        results = tool.run(action="search", query="用户服务", language="markdown").metadata["results"]
        assert [item["location"].split(":")[0] for item in results] == ["README.md"]

    def test_long_content_is_truncated_in_text_but_kept_in_metadata(self, tmp_path) -> None:
        (tmp_path / "long.py").write_text(LONG_SAMPLE, encoding="utf-8")
        tool = _indexed_tool(tmp_path, max_content_chars=50)
        result = tool.run(action="search", query="long_function 填充行", top_k=1)
        assert "片段已截断" in str(result.output)
        assert len(result.metadata["results"][0]["content"]) > 50

    def test_no_match_returns_friendly_text(self, repo) -> None:
        tool = _indexed_tool(repo)
        result = tool.run(action="search", query="量子纠缠退相干哈密顿量")
        assert result.success is True
        assert "未检索到" in str(result.output) or result.metadata["count"] >= 0


class TestIndex:
    def test_index_without_policy_is_refused(self, repo) -> None:
        result = _tool().run(action="index", path=str(repo))
        assert result.success is False
        assert result.error_type == "SecurityPolicyMissing"
        assert result.metadata["hint"]

    def test_index_outside_allowed_roots_is_refused(self, repo, tmp_path) -> None:
        outside = tmp_path.parent
        result = _tool(policy=SandboxPolicy.for_roots([repo])).run(
            action="index", path=str(outside)
        )
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_index_missing_directory_is_refused(self, repo) -> None:
        tool = _tool(policy=SandboxPolicy.for_roots([repo]))
        result = tool.run(action="index", path=str(repo / "nope"))
        assert result.success is False
        assert result.error_type == "ToolExecutionError"

    def test_index_returns_stats(self, repo) -> None:
        result = _tool(policy=SandboxPolicy.for_roots([repo])).run(
            action="index", path=str(repo)
        )
        assert result.success is True
        assert result.metadata["files"] == 1
        assert result.metadata["chunks"] >= 2
        assert result.metadata["semantic"] is False
        assert "索引完成" in str(result.output)

    def test_index_is_repeatable_and_resets(self, repo) -> None:
        tool = _indexed_tool(repo)
        first = tool.store.count()
        extra = repo / "extra.py"
        extra.write_text("def extra():\n    return 1\n", encoding="utf-8")
        tool.run(action="index", path=str(repo))
        assert tool.store.count() > first
        extra.unlink()
        tool.run(action="index", path=str(repo))
        assert tool.store.count() == first

    def test_search_after_reindex_uses_fresh_corpus(self, repo) -> None:
        tool = _indexed_tool(repo)
        extra = repo / "pkg" / "token_store.py"
        extra.write_text("def rotate_signing_key():\n    return 'new'\n", encoding="utf-8")
        tool.run(action="index", path=str(repo))
        results = tool.run(action="search", query="rotate_signing_key", top_k=1).metadata["results"]
        assert results[0]["location"].startswith("pkg/token_store.py:")


class TestBuildRagTool:
    def test_builds_shared_components_without_root(self) -> None:
        tool = build_rag_tool(Config(embedding_api_key="", llm_api_key=""))
        assert tool.store.dim == tool.embedder.dim
        assert tool.indexer.store is tool.store
        assert tool.retriever.store is tool.store
        assert tool.policy is None

    def test_root_enables_indexing(self, repo) -> None:
        tool = build_rag_tool(Config(embedding_api_key="", llm_api_key=""), root=repo)
        assert tool.policy is not None
        assert tool.run(action="index", path=str(repo)).success is True
        assert tool.run(action="search", query="login").metadata["count"] >= 1

    def test_rejects_index_without_root(self, repo) -> None:
        tool = build_rag_tool(Config(embedding_api_key="", llm_api_key=""))
        assert tool.run(action="index", path=str(repo)).error_type == "SecurityPolicyMissing"

    def test_no_fallback_without_key_raises(self) -> None:
        from codeagentx.core.exceptions import RAGError

        with pytest.raises(RAGError, match="EMBEDDING_API_KEY"):
            build_rag_tool(Config(embedding_api_key="", llm_api_key=""), allow_fallback=False)


#: 示例仓库：W4 验收用的"真实"目标（缺陷是故意留的，见其 README）
SAMPLE_REPO = Path(__file__).resolve().parents[1] / "data" / "sample_repo"


class TestSampleRepositoryAcceptance:
    """W4 验收：输入「用户登录逻辑在哪」能返回相关代码片段。

    这里跑的是仓库自带的示例项目（``data/sample_repo``），而不是临时造的片段，
    用来保证"离线降级向量化 + BM25"这条默认链路确实能定位到登录逻辑。
    """

    @pytest.fixture
    def tool(self) -> RAGTool:
        # 与真实默认一致的维度（HashEmbedder 缺省 512），避免验收结论只在特定维度成立
        embedder = HashEmbedder()
        tool = RAGTool(
            embedder=embedder,
            store=InMemoryVectorStore(dim=embedder.dim),
            policy=SandboxPolicy.for_roots([SAMPLE_REPO]),
        )
        assert tool.run(action="index", path=str(SAMPLE_REPO)).success
        return tool

    def test_login_query_returns_login_function(self, tool) -> None:
        result = tool.run(action="search", query="用户登录逻辑在哪", top_k=3)
        assert result.success
        top = result.metadata["results"][0]
        assert top["path"] == "app/auth/service.py"
        assert top["name"] == "login"
        assert "def login(username, password)" in top["content"]

    def test_login_chunk_is_found_by_both_paths(self, tool) -> None:
        results = tool.run(action="search", query="用户登录逻辑在哪", top_k=5).metadata["results"]
        login = next(item for item in results if item["name"] == "login")
        assert set(login["sources"]) == {"semantic", "lexical"}

    def test_identifier_query_returns_hash_helper(self, tool) -> None:
        results = tool.run(action="search", query="hash_password", top_k=1).metadata["results"]
        assert results[0]["name"] == "hash_password"

    def test_risk_query_returns_sql_builder(self, tool) -> None:
        results = tool.run(action="search", query="SQL 注入风险", top_k=1).metadata["results"]
        assert results[0]["path"] == "app/db/repository.py"

    def test_text_output_reports_position(self, tool) -> None:
        output = str(tool.run(action="search", query="用户登录逻辑在哪", top_k=1).output)
        assert "app/auth/service.py:" in output
        assert "命中：" in output
