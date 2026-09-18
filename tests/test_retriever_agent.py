"""Retriever 角色测试。

这个角色刻意不用 LLM，因此测试的重点是**确定性**：
- 片段到 ``Evidence`` 的字段映射（含鸭子类型替身）；
- 去重合并命中来源、按分数稳定排序；
- 「部分查询失败」与「全部查询失败」必须区分；
- 线索文本按预算截断，不能把审查角色的上下文撑爆。
"""

from __future__ import annotations

import pytest

from codeagentx.agents.retriever import (
    DEFAULT_DIGEST_CHARS,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    RetrieverAgent,
)
from codeagentx.agents.schemas import Evidence
from codeagentx.core.agent import ZERO_USAGE
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.registry import ToolRegistry


def chunk(
    path: str,
    start: int,
    end: int,
    *,
    score: float = 1.0,
    name: str = "login",
    sources: tuple[str, ...] = ("semantic",),
    content: str = "def login(): ...",
) -> dict:
    """构造与 RAGTool 返回结构一致的片段 dict。"""
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "kind": "function",
        "name": name,
        "score": score,
        "sources": list(sources),
        "content": content,
    }


class FakeSearchTool(BaseTool):
    """``code_search`` 测试替身：返回预设片段，可指定失败的查询词。"""

    name = "code_search"
    description = "测试替身：返回预设的检索结果。"
    parameters = [
        ToolParameter(name="action", description="search 或 index", required=False, default="search"),
        ToolParameter(name="query", description="查询词", required=False, default=""),
        ToolParameter(name="top_k", description="返回条数", required=False, default=DEFAULT_TOP_K),
        ToolParameter(name="path", description="索引路径", required=False, default=""),
    ]

    def __init__(self, chunks: list[dict] | None = None, *, fail_queries=(), index_ok: bool = True):
        super().__init__()
        self.chunks = list(chunks or [])
        self.fail_queries = set(fail_queries)
        self.index_ok = index_ok
        self.calls: list[dict] = []

    def _run(self, action: str = "search", query: str = "", top_k: int = DEFAULT_TOP_K, path: str = ""):
        self.calls.append({"action": action, "query": query, "top_k": top_k, "path": path})
        if action == "index":
            if not self.index_ok:
                return ToolResult.fail("索引目标不是目录")
            return ToolResult.ok("索引完成：2 个代码块", chunks=2)
        if query in self.fail_queries:
            return ToolResult.fail(f"检索「{query}」失败")
        results = [dict(item) for item in self.chunks]
        return ToolResult.ok(
            f"检索「{query}」命中 {len(results)} 个代码片段",
            results=results,
            count=len(results),
        )


def build(tool: FakeSearchTool, **kwargs) -> RetrieverAgent:
    registry = ToolRegistry(name="retriever-test")
    registry.register(tool)
    return RetrieverAgent(registry, **kwargs)


class TestEvidenceMapping:
    def test_from_chunk_maps_dict_fields(self):
        item = Evidence.from_chunk(chunk("app/auth/service.py", 10, 20, score=0.75), query="登录")

        assert item.path == "app/auth/service.py"
        assert item.location == "app/auth/service.py:10-20"
        assert item.symbol == "login"
        assert item.snippet == "def login(): ..."
        assert item.query == "登录"
        assert item.score == 0.75
        assert item.sources == ["semantic"]

    def test_from_chunk_accepts_duck_typed_object(self):
        """用鸭子类型而不是 import RetrievedChunk，方便注入轻量替身。"""

        class FakeChunk:
            path = "a.py"
            start_line = 3
            end_line = 4
            kind = "class"
            symbol = "Service"
            sources = ("semantic", "lexical")
            score = 0.5
            content = "class Service: ..."

        item = Evidence.from_chunk(FakeChunk(), query="服务")

        assert item.path == "a.py"
        assert item.symbol == "Service"
        assert item.sources == ["semantic", "lexical"]
        assert item.query == "服务"

    def test_location_without_line_numbers_is_path_only(self):
        assert Evidence(path="a.py").location == "a.py"
        assert Evidence().location == ""

    def test_to_text_carries_position_and_source(self):
        text = Evidence.from_chunk(chunk("a.py", 1, 2, content="x = 1")).to_text()
        assert "[a.py:1-2] function login" in text
        assert "命中：semantic" in text
        assert "x = 1" in text


class TestRetrieve:
    def test_single_query_returns_evidence(self):
        tool = FakeSearchTool([chunk("a.py", 1, 5)])
        evidence, errors = build(tool).retrieve("登录")

        assert errors == []
        assert len(evidence) == 1
        assert tool.calls[0]["top_k"] == DEFAULT_TOP_K

    def test_duplicate_hits_are_merged_with_combined_sources(self):
        """同一位置的片段只保留一条：来源合并、分数取高。"""
        tool = FakeSearchTool(
            [
                chunk("a.py", 1, 5, sources=("semantic",), score=0.4),
                chunk("a.py", 1, 5, sources=("lexical",), score=0.6),
            ]
        )
        evidence, errors = build(tool).retrieve("登录")

        assert errors == []
        assert len(evidence) == 1
        assert evidence[0].sources == ["semantic", "lexical"]
        assert evidence[0].score == 0.6

    def test_run_dedupes_across_queries(self):
        tool = FakeSearchTool([chunk("a.py", 1, 5)])
        result = build(tool).run(["登录", "认证"])

        assert len(tool.calls) == 2
        assert result.metadata["count"] == 1
        assert result.metadata["queries"] == ["登录", "认证"]

    def test_evidence_is_sorted_by_score_then_position(self):
        tool = FakeSearchTool(
            [
                chunk("b.py", 1, 2, score=0.2),
                chunk("a.py", 9, 10, score=0.9),
                chunk("a.py", 1, 2, score=0.9),
            ]
        )
        result = build(tool).run("查询")
        paths = [(item["path"], item["start_line"]) for item in result.metadata["evidence"]]

        assert paths == [("a.py", 1), ("a.py", 9), ("b.py", 1)]

    def test_string_query_is_accepted(self):
        tool = FakeSearchTool([chunk("a.py", 1, 2)])
        result = build(tool).run("单条查询")
        assert result.metadata["queries"] == ["单条查询"]

    def test_top_k_is_clamped_to_the_tool_limit(self):
        tool = FakeSearchTool([chunk("a.py", 1, 2)])
        build(tool, top_k=100).retrieve("查询")
        assert tool.calls[0]["top_k"] == MAX_TOP_K

    def test_top_k_must_be_positive(self):
        with pytest.raises(ValueError):
            build(FakeSearchTool(), top_k=0)


class TestFailureSemantics:
    def test_partial_failure_is_not_a_step_failure(self):
        """一个查询失败、另一个成功：这一步仍然算成功（检索是尽力而为）。"""
        tool = FakeSearchTool([chunk("a.py", 1, 2)], fail_queries=["坏查询"])
        result = build(tool).run(["坏查询", "好查询"])

        assert result.success is True
        assert result.metadata["errors"] and "坏查询" in result.metadata["errors"][0]
        assert result.metadata["count"] == 1

    def test_all_queries_failing_marks_failure(self):
        tool = FakeSearchTool(fail_queries=["a", "b"])
        result = build(tool).run(["a", "b"])

        assert result.success is False
        assert result.error
        assert result.metadata["count"] == 0

    def test_no_hit_is_a_normal_result(self):
        result = build(FakeSearchTool()).run("查不到")

        assert result.success is True
        assert result.metadata["count"] == 0
        assert "未检索到" in result.output

    def test_index_failure_is_reported_in_notes(self):
        tool = FakeSearchTool(index_ok=False)
        result = build(tool).run("查询", index_root="not-a-dir")

        assert result.metadata["notes"] and "索引失败" in result.metadata["notes"][0]
        assert result.success is True  # 检索仍可能命中已有索引

    def test_index_is_built_when_root_is_given(self):
        tool = FakeSearchTool([chunk("a.py", 1, 2)])
        result = build(tool).run("查询", index_root="data/sample_repo")

        assert tool.calls[0]["action"] == "index"
        assert tool.calls[0]["path"] == "data/sample_repo"
        assert result.success is True


class TestDigest:
    def test_digest_lists_positions_in_order(self):
        tool = FakeSearchTool([chunk("a.py", 1, 2), chunk("b.py", 5, 6, score=0.3)])
        agent = build(tool)
        result = agent.run("查询")

        assert "[证据 1]" in result.output
        assert result.output.index("a.py") < result.output.index("b.py")

    def test_digest_respects_char_budget(self):
        tool = FakeSearchTool(
            [chunk(f"f{index}.py", 1, 2, content="x" * 200) for index in range(10)]
        )
        result = build(tool, digest_chars=300).run("查询")

        assert "因篇幅省略" in result.output
        assert len(result.output) < 1000

    def test_empty_evidence_has_placeholder(self):
        assert "未检索到" in build(FakeSearchTool()).evidence_digest([])

    def test_default_budget_is_positive(self):
        assert DEFAULT_DIGEST_CHARS > 0


class TestUsage:
    def test_deterministic_role_reports_zero_usage(self):
        result = build(FakeSearchTool([chunk("a.py", 1, 2)])).run("查询")

        assert result.usage["calls"] == ZERO_USAGE["calls"]
        assert result.usage["total_tokens"] == 0
        assert result.metadata["agent"] == "retriever"
