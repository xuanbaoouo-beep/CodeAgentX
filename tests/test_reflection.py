"""Reflection 测试：评审解析容错、终止条件、以及审查场景下的端到端自我优化。

最关键的两条不变量：
1. Reflection 必须能停下来（accept / 打转 / 用完轮次三条出口）；
2. 评审输出不可解析时按"需要修订"处理，绝不乐观地当成通过。
"""

from __future__ import annotations

import json

import pytest

from codeagentx.agents.reflection import (
    Critique,
    Reflection,
    ReflectionReviewer,
    parse_critique,
)
from codeagentx.core.llm import MockLLM

DRAFT = json.dumps(
    {
        "summary": "初稿：发现密钥硬编码。",
        "findings": [
            {
                "title": "硬编码密钥",
                "file": "app/auth/service.py",
                "line": 8,
                "severity": "low",
                "category": "security",
                "description": "SECRET_KEY 写在源码里",
            }
        ],
    },
    ensure_ascii=False,
)

REVISED = json.dumps(
    {
        "summary": "修订稿：严重度修正为 high，并补充密钥轮换建议。",
        "findings": [
            {
                "title": "硬编码密钥",
                "file": "app/auth/service.py",
                "line": 8,
                "severity": "high",
                "category": "security",
                "description": "SECRET_KEY 写在源码里",
                "suggestion": "改为从环境变量读取并支持轮换",
            }
        ],
    },
    ensure_ascii=False,
)

CRITIQUE_REVISE = json.dumps(
    {
        "score": 0.4,
        "issues": ["严重度明显偏低"],
        "missing": ["未提及密钥轮换"],
        "verdict": "revise",
    },
    ensure_ascii=False,
)

CRITIQUE_ACCEPT = json.dumps(
    {"score": 0.92, "issues": [], "missing": [], "verdict": "accept"}, ensure_ascii=False
)

TOOL_CALL = {"tool_calls": [{"id": "c1", "name": "echo", "arguments": {"text": "x"}}]}


class TestCritique:
    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            ("accept", "accept"),
            ("ACCEPT", "accept"),
            ("pass", "accept"),
            ("接受", "accept"),
            ("revise", "revise"),
            ("随便什么", "revise"),
            ("", "revise"),
        ],
    )
    def test_verdict_normalization(self, verdict, expected):
        assert Critique(verdict=verdict).verdict == expected
        assert Critique(verdict=verdict).should_revise is (expected == "revise")

    def test_fields_are_normalized(self):
        critique = Critique(score="85%", issues="a,b", missing=["c"])
        assert critique.score == 0.85
        assert critique.issues == ["a", "b"]
        assert critique.missing == ["c"]

    def test_text_lists_issues(self):
        text = Critique(score=0.4, issues=["严重度偏低"], missing=["缺少轮换"]).to_text()
        assert "评分：0.4" in text
        assert "- 严重度偏低" in text
        assert "- 缺少轮换" in text


class TestParseCritique:
    def test_standard_payload(self):
        critique = parse_critique(CRITIQUE_REVISE)
        assert critique.score == 0.4
        assert critique.verdict == "revise"
        assert critique.missing == ["未提及密钥轮换"]
        assert critique.raw == CRITIQUE_REVISE

    def test_unparsable_is_treated_as_revise(self):
        """评审解析不出来时按"需要修订"处理，绝不能当成通过。"""
        critique = parse_critique("我觉得还行吧")
        assert critique.verdict == "revise"
        assert critique.issues  # 至少留一条说明
        assert critique.score == 0.0

    def test_empty_output_has_explanatory_issue(self):
        critique = parse_critique("   ")
        assert critique.issues == ["评审输出为空"]

    def test_verdict_inferred_from_score_when_absent(self):
        assert parse_critique('{"score": 0.95}').verdict == "accept"
        assert parse_critique('{"score": 0.2}').verdict == "revise"

    def test_bare_list_becomes_issues(self):
        critique = parse_critique('["行号可能不对", "建议太笼统"]')
        assert critique.issues == ["行号可能不对", "建议太笼统"]
        assert critique.verdict == "revise"

    def test_alias_keys(self):
        critique = parse_critique('{"rating": 0.3, "problems": ["x"], "decision": "revise"}')
        assert critique.score == 0.3
        assert critique.issues == ["x"]

    def test_roundtrip(self):
        critique = Critique(score=0.5, issues=["a"], missing=["b"], verdict="revise")
        assert Critique.from_dict(critique.to_dict()).to_dict() == critique.to_dict()


class TestReflectionStopping:
    def test_disabled_when_zero_rounds(self):
        result = Reflection(MockLLM([]), max_rounds=0).improve("t", DRAFT)
        assert result.stopped_reason == "disabled"
        assert result.rounds == []
        assert result.output == DRAFT
        assert result.improved is False

    def test_accept_stops_immediately(self):
        llm = MockLLM([CRITIQUE_ACCEPT])
        result = Reflection(llm, max_rounds=3).improve("t", DRAFT)

        assert result.stopped_reason == "accepted"
        assert result.rounds_used == 1
        assert result.improved is False
        assert result.final_score == 0.92
        assert result.output == DRAFT
        assert len(llm.calls) == 1  # 通过就不再修订

    def test_score_threshold_also_stops(self):
        llm = MockLLM(['{"score": 0.9, "verdict": "revise"}'])
        result = Reflection(llm, max_rounds=2, accept_score=0.85).improve("t", DRAFT)
        assert result.stopped_reason == "accepted"

    def test_revise_then_hit_round_limit(self):
        llm = MockLLM([CRITIQUE_REVISE, REVISED])
        result = Reflection(llm, max_rounds=1).improve("t", DRAFT)

        assert result.stopped_reason == "max_rounds"
        assert result.improved is True
        assert result.output == REVISED
        assert result.final_score == 0.4

    def test_stops_when_model_does_not_change_anything(self):
        llm = MockLLM([CRITIQUE_REVISE, DRAFT])
        result = Reflection(llm, max_rounds=5).improve("t", DRAFT)

        assert result.stopped_reason == "no_change"
        assert result.rounds_used == 1
        assert result.improved is False

    def test_two_rounds_until_accept(self):
        llm = MockLLM([CRITIQUE_REVISE, REVISED, CRITIQUE_ACCEPT])
        result = Reflection(llm, max_rounds=2).improve("t", DRAFT)

        assert result.stopped_reason == "accepted"
        assert result.rounds_used == 2
        assert result.final_score == 0.92
        assert len(llm.calls) == 3  # critique + revise + critique

    def test_rounds_record_drafts_and_changes(self):
        llm = MockLLM([CRITIQUE_REVISE, REVISED])
        result = Reflection(llm, max_rounds=1).improve("t", DRAFT)

        record = result.rounds[0]
        assert record.draft == DRAFT
        assert record.revised == REVISED
        assert record.changed is True
        payload = result.to_dict(include_text=True)
        assert payload["rounds"][0]["draft"] == DRAFT

    @pytest.mark.parametrize("kwargs", [{"max_rounds": -1}, {"accept_score": 0.0}, {"accept_score": 1.5}])
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            Reflection(MockLLM([]), **kwargs)

    def test_unparsable_critique_triggers_revision(self):
        """评审输出坏掉时必须继续修订，而不是误判为通过。"""
        llm = MockLLM(["我拒绝输出 JSON", REVISED])
        result = Reflection(llm, max_rounds=1).improve("t", DRAFT)

        assert result.rounds[0].critique.verdict == "revise"
        assert result.output == REVISED


class TestReflectionReviewer:
    def test_full_flow_without_tools(self):
        llm = MockLLM([DRAFT, CRITIQUE_REVISE, REVISED])
        agent = ReflectionReviewer(llm, tools=None, reflection_rounds=1)
        result = agent.run("审查登录逻辑", target="app/auth/service.py", use_tools=False)

        assert result.success is True
        assert result.output == REVISED

        report = result.metadata["report"]
        assert report["findings"][0]["severity"] == "high"  # 修订后的结论
        assert result.metadata["draft_report"]["findings"][0]["severity"] == "low"  # 初稿
        assert result.metadata["reflection"]["improved"] is True
        assert result.metadata["reflection"]["stopped_reason"] == "max_rounds"

    def test_review_helper_returns_report(self):
        llm = MockLLM([DRAFT, CRITIQUE_REVISE, REVISED])
        agent = ReflectionReviewer(llm, tools=None, reflection_rounds=1)
        report = agent.review("审查", target="a.py", use_tools=False)

        assert report.total == 1
        assert report.findings[0].severity == "high"
        assert report.findings[0].source == "reflection_reviewer"

    def test_no_revision_when_accepted(self):
        llm = MockLLM([DRAFT, CRITIQUE_ACCEPT])
        agent = ReflectionReviewer(llm, tools=None, reflection_rounds=1)
        result = agent.run("审查", use_tools=False)

        assert result.output == DRAFT
        assert result.metadata["reflection"]["improved"] is False
        assert result.metadata["reflection"]["stopped_reason"] == "accepted"

    def test_draft_uses_tools_when_available(self, registry):
        llm = MockLLM([TOOL_CALL, DRAFT, CRITIQUE_REVISE, REVISED])
        agent = ReflectionReviewer(llm, tools=registry, reflection_rounds=1)
        result = agent.run("审查", target="a.py")

        assert result.success is True
        assert len(result.tool_calls) == 1
        assert result.metadata["report"]["metadata"]["tools_used"] == ["echo"]
        assert result.iterations == 2  # 初稿阶段两轮

    def test_unparsable_final_report_is_a_failure(self):
        llm = MockLLM(["乱七八糟的初稿", CRITIQUE_ACCEPT])
        agent = ReflectionReviewer(llm, tools=None, reflection_rounds=1)
        result = agent.run("审查", use_tools=False)

        assert result.success is False
        assert "无法解析" in result.error

    def test_broken_revision_falls_back_to_draft(self):
        """修订稿坏掉（真实模型偶发）时退回初稿，并如实标注这次反思被拒收。"""
        llm = MockLLM([DRAFT, CRITIQUE_REVISE, "修订到一半就断了 {"])
        agent = ReflectionReviewer(llm, tools=None, reflection_rounds=1)
        result = agent.run("审查", target="a.py", use_tools=False)

        assert result.success is True
        assert result.output == DRAFT  # 交付的是初稿，而不是半截修订稿

        report = result.metadata["report"]
        assert report["findings"][0]["severity"] == "low"  # 初稿结论被保住
        assert "revision_rejected" in report["metadata"]
        assert report["metadata"]["rejected_output"]  # 坏样本留痕，便于回溯

    def test_reflection_can_be_disabled(self):
        llm = MockLLM([DRAFT])
        agent = ReflectionReviewer(llm, tools=None, reflection_rounds=0)
        result = agent.run("审查", use_tools=False)

        assert result.output == DRAFT
        assert result.metadata["reflection"]["stopped_reason"] == "disabled"
        assert len(llm.calls) == 1

    def test_usage_is_accumulated_across_phases(self):
        llm = MockLLM([DRAFT, CRITIQUE_REVISE, REVISED])
        agent = ReflectionReviewer(llm, tools=None, reflection_rounds=1)
        result = agent.run("审查", use_tools=False)

        assert result.usage["calls"] == 3
        assert result.usage["total_tokens"] > 0
