"""Reviewer / Security 角色测试。

两个角色共用 :class:`FocusedReviewAgent`，只换系统提示与关注面。
本文件要证明"共用基类"没有把 W5 的两条硬约束丢掉：
- 未收敛（达到最大轮次）→ ``success=False``；
- 结论不可解析 → 报告带 ``parse_error``，不能被当成"没问题"。
另外确认关注面与工具白名单确实写进了请求。
"""

from __future__ import annotations

import json

import pytest

from codeagentx.agents.reviewer import FOCUSED_REVIEW_TOOLS, ReviewerAgent
from codeagentx.agents.schemas import ReviewReport
from codeagentx.agents.security import SecurityAgent
from codeagentx.core.exceptions import ToolNotFoundError
from codeagentx.core.llm import MockLLM
from codeagentx.tools.base import BaseTool
from codeagentx.tools.registry import ToolRegistry

REPORT_JSON = json.dumps(
    {
        "summary": "发现 1 个高危问题。",
        "findings": [
            {
                "title": "SQL 字符串拼接",
                "file": "app/db/query.py",
                "line": 12,
                "severity": "high",
                "category": "security",
                "description": "用户输入直接拼进 SQL",
                "suggestion": "改用参数化查询",
                "confidence": 0.85,
            }
        ],
    },
    ensure_ascii=False,
)

TOOL_CALL = {"tool_calls": [{"id": "c1", "name": "code_search", "arguments": {"query": "登录"}}]}


class _StubTool(BaseTool):
    """只提供名字的占位工具，用于验证工具白名单。"""

    description = "测试替身工具。"
    parameters = []

    def __init__(self, name: str):
        self.name = name
        super().__init__()

    def _run(self) -> str:
        return "ok"


def focused_registry(*names: str) -> ToolRegistry:
    registry = ToolRegistry(name="roles-test")
    for name in names:
        registry.register(_StubTool(name))
    return registry


class TestReviewer:
    def test_identity_and_focus(self):
        agent = ReviewerAgent(MockLLM([REPORT_JSON]))
        assert agent.name == "reviewer"
        assert "逻辑正确性" in agent.focus
        assert "可维护性" in agent.focus

    def test_system_prompt_is_the_reviewer_contract(self):
        llm = MockLLM([REPORT_JSON])
        ReviewerAgent(llm).run(target="repo")
        system = llm.calls[0]["messages"][0]["content"]

        assert "正确性与可维护性" in system
        assert "JSON" in system  # 契约必须随提示一起下发

    def test_focus_and_evidence_reach_the_prompt(self):
        llm = MockLLM([REPORT_JSON])
        ReviewerAgent(llm).run(target="repo", evidence="[证据 1] a.py:1-2", hints="重点看异常处理")

        user = llm.calls[0]["messages"][1]["content"]
        assert "本轮重点关注" in user
        assert "逻辑正确性" in user
        assert "[证据 1] a.py:1-2" in user
        assert "仍需核实" in user
        assert "重点看异常处理" in user

    def test_report_source_is_the_role_name(self):
        report = ReviewerAgent(MockLLM([REPORT_JSON])).review(target="repo")

        assert isinstance(report, ReviewReport)
        assert report.findings[0].source == "reviewer"
        assert report.total == 1

    def test_only_read_only_tools_are_exposed(self):
        llm = MockLLM([REPORT_JSON])
        registry = focused_registry("code_search", "terminal", "git", "static_analyzer", "echo")
        ReviewerAgent(llm, tools=registry).run(target="repo")

        names = sorted(item["function"]["name"] for item in llm.calls[0]["tools"])
        assert names == ["code_search", "static_analyzer", "terminal"]
        assert "git" not in names  # 写操作类工具一律不放到审查角色手里

    def test_no_tools_means_no_tool_schemas(self):
        llm = MockLLM([REPORT_JSON])
        ReviewerAgent(llm, tools=None).run(target="repo")
        assert llm.calls[0]["tools"] is None


class TestSecurity:
    def test_identity_and_focus(self):
        agent = SecurityAgent(MockLLM([REPORT_JSON]))
        assert agent.name == "security"
        assert "注入" in agent.focus
        assert "凭据" in agent.focus

    def test_system_prompt_is_the_security_contract(self):
        llm = MockLLM([REPORT_JSON])
        SecurityAgent(llm).run(target="repo")
        system = llm.calls[0]["messages"][0]["content"]

        assert "安全审查专家" in system
        assert "bandit" in system
        assert "JSON" in system

    def test_report_source_is_security(self):
        report = SecurityAgent(MockLLM([REPORT_JSON])).review(target="repo")
        assert report.findings[0].source == "security"

    def test_default_task_mentions_attack_control(self):
        llm = MockLLM([REPORT_JSON])
        SecurityAgent(llm).run(target="repo")
        assert "攻击者控制" in llm.calls[0]["messages"][1]["content"]


class TestSharedHardConstraints:
    """两条硬约束必须对两个角色一视同仁。"""

    def test_not_converged_marks_failure(self):
        for agent_cls in (ReviewerAgent, SecurityAgent):
            llm = MockLLM([TOOL_CALL], default_response=TOOL_CALL)
            agent = agent_cls(
                llm, tools=focused_registry(*FOCUSED_REVIEW_TOOLS), max_iterations=2
            )
            result = agent.run(target="repo")

            assert result.success is False, agent_cls.__name__
            assert "最大迭代轮次" in result.error

    def test_missing_whitelisted_tool_raises_the_registry_error(self):
        """工具白名单写的是注册表里没有的工具时，必须显式报错而不是静默降级。

        编排层会把这个异常记成阶段 ``failed``（见 test_workflow 的对应用例），
        也就是"少挂了一个工具"一定会被发现。
        """
        agent = ReviewerAgent(MockLLM([REPORT_JSON]), tools=focused_registry("code_search"))
        with pytest.raises(ToolNotFoundError):
            agent.run(target="repo")

    def test_unparsable_output_is_marked(self):
        for agent_cls in (ReviewerAgent, SecurityAgent):
            result = agent_cls(MockLLM(["我觉得还行，就不给 JSON 了。"])).run(target="repo")
            report = result.metadata["report"]

            assert "parse_error" in report["metadata"], agent_cls.__name__
            assert report["total"] == 0
            assert "未产出有效结论" in report["summary"]

    def test_shared_tool_whitelist(self):
        assert FOCUSED_REVIEW_TOOLS == ("code_search", "static_analyzer", "terminal")

    def test_history_is_reset_between_runs(self):
        agent = ReviewerAgent(MockLLM([REPORT_JSON, REPORT_JSON]))
        agent.run(target="repo")
        agent.run(target="repo")
        assert len(agent.history) == 2
