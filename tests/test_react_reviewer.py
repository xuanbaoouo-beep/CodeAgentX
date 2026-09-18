"""ReAct 审查 Agent 测试。

覆盖三条关键路径：
1. 正常收敛（无工具直出 / 调工具后出结论）；
2. 未收敛（达到最大轮次，必须显式失败而不是假装成功）；
3. 输出不可解析（必须带 parse_error 标记，不能表现为"没问题"）。
"""

from __future__ import annotations

import json
from pathlib import Path

from codeagentx.agents.react_reviewer import ReActReviewer
from codeagentx.agents.schemas import ReviewReport
from codeagentx.agents.toolkit import build_review_toolkit
from codeagentx.core.exceptions import LLMError
from codeagentx.core.llm import MockLLM

SAMPLE_REPO = Path(__file__).resolve().parents[1] / "data" / "sample_repo"

REPORT_JSON = json.dumps(
    {
        "summary": "发现 1 个高危问题：密钥硬编码。",
        "findings": [
            {
                "title": "硬编码密钥",
                "file": "app/auth/service.py",
                "line": 8,
                "severity": "high",
                "category": "security",
                "description": "SECRET_KEY 直接写在源码中",
                "suggestion": "改为从环境变量读取",
                "confidence": 0.9,
            }
        ],
    },
    ensure_ascii=False,
)

TOOL_CALL = {"tool_calls": [{"id": "c1", "name": "echo", "arguments": {"text": "hi"}}]}


class TestDirectOutput:
    """模型不需要工具就能给出结论的情形。"""

    def test_parses_report_and_writes_metadata(self):
        agent = ReActReviewer(MockLLM([REPORT_JSON]), tools=None)
        result = agent.run("审查代码", target="app/auth/service.py")

        assert result.success is True
        assert result.iterations == 1
        report = result.metadata["report"]
        assert report["target"] == "app/auth/service.py"
        assert report["total"] == 1
        finding = report["findings"][0]
        assert finding["severity"] == "high"
        assert finding["source"] == "react_reviewer"
        assert result.metadata["agent"] == "react_reviewer"

    def test_review_returns_structured_report(self):
        agent = ReActReviewer(MockLLM([REPORT_JSON]), tools=None)
        report = agent.review("审查代码", target="a.py")

        assert isinstance(report, ReviewReport)
        assert report.total == 1
        assert report.findings[0].title == "硬编码密钥"
        assert report.severity_counts()["high"] == 1

    def test_task_target_and_hints_reach_the_prompt(self):
        llm = MockLLM([REPORT_JSON])
        agent = ReActReviewer(llm, tools=None)
        agent.run("找出安全问题", target="app/auth/service.py", hints="重点看密钥管理")

        user_content = llm.calls[0]["messages"][1]["content"]
        assert "app/auth/service.py" in user_content
        assert "重点看密钥管理" in user_content
        assert "找出安全问题" in user_content

    def test_system_prompt_is_injected(self):
        llm = MockLLM([REPORT_JSON])
        agent = ReActReviewer(llm, tools=None)
        agent.run("审查")

        assert llm.calls[0]["messages"][0]["role"] == "system"
        assert "ReAct" in llm.calls[0]["messages"][0]["content"]


class TestToolLoop:
    """工具调用链路。"""

    def test_tool_result_is_fed_back_before_final_answer(self, registry):
        llm = MockLLM([TOOL_CALL, REPORT_JSON])
        agent = ReActReviewer(llm, tools=registry)
        result = agent.run("审查")

        assert result.success is True
        assert result.iterations == 2
        assert [item["name"] for item in result.tool_calls] == ["echo"]
        assert [message.role.value for message in result.messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert result.metadata["report"]["metadata"]["tools_used"] == ["echo"]

    def test_tools_are_exposed_to_the_model(self, registry):
        llm = MockLLM([REPORT_JSON])
        agent = ReActReviewer(llm, tools=registry)
        agent.run("审查")

        names = sorted(item["function"]["name"] for item in llm.calls[0]["tools"])
        assert names == ["boom", "echo", "guarded"]

    def test_allowed_tools_narrows_exposure(self, registry):
        llm = MockLLM([REPORT_JSON])
        agent = ReActReviewer(llm, tools=registry, allowed_tools=["echo"])
        agent.run("审查")

        names = [item["function"]["name"] for item in llm.calls[0]["tools"]]
        assert names == ["echo"]

    def test_tool_failure_does_not_break_the_run(self, registry):
        """工具自身失败（这里 boom 抛异常）也应回填给模型，而不是击穿 Agent。"""
        llm = MockLLM(
            [
                {"tool_calls": [{"id": "c1", "name": "boom", "arguments": {}}]},
                REPORT_JSON,
            ]
        )
        agent = ReActReviewer(llm, tools=registry)
        result = agent.run("审查")

        assert result.success is True
        assert result.tool_calls[0]["success"] is False

    def test_no_tools_means_no_tool_schemas(self):
        llm = MockLLM([REPORT_JSON])
        ReActReviewer(llm, tools=None).run("审查")
        assert llm.calls[0]["tools"] is None


class TestNotConverged:
    def test_max_iterations_marks_failure(self, registry):
        llm = MockLLM([TOOL_CALL], default_response=TOOL_CALL)
        agent = ReActReviewer(llm, tools=registry, max_iterations=2)
        result = agent.run("审查")

        assert result.success is False
        assert "最大迭代轮次" in result.error
        assert result.iterations == 2

    def test_unparsable_output_is_marked_as_invalid(self):
        agent = ReActReviewer(MockLLM(["这段代码我觉得有问题，但我不想给 JSON。"]), tools=None)
        result = agent.run("审查")

        report = result.metadata["report"]
        assert report["total"] == 0
        assert "parse_error" in report["metadata"]
        assert "未产出有效结论" in report["summary"]


class TestJsonRepair:
    """结论解析不了时补问一次（真实故障：强制收敛后输出自由文本，占比约 5%）。"""

    def test_repair_recovers_findings_when_second_answer_is_json(self):
        llm = MockLLM(["这段代码我觉得有问题，但我不想给 JSON。", REPORT_JSON])
        agent = ReActReviewer(llm, tools=None)
        result = agent.run("审查", target="app/auth/service.py")

        report = result.metadata["report"]
        assert report["total"] == 1, "补问成功后问题不该丢"
        assert "parse_error" not in report["metadata"]
        assert report["metadata"]["json_repaired"] is True
        assert result.metadata["json_repaired"] is True
        assert result.output == REPORT_JSON, "下游该看到被采纳的那一份，而不是坏掉的初稿"

        assert len(llm.calls) == 2, "只补问一次"
        repair_call = llm.calls[1]
        assert repair_call["tools"] is None, "补问不再给工具，逼模型直接写结论"
        assert repair_call["messages"][0]["role"] == "system", "补问沿用同一段对话（含系统契约）"
        assert "重新输出" in repair_call["messages"][-1]["content"]
        assert result.usage["calls"] == 2, "补问的用量必须计入本角色"

    def test_repair_gives_up_after_one_retry(self):
        llm = MockLLM(["不是 JSON", "仍然不是 JSON"])
        agent = ReActReviewer(llm, tools=None)
        result = agent.run("审查")

        report = result.metadata["report"]
        assert report["total"] == 0
        assert "parse_error" in report["metadata"], "补问也失败时必须如实留标记"
        assert len(llm.calls) == 2, "不能无限重试"

    def test_repair_error_from_llm_does_not_break_the_report(self):
        def boom():
            raise LLMError("模拟补问时的网络故障")

        llm = MockLLM(["不是 JSON", boom])
        agent = ReActReviewer(llm, tools=None)
        result = agent.run("审查")

        report = result.metadata["report"]
        assert "parse_error" in report["metadata"], "补问失败不该把阶段变成异常"
        assert result.metadata["report"]["total"] == 0

    def test_no_repair_when_output_parses(self):
        llm = MockLLM([REPORT_JSON])
        agent = ReActReviewer(llm, tools=None)
        agent.run("审查")

        assert len(llm.calls) == 1, "结论正常时不该多花一次调用"


class TestReuse:
    def test_reset_clears_history_between_runs(self):
        agent = ReActReviewer(MockLLM([REPORT_JSON, REPORT_JSON]), tools=None)
        agent.run("第一次审查")
        assert len(agent.history) == 2  # user + assistant

        agent.run("第二次审查")
        assert len(agent.history) == 2

    def test_history_can_be_kept_on_demand(self):
        agent = ReActReviewer(MockLLM([REPORT_JSON, REPORT_JSON]), tools=None)
        agent.run("第一次审查")
        agent.run("第二次审查", reset=False)
        assert len(agent.history) == 4


class TestSampleRepositoryAcceptance:
    """W5 验收：输入代码 → 输出"问题列表 + 修复建议"。

    这里只有 LLM 是脚本化的；检索、索引、沙箱、报告渲染全部走真实实现，
    因此能真正验证"Agent 拿到真实检索结果后产出结构化报告"这条链路。
    """

    def test_retrieval_then_structured_report(self, config):
        toolkit = build_review_toolkit(config, root=SAMPLE_REPO)

        index_result = toolkit.execute("code_search", {"action": "index", "path": str(SAMPLE_REPO)})
        assert index_result.success is True
        assert index_result.metadata["chunks"] > 0

        llm = MockLLM(
            [
                {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "code_search",
                            "arguments": {"query": "用户登录逻辑在哪"},
                        }
                    ]
                },
                REPORT_JSON,
            ]
        )
        agent = ReActReviewer(llm, tools=toolkit)
        result = agent.run("审查登录模块", target="data/sample_repo")

        assert result.success is True
        assert result.metadata["report"]["metadata"]["tools_used"] == ["code_search"]

        # 检索结果是真实的：必须命中示例仓库里的登录实现
        search_record = result.tool_calls[0]
        assert search_record["success"] is True
        assert "app/auth/service.py" in search_record["output"]

        report = ReviewReport.from_dict(result.metadata["report"])
        assert report.total == 1
        assert report.findings[0].severity == "high"
        assert report.findings[0].suggestion
        markdown = report.to_markdown()
        assert "# 代码审查报告" in markdown
        assert "**修复建议**" in markdown

    def test_static_analyzer_is_reachable_from_the_agent(self, config):
        """静态分析工具在 Agent 侧可直接调用（本机装了 ruff 时应真的产出结果）。"""
        toolkit = build_review_toolkit(config, root=SAMPLE_REPO, tools=("static_analyzer",))
        result = toolkit.execute(
            "static_analyzer", {"tool": "ruff", "target": "app/auth/service.py"}
        )

        assert result.success is True
        assert result.metadata["tool"] == "ruff"
        assert "findings" in result.output
