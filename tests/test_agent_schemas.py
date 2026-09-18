"""审查结论数据结构测试：字段容错归一化、报告渲染、模型输出的 JSON 解析。

关注点不是"正常输入能跑通"，而是"模型不听话时会不会丢问题"：
非法严重度、百分比置信度、围栏包裹的 JSON、塞在解释文字里的 JSON，
都必须被正确归一化，而不是抛异常或静默丢弃。
"""

from __future__ import annotations

import pytest

from codeagentx.agents.schemas import (
    Finding,
    PlanStep,
    RefactorPlan,
    ReviewReport,
    normalize_category,
    normalize_confidence,
    normalize_file,
    normalize_severity,
    normalize_str_list,
    parse_json_payload,
    parse_review_report,
    plan_from_payload,
    report_from_payload,
)
from codeagentx.core.exceptions import AgentOutputError


class TestNormalize:
    """字段归一化：把模型的花式写法收敛到标准取值。"""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("high", "high"),
            ("HIGH", "high"),
            ("critical", "high"),
            ("严重", "high"),
            ("warn", "medium"),
            ("major", "medium"),
            ("中等", "medium"),
            ("minor", "low"),
            ("info", "low"),
            ("whatever", "medium"),
            ("", "medium"),
            (None, "medium"),
        ],
    )
    def test_severity_aliases(self, raw, expected):
        assert normalize_severity(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("bug", "bug"),
            ("correctness", "bug"),
            ("漏洞", "security"),
            ("Security", "security"),
            ("perf", "performance"),
            ("pep8", "style"),
            ("refactor", "maintainability"),
            ("design", "maintainability"),
            ("覆盖率", "other"),
            ("", "other"),
        ],
    )
    def test_category_aliases(self, raw, expected):
        assert normalize_category(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0.8, 0.8),
            ("0.8", 0.8),
            ("80%", 0.8),
            (80, 0.8),
            (150, 1.0),
            (-1, 0.0),
            ("abc", 0.5),
            (None, 0.5),
        ],
    )
    def test_confidence(self, raw, expected):
        assert normalize_confidence(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("./app/x.py", "app/x.py"),
            ("app\\auth\\service.py", "app/auth/service.py"),
            ("`app/x.py`", "app/x.py"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_file_path(self, raw, expected):
        assert normalize_file(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a,b", ["a", "b"]),
            ("a、b", ["a", "b"]),
            (["a", "", "b"], ["a", "b"]),
            ("单个", ["单个"]),
            (None, []),
            (5, ["5"]),
        ],
    )
    def test_str_list(self, raw, expected):
        assert normalize_str_list(raw) == expected


class TestFinding:
    def test_normalizes_every_field(self):
        finding = Finding(
            title="  硬编码密钥  ",
            file=".\\app\\auth.py",
            line="42",
            severity="CRITICAL",
            category="漏洞",
            confidence="90%",
        )
        assert finding.title == "硬编码密钥"
        assert finding.file == "app/auth.py"
        assert finding.line == 42
        assert finding.severity == "high"
        assert finding.category == "security"
        assert finding.confidence == 0.9
        assert finding.location == "app/auth.py:42"

    def test_title_falls_back_when_missing(self):
        # 模型只给了 description 时，标题由描述派生，而不是变成空标题
        finding = Finding.from_dict({"description": "SQL 语句由字符串拼接而成"})
        assert finding.title == "SQL 语句由字符串拼接而成"

    def test_unparsable_item_still_kept(self):
        finding = Finding.from_dict("这行代码可能有风险")
        assert finding.title == "这行代码可能有风险"
        assert finding.severity == "medium"

    def test_alias_keys_are_recognized(self):
        finding = Finding.from_dict(
            {
                "message": "未捕获异常",
                "path": "a.py",
                "line_number": 7,
                "level": "warn",
                "type": "correctness",
                "fix": "改为 except ValueError",
            }
        )
        assert finding.title == "未捕获异常"
        assert finding.file == "a.py"
        assert finding.line == 7
        assert finding.severity == "medium"
        assert finding.category == "bug"
        assert finding.suggestion == "改为 except ValueError"

    def test_location_without_line(self):
        assert Finding(title="x", file="a.py").location == "a.py"
        assert Finding(title="x").location == ""

    def test_roundtrip(self):
        original = Finding(title="x", file="a.py", line=3, severity="high", category="security")
        assert Finding.from_dict(original.to_dict()).to_dict() == original.to_dict()

    def test_source_defaults_from_context(self):
        finding = Finding.from_dict({"title": "x"}, source="react_reviewer")
        assert finding.source == "react_reviewer"
        # 自己带了 source 时以自己为准
        assert Finding.from_dict({"title": "x", "source": "静态分析"}, source="agent").source == "静态分析"


class TestReviewReport:
    def _report(self) -> ReviewReport:
        return ReviewReport(
            target="data/sample_repo",
            summary="共发现 3 个问题，其中硬编码密钥风险最高。",
            findings=[
                Finding(title="风格问题", file="b.py", line=10, severity="low", category="style"),
                Finding(title="硬编码密钥", file="a.py", line=8, severity="high", category="security"),
                Finding(title="空异常捕获", file="a.py", line=20, severity="medium", category="bug"),
            ],
        )

    def test_severity_counts_always_has_three_keys(self):
        counts = self._report().severity_counts()
        assert counts == {"high": 1, "medium": 1, "low": 1}
        assert ReviewReport().severity_counts() == {"high": 0, "medium": 0, "low": 0}

    def test_sorted_by_severity_then_location(self):
        titles = [item.title for item in self._report().sorted_findings()]
        assert titles == ["硬编码密钥", "空异常捕获", "风格问题"]

    def test_markdown_contains_position_and_suggestion(self):
        markdown = self._report().to_markdown()
        assert "# 代码审查报告" in markdown
        assert "**问题总数**：3（high 1 / medium 1 / low 1）" in markdown
        assert "### 1. [high] 硬编码密钥" in markdown
        assert "**位置**：`a.py:8`" in markdown

    def test_markdown_when_nothing_found(self):
        markdown = ReviewReport(target="a.py").to_markdown()
        assert "未发现需要报告的问题。" in markdown

    def test_text_output_is_compact(self):
        text = self._report().to_text()
        assert "[1] [high][security] 硬编码密钥 @ a.py:8" in text
        assert "问题总数：3" in text

    def test_dict_roundtrip(self):
        payload = self._report().to_dict()
        restored = ReviewReport.from_dict(payload)
        assert restored.total == 3
        assert restored.target == "data/sample_repo"
        assert [item.title for item in restored.sorted_findings()] == [
            "硬编码密钥",
            "空异常捕获",
            "风格问题",
        ]

    def test_metadata_is_not_lost_on_roundtrip(self):
        report = self._report()
        report.metadata["parse_error"] = "boom"
        assert ReviewReport.from_dict(report.to_dict()).metadata["parse_error"] == "boom"


class TestParseJsonPayload:
    def test_plain_object(self):
        assert parse_json_payload('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        text = '下面是结论：\n```json\n{"findings": []}\n```\n以上。'
        assert parse_json_payload(text) == {"findings": []}

    def test_bare_fence(self):
        assert parse_json_payload("```\n[1, 2]\n```") == [1, 2]

    def test_json_embedded_in_prose(self):
        text = '好的，我的结论是 {"summary": "ok"} ，请查收。'
        assert parse_json_payload(text) == {"summary": "ok"}

    def test_empty_raises(self):
        with pytest.raises(AgentOutputError):
            parse_json_payload("   ")

    def test_tolerates_raw_newlines_inside_strings(self):
        """字符串里带未转义的换行是真实模型的高频毛病，要能救回来。"""
        text = '{"summary": "第一行\n第二行", "findings": []}'
        assert parse_json_payload(text) == {"summary": "第一行\n第二行", "findings": []}

    def test_structural_errors_are_still_rejected(self):
        """放宽只针对裸控制字符；结构性错误（尾逗号等）仍须判为不可解析。"""
        with pytest.raises(AgentOutputError):
            parse_json_payload('{"a": 1,}')

    def test_no_json_raises(self):
        with pytest.raises(AgentOutputError) as excinfo:
            parse_json_payload("我认为这段代码存在安全问题。")
        assert "未找到合法 JSON" in str(excinfo.value)


class TestReportFromPayload:
    def test_standard_shape(self):
        report = report_from_payload(
            {"summary": "结论", "findings": [{"title": "x", "severity": "high"}]},
            target="a.py",
            source="react_reviewer",
        )
        assert report.summary == "结论"
        assert report.total == 1
        assert report.findings[0].source == "react_reviewer"

    @pytest.mark.parametrize("key", ["issues", "problems", "results"])
    def test_alternative_list_keys(self, key):
        report = report_from_payload({key: [{"title": "x"}]})
        assert report.total == 1

    def test_single_finding_object(self):
        report = report_from_payload({"title": "x", "severity": "high"})
        assert report.total == 1
        assert report.findings[0].severity == "high"

    def test_bare_list(self):
        assert report_from_payload([{"title": "x"}, "y"]).total == 2

    def test_unusable_payload_gives_empty_report(self):
        assert report_from_payload(42).total == 0
        assert report_from_payload(None).total == 0

    def test_chinese_keys(self):
        report = report_from_payload({"摘要": "总结", "问题": [{"标题": "x"}]})
        assert report.summary == "总结"
        assert report.total == 1


class TestParseReviewReport:
    def test_success_path(self):
        report = parse_review_report('{"summary": "s", "findings": [{"title": "x"}]}', target="a.py")
        assert report.total == 1
        assert "parse_error" not in report.metadata

    def test_failure_is_marked_not_silent(self):
        """解析失败绝不能表现为"没有发现问题"。"""
        report = parse_review_report("模型今天不想输出 JSON", target="a.py")
        assert report.total == 0
        assert "parse_error" in report.metadata
        assert "raw_output" in report.metadata
        assert "未产出有效结论" in report.summary


class TestRefactorPlan:
    def test_step_from_string(self):
        step = PlanStep.from_dict("拆分长函数")
        assert step.description == "拆分长函数"
        assert step.status == "pending"

    def test_step_normalizes_files_and_status(self):
        step = PlanStep.from_dict(
            {"description": "改参数校验", "files": [".\\a.py"], "status": "DONE"}
        )
        assert step.files == ["a.py"]
        assert step.status == "done"

    def test_step_invalid_status_falls_back(self):
        assert PlanStep(description="x", status="进行中").status == "pending"

    def test_plan_from_dict(self):
        plan = plan_from_payload(
            {
                "goal": "拆解登录逻辑",
                "steps": [{"description": "提取校验函数"}, {"description": "补单元测试"}],
                "risks": "可能改变异常类型",
                "verification": ["pytest tests/test_auth.py"],
            }
        )
        assert plan.total == 2
        assert plan.risks == ["可能改变异常类型"]
        assert plan.verification == ["pytest tests/test_auth.py"]

    def test_plan_from_bare_list(self):
        assert plan_from_payload(["第一步", "第二步"]).total == 2

    def test_plan_roundtrip(self):
        plan = RefactorPlan(goal="g", steps=[PlanStep(description="s1")], risks=["r"])
        assert RefactorPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()

    def test_markdown_renders_steps_and_sections(self):
        plan = RefactorPlan(
            goal="消除硬编码密钥",
            steps=[PlanStep(description="改为读环境变量", files=["app/auth.py"], status="done")],
            risks=["忘记同步 .env.example"],
            verification=["python -m pytest"],
        )
        markdown = plan.to_markdown()
        assert "# 重构计划" in markdown
        assert "1. **[done]** 改为读环境变量" in markdown
        assert "`app/auth.py`" in markdown
        assert "## 风险" in markdown
        assert "## 验证方式" in markdown

    def test_plan_from_scalar_raises(self):
        with pytest.raises(AgentOutputError):
            plan_from_payload("这不是计划")
