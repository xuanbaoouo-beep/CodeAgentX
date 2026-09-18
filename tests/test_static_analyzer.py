"""StaticAnalyzer 测试：结构化解析、严重度映射、安全校验与缺失降级。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeagentx.core.exceptions import ToolValidationError
from codeagentx.tools.sandbox import SandboxPolicy, find_executable
from codeagentx.tools.static_analyzer import (
    DEFAULT_MAX_FINDINGS,
    MAX_FINDINGS_LIMIT,
    StaticAnalyzer,
    _clamp_max_findings,
    _parse_bandit,
    _parse_pylint,
    _parse_ruff,
    _ruff_severity,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    # 两个未使用的导入 -> ruff 至少产出 2 条 F401，便于验证条数截断
    (tmp_path / "sample.py").write_text(
        "import os\nimport sys\n\n\ndef unused() -> None:\n    pass\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def analyzer(workspace: Path) -> StaticAnalyzer:
    return StaticAnalyzer(SandboxPolicy.for_roots([workspace]))


class TestRuffParsing:
    def test_parses_findings(self) -> None:
        payload = [
            {
                "filename": "sample.py",
                "code": "F401",
                "message": "`os` imported but unused",
                "location": {"row": 1, "column": 8},
            }
        ]
        findings = _parse_ruff(payload)
        assert findings == [
            {
                "tool": "ruff",
                "file": "sample.py",
                "line": 1,
                "column": 8,
                "code": "F401",
                "severity": "medium",
                "message": "`os` imported but unused",
            }
        ]

    def test_empty_payload(self) -> None:
        assert _parse_ruff([]) == []

    def test_tolerates_missing_fields(self) -> None:
        findings = _parse_ruff([{"code": "E501"}])
        assert findings[0]["line"] == 0
        assert findings[0]["file"] == ""


class TestPylintParsing:
    def test_parses_findings(self) -> None:
        payload = [
            {
                "type": "warning",
                "module": "sample",
                "line": 5,
                "column": 0,
                "path": "sample.py",
                "symbol": "unused-variable",
                "message-id": "W0612",
                "message": "Unused variable 'x'",
            }
        ]
        findings = _parse_pylint(payload)
        assert findings[0]["code"] == "W0612"
        assert findings[0]["severity"] == "medium"
        assert findings[0]["line"] == 5

    def test_ignores_non_dict_entries(self) -> None:
        assert _parse_pylint(["[", "score"]) == []


class TestBanditParsing:
    def test_parses_findings(self) -> None:
        payload = {
            "results": [
                {
                    "filename": "sample.py",
                    "line_number": 7,
                    "col_offset": 4,
                    "test_id": "B105",
                    "issue_severity": "HIGH",
                    "issue_text": "Possible hardcoded password",
                }
            ]
        }
        findings = _parse_bandit(payload)
        assert findings[0]["severity"] == "high"
        assert findings[0]["code"] == "B105"

    def test_missing_results_key(self) -> None:
        assert _parse_bandit({}) == []


class TestSeverityMapping:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [("F401", "medium"), ("E902", "high"), ("S101", "high"), ("C901", "low")],
    )
    def test_ruff_severity_heuristic(self, code: str, expected: str) -> None:
        assert _ruff_severity(code) == expected


class TestClampMaxFindings:
    def test_caps_at_maximum(self) -> None:
        assert _clamp_max_findings(MAX_FINDINGS_LIMIT + 10) == MAX_FINDINGS_LIMIT

    @pytest.mark.parametrize("bad", [0, -1, "abc"])
    def test_rejects_invalid_value(self, bad: object) -> None:
        with pytest.raises(ToolValidationError):
            _clamp_max_findings(bad)


class TestValidation:
    def test_rejects_unknown_tool(self, analyzer: StaticAnalyzer) -> None:
        result = analyzer.run(tool="mypy")
        assert result.success is False
        assert result.error_type == "ToolValidationError"

    def test_rejects_target_outside_sandbox(self, analyzer: StaticAnalyzer) -> None:
        result = analyzer.run(tool="ruff", target="../../")
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_rejects_missing_target(self, analyzer: StaticAnalyzer) -> None:
        result = analyzer.run(tool="ruff", target="nope.py")
        assert result.success is False
        assert result.error_type == "ToolExecutionError"

    def test_exposes_schema(self, analyzer: StaticAnalyzer) -> None:
        function = analyzer.to_openai_schema()["function"]
        assert function["name"] == "static_analyzer"
        assert function["parameters"]["properties"]["tool"]["enum"] == ["bandit", "pylint", "ruff"]


class TestDegradation:
    @pytest.mark.parametrize("missing", ["pylint", "bandit"])
    def test_missing_tool_returns_tool_unavailable(
        self, analyzer: StaticAnalyzer, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        monkeypatch.setattr("codeagentx.tools.static_analyzer.find_executable", lambda name: None)
        result = analyzer.run(tool=missing, target="sample.py")
        assert result.success is False
        assert result.error_type == "ToolUnavailable"
        assert "hint" in result.metadata


@pytest.mark.skipif(find_executable("ruff") is None, reason="本机未安装 ruff")
class TestWithRealRuff:
    def test_reports_unused_import(self, analyzer: StaticAnalyzer) -> None:
        result = analyzer.run(tool="ruff", target="sample.py")
        assert result.success is True
        payload = result.output
        assert payload["tool"] == "ruff"
        assert payload["finding_count"] >= 1
        assert "F401" in {finding["code"] for finding in payload["findings"]}
        assert payload["severity_counts"]["medium"] >= 1

    def test_respects_max_findings(self, analyzer: StaticAnalyzer) -> None:
        result = analyzer.run(tool="ruff", target="sample.py", max_findings=1)
        assert result.success is True
        assert len(result.output["findings"]) == 1
        assert result.output["truncated"] is True

    def test_default_max_findings_is_used(self, analyzer: StaticAnalyzer) -> None:
        result = analyzer.run(tool="ruff", target="sample.py")
        assert len(result.output["findings"]) <= DEFAULT_MAX_FINDINGS

    def test_clean_file_reports_no_findings(self, analyzer: StaticAnalyzer, workspace: Path) -> None:
        (workspace / "clean.py").write_text("print('ok')\n", encoding="utf-8")
        result = analyzer.run(tool="ruff", target="clean.py")
        assert result.success is True
        assert result.output["finding_count"] == 0
        assert result.output["exit_code"] == 0

    def test_parse_error_falls_back_to_raw_output(
        self, analyzer: StaticAnalyzer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "codeagentx.tools.static_analyzer._parse_findings",
            lambda tool, stdout: json.loads("{ not json"),
        )
        result = analyzer.run(tool="ruff", target="sample.py")
        assert result.success is True
        assert "parse_error" in result.output
        assert "raw_output" in result.output
