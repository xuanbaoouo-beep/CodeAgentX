"""TestRunner 测试：pytest 结果解析、安全校验与缺失降级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagentx.tools.sandbox import SandboxPolicy, find_executable
from codeagentx.tools.test_runner import TestRunner as Runner
from codeagentx.tools.test_runner import _extract_failures, _parse_counts, _summary_line

SAMPLE_SUMMARY = """\
=========================== short test summary info ===========================
FAILED tests/test_login.py::test_bad_password - AssertionError: assert 200 == 401
FAILED tests/test_login.py::test_locked - ValueError: boom
ERROR tests/test_api.py::test_setup - RuntimeError: fixture failed
3 failed, 10 passed, 2 skipped, 1 warning in 1.23s
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "test_sample.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def runner(workspace: Path) -> Runner:
    return Runner(SandboxPolicy.for_roots([workspace]))


class TestParsing:
    def test_parse_counts_from_summary(self) -> None:
        counts = _parse_counts(SAMPLE_SUMMARY)
        assert counts == {"failed": 3, "passed": 10, "skipped": 2}

    def test_warnings_are_not_counted_as_tests(self) -> None:
        assert "warnings" not in _parse_counts("1 warning in 0.10s")

    def test_errors_label_is_normalized(self) -> None:
        assert _parse_counts("2 errors in 0.10s") == {"error": 2}

    def test_empty_text_yields_no_counts(self) -> None:
        assert _parse_counts("") == {}

    def test_summary_line_is_last_matching_line(self) -> None:
        assert _summary_line(SAMPLE_SUMMARY) == "3 failed, 10 passed, 2 skipped, 1 warning in 1.23s"

    def test_summary_line_empty_when_absent(self) -> None:
        assert _summary_line("nothing here") == ""

    def test_extract_failures(self) -> None:
        failures = _extract_failures(SAMPLE_SUMMARY)
        assert len(failures) == 3
        assert failures[0].startswith("FAILED tests/test_login.py::test_bad_password")
        assert any(item.startswith("ERROR tests/test_api.py") for item in failures)

    def test_extract_failures_empty(self) -> None:
        assert _extract_failures("1 passed in 0.01s") == []


class TestValidation:
    def test_rejects_target_outside_sandbox(self, runner: Runner) -> None:
        result = runner.run(target="../../")
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_rejects_missing_target(self, runner: Runner) -> None:
        result = runner.run(target="nope")
        assert result.success is False
        assert result.error_type == "ToolExecutionError"

    def test_exposes_schema(self, runner: Runner) -> None:
        function = runner.to_openai_schema()["function"]
        assert function["name"] == "test_runner"
        # 三个参数都有默认值，因此不存在必填参数
        assert "required" not in function["parameters"]
        assert runner.dangerous is True


class TestDegradation:
    def test_missing_pytest_returns_tool_unavailable(
        self, runner: Runner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("codeagentx.tools.test_runner.find_executable", lambda name: None)
        result = runner.run(target=".")
        assert result.success is False
        assert result.error_type == "ToolUnavailable"
        assert "hint" in result.metadata


@pytest.mark.skipif(find_executable("pytest") is None, reason="本机未安装 pytest")
class TestWithRealPytest:
    def test_passing_suite(self, runner: Runner) -> None:
        result = runner.run(target=".")
        assert result.success is True
        payload = result.output
        assert payload["passed"] is True
        assert payload["counts"].get("passed") == 1
        assert payload["failures"] == []

    def test_failing_suite_is_reported_not_raised(self, runner: Runner, workspace: Path) -> None:
        (workspace / "test_fail.py").write_text(
            "def test_bad():\n    assert 1 == 2\n", encoding="utf-8"
        )
        result = runner.run(target="test_fail.py")
        assert result.success is True  # 工具执行成功，测试本身失败
        payload = result.output
        assert payload["passed"] is False
        assert payload["counts"].get("failed") == 1
        assert payload["failures"] and "test_fail.py::test_bad" in payload["failures"][0]

    def test_extra_args_are_forwarded(self, runner: Runner, workspace: Path) -> None:
        (workspace / "test_two.py").write_text(
            "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n",
            encoding="utf-8",
        )
        result = runner.run(target="test_two.py", args=["-k", "test_a"])
        assert result.success is True
        assert result.output["counts"].get("passed") == 1
        assert result.output["counts"].get("deselected") == 1

    def test_timeout_is_reported(self, runner: Runner, workspace: Path) -> None:
        (workspace / "test_slow.py").write_text(
            "import time\n\n\ndef test_slow():\n    time.sleep(10)\n", encoding="utf-8"
        )
        result = runner.run(target="test_slow.py", timeout=1.0)
        assert result.success is False
        assert result.error_type == "TimeoutExpired"

    def test_parses_counts_when_project_config_quiets_output(
        self, runner: Runner, workspace: Path
    ) -> None:
        # 回归：目标项目自身配置 addopts = "-q" 时会与工具的 -q 叠加成 -qq，
        # pytest 将不再打印摘要行 —— 必须通过清空 addopts 规避
        (workspace / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\naddopts = "-q"\n', encoding="utf-8"
        )
        result = runner.run(target=".")
        assert result.success is True
        assert result.output["counts"].get("passed") == 1
        assert "passed" in result.output["summary_line"]
