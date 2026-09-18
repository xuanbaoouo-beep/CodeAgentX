"""GitTool 测试：参数构造、安全校验、以及"git 未安装/非仓库"两级降级。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codeagentx.core.exceptions import ToolValidationError
from codeagentx.tools.git_tool import (
    MAX_LOG_ENTRIES,
    GitTool,
    _build_argv,
    _clamp_limit,
    _is_not_a_repo,
)
from codeagentx.tools.sandbox import SandboxPolicy, find_executable


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tool(workspace: Path) -> GitTool:
    return GitTool(SandboxPolicy.for_roots([workspace]))


class TestBuildArgv:
    def test_status(self) -> None:
        assert _build_argv("status", None, 20) == ["status", "--short", "--branch"]

    def test_log_respects_limit(self) -> None:
        argv = _build_argv("log", None, 5)
        assert argv[0] == "log"
        assert "--max-count=5" in argv

    def test_diff_without_target(self) -> None:
        assert _build_argv("diff", None, 20) == ["diff", "--no-color"]

    def test_diff_with_target(self) -> None:
        assert _build_argv("diff", "HEAD~1", 20) == ["diff", "--no-color", "HEAD~1"]

    def test_show_defaults_to_head(self) -> None:
        assert "HEAD" in _build_argv("show", None, 20)

    def test_blame_uses_target(self) -> None:
        assert _build_argv("blame", "src/main.py", 20) == ["blame", "--date=short", "src/main.py"]

    def test_unknown_action_raises(self) -> None:
        with pytest.raises(ToolValidationError):
            _build_argv("push", None, 20)


class TestClampLimit:
    def test_accepts_positive_integer(self) -> None:
        assert _clamp_limit(5) == 5

    def test_caps_at_maximum(self) -> None:
        assert _clamp_limit(MAX_LOG_ENTRIES + 500) == MAX_LOG_ENTRIES

    @pytest.mark.parametrize("bad", [0, -3, "abc"])
    def test_rejects_invalid_limit(self, bad: object) -> None:
        with pytest.raises(ToolValidationError):
            _clamp_limit(bad)


class TestValidation:
    def test_rejects_unknown_action(self, tool: GitTool) -> None:
        result = tool.run(action="push")
        assert result.success is False
        assert result.error_type == "ToolValidationError"

    def test_blame_requires_target(self, tool: GitTool) -> None:
        result = tool.run(action="blame")
        assert result.success is False
        assert result.error_type == "ToolValidationError"

    def test_rejects_repo_outside_sandbox(self, tool: GitTool) -> None:
        result = tool.run(action="status", repo="../../")
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_rejects_target_path_escape(self, tool: GitTool) -> None:
        result = tool.run(action="blame", target="../../etc/passwd")
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_tool_exposes_schema(self, tool: GitTool) -> None:
        function = tool.to_openai_schema()["function"]
        assert function["name"] == "git"
        assert "push" not in function["parameters"]["properties"]["action"]["enum"]


class TestDegradation:
    def test_missing_git_returns_tool_unavailable(
        self, tool: GitTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("codeagentx.tools.git_tool.find_executable", lambda name: None)
        result = tool.run(action="status")
        assert result.success is False
        assert result.error_type == "ToolUnavailable"
        assert "hint" in result.metadata

    def test_not_a_repository_detection(self) -> None:
        assert _is_not_a_repo("fatal: not a git repository (or any of the parent directories): .git")
        assert _is_not_a_repo("fatal: Not a git repository")
        assert not _is_not_a_repo("")


@pytest.mark.skipif(find_executable("git") is None, reason="本机未安装 git")
class TestWithRealGit:
    def test_non_repo_directory_reports_not_a_git_repository(
        self, tool: GitTool, workspace: Path
    ) -> None:
        result = tool.run(action="status")
        assert result.success is False
        assert result.error_type == "NotAGitRepository"

    def test_status_on_real_repository(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        (repo / "a.py").write_text("print(1)\n", encoding="utf-8")
        real_tool = GitTool(SandboxPolicy.for_roots([repo]))
        result = real_tool.run(action="status")
        assert result.success is True
        assert "a.py" in result.output
