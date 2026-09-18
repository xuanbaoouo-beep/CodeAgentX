"""TerminalTool 测试。

W3 验收标准：``rm -rf /`` 被拒绝，``cat README.md`` 正常。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagentx.tools.sandbox import SandboxPolicy
from codeagentx.tools.terminal import TerminalTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# Demo 项目\n说明文本\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def tool(workspace: Path) -> TerminalTool:
    return TerminalTool(SandboxPolicy.for_roots([workspace]))


class TestAcceptance:
    """W3 验收用例。"""

    def test_rm_rf_root_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="rm", args=["-rf", "/"])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"
        assert "危险" in (result.error or "")

    def test_cat_readme_succeeds(self, tool: TerminalTool) -> None:
        result = tool.run(command="cat", args=["README.md"])
        assert result.success is True
        assert "# Demo 项目" in result.output
        assert result.metadata["returncode"] == 0


class TestCommandInterception:
    @pytest.mark.parametrize("args", [["-rf", "/"], ["-fr", "/"], ["-r", "-f", "/"]])
    def test_rm_variants_are_rejected(self, tool: TerminalTool, args: list[str]) -> None:
        result = tool.run(command="rm", args=args)
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_dangerous_pattern_is_rejected_even_for_allowed_command(
        self, tool: TerminalTool
    ) -> None:
        result = tool.run(command="echo", args=["rm -rf /"])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    @pytest.mark.parametrize("command", ["curl", "wget", "sudo", "bash", "powershell", "docker"])
    def test_denied_commands_are_rejected(self, tool: TerminalTool, command: str) -> None:
        result = tool.run(command=command, args=[])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_command_outside_whitelist_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="htop", args=[])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"
        assert "不在白名单" in (result.error or "")

    def test_python_inline_code_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="python", args=["-c", "import os"])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_git_write_subcommand_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="git", args=["push"])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"


class TestPathContainment:
    def test_relative_escape_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="cat", args=["../../secret.txt"])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_absolute_path_outside_workspace_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="cat", args=[str(Path.home() / ".bashrc")])
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_cwd_outside_workspace_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="ls", args=[], cwd=str(Path.home()))
        assert result.success is False
        assert result.error_type == "SecurityViolationError"

    def test_cwd_inside_workspace_is_allowed(self, tool: TerminalTool, workspace: Path) -> None:
        result = tool.run(command="dir", args=[], cwd=str(workspace / "src"))
        assert result.success is True


class TestValidation:
    def test_missing_required_argument(self, tool: TerminalTool) -> None:
        result = tool.run()
        assert result.success is False
        assert result.error_type == "ToolValidationError"

    def test_unknown_argument_is_rejected(self, tool: TerminalTool) -> None:
        result = tool.run(command="ls", bogus=1)
        assert result.success is False
        assert result.error_type == "ToolValidationError"

    def test_tool_is_marked_dangerous_and_exposes_schema(self, tool: TerminalTool) -> None:
        assert tool.dangerous is True
        function = tool.to_openai_schema()["function"]
        assert function["name"] == "terminal"
        assert set(function["parameters"]["properties"]) == {"command", "args", "cwd", "timeout"}
        assert function["parameters"]["required"] == ["command"]


class TestExecution:
    def test_nonzero_exit_is_not_treated_as_tool_failure(
        self, tool: TerminalTool, workspace: Path
    ) -> None:
        # 命令自身退出码非零（如 grep 无匹配）说明"命令跑完了"，而非工具故障
        (workspace / "fail.py").write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        result = tool.run(command="python", args=["fail.py"])
        assert result.success is True
        assert result.metadata["returncode"] == 3
        assert "[exit=3]" in result.output

    def test_timeout_returns_failure(self, tool: TerminalTool, workspace: Path) -> None:
        (workspace / "slow.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
        result = tool.run(command="python", args=["slow.py"], timeout=0.5)
        assert result.success is False
        assert result.error_type == "TimeoutExpired"
        assert result.metadata["timed_out"] is True

    def test_metadata_excludes_bulky_output(self, tool: TerminalTool) -> None:
        result = tool.run(command="cat", args=["README.md"])
        assert "stdout" not in result.metadata
        assert "stderr" not in result.metadata

    def test_tool_never_raises(self, tool: TerminalTool) -> None:
        # 未安装的命令也应返回失败结果而不是抛异常
        result = tool.run(command="grep", args=["x", "README.md"])
        if result.success is False:
            assert result.error_type == "ToolExecutionError"
