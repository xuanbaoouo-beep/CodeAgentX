"""安全沙箱内核测试：路径守卫、命令守卫、策略校验与命令执行。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from codeagentx.core.exceptions import SecurityViolationError, ToolExecutionError
from codeagentx.tools.sandbox import (
    CommandGuard,
    PathGuard,
    SandboxPolicy,
    find_executable,
    run_sandboxed,
)


@pytest.fixture
def path_guard(tmp_path: Path) -> PathGuard:
    return PathGuard([tmp_path])


class TestPathGuard:
    def test_resolves_path_inside_root(self, path_guard: PathGuard, tmp_path: Path) -> None:
        target = tmp_path / "a.txt"
        target.write_text("hi", encoding="utf-8")
        assert path_guard.resolve("a.txt", must_exist=True) == target.resolve()

    def test_rejects_parent_escape(self, path_guard: PathGuard) -> None:
        with pytest.raises(SecurityViolationError, match="路径越界"):
            path_guard.resolve("../outside.txt")

    def test_rejects_absolute_path_outside_root(self, path_guard: PathGuard) -> None:
        with pytest.raises(SecurityViolationError, match="路径越界"):
            path_guard.resolve(Path(__file__).resolve())

    def test_rejects_symlink_escape(self, path_guard: PathGuard, tmp_path: Path) -> None:
        outside = tmp_path.parent / "codeagentx-outside-target.txt"
        outside.write_text("secret", encoding="utf-8")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("当前平台不支持创建符号链接（需要管理员权限）")
        with pytest.raises(SecurityViolationError, match="路径越界"):
            path_guard.resolve("link.txt")

    def test_requires_existing_path_when_asked(self, path_guard: PathGuard) -> None:
        with pytest.raises(ToolExecutionError, match="路径不存在"):
            path_guard.resolve("missing.txt", must_exist=True)

    def test_requires_at_least_one_root(self) -> None:
        with pytest.raises(ValueError, match="至少需要一个"):
            PathGuard([])

    def test_validate_path_like_arguments_filters_non_paths(self, path_guard: PathGuard) -> None:
        # 选项、纯单词、纯数字都不应被当成路径
        path_guard.validate_path_like_arguments(["-n", "hello", "12"])

    def test_validate_path_like_arguments_rejects_escape(self, path_guard: PathGuard) -> None:
        with pytest.raises(SecurityViolationError):
            path_guard.validate_path_like_arguments(["../escape"])


class TestCommandGuard:
    @pytest.fixture
    def command_guard(self) -> CommandGuard:
        return CommandGuard()

    @pytest.mark.parametrize("command", ["cat", "ls", "grep", "ruff", "python", "pytest"])
    def test_allows_whitelisted(self, command_guard: CommandGuard, command: str) -> None:
        command_guard.validate(command, [])

    @pytest.mark.parametrize(
        "command",
        ["rm", "rmdir", "curl", "wget", "sudo", "sh", "bash", "powershell", "docker", "make"],
    )
    def test_rejects_denied(self, command_guard: CommandGuard, command: str) -> None:
        with pytest.raises(SecurityViolationError):
            command_guard.validate(command, [])

    def test_rejects_command_outside_whitelist(self, command_guard: CommandGuard) -> None:
        with pytest.raises(SecurityViolationError, match="不在白名单"):
            command_guard.validate("htop", [])

    def test_rejects_empty_command(self, command_guard: CommandGuard) -> None:
        with pytest.raises(SecurityViolationError, match="命令为空"):
            command_guard.validate("", [])

    def test_normalizes_path_and_exe_suffix(self, command_guard: CommandGuard) -> None:
        command_guard.validate(r"C:\Windows\System32\where.exe", [])

    @pytest.mark.parametrize("subcommand", ["status", "log", "diff", "show", "branch"])
    def test_allows_readonly_git_subcommands(
        self, command_guard: CommandGuard, subcommand: str
    ) -> None:
        command_guard.validate("git", [subcommand])

    def test_rejects_git_write_subcommand(self, command_guard: CommandGuard) -> None:
        with pytest.raises(SecurityViolationError, match="不在只读白名单"):
            command_guard.validate("git", ["push"])

    def test_allows_option_only_git(self, command_guard: CommandGuard) -> None:
        command_guard.validate("git", ["--version"])

    def test_allows_readonly_pip_subcommand(self, command_guard: CommandGuard) -> None:
        command_guard.validate("pip", ["list"])

    def test_rejects_pip_install(self, command_guard: CommandGuard) -> None:
        with pytest.raises(SecurityViolationError, match="不在只读白名单"):
            command_guard.validate("pip", ["install", "requests"])

    @pytest.mark.parametrize("flag", ["-c", "-m", "-i"])
    def test_rejects_python_inline_flags(self, command_guard: CommandGuard, flag: str) -> None:
        with pytest.raises(SecurityViolationError, match="python 不允许"):
            command_guard.validate("python", [flag, "print(1)"])

    def test_allows_python_script(self, command_guard: CommandGuard) -> None:
        command_guard.validate("python", ["script.py"])

    def test_rejects_dangerous_pattern_in_arguments(self, command_guard: CommandGuard) -> None:
        with pytest.raises(SecurityViolationError, match="危险命令模式"):
            command_guard.validate("echo", ["rm -rf /"])

    def test_rejects_fork_bomb_pattern(self, command_guard: CommandGuard) -> None:
        with pytest.raises(SecurityViolationError, match="危险命令模式"):
            command_guard.validate("echo", [":() { :|: & };:"])


class TestSandboxPolicy:
    def test_for_roots_requires_existing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="不存在"):
            SandboxPolicy.for_roots([tmp_path / "missing"])
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError, match="不是目录"):
            SandboxPolicy.for_roots([target])

    def test_default_workdir_is_first_root(self, tmp_path: Path) -> None:
        assert SandboxPolicy.for_roots([tmp_path]).default_workdir == tmp_path.resolve()

    def test_workdir_can_be_overridden(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        policy = SandboxPolicy.for_roots([tmp_path], workdir=sub.resolve())
        assert policy.default_workdir == sub.resolve()

    def test_clamp_timeout(self, tmp_path: Path) -> None:
        policy = SandboxPolicy.for_roots([tmp_path], default_timeout=5.0, max_timeout=10.0)
        assert policy.clamp_timeout(None) == 5.0
        assert policy.clamp_timeout(3) == 3.0
        assert policy.clamp_timeout(999) == 10.0

    @pytest.mark.parametrize("bad_timeout", [0, -1, "abc"])
    def test_clamp_timeout_rejects_invalid_value(self, tmp_path: Path, bad_timeout: object) -> None:
        policy = SandboxPolicy.for_roots([tmp_path])
        with pytest.raises(ToolExecutionError):
            policy.clamp_timeout(bad_timeout)  # type: ignore[arg-type]

    def test_rejects_invalid_timeout_configuration(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="超时配置非法"):
            SandboxPolicy(allowed_roots=(tmp_path.resolve(),), default_timeout=0)

    def test_rejects_empty_roots(self) -> None:
        with pytest.raises(ValueError, match="至少需要一个"):
            SandboxPolicy(allowed_roots=())

    def test_guards_are_derived_from_policy(self, tmp_path: Path) -> None:
        policy = SandboxPolicy.for_roots([tmp_path])
        assert isinstance(policy.path_guard(), PathGuard)
        assert isinstance(policy.command_guard(), CommandGuard)


class TestFindExecutable:
    def test_python_maps_to_current_interpreter(self) -> None:
        assert find_executable("python") == sys.executable

    def test_finds_venv_sibling_script(self) -> None:
        # 以 `.venv\Scripts\python.exe -m pytest` 启动时 Scripts 不在 PATH，
        # find_executable 必须仍能找到 pytest
        assert find_executable("pytest") is not None

    def test_returns_none_for_unknown_command(self) -> None:
        assert find_executable("definitely-not-a-real-binary-xyz") is None


class TestRunSandboxed:
    def test_runs_python_script(self, tmp_path: Path) -> None:
        script = tmp_path / "hello.py"
        script.write_text("print('ok')", encoding="utf-8")
        outcome = run_sandboxed("python", ["hello.py"], cwd=tmp_path)
        assert outcome.ok
        assert "ok" in outcome.stdout

    def test_missing_executable_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ToolExecutionError, match="未找到可执行文件"):
            run_sandboxed("definitely-not-a-real-binary-xyz", [], cwd=tmp_path)

    def test_nonzero_exit_is_recorded(self, tmp_path: Path) -> None:
        script = tmp_path / "fail.py"
        script.write_text("import sys\nsys.exit(3)", encoding="utf-8")
        outcome = run_sandboxed("python", ["fail.py"], cwd=tmp_path)
        assert outcome.returncode == 3
        assert not outcome.ok

    def test_timeout_marks_timed_out(self, tmp_path: Path) -> None:
        script = tmp_path / "slow.py"
        script.write_text("import time\ntime.sleep(5)", encoding="utf-8")
        outcome = run_sandboxed("python", ["slow.py"], cwd=tmp_path, timeout=0.5)
        assert outcome.timed_out
        assert outcome.returncode == -1
        assert not outcome.ok

    def test_output_is_truncated(self, tmp_path: Path) -> None:
        script = tmp_path / "big.py"
        script.write_text("print('x' * 5000)", encoding="utf-8")
        outcome = run_sandboxed("python", ["big.py"], cwd=tmp_path, max_output_chars=100)
        assert outcome.truncated
        assert len(outcome.stdout) < 300


class TestFileViewShim:
    """cat/head/tail 在 Windows 上无对应可执行文件，需走 Python 兜底实现。"""

    @pytest.fixture(autouse=True)
    def force_shim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("codeagentx.tools.sandbox.find_executable", lambda name: None)

    @pytest.fixture
    def sample(self, tmp_path: Path) -> Path:
        (tmp_path / "a.txt").write_text("1\n2\n3\n4\n5", encoding="utf-8")
        return tmp_path

    def test_cat_outputs_full_content(self, sample: Path) -> None:
        outcome = run_sandboxed("cat", ["a.txt"], cwd=sample)
        assert outcome.ok
        assert outcome.stdout == "1\n2\n3\n4\n5"

    def test_cat_with_line_numbers(self, sample: Path) -> None:
        outcome = run_sandboxed("cat", ["-n", "a.txt"], cwd=sample)
        assert outcome.ok
        assert "1\t1" in outcome.stdout
        assert "5\t5" in outcome.stdout

    def test_head_and_tail(self, sample: Path) -> None:
        head = run_sandboxed("head", ["-n", "2", "a.txt"], cwd=sample)
        tail = run_sandboxed("tail", ["-n", "2", "a.txt"], cwd=sample)
        assert head.stdout == "1\n2"
        assert tail.stdout == "4\n5"

    def test_head_defaults_to_ten_lines(self, tmp_path: Path) -> None:
        (tmp_path / "b.txt").write_text("\n".join(str(i) for i in range(1, 21)), encoding="utf-8")
        outcome = run_sandboxed("head", ["b.txt"], cwd=tmp_path)
        assert outcome.stdout.splitlines()[-1] == "10"

    def test_missing_file_returns_nonzero(self, sample: Path) -> None:
        outcome = run_sandboxed("cat", ["nope.txt"], cwd=sample)
        assert not outcome.ok
        assert outcome.returncode == 1
        assert outcome.stderr

    def test_unsupported_option_returns_usage_error(self, sample: Path) -> None:
        outcome = run_sandboxed("cat", ["--bogus", "a.txt"], cwd=sample)
        assert outcome.returncode == 2
        assert "不支持选项" in outcome.stderr

    def test_missing_target_returns_usage_error(self, sample: Path) -> None:
        outcome = run_sandboxed("head", ["-n", "3"], cwd=sample)
        assert outcome.returncode == 2
        assert "至少需要一个文件路径" in outcome.stderr

    def test_unknown_command_still_raises(self, sample: Path) -> None:
        with pytest.raises(ToolExecutionError):
            run_sandboxed("grep", ["x", "a.txt"], cwd=sample)


class TestDirectoryShim:
    """ls/dir 在 Windows 上无对应可执行文件，同样走 Python 兜底实现。"""

    @pytest.fixture(autouse=True)
    def force_shim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("codeagentx.tools.sandbox.find_executable", lambda name: None)

    @pytest.fixture
    def sample(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x", encoding="utf-8")
        (tmp_path / "README.md").write_text("# x", encoding="utf-8")
        (tmp_path / ".hidden").write_text("h", encoding="utf-8")
        return tmp_path

    def test_lists_entries_and_marks_directories(self, sample: Path) -> None:
        outcome = run_sandboxed("ls", [], cwd=sample)
        assert outcome.ok
        assert "src/" in outcome.stdout
        assert "README.md" in outcome.stdout

    def test_hides_dotfiles_by_default(self, sample: Path) -> None:
        assert ".hidden" not in run_sandboxed("ls", [], cwd=sample).stdout

    def test_all_flag_shows_dotfiles(self, sample: Path) -> None:
        assert ".hidden" in run_sandboxed("ls", ["-a"], cwd=sample).stdout

    def test_long_format_shows_kind_and_size(self, sample: Path) -> None:
        stdout = run_sandboxed("ls", ["-l"], cwd=sample).stdout
        assert "d " in stdout
        assert "- " in stdout

    def test_dir_is_alias_of_ls(self, sample: Path) -> None:
        assert run_sandboxed("dir", [], cwd=sample).stdout == run_sandboxed("ls", [], cwd=sample).stdout

    def test_lists_subdirectory(self, sample: Path) -> None:
        assert run_sandboxed("ls", ["src"], cwd=sample).stdout.strip() == "main.py"

    def test_missing_path_returns_usage_error(self, sample: Path) -> None:
        outcome = run_sandboxed("ls", ["nope"], cwd=sample)
        assert outcome.returncode == 2
        assert "路径不存在" in outcome.stderr
