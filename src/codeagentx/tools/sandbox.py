"""安全沙箱：路径守卫（PathGuard）与命令守卫（CommandGuard）。

威胁模型与边界（重要）
----------------------
本模块防的是**误用与常见破坏性操作**：路径逃逸、危险命令、无限执行。
它**不是**不可逃逸的强隔离——被测仓库自身的测试代码一旦运行，
仍以当前用户权限执行。真正的隔离请在容器中运行（见 W10 的 Dockerfile）。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeagentx.core.exceptions import SecurityViolationError, ToolExecutionError

__all__ = [
    "ALLOWED_COMMANDS",
    "DENIED_COMMANDS",
    "DENY_PATTERNS",
    "CommandOutcome",
    "CommandGuard",
    "PathGuard",
    "SandboxPolicy",
    "find_executable",
    "run_sandboxed",
]

#: 白名单命令（POSIX 与 Windows 双份；仅允许只读/分析类操作）
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        # 运行时与包信息
        "python",
        "python3",
        "py",
        "pip",
        "pip3",
        # 代码质量与测试
        "pytest",
        "ruff",
        "pylint",
        "bandit",
        "mypy",
        "flake8",
        # 版本控制（子命令另有限制）
        "git",
        # 文件与文本查看
        "ls",
        "dir",
        "cat",
        "type",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "grep",
        "rg",
        "find",
        "findstr",
        "tree",
        "stat",
        "file",
        "diff",
        "cmp",
        "echo",
        "pwd",
        "where",
        "which",
        "basename",
        "dirname",
    }
)

#: 明确拒绝的命令：即使白名单被误改也不放行
DENIED_COMMANDS: frozenset[str] = frozenset(
    {
        # 破坏性文件操作
        "rm",
        "rmdir",
        "del",
        "erase",
        "format",
        "mkfs",
        "dd",
        "shred",
        "truncate",
        # 提权与账户
        "sudo",
        "su",
        "doas",
        "runas",
        "chmod",
        "chown",
        "chgrp",
        "useradd",
        "userdel",
        "passwd",
        # 进程与服务
        "kill",
        "pkill",
        "killall",
        "taskkill",
        "shutdown",
        "reboot",
        "halt",
        "systemctl",
        "service",
        "crontab",
        "schtasks",
        # 网络下载与远程
        "curl",
        "wget",
        "nc",
        "netcat",
        "ncat",
        "ssh",
        "scp",
        "sftp",
        "telnet",
        "ftp",
        # shell 解释器（阻断命令链）
        "sh",
        "bash",
        "zsh",
        "csh",
        "fish",
        "cmd",
        "powershell",
        "pwsh",
        # 包管理与任意代码执行
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "brew",
        "choco",
        "npm",
        "npx",
        "yarn",
        "pnpm",
        "docker",
        "kubectl",
        "make",
        # 注册表
        "reg",
        "regedit",
    }
)

#: 危险字符串模式（纵深防御：即便命令被误放行也拦截）
DENY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-[a-z]*r[a-z]*f", re.IGNORECASE),
    re.compile(r"\brm\s+-[a-z]*f[a-z]*r", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"--no-preserve-root", re.IGNORECASE),
    re.compile(r":\(\)\s*\{"),  # fork bomb
    re.compile(r">\s*/dev/(sd|nvme|hd)", re.IGNORECASE),
    re.compile(r"\bmkfs(\.\w+)?\b", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bchmod\s+-R\s+777\s+/"),
)

#: git 只读子命令
GIT_READONLY_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "branch",
        "rev-parse",
        "rev-list",
        "ls-files",
        "ls-tree",
        "remote",
        "describe",
        "shortlog",
        "blame",
        "tag",
        "show-ref",
        "cat-file",
        "grep",
        "whatchanged",
    }
)

#: pip 只读子命令
PIP_READONLY_SUBCOMMANDS: frozenset[str] = frozenset({"list", "show", "freeze", "check"})

#: python 禁止的选项（可执行任意代码/进入交互）
PYTHON_DENIED_FLAGS: frozenset[str] = frozenset({"-c", "-m", "-i", "-", "--eval"})

#: 跨平台补齐的命令。
#: Windows 上不存在 cat/head/tail/ls 可执行文件（dir/type 是 cmd 内建，shell=False 下无法调用），
#: 这里用等价的 Python 只读实现兜底，**仅在系统确实找不到该命令时**才会启用。
FILE_VIEW_SHIMS: frozenset[str] = frozenset({"cat", "head", "tail"})
DIRECTORY_SHIMS: frozenset[str] = frozenset({"ls", "dir"})
SHIM_COMMANDS: frozenset[str] = FILE_VIEW_SHIMS | DIRECTORY_SHIMS


def _looks_like_path(value: str) -> bool:
    """粗略判断一个参数是否像文件路径（用于路径越界检查）。"""
    if not value or value.startswith("-"):
        return False
    if value.startswith(("~", ".", "/", "\\")):
        return True
    if len(value) >= 2 and value[0].isalpha() and value[1] == ":":
        return True
    return "/" in value or "\\" in value


class PathGuard:
    """路径守卫：任何被访问的路径都必须落在允许的根目录之内。"""

    def __init__(self, allowed_roots: Iterable[str | Path]) -> None:
        roots = tuple(_resolve_root(root) for root in allowed_roots)
        if not roots:
            raise ValueError("PathGuard 至少需要一个允许根目录")
        self.allowed_roots = roots

    def resolve(
        self,
        raw: str | Path,
        *,
        base: str | Path | None = None,
        must_exist: bool = False,
    ) -> Path:
        """规范化路径并校验是否越界。

        Args:
            raw: 待校验路径，可为相对路径。
            base: 相对路径的基准目录，缺省用第一个允许根目录。
            must_exist: 是否要求路径必须存在。

        Raises:
            SecurityViolationError: 路径越界（含通过 ``..`` 或符号链接逃逸）。
            ToolExecutionError: ``must_exist=True`` 但路径不存在。
        """
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            anchor = Path(base) if base is not None else self.allowed_roots[0]
            candidate = anchor / candidate
        # resolve() 会展开符号链接，因此链接逃逸也会被下面的前缀检查拦下
        resolved = candidate.resolve()

        if not self.contains(resolved):
            raise SecurityViolationError(
                f"路径越界：{raw} 不在允许范围内",
                detail=f"允许根目录：{[str(root) for root in self.allowed_roots]}",
            )
        if must_exist and not resolved.exists():
            raise ToolExecutionError(f"路径不存在：{resolved}")
        return resolved

    def contains(self, path: Path) -> bool:
        """判断已解析的绝对路径是否位于某个允许根目录内。"""
        return any(path == root or root in path.parents for root in self.allowed_roots)

    def validate_path_like_arguments(self, values: Sequence[str], *, base: str | Path | None = None) -> None:
        """对一组命令行参数做路径越界检查（只检查"像路径"的参数）。"""
        for value in values:
            if _looks_like_path(value):
                self.resolve(value, base=base)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<PathGuard roots={[str(root) for root in self.allowed_roots]}>"


def _command_basename(command: str) -> str:
    """取命令字符串里的"文件名"部分（保留原大小写与 ``.exe`` 后缀）。

    不能直接写 ``Path(command).name``：``\\`` **只在 Windows 上是路径分隔符**，
    于是 Linux 下传 ``C:\\Windows\\System32\\where.exe`` 会把整串当成一个文件名，
    拿它去比对白名单必然不通过（CI 在 ubuntu 上就是这么挂的）。
    先把 ``\\`` 统一成 ``/``（两种平台都认它作分隔符）再取 basename。
    """
    return Path(command.replace("\\", "/")).name


class CommandGuard:
    """命令守卫：白名单 + 明确拒绝名单 + 子命令收窄 + 危险字符串拦截。"""

    def __init__(
        self,
        *,
        allowed_commands: frozenset[str] = ALLOWED_COMMANDS,
        denied_commands: frozenset[str] = DENIED_COMMANDS,
        deny_patterns: Sequence[re.Pattern[str]] = DENY_PATTERNS,
    ) -> None:
        self.allowed_commands = allowed_commands
        self.denied_commands = denied_commands
        self.deny_patterns = tuple(deny_patterns)

    def validate(self, command: str, args: Sequence[str]) -> None:
        """校验一条命令是否允许执行。"""
        name = _command_basename(command).lower().removesuffix(".exe")
        normalized_args = [str(arg) for arg in args]

        if not name:
            raise SecurityViolationError("安全策略拒绝：命令为空")
        if name in self.denied_commands:
            raise SecurityViolationError(f"安全策略拒绝：命令 {name!r} 属于危险命令")
        if name not in self.allowed_commands:
            raise SecurityViolationError(
                f"安全策略拒绝：命令 {name!r} 不在白名单",
                detail=f"白名单：{sorted(self.allowed_commands)}",
            )

        self._reject_dangerous_text([name, *normalized_args])

        if name == "git":
            _require_readonly_subcommand(name, normalized_args, GIT_READONLY_SUBCOMMANDS)
        elif name in {"pip", "pip3"}:
            _require_readonly_subcommand(name, normalized_args, PIP_READONLY_SUBCOMMANDS)
        elif name in {"python", "python3", "py"}:
            denied = [arg for arg in normalized_args if arg in PYTHON_DENIED_FLAGS]
            if denied:
                raise SecurityViolationError(
                    f"安全策略拒绝：python 不允许使用 {denied}",
                    detail="禁止通过 -c/-m 执行任意代码或进入交互模式",
                )

    def _reject_dangerous_text(self, tokens: Sequence[str]) -> None:
        joined = " ".join(tokens)
        for pattern in self.deny_patterns:
            if pattern.search(joined):
                raise SecurityViolationError(
                    "安全策略拒绝：命中危险命令模式",
                    detail=f"pattern={pattern.pattern!r} input={joined!r}",
                )


def _require_readonly_subcommand(name: str, args: Sequence[str], allowlist: frozenset[str]) -> None:
    """校验首个非选项参数是否属于只读子命令。"""
    positional = [arg for arg in args if not arg.startswith("-")]
    if not positional:
        return  # 仅含选项（如 --version）
    if positional[0] not in allowlist:
        raise SecurityViolationError(
            f"安全策略拒绝：{name} 子命令 {positional[0]!r} 不在只读白名单",
            detail=f"允许的子命令：{sorted(allowlist)}",
        )


def _resolve_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if not resolved.exists():
        raise ValueError(f"允许根目录不存在：{resolved}")
    if not resolved.is_dir():
        raise ValueError(f"允许根目录不是目录：{resolved}")
    return resolved


@dataclass(frozen=True)
class SandboxPolicy:
    """沙箱策略：把"允许访问哪些目录""允许跑哪些命令"集中成一处配置。"""

    allowed_roots: tuple[Path, ...]
    allowed_commands: frozenset[str] = ALLOWED_COMMANDS
    denied_commands: frozenset[str] = DENIED_COMMANDS
    default_timeout: float = 30.0
    max_timeout: float = 120.0
    max_output_chars: int = 20000
    #: 默认工作目录，缺省取第一个允许根目录
    workdir: Path | None = None

    def __post_init__(self) -> None:
        if not self.allowed_roots:
            raise ValueError("SandboxPolicy 至少需要一个 allowed_root")
        if self.default_timeout <= 0 or self.max_timeout < self.default_timeout:
            raise ValueError("超时配置非法：需满足 0 < default_timeout <= max_timeout")

    @classmethod
    def for_roots(cls, roots: Iterable[str | Path], **overrides: Any) -> SandboxPolicy:
        """以一组根目录构造策略（会做存在性校验与绝对化）。"""
        return cls(allowed_roots=tuple(_resolve_root(root) for root in roots), **overrides)

    @property
    def default_workdir(self) -> Path:
        return self.workdir or self.allowed_roots[0]

    def clamp_timeout(self, timeout: float | None) -> float:
        """把请求的超时收敛到 [0, max_timeout] 区间。"""
        if timeout is None:
            return self.default_timeout
        try:
            value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(f"超时参数非法：{timeout!r}") from exc
        if value <= 0:
            raise ToolExecutionError(f"超时参数必须为正数：{timeout!r}")
        return min(value, self.max_timeout)

    def path_guard(self) -> PathGuard:
        return PathGuard(self.allowed_roots)

    def command_guard(self) -> CommandGuard:
        return CommandGuard(
            allowed_commands=self.allowed_commands,
            denied_commands=self.denied_commands,
        )


@dataclass
class CommandOutcome:
    """一次沙箱命令执行的结果。"""

    command: str
    args: list[str]
    cwd: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_text(self) -> str:
        """整形为喂给模型的文本：首行是退出码，命令自身错误放入 ``[stderr]`` 段。"""
        stdout = (self.stdout or "").strip()
        stderr = (self.stderr or "").strip()
        parts = [stdout] if stdout else []
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        body = "\n".join(parts).strip() or f"（无输出，exit={self.returncode}）"
        return f"[exit={self.returncode}]\n{body}"

    def to_meta(self) -> dict[str, Any]:
        """只含元数据的字典（剔除体积较大的 stdout/stderr）。"""
        meta = self.to_dict()
        meta.pop("stdout", None)
        meta.pop("stderr", None)
        return meta

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": " ".join([self.command, *self.args]).strip(),
            "cwd": self.cwd,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "duration": round(self.duration, 3),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def run_sandboxed(
    command: str,
    args: Sequence[str] = (),
    *,
    cwd: str | Path,
    timeout: float = 30.0,
    max_output_chars: int = 20000,
    extra_env: dict[str, str] | None = None,
) -> CommandOutcome:
    """在沙箱约束下执行外部命令（始终 ``shell=False``，不存在 shell 注入面）。

    Raises:
        ToolExecutionError: 可执行文件不存在或启动失败。
    """
    normalized_args = [str(arg) for arg in args]
    executable = find_executable(command)
    if executable is None:
        if _command_basename(command).lower() in SHIM_COMMANDS:
            return _run_shim(
                command,
                normalized_args,
                cwd=cwd,
                max_output_chars=max_output_chars,
            )
        raise ToolExecutionError(
            f"未找到可执行文件：{command}",
            detail="请确认该命令已安装且在 PATH 中",
        )

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if extra_env:
        env.update(extra_env)

    start = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603 - 已做白名单校验，且 shell=False
            [executable, *normalized_args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout, truncated_out = _clip(_as_text(exc.stdout), max_output_chars)
        stderr, truncated_err = _clip(_as_text(exc.stderr), max_output_chars // 2)
        return CommandOutcome(
            command=command,
            args=normalized_args,
            cwd=str(cwd),
            returncode=-1,
            stdout=stdout,
            stderr=stderr or f"命令执行超时（>{timeout}s），进程已被终止",
            duration=time.perf_counter() - start,
            timed_out=True,
            truncated=truncated_out or truncated_err,
        )
    except OSError as exc:
        raise ToolExecutionError(f"命令启动失败：{command}", detail=str(exc)) from exc

    stdout, truncated_out = _clip(completed.stdout or "", max_output_chars)
    stderr, truncated_err = _clip(completed.stderr or "", max_output_chars // 2)
    return CommandOutcome(
        command=command,
        args=normalized_args,
        cwd=str(cwd),
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration=time.perf_counter() - start,
        truncated=truncated_out or truncated_err,
    )


def find_executable(command: str) -> str | None:
    """查找可执行文件，找不到返回 ``None``（供工具做"未安装"降级判断）。

    查找顺序：PATH -> 当前解释器同级目录（虚拟环境的 Scripts/bin）。
    后者是必需的：以 ``.venv\\Scripts\\python.exe -m pytest`` 方式启动时，
    ``Scripts`` 目录通常**不在** PATH 中，直接用 ``shutil.which`` 会误判为未安装。
    """
    name = _command_basename(command)
    if name in {"python", "python3", "py"}:
        return sys.executable
    found = shutil.which(command)
    if found:
        return found
    sibling = Path(sys.executable).parent / name
    for candidate in (sibling, sibling.with_name(sibling.name + ".exe")):
        if candidate.is_file():
            return str(candidate)
    return None


class _ShimUsageError(Exception):
    """兜底命令的用法错误（等价于命令以退出码 2 退出）。"""


def _run_shim(
    command: str,
    args: Sequence[str],
    *,
    cwd: str | Path,
    max_output_chars: int,
) -> CommandOutcome:
    """用 Python 实现 cat/head/tail/ls/dir 的只读等价行为（仅当系统缺少该命令时调用）。"""
    workdir = Path(cwd)
    start = time.perf_counter()
    try:
        if command in FILE_VIEW_SHIMS:
            text = _file_view(command, args, workdir)
        else:
            text = _list_directory(command, args, workdir)
    except _ShimUsageError as exc:
        return CommandOutcome(
            command=command,
            args=list(args),
            cwd=str(workdir),
            returncode=2,
            stderr=str(exc),
            duration=time.perf_counter() - start,
        )
    except OSError as exc:
        return CommandOutcome(
            command=command,
            args=list(args),
            cwd=str(workdir),
            returncode=1,
            stderr=str(exc),
            duration=time.perf_counter() - start,
        )
    clipped, truncated = _clip(text, max_output_chars)
    return CommandOutcome(
        command=command,
        args=list(args),
        cwd=str(workdir),
        returncode=0,
        stdout=clipped,
        duration=time.perf_counter() - start,
        truncated=truncated,
    )


def _file_view(command: str, args: Sequence[str], cwd: Path) -> str:
    """解析 cat/head/tail 的参数并读取文件内容。"""
    show_numbers = False
    limit: int | None = None
    targets: list[str] = []

    index = 0
    while index < len(args):
        arg = args[index]
        if command == "cat" and arg in {"-n", "--number"}:
            show_numbers = True
        elif command in {"head", "tail"} and arg in {"-n", "--lines"}:
            index += 1
            limit = _positive_int(args[index] if index < len(args) else None, command)
        elif command in {"head", "tail"} and arg.startswith("-") and arg[1:].isdigit():
            limit = _positive_int(arg[1:], command)
        elif arg.startswith("-") and arg != "-":
            raise _ShimUsageError(f"{command} 不支持选项：{arg}")
        else:
            targets.append(arg)
        index += 1

    if not targets:
        raise _ShimUsageError(f"{command} 至少需要一个文件路径")
    if command in {"head", "tail"} and limit is None:
        limit = 10  # 与 POSIX 默认值一致

    blocks: list[str] = []
    for raw in targets:
        path = Path(raw)
        if not path.is_absolute():
            path = cwd / path
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if command == "head":
            lines = lines[:limit]
        elif command == "tail":
            lines = lines[-limit:] if limit else []
        if show_numbers:
            lines = [f"{number:>6}\t{line}" for number, line in enumerate(lines, 1)]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _list_directory(command: str, args: Sequence[str], cwd: Path) -> str:
    """解析 ls/dir 的参数并列出目录内容。"""
    show_all = False
    long_format = False
    targets: list[str] = []

    for arg in args:
        if arg in {"-a", "--all"}:
            show_all = True
        elif arg in {"-l", "--long"}:
            long_format = True
        elif arg.startswith("-") and arg != "-":
            raise _ShimUsageError(f"{command} 不支持选项：{arg}")
        else:
            targets.append(arg)

    if not targets:
        targets = ["."]

    blocks: list[str] = []
    for raw in targets:
        path = Path(raw)
        if not path.is_absolute():
            path = cwd / path
        if not path.exists():
            raise _ShimUsageError(f"路径不存在：{raw}")
        if path.is_file() or path.is_symlink():
            blocks.append(_format_entry(path, long_format))
            continue
        entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
        if not show_all:
            entries = [entry for entry in entries if not entry.name.startswith(".")]
        lines = [_format_entry(entry, long_format) for entry in entries]
        if len(targets) > 1:
            blocks.append(f"{path}:")
            blocks.extend(lines)
        else:
            blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _format_entry(path: Path, long_format: bool) -> str:
    if not long_format:
        return f"{path.name}/" if path.is_dir() else path.name
    kind = "-"
    if path.is_symlink():
        kind = "l"
    elif path.is_dir():
        kind = "d"
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return f"{kind} {size:>10} {path.name}"


def _positive_int(value: Any, command: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise _ShimUsageError(f"{command} 的行数参数非法：{value!r}") from exc
    if number <= 0:
        raise _ShimUsageError(f"{command} 的行数参数必须为正整数：{value!r}")
    return number


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _clip(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return f"{text[:limit]}\n...（输出被截断，原始长度 {len(text)} 字符）", True
