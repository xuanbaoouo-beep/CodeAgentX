"""GitTool：以**只读**方式读取仓库信息（status / log / diff / show / branch / blame / ls-files）。

降级约定（本机 git 可能未安装）：
- git 不在 PATH 时返回 ``error_type="ToolUnavailable"``，而不是抛异常；
- 目标目录不是 Git 仓库时返回 ``error_type="NotAGitRepository"``；
两者都是**环境信息**而非工具故障，上层（Agent）据此可以切换到纯文件读取策略。

写操作（commit / push / checkout 等）在沙箱层就被拒绝，本工具也不提供对应参数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from codeagentx.core.exceptions import ToolValidationError
from codeagentx.core.logger import get_logger
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.sandbox import SandboxPolicy, find_executable, run_sandboxed

logger = get_logger("tools.git")

#: 支持的只读动作
ACTIONS: frozenset[str] = frozenset(
    {"status", "log", "diff", "show", "branch", "blame", "ls-files"}
)

#: log 最多返回的提交条数上限
MAX_LOG_ENTRIES = 200

#: 需要 target 参数的动作
_NEEDS_TARGET: frozenset[str] = frozenset({"blame"})

_NOT_A_REPO_HINTS = ("not a git repository", "not a git repo")

#: git 返回该退出码表示致命错误（如引用不存在、不是仓库）
_GIT_FATAL = 128


class GitTool(BaseTool):
    """读取 Git 仓库信息（只读）。"""

    name = "git"
    description = (
        "读取 Git 仓库信息（只读）：status 看工作区改动，log 看提交历史，"
        "diff 看差异，show 看某次提交，branch 看分支，blame 看某文件逐行归属，"
        "ls-files 列出被跟踪文件。不支持任何写操作。"
    )
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="要执行的只读动作",
            enum=sorted(ACTIONS),
        ),
        ToolParameter(
            name="target",
            type="string",
            description="引用或文件路径，例如 HEAD~1 / main / src/main.py（diff/show/blame 使用）",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description=f"log 返回的最大提交条数（1~{MAX_LOG_ENTRIES}），默认 20",
            required=False,
            default=20,
        ),
        ToolParameter(
            name="repo",
            type="string",
            description="仓库目录（必须位于沙箱允许范围内），缺省用沙箱默认工作目录",
            required=False,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="超时秒数，会被收敛到策略允许的最大值",
            required=False,
        ),
    ]

    def __init__(self, policy: SandboxPolicy) -> None:
        super().__init__()
        self.policy = policy
        self._paths = policy.path_guard()
        self._commands = policy.command_guard()

    def _run(
        self,
        action: str,
        target: str | None = None,
        limit: int = 20,
        repo: str | None = None,
        timeout: float | None = None,
    ) -> ToolResult:
        normalized_action = str(action).strip().lower()
        if normalized_action not in ACTIONS:
            raise ToolValidationError(
                f"不支持的 git 动作：{action!r}",
                detail=f"可选动作：{sorted(ACTIONS)}",
            )
        if normalized_action in _NEEDS_TARGET and not target:
            raise ToolValidationError(f"git {normalized_action} 需要 target 参数")

        # 1. 定位仓库目录并构造参数（安全与参数校验必须先于环境检查）
        workdir = self._resolve_workdir(repo)
        argv = _build_argv(normalized_action, target, _clamp_limit(limit))
        self._commands.validate("git", argv)
        self._paths.validate_path_like_arguments(argv, base=workdir)

        # 2. 环境降级：git 未安装时不抛异常，交回一个可解释的失败结果
        if find_executable("git") is None:
            return ToolResult.fail(
                "未找到 git 可执行文件，GitTool 当前不可用",
                error_type="ToolUnavailable",
                hint="请安装 Git 并确保其位于 PATH 中；也可改用 terminal/static_analyzer 走纯文件分析",
                action=normalized_action,
            )

        # 3. 执行
        outcome = run_sandboxed(
            "git",
            argv,
            cwd=workdir,
            timeout=self.policy.clamp_timeout(timeout),
            max_output_chars=self.policy.max_output_chars,
        )
        meta = {**outcome.to_meta(), "action": normalized_action, "repo": str(workdir)}

        if outcome.timed_out:
            return ToolResult.fail(
                f"git {normalized_action} 执行超时", error_type="TimeoutExpired", **meta
            )
        if _is_not_a_repo(outcome.stderr):
            return ToolResult.fail(
                f"{workdir} 不是 Git 仓库",
                error_type="NotAGitRepository",
                hint="请确认 repo 参数指向仓库根目录",
                **meta,
            )
        if outcome.returncode == _GIT_FATAL:
            return ToolResult.fail(
                f"git {normalized_action} 执行失败：{outcome.stderr.strip() or 'git 返回致命错误'}",
                error_type="GitCommandError",
                **meta,
            )
        return ToolResult.ok(outcome.to_text(), **meta)

    def _resolve_workdir(self, repo: str | None) -> Path:
        if not repo:
            return self.policy.default_workdir
        return self._paths.resolve(repo, base=self.policy.default_workdir, must_exist=True)


def _build_argv(action: str, target: str | None, limit: int) -> list[str]:
    """把动作翻译成 git 参数列表（参数经命令守卫二次校验）。"""
    if action == "status":
        return ["status", "--short", "--branch"]
    if action == "log":
        return ["log", f"--max-count={limit}", "--date=short", "--pretty=format:%h %ad %an %s"]
    if action == "diff":
        return ["diff", "--no-color", *([target] if target else [])]
    if action == "show":
        return ["show", "--no-color", "--stat", target or "HEAD"]
    if action == "branch":
        return ["branch", "--list", "--no-color"]
    if action == "blame":
        return ["blame", "--date=short", str(target)]
    if action == "ls-files":
        return ["ls-files"]
    raise ToolValidationError(f"不支持的 git 动作：{action!r}")


def _clamp_limit(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"limit 必须是整数：{value!r}") from exc
    if limit <= 0:
        raise ToolValidationError(f"limit 必须为正整数：{value!r}")
    return min(limit, MAX_LOG_ENTRIES)


def _is_not_a_repo(stderr: str) -> bool:
    text = (stderr or "").lower()
    return any(hint in text for hint in _NOT_A_REPO_HINTS)
