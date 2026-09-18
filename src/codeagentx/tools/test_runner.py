"""TestRunner：在沙箱内运行 pytest，并把结果解析成结构化统计。

为什么需要它：Agent 判断"某个改动是否安全"时，最硬的证据就是测试是否通过。
这里只做两件事：**约束执行**（沙箱、超时、路径）与**结果结构化**
（passed/failed 计数、失败用例清单、输出尾部），不解测试本身。

pytest 未安装时返回 ``error_type="ToolUnavailable"``，不抛异常。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from codeagentx.core.logger import get_logger
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.sandbox import SandboxPolicy, find_executable, run_sandboxed

logger = get_logger("tools.test_runner")

#: pytest 摘要行中的计数项（"3 failed, 10 passed in 1.23s"）
_COUNT_LABELS = (
    "passed",
    "failed",
    "error",
    "errors",
    "skipped",
    "xfailed",
    "xpassed",
    "deselected",
    "warnings",
)
_COUNT_PATTERN = re.compile(r"(\d+)\s+(" + "|".join(_COUNT_LABELS) + r")\b")

#: 失败用例清单行（pytest --tb=short 的 short test summary info 段）
_FAILURE_PATTERN = re.compile(r"^(FAILED|ERROR)\s+(.+)$", re.MULTILINE)

#: 输出尾部保留的行数
_TAIL_LINES = 60


class TestRunner(BaseTool):
    """在沙箱内运行 pytest 测试并返回结果统计。"""

    name = "test_runner"
    description = (
        "在沙箱内运行 pytest 测试，返回通过/失败/跳过计数、失败用例清单与输出尾部。"
        "用于验证改动是否破坏现有行为。只读运行，不会修改被测代码。"
    )
    dangerous = True
    parameters = [
        ToolParameter(
            name="target",
            type="string",
            description="测试目标（文件或目录，必须位于沙箱允许范围内），默认当前目录",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="args",
            type="array",
            description="额外的 pytest 参数，例如 [\"-k\", \"login\"] 或 [\"--maxfail=1\"]",
            required=False,
            default=[],
            items={"type": "string"},
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
        target: str = ".",
        args: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> ToolResult:
        # 1. 定位目标并组装参数：先做路径/参数校验，再判断环境
        workdir = self.policy.default_workdir
        resolved_target = self._paths.resolve(target, base=workdir, must_exist=True)
        argv = [
            _display_path(resolved_target, workdir),
            # 目标项目的 addopts 可能改变输出格式：例如其自身配置了 -q，
            # 再叠加我们的 -q 会变成 -qq，pytest 将**完全不打印摘要行**，导致结果无法解析。
            # 因此这里清空 addopts，由本工具自行控制输出格式。
            "--override-ini=addopts=",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            *[str(item) for item in (args or [])],
        ]
        self._commands.validate("pytest", argv)
        self._paths.validate_path_like_arguments(argv, base=workdir)

        # 2. 环境降级：pytest 未安装时给出可解释的失败
        if find_executable("pytest") is None:
            return ToolResult.fail(
                "未找到 pytest，无法运行测试",
                error_type="ToolUnavailable",
                hint="请安装 pytest（pip install pytest）后重试",
            )

        # 3. 执行
        outcome = run_sandboxed(
            "pytest",
            argv,
            cwd=workdir,
            timeout=self.policy.clamp_timeout(timeout),
            max_output_chars=self.policy.max_output_chars,
        )
        combined = f"{outcome.stdout}\n{outcome.stderr}".strip()
        meta = {
            **outcome.to_meta(),
            "target": _display_path(resolved_target, workdir),
            "pytest_passed": outcome.returncode == 0,
        }

        if outcome.timed_out:
            return ToolResult.fail("pytest 执行超时", error_type="TimeoutExpired", **meta)

        return ToolResult.ok(
            {
                "target": _display_path(resolved_target, workdir),
                "exit_code": outcome.returncode,
                "passed": outcome.returncode == 0,
                "counts": _parse_counts(combined),
                "summary_line": _summary_line(combined),
                "failures": _extract_failures(combined),
                "output_tail": _tail(combined, _TAIL_LINES),
            },
            **meta,
        )


def _display_path(path: Path, workdir: Path) -> str:
    """尽量输出相对路径；目标即工作目录时用 ``.``。"""
    try:
        return path.relative_to(workdir).as_posix() or "."
    except ValueError:
        return str(path)


def _parse_counts(text: str) -> dict[str, int]:
    """从摘要行中提取计数，例如 ``3 failed, 10 passed`` -> {failed: 3, passed: 10}。"""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        for number, label in _COUNT_PATTERN.findall(line):
            key = "error" if label == "errors" else label
            if key == "warnings":
                continue  # warnings 与用例通过/失败无关，不参与统计
            counts[key] = counts.get(key, 0) + int(number)
    return counts


def _summary_line(text: str) -> str:
    """返回最后一行含计数信息的摘要行。"""
    for line in reversed(text.splitlines()):
        if _COUNT_PATTERN.search(line):
            return line.strip()
    return ""


def _extract_failures(text: str) -> list[str]:
    """提取失败用例清单（``FAILED <nodeid> - <原因>``）。"""
    return [f"{kind} {detail}".strip() for kind, detail in _FAILURE_PATTERN.findall(text)]


def _tail(text: str, lines: int) -> str:
    if lines <= 0:
        return ""
    return "\n".join(text.splitlines()[-lines:])
