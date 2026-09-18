"""StaticAnalyzer：封装 ruff / pylint / bandit，输出统一的结构化问题列表。

设计要点
--------
1. **统一结构**：三个工具的原始输出格式各不相同，这里统一成
   ``{tool, file, line, column, code, severity, message}``，
   让上层 Agent 不必为每个工具写一套解析逻辑。
2. **缺失降级**：工具未安装时返回 ``error_type="ToolUnavailable"`` 而**不抛异常**，
   Agent 可以据此跳过该静态检查，或改用其他工具继续审查。
3. **严重度是启发式**：ruff 本身不输出严重度，由规则前缀推断（见 :func:`_ruff_severity`），
   仅用于排序与聚合，不应作为最终结论。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeagentx.core.exceptions import ToolValidationError
from codeagentx.core.logger import get_logger
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.sandbox import SandboxPolicy, find_executable, run_sandboxed

logger = get_logger("tools.static_analyzer")

#: 支持的静态分析工具
ANALYZERS: frozenset[str] = frozenset({"ruff", "pylint", "bandit"})

#: 默认返回的问题条数上限（避免把上下文塞满）
DEFAULT_MAX_FINDINGS = 100
MAX_FINDINGS_LIMIT = 500

#: pylint 的 type -> 统一严重度
_PYLINT_SEVERITY = {
    "fatal": "high",
    "error": "high",
    "warning": "medium",
    "refactor": "low",
    "convention": "low",
    "info": "low",
}

#: bandit 的 issue_severity -> 统一严重度
_BANDIT_SEVERITY = {"high": "high", "medium": "medium", "low": "low"}


class StaticAnalyzer(BaseTool):
    """对指定路径运行静态检查并返回结构化问题列表。"""

    name = "static_analyzer"
    description = (
        "对代码运行静态分析并返回结构化问题列表（文件/行号/规则号/严重度/描述）。"
        "可选 ruff（快速 lint）、pylint（深度检查）、bandit（安全扫描）。"
        "只读操作，不会修改任何文件。"
    )
    parameters = [
        ToolParameter(
            name="tool",
            type="string",
            description="使用的分析工具，默认 ruff",
            required=False,
            default="ruff",
            enum=sorted(ANALYZERS),
        ),
        ToolParameter(
            name="target",
            type="string",
            description="待分析的路径（文件或目录，必须位于沙箱允许范围内），默认当前目录",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="max_findings",
            type="integer",
            description=f"最多返回的问题条数（1~{MAX_FINDINGS_LIMIT}），默认 {DEFAULT_MAX_FINDINGS}",
            required=False,
            default=DEFAULT_MAX_FINDINGS,
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
        tool: str = "ruff",
        target: str = ".",
        max_findings: int = DEFAULT_MAX_FINDINGS,
        timeout: float | None = None,
    ) -> ToolResult:
        analyzer = str(tool).strip().lower()
        if analyzer not in ANALYZERS:
            raise ToolValidationError(
                f"不支持的静态分析工具：{tool!r}",
                detail=f"可选：{sorted(ANALYZERS)}",
            )
        limit = _clamp_max_findings(max_findings)

        # 1. 目标路径必须落在沙箱允许范围内，并构造参数（先于环境检查）
        workdir = self.policy.default_workdir
        resolved_target = self._paths.resolve(target, base=workdir, must_exist=True)
        argv = _build_argv(analyzer, resolved_target, workdir)
        self._commands.validate(analyzer, argv)
        self._paths.validate_path_like_arguments(argv, base=workdir)

        # 2. 环境降级：工具未安装时给出可解释的失败，而非异常
        if find_executable(analyzer) is None:
            return ToolResult.fail(
                f"未找到 {analyzer}，本次静态检查已跳过",
                error_type="ToolUnavailable",
                hint=f"请安装 {analyzer}（例如 pip install {analyzer}）后重试",
                tool=analyzer,
            )

        # 3. 执行
        outcome = run_sandboxed(
            analyzer,
            argv,
            cwd=workdir,
            timeout=self.policy.clamp_timeout(timeout),
            max_output_chars=self.policy.max_output_chars,
        )
        if outcome.timed_out:
            return ToolResult.fail(
                f"{analyzer} 执行超时", error_type="TimeoutExpired", tool=analyzer
            )

        # 5. 解析（解析失败不丢信息：把原始输出一并交回）
        try:
            findings = _parse_findings(analyzer, outcome.stdout)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            logger.debug("解析 %s 输出失败", analyzer, exc_info=True)
            return ToolResult.ok(
                {
                    "tool": analyzer,
                    "target": _display_path(resolved_target, workdir),
                    "exit_code": outcome.returncode,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                    "raw_output": outcome.stdout.strip()[:4000],
                    "raw_stderr": outcome.stderr.strip()[:1000],
                },
                tool=analyzer,
                parse_error=True,
            )

        return ToolResult.ok(
            {
                "tool": analyzer,
                "target": _display_path(resolved_target, workdir),
                "exit_code": outcome.returncode,
                "finding_count": len(findings),
                "truncated": len(findings) > limit,
                "severity_counts": _severity_counts(findings),
                "findings": findings[:limit],
            },
            tool=analyzer,
            finding_count=len(findings),
        )


def _build_argv(analyzer: str, target: Path, workdir: Path) -> list[str]:
    """把目标路径转成相对于工作目录的形式（输出更短、更可读）。"""
    display = _display_path(target, workdir)
    if analyzer == "ruff":
        return ["check", display, "--output-format=json", "--no-cache"]
    if analyzer == "pylint":
        return [display, "--output-format=json", "--score=n", "--reports=n"]
    if analyzer == "bandit":
        # -r 递归扫描目录；对单文件同样适用
        return ["-r", display, "-f", "json", "-q"]
    raise ToolValidationError(f"不支持的静态分析工具：{analyzer!r}")


def _parse_findings(analyzer: str, stdout: str) -> list[dict[str, Any]]:
    payload = json.loads(stdout or "[]")
    if analyzer == "ruff":
        return _parse_ruff(payload)
    if analyzer == "pylint":
        return _parse_pylint(payload)
    if analyzer == "bandit":
        return _parse_bandit(payload)
    raise ToolValidationError(f"不支持的静态分析工具：{analyzer!r}")


def _parse_ruff(payload: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in payload or []:
        code = str(item.get("code") or "")
        location = item.get("location") or {}
        findings.append(
            {
                "tool": "ruff",
                "file": str(item.get("filename") or ""),
                "line": int(location.get("row") or 0),
                "column": int(location.get("column") or 0),
                "code": code,
                "severity": _ruff_severity(code),
                "message": str(item.get("message") or ""),
            }
        )
    return findings


def _parse_pylint(payload: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").lower()
        findings.append(
            {
                "tool": "pylint",
                "file": str(item.get("path") or ""),
                "line": int(item.get("line") or 0),
                "column": int(item.get("column") or 0),
                "code": str(item.get("message-id") or item.get("symbol") or ""),
                "severity": _PYLINT_SEVERITY.get(kind, "low"),
                "message": str(item.get("message") or ""),
            }
        )
    return findings


def _parse_bandit(payload: Any) -> list[dict[str, Any]]:
    results = (payload or {}).get("results") or []
    findings: list[dict[str, Any]] = []
    for item in results:
        severity = str(item.get("issue_severity") or "").lower()
        findings.append(
            {
                "tool": "bandit",
                "file": str(item.get("filename") or ""),
                "line": int(item.get("line_number") or 0),
                "column": int(item.get("col_offset") or 0),
                "code": str(item.get("test_id") or ""),
                "severity": _BANDIT_SEVERITY.get(severity, "low"),
                "message": str(item.get("issue_text") or ""),
            }
        )
    return findings


def _ruff_severity(code: str) -> str:
    """由 ruff 规则号推断严重度（启发式，仅用于排序与聚合）。"""
    upper = code.upper()
    if upper.startswith("E9") or upper.startswith("S"):  # 语法错误 / bandit 安全规则
        return "high"
    if upper.startswith("F"):  # pyflakes：未定义名、未使用导入等
        return "medium"
    return "low"


def _severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    return counts


def _clamp_max_findings(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"max_findings 必须是整数：{value!r}") from exc
    if limit <= 0:
        raise ToolValidationError(f"max_findings 必须为正整数：{value!r}")
    return min(limit, MAX_FINDINGS_LIMIT)


def _display_path(path: Path, workdir: Path) -> str:
    """尽量输出相对路径，避免把绝对路径带进上下文。"""
    try:
        return path.relative_to(workdir).as_posix() or "."
    except ValueError:
        return str(path)
