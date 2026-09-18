"""TerminalTool：受安全沙箱约束的命令执行工具。

与 :mod:`codeagentx.tools.sandbox` 的分工：
- 沙箱负责"能不能执行"（白名单、路径、超时）；
- 本工具负责"怎么暴露给模型"（参数 schema、输出整形）。

语义约定：``ToolResult.success`` 表示**工具是否正常执行完**，
命令自身的退出码（如 ``grep`` 无匹配返回 1）放在输出里由模型判断，
避免把"正常但非零退出"误报成工具故障。
"""

from __future__ import annotations

from collections.abc import Sequence

from codeagentx.core.exceptions import ToolExecutionError
from codeagentx.core.logger import get_logger
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.sandbox import SandboxPolicy, run_sandboxed

logger = get_logger("tools.terminal")


class TerminalTool(BaseTool):
    """在沙箱内执行白名单命令。"""

    name = "terminal"
    description = (
        "在安全沙箱内执行白名单命令（只读/分析类），用于查看文件、搜索代码、"
        "运行测试与静态检查。禁止 rm/curl/sudo/管道下载等危险操作，"
        "禁止访问工作区之外的路径，单条命令有超时上限。"
        "工作目录默认就是审查目标根目录：路径请写相对路径，不要重复拼目标目录名。"
        "跨平台可靠的选择：ls（列目录，不带参数即当前目录）、cat / head / tail（读文件内容）、"
        "pytest / ruff（检查与测试）。"
        "注意：ls 不支持 -R；python 禁止 -c / -m；find / grep / dir /s 在部分平台不可用或"
        "语义不同，不要依赖它们做递归遍历——需要批量看代码请用 code_search。"
    )
    dangerous = True
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="可执行命令名，例如 ls / cat / head / tail / pytest / ruff / git",
        ),
        ToolParameter(
            name="args",
            type="array",
            description="命令参数列表，例如 [\"-n\", \"src/main.py\"]",
            required=False,
            default=[],
            items={"type": "string"},
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="工作目录（必须位于允许根目录内）。缺省为沙箱默认工作目录。",
            required=False,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="超时秒数，会被收敛到策略允许的最大值。",
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
        command: str,
        args: Sequence[str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> ToolResult:
        argv = [str(item) for item in (args or [])]

        # 1. 命令层校验（白名单 / 黑名单 / 子命令收窄 / 危险模式）
        self._commands.validate(command, argv)

        # 2. 工作目录必须落在允许范围内
        workdir = self.policy.default_workdir
        if cwd:
            workdir = self._paths.resolve(cwd, base=self.policy.default_workdir, must_exist=True)
        if not workdir.is_dir():
            raise ToolExecutionError(f"工作目录不是目录：{workdir}")

        # 3. 路径型参数不得越界
        self._paths.validate_path_like_arguments(argv, base=workdir)

        # 4. 执行
        effective_timeout = self.policy.clamp_timeout(timeout)
        outcome = run_sandboxed(
            command,
            argv,
            cwd=workdir,
            timeout=effective_timeout,
            max_output_chars=self.policy.max_output_chars,
        )

        if outcome.timed_out:
            return ToolResult.fail(
                f"命令执行超时（>{effective_timeout}s）：{command}",
                error_type="TimeoutExpired",
                **outcome.to_dict(),
            )

        return ToolResult.ok(
            outcome.to_text(),
            command_line=" ".join([command, *argv]),
            **outcome.to_meta(),
        )
