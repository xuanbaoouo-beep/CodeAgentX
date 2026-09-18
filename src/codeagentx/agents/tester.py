"""Tester 角色：为已发现的问题写"可运行的复现测试"，并能跑一遍现有测试套件。

能力边界（写在最前面，避免误解）
--------------------------------
本项目的沙箱**没有写文件工具**（工具层只有只读的 terminal / git / static_analyzer /
code_search / test_runner），因此 Tester 不会把生成的测试文件写进仓库，而是：

1. 产出**测试代码文本**，随报告一起交付（人可以复制到仓库里直接跑）；
2. 可选地调用 ``test_runner`` 跑**仓库现有的**测试套件，给出回归基线
   （现有用例是否通过，是判断重构是否破坏行为的关键证据）。

为什么不假装"已执行你写的测试"：那是本次任务无法兑现的承诺，
报告里必须能区分"我写了测试"与"我跑了测试"，这两件事的证据完全不同。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from codeagentx.core.agent import Agent, AgentResult
from codeagentx.core.llm import BaseLLM
from codeagentx.core.logger import get_logger, log_event
from codeagentx.prompts.agents import TESTER_SYSTEM, build_test_task
from codeagentx.tools.registry import ToolRegistry

logger = get_logger("agents.tester")

#: 运行器的工具名（tools 层实现）
TEST_TOOL = "test_runner"
#: 未指定测试目标时
DEFAULT_TEST_TARGET = "（未指定，针对目标仓库整体）"

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_TEST_HINT_RE = re.compile(r"^\s*(?:async\s+)?def\s+test_\w+\s*\(", re.MULTILINE)


def extract_test_code(text: str) -> str:
    """从模型输出中提取测试代码。

    优先取第一个 ```python 代码块；没有围栏时，如果正文里出现 ``def test_xxx(``
    就认为整段输出即是测试代码（模型直接贴代码也是常见情况）。
    """
    if not text:
        return ""
    blocks = _CODE_FENCE_RE.findall(text)
    if blocks:
        return blocks[0].strip()
    if _TEST_HINT_RE.search(text):
        return text.strip()
    return ""


class TesterAgent(Agent):
    """测试生成 + 回归验证角色。"""

    name = "tester"
    #: 类名以 Test 开头会让 pytest 误当成测试类收集（会报警告并跳过），显式声明不是测试
    __test__ = False

    def __init__(
        self,
        llm: BaseLLM,
        *,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 3,
    ) -> None:
        super().__init__(
            llm,
            system_prompt=system_prompt or TESTER_SYSTEM,
            tools=tools,
            max_iterations=max_iterations,
        )

    # ------------------------------------------------------------ 程序接口
    def verify(
        self,
        target: str | None = None,
        *,
        args: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """运行仓库现有测试，返回结构化结果。

        返回 ``available=False`` 表示当前 Agent 没挂载 ``test_runner``
        （或环境不支持），此时调用方必须把它当作"未验证"，而不是"通过"。
        """
        if self.tools is None or TEST_TOOL not in self.tools.names():
            return {
                "available": False,
                "success": False,
                "error": f"当前未挂载 {TEST_TOOL} 工具，无法运行测试",
            }
        payload: dict[str, Any] = {"target": target or "."}
        if args:
            payload["args"] = list(args)
        if timeout is not None:
            payload["timeout"] = timeout
        outcome = self.tools.execute(TEST_TOOL, payload)
        if not outcome.success:
            return {"available": True, "success": False, "error": outcome.error}
        detail = outcome.output if isinstance(outcome.output, dict) else {}
        return {"available": True, **detail}

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        target: str = DEFAULT_TEST_TARGET,
        *,
        finding: str = "",
        context: str = "",
        run_existing: bool = False,
        test_path: str | None = None,
        reset: bool = True,
        **_: Any,
    ) -> AgentResult:
        """生成测试代码，并（可选）运行现有测试作为回归基线。

        Args:
            target: 测试目标（文件或模块）。
            finding: 需要复现/验证的具体问题（通常是一条 Finding 的文本）。
            context: 相关代码片段等上下文。
            run_existing: 是否顺带跑一遍现有测试套件。
            test_path: 运行测试的路径，缺省用 ``.``（仓库根）。
            reset: 是否清空历史。
        """
        if reset:
            self.reset()
        prompt = build_test_task(target or DEFAULT_TEST_TARGET, finding=finding, context=context)
        result = self.tool_loop(self._seed_messages(prompt))

        code = extract_test_code(result.output)
        test_result = self.verify(test_path) if run_existing else None
        result.metadata["agent"] = self.name
        result.metadata["test_code"] = code
        result.metadata["test_result"] = test_result
        result.metadata["target"] = target or DEFAULT_TEST_TARGET
        if not code:
            # 没写出测试就不能算完成：宁可报告"这一步失败"，也不要给出一份空交付
            result.success = False
            result.error = "未能从模型输出中提取到测试代码"
        log_event(
            logger,
            "tester_finished",
            agent=self.name,
            success=result.success,
            has_code=bool(code),
            ran_existing=bool(test_result),
        )
        return result


__all__ = ["DEFAULT_TEST_TARGET", "TEST_TOOL", "TesterAgent", "extract_test_code"]
