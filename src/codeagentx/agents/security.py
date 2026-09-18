"""Security 角色：主责安全风险，机制完全复用 :class:`FocusedReviewAgent`。

为什么安全必须单独一个角色
--------------------------
"顺手也看看安全"在实际运行中等于"没人看安全"：Reviewer 的注意力会被
可读性、异常处理这些更容易发现的问题占满，而 SQL 拼接、硬编码密钥这类问题
往往藏在看起来正常的数据流里，需要专门沿着"外部输入 → 敏感操作"这条线去追。

行为约束与 Reviewer 一致（同基类）：工具轮次有上限、未收敛即 ``success=False``、
结论解析失败带 ``parse_error``。安全结论尤其怕"假装没问题"，
所以这里不做任何"兜底理解"式的解析放宽。
"""

from __future__ import annotations

from collections.abc import Sequence

from codeagentx.agents.reviewer import FOCUSED_REVIEW_TOOLS, FocusedReviewAgent
from codeagentx.core.llm import BaseLLM
from codeagentx.prompts.agents import SECURITY_SYSTEM
from codeagentx.tools.registry import ToolRegistry


class SecurityAgent(FocusedReviewAgent):
    """安全方向的审查角色。"""

    name = "security"
    focus = "注入（SQL/命令/路径）、硬编码凭据、不安全反序列化、弱加密、鉴权缺失、敏感信息泄露"
    default_task = "审查目标代码中的安全风险，确认外部输入能否被攻击者控制、影响范围是什么。"

    def __init__(
        self,
        llm: BaseLLM,
        *,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 6,
        allowed_tools: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            llm,
            tools=tools,
            system_prompt=system_prompt or SECURITY_SYSTEM,
            max_iterations=max_iterations,
            allowed_tools=allowed_tools or FOCUSED_REVIEW_TOOLS,
        )


__all__ = ["SecurityAgent"]
