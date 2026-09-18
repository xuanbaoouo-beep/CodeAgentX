"""Refactor 角色：把审查结论翻译成"可执行、可验收"的重构计划。

W5 已经有完整的 Plan-and-Solve 实现（:class:`~codeagentx.agents.plan_solve_refactor.PlanSolveRefactor`），
W6 的 Refactor 角色不重复造轮子，只补一件多 Agent 场景必需的事：
**把上一步产出的 Finding 列表转成重构目标**（含位置清单），
否则"重构一下这些问题"这种模糊目标会让规划阶段失焦。

另外约定：工作流默认只取**计划**（``execute=False``），不逐步骤执行——
重构是写操作的前置，必须在人确认后手动触发，自动跑完会让人措手不及。
"""

from __future__ import annotations

from collections.abc import Sequence

from codeagentx.agents.plan_solve_refactor import DEFAULT_GOAL, DEFAULT_MAX_STEPS, PlanSolveRefactor
from codeagentx.agents.schemas import Finding, RefactorPlan, ReviewReport
from codeagentx.core.llm import BaseLLM
from codeagentx.tools.registry import ToolRegistry

#: 目标描述里最多列出的问题条数（其余合并说明，避免目标文本过长）
MAX_FINDINGS_IN_GOAL = 5


def build_goal_from_findings(findings: Sequence[Finding], *, limit: int = MAX_FINDINGS_IN_GOAL) -> str:
    """把审查发现转成重构目标描述（按严重度排序，带位置，便于模型定位）。"""
    ordered = ReviewReport(findings=list(findings)).sorted_findings()
    if not ordered:
        return DEFAULT_GOAL
    lines = ["在保持外部行为不变的前提下，修复以下审查发现的问题："]
    for item in ordered[:limit]:
        location = item.location or "未定位"
        lines.append(f"- [{item.severity}][{item.category}] {item.title} @ {location}")
    if len(ordered) > limit:
        lines.append(f"- （另有 {len(ordered) - limit} 条同类问题，合并处理）")
    return "\n".join(lines)


class RefactorAgent(PlanSolveRefactor):
    """重构计划角色（复用 Plan-and-Solve 的规划与执行能力）。"""

    name = "refactor"

    def __init__(
        self,
        llm: BaseLLM,
        *,
        tools: ToolRegistry | None = None,
        max_iterations: int = 4,
        max_steps: int = DEFAULT_MAX_STEPS,
        plan_system_prompt: str | None = None,
        solve_system_prompt: str | None = None,
        allowed_tools: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            llm,
            tools=tools,
            max_iterations=max_iterations,
            max_steps=max_steps,
            plan_system_prompt=plan_system_prompt,
            solve_system_prompt=solve_system_prompt,
            allowed_tools=allowed_tools,
        )

    def plan_from_findings(
        self,
        findings: Sequence[Finding],
        *,
        goal: str = "",
        scope: str = "",
        context: str = "",
    ) -> RefactorPlan:
        """按审查结论制定重构计划（只规划，不执行）。"""
        resolved = goal.strip() or build_goal_from_findings(findings)
        return self.plan(resolved, scope=scope, context=context)


__all__ = ["MAX_FINDINGS_IN_GOAL", "RefactorAgent", "build_goal_from_findings"]
