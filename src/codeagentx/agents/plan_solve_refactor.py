"""Plan-and-Solve 重构 Agent：先出计划，再逐步执行，每步单独取证。

为什么重构要用 Plan-and-Solve
-----------------------------
重构是**多步、有顺序依赖**的任务：先看清现状 → 再等价改写 → 最后验证。
让模型一口气输出"重构后的代码"，得到的往往是大而全、无法验证的一坨。
Plan-and-Solve 把它拆成"规划 → 逐步执行"，每步都有独立产出与完成标志，
既能逐步审查，也能中途停下（对应 W6 的中断恢复需求）。

阶段划分
--------
1. :meth:`plan`（规划，纯 LLM）：产出 :class:`~codeagentx.agents.schemas.RefactorPlan`；
   规划阶段不挂工具——此时还不该动代码，先把意图和步骤定清楚。
2. :meth:`execute_step`（执行，可挂工具）：逐步执行，允许 :code:`code_search` /
   :code:`terminal` 查看真实代码。

用量统计
--------
规划 + N 步执行会发起多次 LLM 调用，因此用量取"进入 :meth:`run` 前后的快照差"，
而不是单次调用的用量。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from codeagentx.agents.schemas import RefactorPlan, parse_json_payload, plan_from_payload
from codeagentx.core.agent import Agent, AgentResult, usage_delta
from codeagentx.core.exceptions import AgentOutputError
from codeagentx.core.llm import BaseLLM
from codeagentx.core.logger import get_logger, log_event
from codeagentx.core.message import Message
from codeagentx.prompts.refactor import PLAN_SYSTEM, SOLVE_SYSTEM, build_plan_task, build_solve_task
from codeagentx.tools.registry import ToolRegistry

logger = get_logger("agents.plan_solve_refactor")

#: 未指定目标时的默认重构目标
DEFAULT_GOAL = "在不改变外部行为的前提下，重构目标代码以消除已发现的缺陷与坏味道。"

#: 计划步骤数上限（防止模型规划出几十步，既跑不完也无法验收）
DEFAULT_MAX_STEPS = 7


class PlanSolveRefactor(Agent):
    """Plan-and-Solve 模式的重构 Agent。"""

    name = "plan_solve_refactor"

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
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")
        super().__init__(llm, tools=tools, system_prompt=None, max_iterations=max_iterations)
        self.max_steps = max_steps
        self.plan_system_prompt = plan_system_prompt or PLAN_SYSTEM
        self.solve_system_prompt = solve_system_prompt or SOLVE_SYSTEM
        self.allowed_tools = list(allowed_tools) if allowed_tools else None

    # ------------------------------------------------------------ 规划阶段
    def plan(self, goal: str, *, scope: str = "", context: str = "") -> RefactorPlan:
        """产出重构计划（不调用工具，纯 LLM）。"""
        prompt = build_plan_task(goal, scope=scope, context=context)
        response = self.llm.chat(
            [Message.system(self.plan_system_prompt), Message.user(prompt)]
        )
        return self._parse_plan(response.content, goal)

    def _parse_plan(self, content: str, goal: str) -> RefactorPlan:
        try:
            plan = plan_from_payload(parse_json_payload(content))
        except AgentOutputError as exc:
            log_event(
                logger,
                "plan_parse_failed",
                level=logging.WARNING,
                agent=self.name,
                error=str(exc),
            )
            plan = RefactorPlan(goal=goal, notes=f"规划阶段输出无法解析为 JSON：{exc}")

        if not plan.goal:
            plan.goal = goal
        if plan.total > self.max_steps:
            truncated = plan.total - self.max_steps
            plan.steps = plan.steps[: self.max_steps]
            note = f"原计划 {plan.total + truncated} 步，按 max_steps={self.max_steps} 截断。"
            plan.notes = "\n".join(item for item in (plan.notes, note) if item)
        return plan

    # ------------------------------------------------------------ 执行阶段
    def execute_step(self, plan: RefactorPlan, index: int, *, context: str = "") -> AgentResult:
        """执行计划中的第 ``index`` 步（从 0 开始），并把结果写回该步骤。"""
        if not 0 <= index < plan.total:
            raise IndexError(f"步骤下标越界：{index}（共 {plan.total} 步）")

        step = plan.steps[index]
        prompt = build_solve_task(
            plan.goal,
            step.description,
            step_index=index + 1,
            step_total=plan.total,
            rationale=step.rationale,
            context=context,
        )
        self.reset()
        result = self.tool_loop(
            [Message.system(self.solve_system_prompt), Message.user(prompt)],
            allowed_tools=self.allowed_tools,
        )
        step.result = result.output.strip()
        step.status = "done" if result.success else "failed"
        log_event(
            logger,
            "plan_step_finished",
            agent=self.name,
            index=index + 1,
            total=plan.total,
            status=step.status,
        )
        return result

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        goal: str = DEFAULT_GOAL,
        *,
        scope: str = "",
        context: str = "",
        execute: bool = True,
        reset: bool = True,
        **_: Any,
    ) -> AgentResult:
        """执行一次重构：规划 + （可选）逐步执行。

        Args:
            goal: 重构目标。
            scope: 约束范围（例如"只改 app/auth 目录"）。
            context: 已有线索（审查结论、代码片段），供规划阶段参考。
            execute: 是否执行计划；``False`` 时只出计划（"先看方案再动手"）。
            reset: 是否清空历史，默认清空。
        """
        if reset:
            self.reset()
        goal = goal or DEFAULT_GOAL
        before = self.llm.stats.snapshot()

        plan = self.plan(goal, scope=scope, context=context)
        iterations = 1
        tool_records: list[dict[str, Any]] = []
        if execute:
            for index in range(plan.total):
                outcome = self.execute_step(plan, index, context=context)
                iterations += outcome.iterations
                tool_records.extend(outcome.tool_calls)

        failed = [step for step in plan.steps if step.status == "failed"]
        success = not failed
        error = None
        if failed:
            error = f"{len(failed)} 个步骤未成功完成：" + "；".join(step.description for step in failed)

        result = AgentResult(
            output=plan.to_markdown(),
            success=success,
            error=error,
            iterations=iterations,
            messages=[],
            tool_calls=tool_records,
            usage=usage_delta(before, self.llm.stats.snapshot()),
            metadata={
                "agent": self.name,
                "goal": goal,
                "executed": execute,
                "plan": plan.to_dict(),
            },
        )
        log_event(
            logger,
            "plan_solve_finished",
            agent=self.name,
            success=success,
            steps=plan.total,
            executed=execute,
        )
        return result

    def refactor(
        self,
        goal: str = DEFAULT_GOAL,
        *,
        scope: str = "",
        context: str = "",
        execute: bool = True,
    ) -> RefactorPlan:
        """只取结构化计划的便捷入口。"""
        result = self.run(goal, scope=scope, context=context, execute=execute)
        payload = result.metadata.get("plan")
        if not isinstance(payload, dict):  # pragma: no cover - 防御
            return RefactorPlan(goal=goal)
        return RefactorPlan.from_dict(payload)


__all__ = ["DEFAULT_GOAL", "DEFAULT_MAX_STEPS", "PlanSolveRefactor"]
