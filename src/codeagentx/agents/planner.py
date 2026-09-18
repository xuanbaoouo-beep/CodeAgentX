"""Planner Agent：把"审查一个仓库"拆成若干条可验收的子任务。

为什么多 Agent 需要一个 Planner
-------------------------------
单 Agent 拿到"审查这个仓库"时，会本能地挑几处显眼的代码说几句，
既不保证覆盖面（认证、数据存取、外部输入这些高风险面可能整块漏掉），
也无法说明"这次到底查了什么"。Planner 把范围显式化：
产出 3~5 条子任务，每条写清关注面、涉及文件与负责角色，
下游角色照单执行，最终报告可以回答"覆盖了什么、哪里没覆盖"。

降级策略（关键）
----------------
模型输出解析失败时**不抛异常**，而是：
1. 记下 ``metadata["parse_error"]`` / ``["raw_output"]``；
2. 换用 :data:`FALLBACK_TASKS` 这套默认计划让流水线继续走；
3. 同时把 ``AgentResult.success`` 置为 ``False`` 并写明原因。
"降级"与"成功"必须能被区分，否则评估阶段会把兜底计划当成模型的规划能力。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from codeagentx.agents.schemas import ReviewPlan, SubTask, parse_review_plan
from codeagentx.core.agent import Agent, AgentResult
from codeagentx.core.llm import BaseLLM
from codeagentx.core.logger import get_logger, log_event
from codeagentx.prompts.agents import PLANNER_SYSTEM, build_planner_task

logger = get_logger("agents.planner")

#: 未指定目标时的审查范围
DEFAULT_TARGET = "（未指定，按目标仓库整体审查）"
#: 子任务条数上限（与提示词里的约束保持一致）
DEFAULT_MAX_TASKS = 5

#: 规划不可用时的兜底子任务：覆盖三类高风险面，保证流水线不至于空转
FALLBACK_TASKS: tuple[SubTask, ...] = (
    SubTask(
        description="通读目标代码，梳理主要入口、数据流与外部输入",
        reason="没有整体认知时，局部结论容易失真",
        assignee="reviewer",
    ),
    SubTask(
        description="排查安全风险：注入、硬编码凭据、鉴权缺失、信息泄露",
        focus="security",
        assignee="security",
    ),
    SubTask(
        description="排查正确性、异常处理与可维护性问题",
        focus="bug",
        assignee="reviewer",
    ),
)


class PlannerAgent(Agent):
    """审查规划 Agent（只规划、不执行、不调用工具）。"""

    name = "planner"

    def __init__(
        self,
        llm: BaseLLM,
        *,
        system_prompt: str | None = None,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_iterations: int = 1,
    ) -> None:
        if max_tasks < 1:
            raise ValueError("max_tasks 必须 >= 1")
        super().__init__(
            llm,
            system_prompt=system_prompt or PLANNER_SYSTEM,
            tools=None,  # 规划阶段不该动工具：先把"要查什么"想清楚
            max_iterations=max_iterations,
        )
        self.max_tasks = max_tasks

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        target: str = DEFAULT_TARGET,
        *,
        file_list: Sequence[str] | None = None,
        context: str = "",
        reset: bool = True,
        **_: Any,
    ) -> AgentResult:
        """执行一次规划。

        Args:
            target: 审查目标（仓库或文件路径）。
            file_list: 目标下的文件清单，供模型判断该覆盖哪些文件。
            context: 额外背景（例如用户关注点）。
            reset: 是否清空历史，默认清空（每次规划相互独立）。
        """
        if reset:
            self.reset()
        prompt = build_planner_task(
            target, file_list=file_list, context=context, max_tasks=self.max_tasks
        )
        result = self.tool_loop(self._seed_messages(prompt), max_iterations=self.max_iterations)

        plan = self._finalize_plan(result.output, target)
        result.metadata["agent"] = self.name
        result.metadata["plan"] = plan.to_dict()
        if plan.metadata.get("fallback"):
            # 兜底计划不是模型的规划成果，必须如实标记失败
            result.success = False
            result.error = plan.metadata.get("fallback_reason") or "审查计划不可用，已降级为默认计划"
        log_event(
            logger,
            "planner_finished",
            agent=self.name,
            success=result.success,
            tasks=plan.total,
            fallback=bool(plan.metadata.get("fallback")),
        )
        return result

    def plan(
        self,
        target: str = DEFAULT_TARGET,
        *,
        file_list: Sequence[str] | None = None,
        context: str = "",
        reset: bool = True,
    ) -> ReviewPlan:
        """只取结构化计划的便捷入口。"""
        result = self.run(target, file_list=file_list, context=context, reset=reset)
        payload = result.metadata.get("plan")
        if isinstance(payload, dict):
            return ReviewPlan.from_dict(payload)
        return parse_review_plan(result.output, target=target)  # pragma: no cover - 防御

    # ------------------------------------------------------------ 内部
    def _finalize_plan(self, text: str, target: str) -> ReviewPlan:
        """解析 + 兜底 + 截断，返回一份"一定能用"的计划（是否可信看 metadata）。"""
        plan = parse_review_plan(text, target=target)
        unusable = bool(plan.metadata.get("parse_error")) or not plan.tasks
        if unusable:
            metadata = dict(plan.metadata)
            if not plan.metadata.get("parse_error"):
                metadata["empty_plan"] = True
            metadata["fallback"] = True
            metadata["fallback_reason"] = (
                "模型输出无法解析为审查计划" if plan.metadata.get("parse_error") else "模型未产出任何子任务"
            )
            return ReviewPlan(
                target=target,
                tasks=[SubTask.from_dict(item.to_dict()) for item in FALLBACK_TASKS],
                notes=plan.notes,
                metadata=metadata,
            )
        if plan.total > self.max_tasks:
            metadata = dict(plan.metadata)
            metadata["truncated"] = plan.total - self.max_tasks
            plan = ReviewPlan(
                target=plan.target or target,
                tasks=plan.tasks[: self.max_tasks],
                notes=plan.notes,
                metadata=metadata,
            )
        return plan


__all__ = ["DEFAULT_MAX_TASKS", "DEFAULT_TARGET", "FALLBACK_TASKS", "PlannerAgent"]
