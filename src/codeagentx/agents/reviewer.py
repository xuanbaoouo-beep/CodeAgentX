"""Reviewer 角色：主责正确性与可维护性，同时复用同一套"带关注面的审查"机制。

W6 引入七个角色后，最容易失控的是"每个角色各自实现一遍工具取证 + JSON 解析"。
这里把公共部分固化成 :class:`FocusedReviewAgent`：

- 关注面由类属性 :attr:`FocusedReviewAgent.focus` 决定，写进提示词；
- 取证方式、终止条件、结论解析全部继承自 W5 的
  :class:`~codeagentx.agents.react_reviewer.ReActReviewer`（含"未收敛即 success=False"、
  "解析失败带 parse_error"两条硬约束）；
- 角色只负责两件事：**系统提示**与**默认关注面**。

这样 Security 角色（见 :mod:`codeagentx.agents.security`）只需换一个系统提示，
行为约束与 Reviewer 完全一致，不会出现"某个角色的失败被静默吞掉"。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from codeagentx.agents.react_reviewer import ReActReviewer
from codeagentx.agents.schemas import ReviewReport, parse_review_report
from codeagentx.core.agent import AgentResult
from codeagentx.core.llm import BaseLLM
from codeagentx.prompts.agents import REVIEWER_SYSTEM, build_focused_review_task
from codeagentx.tools.registry import ToolRegistry

#: 带关注面的审查 Agent 默认任务描述
DEFAULT_FOCUSED_TASK = "审查目标代码，梳理本角色关注范围内的问题并给出可落地的修复建议。"
#: 审查角色默认可见的工具（写操作类工具一律不放）
FOCUSED_REVIEW_TOOLS: tuple[str, ...] = ("code_search", "static_analyzer", "terminal")


class FocusedReviewAgent(ReActReviewer):
    """带关注面的审查 Agent 基类（Reviewer / Security 共用）。"""

    #: 本角色的关注面，会写进任务提示词
    focus: str = ""
    #: 未传 task 时使用的默认任务
    default_task: str = DEFAULT_FOCUSED_TASK

    def run(
        self,
        task: str = "",
        *,
        target: str = "",
        hints: str = "",
        evidence: str = "",
        reset: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """执行一次带关注面的审查。

        Args:
            task: 审查任务；留空用 :attr:`default_task`。
            target: 审查目标（写进报告与提示词）。
            hints: 额外线索。
            evidence: 已有证据文本（例如 Retriever 的检索结果），
                提示词会明确要求"仍需核实，不要直接当成结论"。
            reset: 是否清空历史，默认清空。
        """
        prompt = build_focused_review_task(
            task or self.default_task,
            hints=hints,
            focus=self.focus,
            evidence=evidence,
        )
        return super().run(prompt, target=target, reset=reset, **kwargs)

    def review(
        self,
        task: str = "",
        *,
        target: str = "",
        hints: str = "",
        evidence: str = "",
        reset: bool = True,
    ) -> ReviewReport:
        """只取结构化报告的便捷入口。"""
        result = self.run(task, target=target, hints=hints, evidence=evidence, reset=reset)
        payload = result.metadata.get("report")
        if isinstance(payload, dict):
            return ReviewReport.from_dict(payload)
        return parse_review_report(result.output, target=target, source=self.name)  # pragma: no cover


class ReviewerAgent(FocusedReviewAgent):
    """正确性与可维护性方向的审查角色。"""

    name = "reviewer"
    focus = "逻辑正确性（边界、异常、资源、并发）与可维护性（结构、命名、重复代码），兼顾明显性能问题"
    default_task = "审查目标代码，找出逻辑错误、异常处理缺陷、资源管理问题与可维护性隐患。"

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
            system_prompt=system_prompt or REVIEWER_SYSTEM,
            max_iterations=max_iterations,
            allowed_tools=allowed_tools or FOCUSED_REVIEW_TOOLS,
        )


__all__ = [
    "DEFAULT_FOCUSED_TASK",
    "FOCUSED_REVIEW_TOOLS",
    "FocusedReviewAgent",
    "ReviewerAgent",
]
