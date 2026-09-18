"""ReAct 审查 Agent：推理（Reason）与行动（Act）交替，证据充分后再产出结论。

为什么审查场景适合 ReAct
------------------------
"这段登录逻辑有安全问题吗"这类问题，模型凭记忆回答必然是猜的；
必须先 :code:`code_search` 定位代码、:code:`static_analyzer` 拿到检查结果，
再基于**实际观察到的内容**下结论。ReAct 把"取证"变成流程的一部分，
而不是寄希望于模型自觉。

终止条件（双重保证，绝不无限循环）
----------------------------------
1. 模型不再请求工具调用（认为证据已充分）；
2. 达到 ``max_iterations``（此时 ``AgentResult.success=False``，由上层决定是否重试）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from codeagentx.agents.schemas import ReviewReport, parse_review_report
from codeagentx.core.agent import Agent, AgentResult, usage_delta
from codeagentx.core.exceptions import CodeAgentXError
from codeagentx.core.llm import BaseLLM
from codeagentx.core.logger import get_logger, log_event
from codeagentx.prompts.review import (
    JSON_REPAIR_PROMPT,
    REACT_REVIEW_SYSTEM,
    build_react_review_task,
)
from codeagentx.tools.registry import ToolRegistry

logger = get_logger("agents.react_reviewer")

#: 未指定任务时的默认目标
DEFAULT_TASK = "审查目标代码，找出缺陷、安全风险与可维护性问题，并给出可落地的修复建议。"


class ReActReviewer(Agent):
    """ReAct 模式的代码审查 Agent。"""

    name = "react_reviewer"

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
            system_prompt=system_prompt or REACT_REVIEW_SYSTEM,
            max_iterations=max_iterations,
        )
        #: 限制模型只能看到部分工具（None 表示注册表里全部可见）
        self.allowed_tools = list(allowed_tools) if allowed_tools else None

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        task: str = DEFAULT_TASK,
        *,
        target: str = "",
        hints: str = "",
        reset: bool = True,
        **_: Any,
    ) -> AgentResult:
        """执行一次 ReAct 审查。

        Args:
            task: 审查任务描述。
            target: 审查目标（文件或仓库路径），会写进报告与提示词。
            hints: 额外线索（例如已知的静态检查结论）。
            reset: 是否清空历史。审查任务之间通常相互独立，默认清空。
        """
        if reset:
            self.reset()
        prompt = build_react_review_task(task or DEFAULT_TASK, target=target, hints=hints)
        result = self.tool_loop(self._seed_messages(prompt), allowed_tools=self.allowed_tools)

        report = parse_review_report(result.output, target=target, source=self.name)
        if report.metadata.get("parse_error"):
            # 解析失败 ≠ 没有发现问题（报告会带 parse_error，编排层据此把阶段判为失败）。
            # 但在判失败之前还值得花**一次**调用把结论要回来：强制收敛那一路产出的常常是
            # 一段"总结式"自由文本，模型手里证据齐全，只是没按契约输出 JSON。
            report = self._repair_report(result, target=target) or report
        report.metadata.setdefault("iterations", result.iterations)
        report.metadata.setdefault("tools_used", _tool_names(result))
        result.metadata["agent"] = self.name
        result.metadata["report"] = report.to_dict()
        log_event(
            logger,
            "react_review_finished",
            agent=self.name,
            success=result.success,
            iterations=result.iterations,
            findings=report.total,
        )
        return result

    # ------------------------------------------------------------ 解析失败补救
    def _repair_report(self, result: AgentResult, *, target: str) -> ReviewReport | None:
        """结论不是合法 JSON 时，用同一段对话历史**再问一次、这次只要 JSON**。

        触发条件严格限定在"上一条回复解析不了"（实测约 5% 的执行会走到这里），
        所以正常情况下不会多花一次调用；补问仍失败就返回 ``None``，
        由调用方继续按"解析失败"如实处理——**不假装成功，也不把空结果说成没问题**。
        """
        log_event(
            logger,
            "review_json_repair_requested",
            level=logging.WARNING,
            agent=self.name,
        )
        usage_before = self.llm.stats.snapshot()
        try:
            reply = self.llm.chat(self._seed_messages(JSON_REPAIR_PROMPT), tools=None)
        except CodeAgentXError as exc:  # 补问本身失败不该盖掉原有结论
            log_event(
                logger,
                "review_json_repair_failed",
                level=logging.WARNING,
                agent=self.name,
                error=str(exc),
            )
            return None
        self._merge_usage(result, usage_before)

        text = (reply.content or "").strip()
        report = parse_review_report(text, target=target, source=self.name)
        if report.metadata.get("parse_error"):
            log_event(
                logger,
                "review_json_repair_failed",
                level=logging.WARNING,
                agent=self.name,
                error=str(report.metadata["parse_error"]),
            )
            return None
        # 补问生效后，``output`` 换成真正被采纳的那一份：下游（原始输出留存、日志、复核）
        # 看到的应当是结论本身，而不是那段解析不了的初稿。
        result.output = text
        result.metadata["json_repaired"] = True
        report.metadata["json_repaired"] = True
        log_event(
            logger,
            "review_json_repaired",
            level=logging.WARNING,
            agent=self.name,
            findings=report.total,
        )
        return report

    def _merge_usage(self, result: AgentResult, usage_before: dict[str, Any]) -> None:
        """把补问这一次调用的用量并进本角色的用量——补问也是真花钱，不能少记。"""
        extra = usage_delta(usage_before, self.llm.stats.snapshot())
        for key, value in extra.items():
            result.usage[key] = result.usage.get(key, 0) + value

    def review(
        self,
        task: str = DEFAULT_TASK,
        *,
        target: str = "",
        hints: str = "",
        reset: bool = True,
    ) -> ReviewReport:
        """只取结构化报告的便捷入口。"""
        result = self.run(task, target=target, hints=hints, reset=reset)
        if not isinstance(result.metadata.get("report"), dict):  # pragma: no cover - 防御
            return parse_review_report(result.output, target=target, source=self.name)
        return ReviewReport.from_dict(result.metadata["report"])


def _tool_names(result: AgentResult) -> list[str]:
    """本次实际调用过的工具名（去重且保持首次调用顺序）。"""
    names: list[str] = []
    for record in result.tool_calls:
        name = str(record.get("name") or "")
        if name and name not in names:
            names.append(name)
    return names


__all__ = ["DEFAULT_TASK", "ReActReviewer"]
