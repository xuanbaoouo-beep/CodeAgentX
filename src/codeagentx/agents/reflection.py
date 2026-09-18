"""Reflection 自我优化：生成 → 批判 → 修订，直到达标或用完轮次。

两个层次
--------
- :class:`Reflection`：与审查领域无关的通用优化器，输入"任务 + 初稿"，
  输出"修订后的稿 + 每轮评审记录"。它只做文本层面的自我批判，
  **不调用工具**——批判的对象是"你刚才声称的结论"，不是新的代码。
- :class:`ReflectionReviewer`：把上面的能力接到审查流程里，
  初稿阶段仍然可以挂工具取证，随后自我批判并修订，最终产出结构化报告。

设计取舍
--------
1. **必须能停下来**：``max_rounds`` 是硬上限；此外只要评审给出 accept、
   或修订结果与上一版完全一致（模型在打转），就立刻停止。没有终止条件的
   Reflection 是纯粹的烧钱机器。
2. **评价透明**：每轮的分数、问题清单、是否修订都记进
   :class:`ReflectionResult`，W8 的"单 Agent vs +Reflection"消融实验直接取这组数据。
3. **解析失败不装作通过**：Critique 解析不出来时判定为"需要修订"，
   而不是乐观地当成 accept。
4. **修订改坏报告时退回初稿**：反思不是免费的——真实模型偶发把一份合格报告
   改成坏 JSON。这时退回初稿交付，成果不至于归零，但"退回"会写进
   ``report.metadata["revision_rejected"]``，避免评估层把负收益记成正收益。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from codeagentx.agents.react_reviewer import DEFAULT_TASK
from codeagentx.agents.schemas import (
    ReviewReport,
    normalize_confidence,
    normalize_str_list,
    parse_json_payload,
    parse_review_report,
)
from codeagentx.core.agent import Agent, AgentResult, usage_delta
from codeagentx.core.exceptions import AgentOutputError
from codeagentx.core.llm import BaseLLM
from codeagentx.core.logger import get_logger, log_event
from codeagentx.core.message import Message
from codeagentx.prompts.review import (
    CRITIQUE_SYSTEM,
    REACT_REVIEW_SYSTEM,
    REVISE_SYSTEM,
    build_critique_task,
    build_react_review_task,
    build_revise_task,
)
from codeagentx.tools.registry import ToolRegistry

logger = get_logger("agents.reflection")

#: 评审结论中"通过"的写法
_ACCEPT_WORDS = frozenset({"accept", "accepted", "ok", "pass", "passed", "通过", "接受"})
#: 默认质量阈值：达到该分数即停止修订
DEFAULT_ACCEPT_SCORE = 0.85
#: 默认最大反思轮次
DEFAULT_MAX_ROUNDS = 1


# ---------------------------------------------------------------- 数据结构
@dataclass
class Critique:
    """一轮评审的结论。"""

    score: float = 0.0
    issues: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    verdict: str = "revise"
    raw: str = ""

    def __post_init__(self) -> None:
        self.score = normalize_confidence(self.score)
        self.issues = normalize_str_list(self.issues)
        self.missing = normalize_str_list(self.missing)
        verdict = str(self.verdict or "").strip().lower()
        self.verdict = "accept" if verdict in _ACCEPT_WORDS else "revise"

    @property
    def should_revise(self) -> bool:
        return self.verdict == "revise"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "issues": list(self.issues),
            "missing": list(self.missing),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Critique:
        return cls(
            score=data.get("score", 0.0),
            issues=data.get("issues") or [],
            missing=data.get("missing") or [],
            verdict=data.get("verdict") or "revise",
        )

    def to_text(self) -> str:
        lines = [f"评分：{self.score:g}（结论：{self.verdict}）"]
        if self.issues:
            lines.append("报告自身的问题：")
            lines.extend(f"- {item}" for item in self.issues)
        if self.missing:
            lines.append("遗漏的问题点：")
            lines.extend(f"- {item}" for item in self.missing)
        return "\n".join(lines)


@dataclass
class ReflectionRound:
    """一轮"批判 + 修订"的完整记录。"""

    index: int
    draft: str
    critique: Critique
    revised: str = ""

    @property
    def changed(self) -> bool:
        """本次修订是否真的改变了内容（相同则说明模型在打转）。"""
        return bool(self.revised.strip()) and self.revised.strip() != self.draft.strip()

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "critique": self.critique.to_dict(),
            "changed": self.changed,
            "draft_chars": len(self.draft),
            "revised_chars": len(self.revised),
        }
        if include_text:
            payload["draft"] = self.draft
            payload["revised"] = self.revised
        return payload


@dataclass
class ReflectionResult:
    """一次完整反思流程的结果。"""

    output: str
    rounds: list[ReflectionRound] = field(default_factory=list)
    stopped_reason: str = "max_rounds"

    @property
    def improved(self) -> bool:
        return any(item.changed for item in self.rounds)

    @property
    def rounds_used(self) -> int:
        return len(self.rounds)

    @property
    def final_score(self) -> float:
        return self.rounds[-1].critique.score if self.rounds else 0.0

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stopped_reason": self.stopped_reason,
            "improved": self.improved,
            "rounds_used": self.rounds_used,
            "final_score": self.final_score,
            "rounds": [item.to_dict(include_text=include_text) for item in self.rounds],
        }
        if include_text:
            payload["output"] = self.output
        return payload


# ---------------------------------------------------------------- 评审解析
def _extract_critique(payload: Any) -> Critique:
    if isinstance(payload, list):
        payload = {"issues": payload}
    if not isinstance(payload, dict):
        raise AgentOutputError(f"评审输出不是 JSON 对象：{type(payload).__name__}")

    score = payload.get("score", payload.get("rating", payload.get("quality", 0.0)))
    verdict = payload.get("verdict") or payload.get("decision") or payload.get("result") or ""
    if not verdict:
        # 没给结论时按分数推断，默认保守地要求修订
        verdict = "accept" if normalize_confidence(score) >= DEFAULT_ACCEPT_SCORE else "revise"

    return Critique(
        score=score,
        issues=payload.get("issues") or payload.get("problems") or payload.get("criticism") or [],
        missing=payload.get("missing") or payload.get("missing_findings") or [],
        verdict=verdict,
    )


def parse_critique(text: str) -> Critique:
    """解析评审输出；解析失败时保守判定为"需要修订"。"""
    try:
        payload = parse_json_payload(text)
        critique = _extract_critique(payload)
    except AgentOutputError as exc:
        preview = (text or "").strip()[:400]
        return Critique(
            score=0.0,
            issues=[f"评审输出无法解析（{exc}）"] if preview else ["评审输出为空"],
            verdict="revise",
            raw=text or "",
        )
    critique.raw = text or ""
    return critique


# ---------------------------------------------------------------- 通用优化器
class Reflection:
    """通用的"生成 → 批判 → 修订"优化器（不挂工具）。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        accept_score: float = DEFAULT_ACCEPT_SCORE,
        critique_system: str | None = None,
        revise_system: str | None = None,
    ) -> None:
        if max_rounds < 0:
            raise ValueError("max_rounds 必须 >= 0")
        if not 0.0 < accept_score <= 1.0:
            raise ValueError(f"accept_score 必须落在 (0, 1]，收到 {accept_score}")
        self.llm = llm
        self.max_rounds = max_rounds
        self.accept_score = accept_score
        self.critique_system = critique_system or CRITIQUE_SYSTEM
        self.revise_system = revise_system or REVISE_SYSTEM

    # ------------------------------------------------------------ 两个原语
    def critique(self, task: str, draft: str) -> Critique:
        """对给定稿件给出评审结论。"""
        response = self.llm.chat(
            _system_user_messages(self.critique_system, build_critique_task(task, draft))
        )
        critique = parse_critique(response.content)
        log_event(
            logger,
            "reflection_critique",
            score=critique.score,
            verdict=critique.verdict,
            issues=len(critique.issues),
            missing=len(critique.missing),
        )
        return critique

    def revise(self, task: str, draft: str, critique: Critique) -> str:
        """按评审意见修订稿件，返回新稿。"""
        response = self.llm.chat(
            _system_user_messages(
                self.revise_system,
                build_revise_task(task, draft, json.dumps(critique.to_dict(), ensure_ascii=False)),
            )
        )
        return response.content

    # ------------------------------------------------------------ 主流程
    def improve(self, task: str, draft: str) -> ReflectionResult:
        """反复"批判 → 修订"，直到通过、打转或用完轮次。"""
        if self.max_rounds == 0:
            return ReflectionResult(output=draft, rounds=[], stopped_reason="disabled")

        current = draft
        rounds: list[ReflectionRound] = []
        reason = "max_rounds"
        for index in range(1, self.max_rounds + 1):
            critique = self.critique(task, current)
            if not critique.should_revise or critique.score >= self.accept_score:
                rounds.append(ReflectionRound(index=index, draft=current, critique=critique))
                reason = "accepted"
                break

            revised = self.revise(task, current, critique)
            record = ReflectionRound(index=index, draft=current, critique=critique, revised=revised)
            rounds.append(record)
            if not record.changed:
                reason = "no_change"
                break
            current = revised

        result = ReflectionResult(output=current, rounds=rounds, stopped_reason=reason)
        log_event(
            logger,
            "reflection_finished",
            rounds=result.rounds_used,
            improved=result.improved,
            stopped_reason=reason,
            final_score=result.final_score,
        )
        return result


def _system_user_messages(system: str, user: str) -> list[Message]:
    return [Message.system(system), Message.user(user)]


# ---------------------------------------------------------------- 审查 Agent
class ReflectionReviewer(Agent):
    """先取证出初稿，再自我批判并修订的审查 Agent。"""

    name = "reflection_reviewer"

    def __init__(
        self,
        llm: BaseLLM,
        *,
        tools: ToolRegistry | None = None,
        max_iterations: int = 6,
        reflection_rounds: int = DEFAULT_MAX_ROUNDS,
        accept_score: float = DEFAULT_ACCEPT_SCORE,
        system_prompt: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        super().__init__(
            llm,
            tools=tools,
            system_prompt=system_prompt or REACT_REVIEW_SYSTEM,
            max_iterations=max_iterations,
        )
        self.reflection = Reflection(llm, max_rounds=reflection_rounds, accept_score=accept_score)
        self.allowed_tools = list(allowed_tools) if allowed_tools else None

    # ------------------------------------------------------------ 初稿
    def draft(self, prompt: str, *, use_tools: bool = True) -> AgentResult:
        """产出初稿：有工具时先取证，无工具时单轮直出。"""
        if use_tools and self.tools is not None and len(self.tools) > 0:
            return self.tool_loop(self._seed_messages(prompt), allowed_tools=self.allowed_tools)
        response = self.llm.chat(self._seed_messages(prompt))
        return AgentResult(
            output=response.content,
            success=True,
            iterations=1,
            messages=[],
            tool_calls=[],
            usage={},
        )

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        task: str = DEFAULT_TASK,
        *,
        target: str = "",
        hints: str = "",
        use_tools: bool = True,
        reset: bool = True,
        **_: Any,
    ) -> AgentResult:
        """执行一次"取证 → 自我批判 → 修订"的审查。"""
        if reset:
            self.reset()
        prompt = build_react_review_task(task or DEFAULT_TASK, target=target, hints=hints)
        before = self.llm.stats.snapshot()

        draft_result = self.draft(prompt, use_tools=use_tools)
        outcome = self.reflection.improve(prompt, draft_result.output)

        report = parse_review_report(outcome.output, target=target, source=self.name)
        draft_report = parse_review_report(draft_result.output, target=target, source=self.name)
        accepted_output = outcome.output

        if "parse_error" in report.metadata and "parse_error" not in draft_report.metadata:
            # 修订把一份**本来合格**的报告改坏了（真实模型偶发吐出坏 JSON）：
            # 退回初稿继续交付，但绝不装作没发生——"退回"这件事必须留在 metadata 里，
            # 否则 W8 的消融实验会把一次负收益的反思记成正收益。
            rejected_output = report.metadata.get("raw_output", "")
            log_event(
                logger,
                "reflection_revision_rejected",
                level=logging.WARNING,
                agent=self.name,
                error=str(report.metadata.get("parse_error") or ""),
            )
            report = draft_report
            accepted_output = draft_result.output
            report.metadata["revision_rejected"] = "修订稿无法解析为结构化 JSON，已回退到初稿"
            report.metadata["rejected_output"] = rejected_output

        report.metadata.setdefault("reflection", outcome.to_dict())
        report.metadata.setdefault("tools_used", _tool_names(draft_result))

        success = "parse_error" not in report.metadata
        result = AgentResult(
            output=accepted_output,
            success=success,
            error=None if success else "最终报告无法解析为结构化 JSON",
            # iterations 只反映初稿阶段的工具循环轮次；反思轮次见 metadata["reflection"]
            iterations=draft_result.iterations,
            messages=[],
            tool_calls=draft_result.tool_calls,
            usage=usage_delta(before, self.llm.stats.snapshot()),
            metadata={
                "agent": self.name,
                "report": report.to_dict(),
                "draft_report": draft_report.to_dict(),
                "draft_output": draft_result.output,
                "reflection": outcome.to_dict(),
            },
        )
        log_event(
            logger,
            "reflection_review_finished",
            agent=self.name,
            success=success,
            findings=report.total,
            improved=outcome.improved,
            stopped_reason=outcome.stopped_reason,
            rejected=bool(report.metadata.get("revision_rejected")),
        )
        return result

    def review(
        self,
        task: str = DEFAULT_TASK,
        *,
        target: str = "",
        hints: str = "",
        use_tools: bool = True,
    ) -> ReviewReport:
        """只取最终结构化报告的便捷入口。"""
        result = self.run(task, target=target, hints=hints, use_tools=use_tools)
        payload = result.metadata.get("report")
        if not isinstance(payload, dict):  # pragma: no cover - 防御
            return parse_review_report(result.output, target=target, source=self.name)
        return ReviewReport.from_dict(payload)


def _tool_names(result: AgentResult) -> list[str]:
    names: list[str] = []
    for record in result.tool_calls:
        name = str(record.get("name") or "")
        if name and name not in names:
            names.append(name)
    return names


__all__ = [
    "DEFAULT_ACCEPT_SCORE",
    "DEFAULT_MAX_ROUNDS",
    "Critique",
    "Reflection",
    "ReflectionResult",
    "ReflectionReviewer",
    "ReflectionRound",
    "parse_critique",
]
