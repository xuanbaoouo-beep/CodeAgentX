"""LLM Judge：对审查**报告本身的质量**做多维度打分。

职责边界（很重要，别混）
------------------------
- "有没有找到真缺陷"由 :mod:`codeagentx.evaluation.metrics` 对**人工标注**比对得出（P/R/F1）。
- "报告写得怎么样"由本模块打分：是否说清楚、给没给依据、建议能否落地、严重度是否夸大。

因此判官**只看上报的结论 + 相关代码片段，绝不看人工标注**。
把标注塞进判官上下文等于把标准答案递过去，分数会好看但毫无意义。
正确性维度依据的是随结论一起给出的代码片段（由 ``target_root`` 就地读取），
所以它判的是"结论与所引代码是否自洽"，而不是"是否穷尽了所有缺陷"。

判官失败也算失败
----------------
模型没按契约输出时记 ``metadata["parse_error"]``、``valid=False``、``overall=0.0``，
由调用方决定是否把这次评分计入统计；绝不把解析失败当成"质量差"来拉低均分。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagentx.agents.schemas import Finding, parse_json_payload
from codeagentx.core.agent import usage_delta
from codeagentx.core.exceptions import AgentOutputError
from codeagentx.core.llm import BaseLLM
from codeagentx.core.logger import get_logger, log_event
from codeagentx.core.message import Message

logger = get_logger("evaluation.judge")

__all__ = [
    "DIMENSIONS",
    "JUDGE_SYSTEM",
    "SCORE_MAX",
    "SCORE_MIN",
    "JudgeScore",
    "JudgeVerdict",
    "LLMJudge",
    "build_judge_prompt",
    "parse_judge_output",
]

#: 评分维度 → 说明（同时用于渲染提示词与解释报告）
DIMENSIONS: dict[str, str] = {
    "correctness": "结论与随附代码片段是否自洽：有没有说错事实、把正常代码判成缺陷",
    "specificity": "是否给出具体位置与具体原因，而不是「代码不规范」这类空话",
    "actionability": "修改建议能否直接落地：改哪里、怎么改、改完怎么验",
    "evidence": "是否有代码事实作依据，而不是凭印象断言",
    "severity_calibration": "严重度与影响是否匹配：既不夸大也不轻描淡写",
}

SCORE_MIN = 1.0
SCORE_MAX = 5.0

#: 每条结论默认附带的代码上下文行数（上下各取这么多行）
DEFAULT_CONTEXT_LINES = 4

#: 送入判官的最大结论条数（避免长报告把上下文撑爆）
DEFAULT_MAX_FINDINGS = 20

JUDGE_SYSTEM = """你是 CodeAgentX 的审查报告评审员，只做一件事：给下面这份审查报告的质量打分。

打分维度（每项 1~5 分）：
{dimensions}

评分锚点：1 = 明显不合格；3 = 基本可用但有明显欠缺；5 = 优秀，无需返工。

硬性规则：
1. **只依据给出的结论与代码片段判断**，不要臆测没给出的代码，也不要因为"没提到某个问题"扣分。
2. 若某条结论与它引用的代码片段对不上（说了代码里没有的事），correctness 必须压低。
3. 不要输出任何解释性文字，只输出一个 JSON：
{{
  "scores": {{
    "correctness": 4,
    "specificity": 3,
    "actionability": 4,
    "evidence": 3,
    "severity_calibration": 4
  }},
  "comment": "一句话总评，指出最该改的一点"
}}
"""


@dataclass
class JudgeScore:
    """单个维度的得分。"""

    dimension: str
    score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "score": self.score, "reason": self.reason}


@dataclass
class JudgeVerdict:
    """一次评分的完整结果。"""

    scores: list[JudgeScore] = field(default_factory=list)
    comment: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return "parse_error" not in self.metadata

    @property
    def overall(self) -> float:
        """各维度均分；解析失败时为 0.0（配合 ``valid`` 判断，别单独看这个数）。"""
        if not self.scores:
            return 0.0
        return round(sum(item.score for item in self.scores) / len(self.scores), 3)

    def score_of(self, dimension: str) -> float:
        return next((item.score for item in self.scores if item.dimension == dimension), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "overall": self.overall,
            "scores": [item.to_dict() for item in self.scores],
            "comment": self.comment,
            "usage": dict(self.usage),
            "metadata": dict(self.metadata),
        }


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(max(number, SCORE_MIN), SCORE_MAX), 2)


def _excerpt(target_root: Path | None, finding: Finding, context_lines: int) -> str:
    """就地取该结论附近的代码，供判官核对事实（读不到就如实说明）。"""
    if target_root is None:
        return "（未提供代码目录，无法附代码片段）"
    if not finding.file:
        return "（该结论没有给出文件位置）"
    if finding.line <= 0:
        return "（该结论没有给出行号，无法定位片段）"
    path = Path(target_root) / finding.file
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return f"（代码片段读取失败：{type(exc).__name__}）"
    start = max(finding.line - context_lines, 1)
    end = min(finding.line + context_lines, len(lines))
    body = [f"{number:>4}| {lines[number - 1]}" for number in range(start, end + 1)]
    return "\n".join(body) if body else "（行号超出文件范围）"


def build_judge_prompt(
    findings: Sequence[Finding],
    *,
    target: str = "",
    target_root: str | Path | None = None,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    max_findings: int = DEFAULT_MAX_FINDINGS,
) -> str:
    """组装判官任务：每条结论 + 它引用的代码片段。"""
    root = Path(target_root) if target_root is not None else None
    shown = list(findings)[:max_findings]
    lines = [
        f"审查目标：{target or '（未说明）'}",
        f"报告共 {len(findings)} 条结论，下面展示 {len(shown)} 条：",
        "",
    ]
    if not shown:
        lines.append("（报告里一条结论都没有。只依据这一点评分，不要臆测没报出的问题。）")
        return "\n".join(lines)

    for index, finding in enumerate(shown, start=1):
        lines.append(f"### {index}. {finding.title}")
        lines.append(f"- 位置：{finding.location or '（未给出）'}")
        lines.append(f"- 类别/严重度：{finding.category} / {finding.severity}")
        if finding.description:
            lines.append(f"- 描述：{finding.description}")
        if finding.suggestion:
            lines.append(f"- 建议：{finding.suggestion}")
        lines.append("- 该位置附近的代码：")
        lines.append("```")
        lines.append(_excerpt(root, finding, context_lines))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def parse_judge_output(text: str) -> JudgeVerdict:
    """把判官输出解析成 :class:`JudgeVerdict`；失败时标记而不抛异常。"""
    try:
        payload = parse_json_payload(text)
    except AgentOutputError as exc:
        return JudgeVerdict(
            metadata={"parse_error": str(exc), "raw_output": (text or "")[:2000]}
        )

    scores_payload: Any = None
    comment = ""
    if isinstance(payload, dict):
        scores_payload = payload.get("scores") or payload.get("dimensions")
        comment = str(payload.get("comment") or payload.get("summary") or "")
        if scores_payload is None:
            # 容忍模型直接把维度写成顶层键
            scores_payload = {key: payload[key] for key in DIMENSIONS if key in payload} or None
    elif isinstance(payload, list):
        scores_payload = payload

    scores: list[JudgeScore] = []
    if isinstance(scores_payload, dict):
        for name, value in scores_payload.items():
            key = str(name).strip().lower()
            if key not in DIMENSIONS:
                continue
            if isinstance(value, dict):
                scores.append(
                    JudgeScore(
                        dimension=key,
                        score=_clamp(value.get("score")),
                        reason=str(value.get("reason") or ""),
                    )
                )
            else:
                scores.append(JudgeScore(dimension=key, score=_clamp(value)))
    elif isinstance(scores_payload, list):
        for item in scores_payload:
            if not isinstance(item, dict):
                continue
            key = str(item.get("dimension") or item.get("name") or "").strip().lower()
            if key not in DIMENSIONS:
                continue
            scores.append(
                JudgeScore(
                    dimension=key,
                    score=_clamp(item.get("score")),
                    reason=str(item.get("reason") or ""),
                )
            )

    if not scores:
        # 输出不是 JSON 也好、JSON 里没有可识别维度也好，都属"这次评分不可用"
        return JudgeVerdict(
            metadata={
                "parse_error": "判官输出中没有可识别的维度得分",
                "raw_output": (text or "")[:2000],
            }
        )

    return JudgeVerdict(scores=scores, comment=comment)


class LLMJudge:
    """用 LLM 给审查报告打多维度分。"""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        system_prompt: str | None = None,
        context_lines: int = DEFAULT_CONTEXT_LINES,
        max_findings: int = DEFAULT_MAX_FINDINGS,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt or JUDGE_SYSTEM.format(
            dimensions="\n".join(f"- {name}：{desc}" for name, desc in DIMENSIONS.items())
        )
        self.context_lines = context_lines
        self.max_findings = max_findings

    def judge(
        self,
        findings: Iterable[Finding],
        *,
        target: str = "",
        target_root: str | Path | None = None,
    ) -> JudgeVerdict:
        """给一组结论打分（一次调用判一份报告）。"""
        items = list(findings)
        prompt = build_judge_prompt(
            items,
            target=target,
            target_root=target_root,
            context_lines=self.context_lines,
            max_findings=self.max_findings,
        )
        usage_before = self.llm.stats.snapshot()
        response = self.llm.chat([Message.system(self.system_prompt), Message.user(prompt)])
        verdict = parse_judge_output(response.content)
        verdict.usage = usage_delta(usage_before, self.llm.stats.snapshot())
        verdict.metadata.setdefault("model", getattr(self.llm, "model_id", ""))
        verdict.metadata["findings_judged"] = min(len(items), self.max_findings)

        if not verdict.valid:
            log_event(
                logger,
                "judge_parse_failed",
                error=verdict.metadata.get("parse_error", ""),
                findings=len(items),
            )
        else:
            log_event(
                logger,
                "judge_finished",
                overall=verdict.overall,
                dimensions=len(verdict.scores),
                findings=len(items),
                total_tokens=verdict.usage.get("total_tokens", 0),
            )
        return verdict
