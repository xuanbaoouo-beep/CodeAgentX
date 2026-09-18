"""评估执行器：跑一次审查 → 对齐标注 → 汇总指标与 Token 成本 → 落盘。

设计要点
--------
1. **审查过程靠注入**：``review`` 是 "目标目录 → :class:`ReviewOutcome`" 的可调用对象，
   评估层不关心背后是单 Agent、ReAct 还是七阶段流水线；
   离线测试注入假实现，真实评估注入真实流程（与工具层同一套可注入思路）。
2. **解析失败 ≠ 没发现问题**：报告带 ``metadata["parse_error"]`` 时该次结果**无效**，
   记 ``valid=False`` 并单列。否则"模型没按契约输出"会被算成"零误报、零问题"，
   把一次失败伪装成一次完美表现。
3. **成本如实记账**：只报 token 数与调用次数，**不臆造单价换算成金额**——
   单价随模型与渠道变化，换算是使用方的事。
4. **不导入编排层**：适配器按鸭子类型读 ``.report`` / ``.state``，避免评估层反向依赖上层
   （评估要能评估任何形态的审查流程）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagentx.agents.schemas import Finding, ReviewReport, parse_review_report
from codeagentx.core.agent import AgentResult
from codeagentx.core.logger import get_logger, log_event
from codeagentx.evaluation.metrics import (
    DEFAULT_LINE_TOLERANCE,
    EvaluationDataset,
    EvaluationResult,
    evaluate,
)

logger = get_logger("evaluation.runner")

__all__ = [
    "FLAG_KEYS",
    "EvaluationRun",
    "ReviewOutcome",
    "outcome_from_agent",
    "outcome_from_workflow",
    "run_evaluation",
]

#: 会改变"这次结果能不能用"判断的信号，出现在报告 metadata 里就透传到评估结论
FLAG_KEYS: tuple[str, ...] = (
    "parse_error",
    "degraded",
    "truncated",
    "forced_convergence",
    "revision_rejected",
)

#: 用量里被承认的键（其余原样保留在 usage 中，不做取舍）
_USAGE_KEYS: tuple[str, ...] = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")


@dataclass
class ReviewOutcome:
    """一次审查的产出（报告）+ 用量，评估层的统一输入。"""

    report: ReviewReport
    protocol: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRun:
    """一次评估运行：指标 + 成本 + 信号。"""

    dataset: str = ""
    protocol: str = ""
    target: str = ""
    evaluation: EvaluationResult = field(default_factory=EvaluationResult)
    findings: list[Finding] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    flags: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    output_path: str = ""

    @property
    def valid(self) -> bool:
        """本次结果是否可用（解析失败即视为无效）。"""
        return "parse_error" not in self.flags

    @property
    def total_tokens(self) -> int:
        return int(self.usage.get("total_tokens") or 0)

    @property
    def tokens_per_finding(self) -> float:
        """每条上报结论摊到的 token（成本效率的直接口径）。"""
        reported = self.evaluation.reported
        return round(self.total_tokens / reported, 1) if reported else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "target": self.target,
            "valid": self.valid,
            "seconds": self.seconds,
            "usage": dict(self.usage),
            "cost": {
                "total_tokens": self.total_tokens,
                "tokens_per_finding": self.tokens_per_finding,
            },
            "flags": dict(self.flags),
            "judge": dict(self.judge),
            "evaluation": self.evaluation.to_dict(),
            "findings": [item.to_dict() for item in self.findings],
        }


def _collect_flags(*sources: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for source in sources:
        for key in FLAG_KEYS:
            if key in source and key not in flags:
                flags[key] = source[key]
    return flags


def _normalize_usage(raw: dict[str, Any]) -> dict[str, Any]:
    """只补齐缺失的计数字段，不丢原始键。"""
    usage = dict(raw or {})
    for key in _USAGE_KEYS:
        usage.setdefault(key, 0)
    return usage


def outcome_from_agent(
    result: AgentResult,
    *,
    protocol: str = "",
    seconds: float = 0.0,
    target: str = "",
) -> ReviewOutcome:
    """把单 Agent（或 ReAct）的执行结果适配成评估输入。"""
    report = parse_review_report(result.output, target=target, source=protocol)
    return ReviewOutcome(
        report=report,
        protocol=protocol,
        usage=_normalize_usage(result.usage),
        seconds=seconds,
        flags=_collect_flags(report.metadata, result.metadata),
    )


def outcome_from_workflow(result: Any, *, protocol: str = "", seconds: float = 0.0) -> ReviewOutcome:
    """把编排层结果适配成评估输入（鸭子类型：只要求 ``.report`` 与 ``.state``）。

    用法与用量口径来自编排层：``state.metadata["usage"]`` 是各角色用量之和，
    ``llm_usage`` 是外部注入 LLM 时的直连调用差额，两者都保留。
    """
    report = result.report
    state = getattr(result, "state", None)
    metadata = dict(getattr(state, "metadata", {}) or {})
    usage = dict(metadata.get("usage") or {})
    if metadata.get("llm_usage"):
        usage["llm_usage"] = metadata["llm_usage"]
    if not usage:
        usage = _normalize_usage(usage)
    flags = _collect_flags(report.metadata, metadata)
    if not bool(getattr(result, "success", True)):
        flags.setdefault("stage_failure", True)
    return ReviewOutcome(
        report=report,
        protocol=protocol,
        usage=usage,
        seconds=float(seconds or metadata.get("duration") or 0.0),
        flags=flags,
    )


def run_evaluation(
    dataset: EvaluationDataset,
    review: Callable[[Path], ReviewOutcome],
    *,
    root: str | Path | None = None,
    out_dir: str | Path | None = None,
    tolerance: int = DEFAULT_LINE_TOLERANCE,
) -> EvaluationRun:
    """在 ``dataset`` 上跑一次评估，必要时把结果落盘成 JSON。"""
    target = Path(dataset.target)
    if root is not None and not target.is_absolute():
        target = Path(root) / target
    if not target.exists():
        raise FileNotFoundError(f"评估目标不存在：{target}")

    started = time.monotonic()
    outcome = review(target)
    seconds = outcome.seconds or round(time.monotonic() - started, 3)

    result = evaluate(
        list(outcome.report.findings),
        dataset,
        protocol=outcome.protocol,
        tolerance=tolerance,
    )
    run = EvaluationRun(
        dataset=dataset.name,
        protocol=outcome.protocol,
        target=target.as_posix(),
        evaluation=result,
        findings=list(outcome.report.findings),
        usage=_normalize_usage(outcome.usage),
        seconds=seconds,
        # 再从报告 metadata 收一遍 signals：直接构造 ReviewOutcome 的调用方可能忘了传，
        # 而 parse_error 这类信号一旦丢失，"失败"就会被当成"零问题零误报"的好成绩。
        # outcome.flags 已是适配器整理过的结果，整体并入而不是按 FLAG_KEYS 过滤。
        flags={**_collect_flags(outcome.report.metadata), **outcome.flags},
    )

    if out_dir is not None:
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = outcome.protocol or "run"
        path = directory / f"{dataset.name}_{suffix}.json"
        path.write_text(
            json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run.output_path = path.as_posix()

    log_event(
        logger,
        "evaluation_finished",
        dataset=dataset.name,
        protocol=outcome.protocol,
        valid=run.valid,
        reported=result.reported,
        tp=result.true_positives,
        fp=result.false_positives,
        fn=result.false_negatives,
        f1=result.f1,
        total_tokens=run.total_tokens,
        seconds=seconds,
    )
    return run
