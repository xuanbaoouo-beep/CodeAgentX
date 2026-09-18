"""Win Rate：把多个审查方案放在同一数据集上对比，给出"谁赢"的结论。

三条口径（都很容易不小心做错）
------------------------------
1. **无效运行不进均分**：``parse_error`` 的那次结果本身不可用，
   把它当 0 分算进均值会伪造出"方案 A 更差"的结论。无效次数单独统计、单独披露。
2. **只在共同数据集上两两比**：方案 A 跑了 3 个仓库、B 只跑了 1 个时，
   拿 A 的平均值去比 B 的那个仓库属于口径不一致；胜率只统计两者都跑过且都有效的样本。
3. **样本量必须写进结论**：单仓库、单次运行得到的"胜率 100%"没有任何统计意义，
   因此 :class:`ComparisonResult` 自带 ``caveat`` 字段，报告里照抄即可，别只贴一个百分比。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from codeagentx.core.logger import get_logger, log_event
from codeagentx.evaluation.runner import EvaluationRun

logger = get_logger("evaluation.win_rate")

__all__ = [
    "ComparisonResult",
    "PairwiseResult",
    "ProtocolSummary",
    "compare_runs",
    "pairwise_win_rate",
    "summarize_runs",
]

#: 支持的比较指标：指标名 → (取值的函数名, 越大越好)
METRICS: dict[str, tuple[str, bool]] = {
    "f1": ("f1", True),
    "precision": ("precision", True),
    "recall": ("recall", True),
    "false_positive_rate": ("false_positive_rate", False),
    # 位置偏差不算误报（见 metrics 的两轮匹配口径），因此定位质量单列一项：
    # 汇总表要能报出"结论对了、但行号要重新定位"的比例。
    "location_precision": ("location_precision", True),
    "tokens_per_finding": ("tokens_per_finding", False),
}


@dataclass
class ProtocolSummary:
    """一个方案在若干数据集上的汇总。"""

    protocol: str
    runs: int = 0
    valid_runs: int = 0
    invalid_runs: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    seconds: float = 0.0
    judge_overall: float = 0.0
    judge_runs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "runs": self.runs,
            "valid_runs": self.valid_runs,
            "invalid_runs": self.invalid_runs,
            "metrics": dict(self.metrics),
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 3),
            "judge_overall": self.judge_overall,
            "judge_runs": self.judge_runs,
        }


@dataclass
class PairwiseResult:
    """两个方案在共同数据集上的胜负。"""

    a: str
    b: str
    metric: str
    compared: int = 0
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    skipped: int = 0  # 因某侧无效而剔除的样本数

    @property
    def win_rate(self) -> float:
        """A 的胜率（并列各记半场）。"""
        if not self.compared:
            return 0.0
        return round((self.a_wins + 0.5 * self.ties) / self.compared, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "metric": self.metric,
            "compared": self.compared,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "skipped": self.skipped,
            "a_win_rate": self.win_rate,
        }


@dataclass
class ComparisonResult:
    """对比总结果。"""

    metric: str = "f1"
    protocols: list[ProtocolSummary] = field(default_factory=list)
    pairs: list[PairwiseResult] = field(default_factory=list)
    winner: str = ""
    caveat: str = ""

    def summary_of(self, protocol: str) -> ProtocolSummary | None:
        return next((item for item in self.protocols if item.protocol == protocol), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "winner": self.winner,
            "caveat": self.caveat,
            "protocols": [item.to_dict() for item in self.protocols],
            "pairs": [item.to_dict() for item in self.pairs],
        }


def _metric_value(run: EvaluationRun, metric: str) -> float:
    attribute, _ = METRICS.get(metric, METRICS["f1"])
    if attribute == "tokens_per_finding":
        return float(run.tokens_per_finding)
    return float(getattr(run.evaluation, attribute, 0.0))


def _judge_overall(run: EvaluationRun) -> float:
    """判官均分：只认跑过且有效的那次（判官结果存在 ``run.judge`` 里）。"""
    verdict = run.judge or {}
    if isinstance(verdict, dict) and verdict.get("valid"):
        return float(verdict.get("overall") or 0.0)
    return 0.0


def summarize_runs(runs: Sequence[EvaluationRun]) -> list[ProtocolSummary]:
    """按方案汇总；无效运行只计数、不进均值。"""
    buckets: dict[str, list[EvaluationRun]] = {}
    for run in runs:
        buckets.setdefault(run.protocol or "unknown", []).append(run)

    summaries: list[ProtocolSummary] = []
    for protocol, items in buckets.items():
        valid = [item for item in items if item.valid]
        summary = ProtocolSummary(
            protocol=protocol,
            runs=len(items),
            valid_runs=len(valid),
            invalid_runs=len(items) - len(valid),
            total_tokens=sum(item.total_tokens for item in valid),
            seconds=sum(item.seconds for item in valid),
        )
        for metric in METRICS:
            scores = [_metric_value(item, metric) for item in valid]
            summary.metrics[metric] = round(sum(scores) / len(scores), 4) if scores else 0.0
        judged = [item for item in valid if _judge_overall(item) > 0]
        summary.judge_runs = len(judged)
        if judged:
            summary.judge_overall = round(
                sum(_judge_overall(item) for item in judged) / len(judged), 3
            )
        summaries.append(summary)
    summaries.sort(key=lambda item: item.protocol)
    return summaries


def _mean_metric(runs: Sequence[EvaluationRun], metric: str) -> float:
    values = [_metric_value(item, metric) for item in runs]
    return sum(values) / len(values)


def pairwise_win_rate(
    runs: Sequence[EvaluationRun], *, metric: str = "f1"
) -> list[PairwiseResult]:
    """两两比较，且只在双方都有效的同一数据集上统计。

    同一数据集内同一方案跑了多次（``--repeat``）时，先取**均值**再比：
    否则胜负会由"谁的哪一次恰好留在了映射里"决定，重复运行等于白跑。
    """
    higher_is_better = METRICS.get(metric, METRICS["f1"])[1]
    by_dataset: dict[str, dict[str, list[EvaluationRun]]] = {}
    protocols: list[str] = []
    for run in runs:
        protocol = run.protocol or "unknown"
        if protocol not in protocols:
            protocols.append(protocol)
        by_dataset.setdefault(run.dataset, {}).setdefault(protocol, []).append(run)

    results: list[PairwiseResult] = []
    for index, a in enumerate(protocols):
        for b in protocols[index + 1 :]:
            pair = PairwiseResult(a=a, b=b, metric=metric)
            for mapping in by_dataset.values():
                runs_a, runs_b = mapping.get(a, []), mapping.get(b, [])
                if not runs_a or not runs_b:
                    continue
                valid_a = [item for item in runs_a if item.valid]
                valid_b = [item for item in runs_b if item.valid]
                if not valid_a or not valid_b:
                    pair.skipped += 1
                    continue
                value_a, value_b = _mean_metric(valid_a, metric), _mean_metric(valid_b, metric)
                pair.compared += 1
                if value_a == value_b:
                    pair.ties += 1
                elif (value_a > value_b) == higher_is_better:
                    pair.a_wins += 1
                else:
                    pair.b_wins += 1
            results.append(pair)
    return results


def _caveat(runs: Sequence[EvaluationRun]) -> str:
    datasets = {run.dataset for run in runs}
    valid = [run for run in runs if run.valid]
    counts = Counter((run.dataset, run.protocol) for run in valid)
    repeats = max(counts.values()) if counts else 0
    sample = (
        f"样本量：{len(datasets)} 个数据集 × {len(valid)} 次有效运行"
        f"（另有 {len(runs) - len(valid)} 次无效运行未计入均分）"
    )
    if repeats > 1:
        sample += f"，同一数据集同一方案最多重复 {repeats} 次"
    return (
        f"{sample}。样本量这么小时，胜负只能作为方向性参考，不构成统计结论；"
        "可信度取决于同一配置重复运行之间的波动，而不是均值差了多少。"
    )


def compare_runs(
    runs: Sequence[EvaluationRun], *, metric: str = "f1", tie_break: str = "tokens_per_finding"
) -> ComparisonResult:
    """汇总并对比多个方案的运行结果，给出胜者（并列时用次指标打破）。"""
    summaries = summarize_runs(runs)
    pairs = pairwise_win_rate(runs, metric=metric)
    higher_is_better = METRICS.get(metric, METRICS["f1"])[1]

    ranked = [item for item in summaries if item.valid_runs]
    winner = ""
    if ranked:
        def sort_key(item: ProtocolSummary) -> tuple[float, float]:
            primary = item.metrics.get(metric, 0.0)
            secondary = item.metrics.get(tie_break, 0.0)
            return (
                primary if higher_is_better else -primary,
                -secondary if tie_break == "tokens_per_finding" else secondary,
            )

        winner = max(ranked, key=sort_key).protocol

    result = ComparisonResult(
        metric=metric,
        protocols=summaries,
        pairs=pairs,
        winner=winner,
        caveat=_caveat(runs),
    )
    log_event(
        logger,
        "comparison_finished",
        metric=metric,
        winner=winner,
        protocols=len(summaries),
        pairs=len(pairs),
    )
    return result
