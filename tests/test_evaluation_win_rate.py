"""Win Rate 对比实验测试：三条口径各盯一条。

1. 无效运行不进均分（否则"模型没按契约输出"会被算成"方案更差"）；
2. 只在双方都有效的同一数据集上两两比（口径不一致的比较没有意义）；
3. 结论必须带样本量（单仓库单次的"胜率 100%"没有统计意义）。
"""

from __future__ import annotations

from codeagentx.evaluation.metrics import EvaluationResult, FindingVerdict
from codeagentx.evaluation.runner import EvaluationRun
from codeagentx.evaluation.win_rate import (
    METRICS,
    compare_runs,
    pairwise_win_rate,
    summarize_runs,
)


def _result(tp: int, fp: int, fn: int) -> EvaluationResult:
    """按 (命中, 误报, 漏报) 造一个评估结果。"""
    verdicts = [
        FindingVerdict(
            index=index,
            title=f"命中{index}",
            file="app/auth/service.py",
            line=index + 1,
            defect_id=f"d{index}",
            reason="line",
        )
        for index in range(tp)
    ]
    verdicts += [
        FindingVerdict(index=100 + index, title=f"误报{index}", file="app/api/routes.py", line=index + 1)
        for index in range(fp)
    ]
    return EvaluationResult(
        verdicts=verdicts,
        matched={f"d{index}": f"命中{index}" for index in range(tp)},
        missed=[f"m{index}" for index in range(fn)],
    )


def _run(
    dataset: str,
    protocol: str,
    *,
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    tokens: int = 0,
    flags: dict[str, object] | None = None,
    judge: dict[str, object] | None = None,
) -> EvaluationRun:
    return EvaluationRun(
        dataset=dataset,
        protocol=protocol,
        evaluation=_result(tp, fp, fn),
        usage={"total_tokens": tokens},
        flags=dict(flags or {}),
        judge=dict(judge or {}),
    )


# --------------------------------------------------------------- 口径 1：无效运行


def test_invalid_runs_are_counted_but_excluded_from_averages() -> None:
    runs = [
        _run("repo_a", "single", tp=7, fn=0, tokens=1000),
        _run("repo_a", "single", flags={"parse_error": "不是 JSON"}, fn=7, tokens=9999),
    ]

    summary = summarize_runs(runs)[0]

    assert summary.protocol == "single"
    assert (summary.runs, summary.valid_runs, summary.invalid_runs) == (2, 1, 1)
    # 关键：均分只按那一次有效运行算，不能被 0 分拖到 0.5
    assert summary.metrics["f1"] == 1.0
    assert summary.total_tokens == 1000  # 失败那次的 token 也不进成本均值


def test_all_invalid_protocol_reports_zero_metrics_without_crashing() -> None:
    summary = summarize_runs([_run("repo_a", "single", flags={"parse_error": "x"})])[0]

    assert summary.valid_runs == 0
    assert summary.invalid_runs == 1
    assert summary.metrics["f1"] == 0.0


def test_judge_score_only_counted_when_valid() -> None:
    runs = [
        _run("repo_a", "single", tp=1, judge={"valid": True, "overall": 4.5}),
        _run("repo_a", "single", tp=1, judge={"valid": False, "overall": 1.0}),
    ]

    summary = summarize_runs(runs)[0]

    assert summary.judge_runs == 1
    assert summary.judge_overall == 4.5


def test_protocols_are_summarized_separately_and_sorted() -> None:
    runs = [
        _run("repo_a", "workflow", tp=7),
        _run("repo_a", "single", tp=3, fn=4),
    ]

    summaries = summarize_runs(runs)

    assert [item.protocol for item in summaries] == ["single", "workflow"]
    assert summaries[0].metrics["recall"] == 0.4286
    assert summaries[1].metrics["f1"] == 1.0


# --------------------------------------------------------------- 口径 2：共同数据集


def test_pairwise_skips_datasets_missing_on_one_side() -> None:
    runs = [
        _run("repo_a", "single", tp=7),
        _run("repo_a", "workflow", tp=3, fn=4),
        _run("repo_b", "single", tp=7),  # workflow 没跑 repo_b
    ]

    pair = pairwise_win_rate(runs)[0]

    assert pair.a == "single" and pair.b == "workflow"
    assert pair.compared == 1
    assert pair.a_wins == 1
    assert pair.skipped == 0  # 缺跑的数据集直接不计，不算"被跳过"


def test_pairwise_skips_datasets_where_one_side_is_invalid() -> None:
    runs = [
        _run("repo_a", "single", tp=7),
        _run("repo_a", "workflow", flags={"parse_error": "坏输出"}, fn=7),
        _run("repo_b", "single", tp=2, fn=5),
        _run("repo_b", "workflow", tp=5),
    ]

    pair = pairwise_win_rate(runs)[0]

    assert pair.compared == 1
    assert pair.skipped == 1
    assert pair.b_wins == 1  # repo_b 上 workflow 更全
    assert pair.win_rate == 0.0


def test_pairwise_tie_counts_as_half_a_win() -> None:
    runs = [
        _run("repo_a", "single", tp=4, fp=1),
        _run("repo_a", "workflow", tp=4, fp=1),
        _run("repo_b", "single", tp=7),
        _run("repo_b", "workflow", tp=3, fn=4),
    ]

    pair = pairwise_win_rate(runs)[0]

    assert (pair.compared, pair.a_wins, pair.b_wins, pair.ties) == (2, 1, 0, 1)
    assert pair.win_rate == 0.75  # (1 + 0.5) / 2


def test_no_pair_when_only_one_protocol_present() -> None:
    assert pairwise_win_rate([_run("repo_a", "single", tp=7)]) == []


def test_repeats_are_averaged_before_comparing() -> None:
    """同一仓库同一方案跑多次时取均值再比：否则胜负由"哪一次留在映射里"决定。

    这里 B 的重复均值（0.873）高于 A 的均值（0.7），B 必须赢；
    若实现只保留最后一次（A 的 1.0 > B 的 0.857）就会判 A 赢，这个用例就会挂。
    """
    runs = [
        _run("repo_a", "a", tp=1, fp=3),  # F1 0.4
        _run("repo_a", "a", tp=4, fp=0),  # F1 1.0 → 两次均值 0.7
        _run("repo_a", "b", tp=4, fp=1),  # F1 0.889
        _run("repo_a", "b", tp=3, fp=1),  # F1 0.857 → 两次均值 0.873
    ]

    pair = pairwise_win_rate(runs)[0]

    assert pair.compared == 1
    assert pair.b_wins == 1


def test_repeats_with_one_invalid_run_still_compare() -> None:
    """重复运行中只要还有有效的那次，这一仓库就不算"跳过"。"""
    runs = [
        _run("repo_a", "a", tp=1, fp=3, flags={"parse_error": "输出无法解析"}),
        _run("repo_a", "a", tp=4, fp=0),
        _run("repo_a", "b", tp=1, fp=1),
    ]

    pair = pairwise_win_rate(runs)[0]

    assert (pair.compared, pair.skipped) == (1, 0)
    assert pair.a_wins == 1


def test_caveat_discloses_repeat_count() -> None:
    runs = [
        _run("repo_a", "a", tp=4),
        _run("repo_a", "a", tp=4),
        _run("repo_a", "b", tp=4),
        _run("repo_a", "b", tp=4),
    ]

    assert "最多重复 2 次" in compare_runs(runs).caveat


# --------------------------------------------------------------- 方向性：越小越好


def test_false_positive_rate_direction_is_lower_is_better() -> None:
    runs = [
        _run("repo_a", "noisy", tp=4, fp=4),  # 误报率 0.5
        _run("repo_a", "quiet", tp=4, fp=1),  # 误报率 0.2
    ]

    pair = pairwise_win_rate(runs, metric="false_positive_rate")[0]

    assert pair.b_wins == 1
    assert pair.a_wins == 0


def test_tokens_per_finding_direction_is_lower_is_better() -> None:
    runs = [
        _run("repo_a", "chatty", tp=1, tokens=9000),
        _run("repo_a", "frugal", tp=1, tokens=1000),
    ]

    pair = pairwise_win_rate(runs, metric="tokens_per_finding")[0]

    assert pair.b_wins == 1


# --------------------------------------------------------------- 结论与样本量


def test_winner_uses_tie_break_when_primary_metric_ties() -> None:
    runs = [
        _run("repo_a", "chatty", tp=7, tokens=14000),
        _run("repo_a", "frugal", tp=7, tokens=1400),
    ]

    comparison = compare_runs(runs, metric="f1")

    assert comparison.winner == "frugal"  # f1 齐平，靠 tokens_per_finding 决胜
    assert [item.protocol for item in comparison.protocols] == ["chatty", "frugal"]


def test_caveat_must_disclose_sample_size() -> None:
    runs = [
        _run("repo_a", "single", tp=7),
        _run("repo_a", "workflow", tp=7),
        _run("repo_b", "single", flags={"parse_error": "x"}),
    ]

    comparison = compare_runs(runs)

    assert "2 个数据集" in comparison.caveat
    assert "2 次有效运行" in comparison.caveat
    assert "1 次无效运行" in comparison.caveat
    assert "不构成统计结论" in comparison.caveat


def test_no_winner_when_every_protocol_invalid() -> None:
    runs = [
        _run("repo_a", "single", flags={"parse_error": "x"}),
        _run("repo_a", "workflow", flags={"parse_error": "y"}),
    ]

    comparison = compare_runs(runs)

    assert comparison.winner == ""
    assert all(item.valid_runs == 0 for item in comparison.protocols)


def test_empty_input_is_safe() -> None:
    comparison = compare_runs([])

    assert comparison.protocols == []
    assert comparison.pairs == []
    assert comparison.winner == ""
    assert "0 个数据集" in comparison.caveat


def test_comparison_serializes_to_dict() -> None:
    runs = [
        _run("repo_a", "single", tp=7, tokens=1000, judge={"valid": True, "overall": 4.0}),
        _run("repo_a", "workflow", tp=5, fn=2, tokens=8000),
    ]

    payload = compare_runs(runs, metric="f1").to_dict()

    assert payload["metric"] == "f1"
    assert payload["winner"] == "single"
    assert payload["pairs"][0]["a_win_rate"] == 1.0
    assert payload["protocols"][0]["judge_overall"] == 4.0
    assert "tokens_per_finding" in payload["protocols"][0]["metrics"]


def test_metrics_registry_covers_reported_metrics() -> None:
    """指标登记表必须能对上汇总里输出的键，否则胜率会静默按 0 比。"""
    summary = summarize_runs([_run("repo_a", "single", tp=1, tokens=100)]).pop()

    assert set(METRICS) == set(summary.metrics)
