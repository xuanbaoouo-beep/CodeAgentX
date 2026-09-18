"""W8 端到端评估：在多个标注集上跑多个审查方案，给出 P/R/F1、误报率与 Token 成本。

默认对 ``data/evaluation/*/labels.json`` **全部数据集**逐一套跑：

* 逐数据集建索引（索引是全局单集合，换数据集必须重建，否则会串味）；
* 逐数据集跑每个方案，再跨数据集汇总（胜率只在双方都有效的同一数据集上比）。

用法::

    # 真实评估（需在 .env 配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID）
    python examples/evaluate_sample_repo.py
    # 只跑其中两个方案
    python examples/evaluate_sample_repo.py --protocols react,workflow
    # 只跑指定数据集（省略即全部）
    python examples/evaluate_sample_repo.py --labels data/evaluation/blog_api/labels.json
    # 额外请 LLM 判官给报告本身打分（每个数据集多花一次调用）
    python examples/evaluate_sample_repo.py --judge

为什么**没有** ``--mock``
------------------------
其他示例提供 ``--mock`` 是为了让没配密钥的人看到输出格式；但评估脚本一旦用脚本化输出，
得到的 F1 是"脚本对着标注算出来的"，看着像指标、实际是自证，比没有指标更危险。
所以这里不提供离线模式：没配密钥就直接退出并说明原因。

口径（写在输出里，不藏在代码里）
--------------------------------
* **无效运行不进均分**：``parse_error`` 的那次单列 ``invalid_runs``，不当 0 分拉低均值；
* **胜率只在共同数据集上比**：双方都跑过且都有效才计入 ``compared``；
* **结论必须带样本量**：``compare_runs`` 的 ``caveat`` 原样打印，别只贴一个百分比。

批跑时长按数据集线性增长（一个数据集 × 三个方案约 3 分钟），
因此每次运行结束都会把 ``comparison.json`` 重写一遍：中途中断也只丢当前这一次。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from codeagentx.agents.react_reviewer import ReActReviewer  # noqa: E402
from codeagentx.agents.reflection import ReflectionReviewer  # noqa: E402
from codeagentx.agents.toolkit import build_review_toolkit  # noqa: E402
from codeagentx.config import get_config  # noqa: E402
from codeagentx.core.llm import build_llm  # noqa: E402
from codeagentx.evaluation import (  # noqa: E402
    LLMJudge,
    ReviewOutcome,
    compare_runs,
    load_dataset,
    outcome_from_agent,
    outcome_from_workflow,
    run_evaluation,
)
from codeagentx.orchestrator import CodeReviewWorkflow  # noqa: E402
from codeagentx.rag.rag_tool import build_rag_tool  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_OUT_DIR = EVAL_ROOT / "runs"
DEFAULT_TASK = "找出目标代码中的缺陷、安全风险与可维护性问题，并给出可落地的修复建议。"
REVIEW_TOOLS = ("code_search", "static_analyzer", "terminal")

#: 索引由脚本预先建好，必须告诉 Agent 一声；否则模型会自己再 index 一次，
#: 而它传的 path 是"相对项目根"的显示路径（工具沙箱根是仓库目录本身），
#: 结果白跑一轮。这条提示对所有方案一致，不影响公平性。
REVIEW_HINTS = (
    "向量索引已由调用方建立（action=index 无需再调用，直接用默认的 search 检索）；"
    "所有工具的路径都相对仓库根目录，不要拼上仓库目录名。"
)

#: 可选方案：名字 → 构造函数
PROTOCOLS = ("react", "reflection", "workflow")


def _display(path: Path) -> str:
    """把绝对路径显示成相对项目根的路径（提示词与报告里都用这个写法）。"""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_labels(spec: str) -> list[Path]:
    """把 ``--labels`` 解析成标注集路径列表；``all`` 表示扫全部数据集。"""
    if spec.strip().lower() in ("", "all"):
        return sorted(EVAL_ROOT.glob("*/labels.json"))
    paths: list[Path] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    return paths


def _write_payload(path: Path, runs: list, comparison, failures: list[tuple[str, str, str]], judge_tokens: int) -> None:
    """把当前进度落盘（每跑完一次就写一遍，中断不会丢已完成的运行）。"""
    payload = {
        "datasets": sorted({run.dataset for run in runs}),
        "judge_tokens": judge_tokens,
        "failures": [
            {"dataset": item[0], "protocol": item[1], "error": item[2]} for item in failures
        ],
        "runs": [run.to_dict() for run in runs],
        "comparison": comparison.to_dict() if comparison is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_reviewer(name: str, llm, config, root: Path) -> Callable[[Path], ReviewOutcome]:
    """按方案名造一个"目标目录 → ReviewOutcome"的可调用对象。"""

    def review(target: Path) -> ReviewOutcome:
        started = time.monotonic()
        tools = build_review_toolkit(config, root=root, tools=REVIEW_TOOLS)
        if name == "workflow":
            workflow = CodeReviewWorkflow(
                llm, root=root, target=_display(target), config=config, reflect=False
            )
            result = workflow.run()
            return outcome_from_workflow(
                result, protocol=name, seconds=round(time.monotonic() - started, 3)
            )
        agent = (
            ReflectionReviewer(llm, tools=tools)
            if name == "reflection"
            else ReActReviewer(llm, tools=tools)
        )
        result = agent.run(DEFAULT_TASK, target=_display(target), hints=REVIEW_HINTS)
        return outcome_from_agent(
            result,
            protocol=name,
            seconds=round(time.monotonic() - started, 3),
            target=_display(target),
        )

    return review


def _print_metrics(run, dataset) -> None:
    evaluation = run.evaluation
    severity = evaluation.recall_by_severity(dataset)
    severity_text = " ".join(
        f"{level} {bucket['recall']:.2f}({bucket['matched']}/{bucket['total']})"
        for level, bucket in sorted(severity.items())
    )
    print(
        f"  P={evaluation.precision:.4f} R={evaluation.recall:.4f} F1={evaluation.f1:.4f}"
        f" 误报率={evaluation.false_positive_rate:.4f}"
        f" | TP={evaluation.true_positives} FP={evaluation.false_positives}"
        f" FN={evaluation.false_negatives}"
    )
    print(
        f"  定位准确率={evaluation.location_precision:.4f}"
        f" | 查全率(按严重度)：{severity_text or '（无标注）'}"
    )
    print(
        f"  token={run.total_tokens} 每条结论={run.tokens_per_finding}"
        f" | 耗时={run.seconds}s | 有效={run.valid}"
    )
    if evaluation.missed:
        print(f"  漏报：{', '.join(evaluation.missed)}")
    extra = {key: value for key, value in run.flags.items() if key not in ("parse_error",)}
    if run.flags.get("parse_error"):
        print(f"  ⚠ 解析失败：{run.flags['parse_error']}")
    if extra:
        print(f"  信号：{extra}")


def _print_comparison(comparison) -> None:
    print(f"[胜率] 指标={comparison.metric}（只在双方都有效的同一数据集上比较）")
    if not comparison.pairs:
        print("  只有一个方案，无法两两比较。")
    for pair in comparison.pairs:
        print(
            f"  {pair.a} vs {pair.b}：胜率 {pair.win_rate:.2%}"
            f"（{pair.a_wins} 胜 / {pair.b_wins} 负 / {pair.ties} 平 / 比较 {pair.compared} 个"
            f"，跳过 {pair.skipped} 个）"
        )
    print(f"[胜者] {comparison.winner or '（无可比方案）'}")
    print(f"[口径] {comparison.caveat}")


def _print_summary(comparison) -> None:
    """跨数据集的方案均分表（同权平均；无效运行只计数、不进均分）。"""
    print("[汇总] 各方案跨数据集均分（同权平均，无效运行不计入）")
    print(
        f"  {'方案':<12}{'F1':>8}{'P':>8}{'R':>8}{'误报率':>9}"
        f"{'定位':>8}{'token/条':>10}{'判官':>7}  有效/总数"
    )
    for item in comparison.protocols:
        metrics = item.metrics
        judge = f"{item.judge_overall:.2f}" if item.judge_runs else "—"
        print(
            f"  {item.protocol:<12}{metrics['f1']:>8.4f}{metrics['precision']:>8.4f}"
            f"{metrics['recall']:>8.4f}{metrics['false_positive_rate']:>9.4f}"
            f"{metrics['location_precision']:>8.4f}{metrics['tokens_per_finding']:>10.1f}"
            f"{judge:>7}  {item.valid_runs}/{item.runs}"
        )
    print("  注：'定位'=位置精确命中（行号 ±3）的比例；位置偏差不计误报，故单列。")


def _print_variance(runs: list) -> None:
    """同一「数据集 × 方案」重复运行之间的波动（没有重复运行时不打印）。

    这是判断"均值差是否可信"的唯一依据：两个方案的均值差如果小于同一配置
    重复运行之间的极差，那点差距就是模型的随机性，不是方案差异。
    """
    grouped: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        if run.valid:
            grouped.setdefault((run.dataset, run.protocol), []).append(run.evaluation.f1)
    spreads = [max(values) - min(values) for values in grouped.values() if len(values) > 1]
    if not spreads:
        return

    print()
    print("[波动] 同一数据集同一方案重复运行的 F1（用于判断均值差是否可信）")
    for (dataset, protocol), values in sorted(grouped.items()):
        if len(values) < 2:
            continue
        print(
            f"  {dataset:<14}{protocol:<11}F1={[round(value, 4) for value in values]}"
            f" 极差={max(values) - min(values):.4f}"
        )
    print(
        f"  平均极差={sum(spreads) / len(spreads):.4f}"
        f" → **小于这个数的方案差距不能当作方案差异**（那是模型自身的随机性）"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeAgentX 评估示例（W8）")
    parser.add_argument(
        "--labels",
        default="all",
        help="标注集路径，逗号分隔；all（默认）表示 data/evaluation/*/labels.json 全部",
    )
    parser.add_argument(
        "--protocols", default=",".join(PROTOCOLS), help="要对比的方案，逗号分隔（react/reflection/workflow）"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="同一「数据集 × 方案」重复运行几次（默认 1；≥2 才能报出重复间波动）",
    )
    parser.add_argument("--judge", action="store_true", help="额外用 LLM 判官给报告本身打分")
    parser.add_argument(
        "--no-index", action="store_true", help="跳过建索引（沿用现有索引，可能维度不匹配）"
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="结果落盘目录")
    args = parser.parse_args(argv)

    if args.repeat < 1:
        print("[错误] --repeat 至少为 1", file=sys.stderr)
        return 2
    repeats = args.repeat

    names = [item.strip() for item in args.protocols.split(",") if item.strip()]
    unknown = [item for item in names if item not in PROTOCOLS]
    if unknown:
        print(f"[错误] 不认识的方案：{', '.join(unknown)}（可选：{', '.join(PROTOCOLS)}）", file=sys.stderr)
        return 2

    label_paths = _resolve_labels(args.labels)
    if not label_paths:
        print(f"[错误] 没有找到任何标注集（在 {_display(EVAL_ROOT)} 下找 */labels.json）", file=sys.stderr)
        return 2
    missing = [path for path in label_paths if not path.is_file()]
    if missing:
        print(f"[错误] 标注集不存在：{', '.join(_display(path) for path in missing)}", file=sys.stderr)
        return 2
    datasets = [load_dataset(path) for path in label_paths]

    config = get_config()
    if not config.is_llm_configured:
        print(
            "[错误] 未配置 LLM_API_KEY，无法做真实评估。\n"
            "       本脚本不提供 --mock：用脚本化输出算出来的 F1 是自证，不是指标。",
            file=sys.stderr,
        )
        return 2

    llm = build_llm(config)
    out_dir = Path(args.out_dir)
    summary_path = out_dir / "comparison.json"
    print(
        f"[配置] 模型={llm.model_id} | 数据集={len(datasets)} 个"
        f"（{'、'.join(dataset.name for dataset in datasets)}）"
        f" | 方案={'、'.join(names)} | 重复={repeats} 次"
        f" | 判官={'开' if args.judge else '关'}"
    )
    print()

    judge = LLMJudge(llm) if args.judge else None
    judge_tokens = 0
    runs: list = []
    failures: list[tuple[str, str, str]] = []

    for dataset in datasets:
        root = PROJECT_ROOT / dataset.target
        print(f"══ 数据集 {dataset.name}：{dataset.file_count} 个文件 / {dataset.total} 条标注缺陷"
              f" | 目录={_display(root)}")
        if not root.is_dir():
            print(f"  [跳过] 标注集指向的目标不存在：{root}", file=sys.stderr)
            failures.append((dataset.name, "*", f"目标目录不存在：{_display(root)}"))
            print()
            continue

        if not args.no_index:
            # 向量库是全局单集合，换数据集必须重建，否则检索到的是上一个仓库的代码。
            # 同一数据集内的各方案共用这一份索引：否则"谁先跑谁付建索引成本"，对比不公平。
            print("[索引] 为数据集目标重建索引（本数据集各方案共用，不计入任一方案的成本）")
            tool = build_rag_tool(config, root=root)
            indexed = tool.run(action="index", path=str(root))
            print(f"  {indexed.output if indexed.success else '[索引失败] ' + str(indexed.error)}")
            if not indexed.success:
                print("  ⚠ 检索不可用，code_search 会失败；本次结果不能代表完整链路的表现。")
        print()

        for name in names:
            for index in range(repeats):
                round_label = f" 第 {index + 1}/{repeats} 次" if repeats > 1 else ""
                print(f"── 方案 {name} @ {dataset.name}{round_label} " + "─" * 30)
                # 重复运行分目录存放，否则后一次会覆盖前一次的明细
                run_dir = out_dir if repeats == 1 else out_dir / f"r{index + 1}"
                try:
                    run = run_evaluation(
                        dataset,
                        _build_reviewer(name, llm, config, root),
                        root=PROJECT_ROOT,
                        out_dir=run_dir,
                    )
                except Exception as exc:  # 批跑要能"坏一个不塌全场"，失败项单独披露
                    print(f"  [失败] {type(exc).__name__}: {exc}", file=sys.stderr)
                    failures.append((dataset.name, name, f"{type(exc).__name__}: {exc}"))
                    _write_payload(summary_path, runs, None, failures, judge_tokens)
                    print()
                    continue
                if judge is not None:
                    verdict = judge.judge(run.findings, target=run.target, target_root=root)
                    run.judge = verdict.to_dict()
                    judge_tokens += int(verdict.usage.get("total_tokens") or 0)
                _print_metrics(run, dataset)
                if run.judge:
                    print(
                        f"  判官均分={run.judge['overall']}（有效={run.judge['valid']}）"
                        f" | {run.judge.get('comment', '')}"
                    )
                print(f"  明细已写入 {run.output_path}")
                print()
                runs.append(run)
                _write_payload(summary_path, runs, None, failures, judge_tokens)

    if not runs:
        print("[错误] 没有任何一次有效运行，无法汇总。", file=sys.stderr)
        return 1

    comparison = compare_runs(runs, metric="f1")
    print()
    _print_summary(comparison)
    _print_variance(runs)
    print()
    _print_comparison(comparison)
    _write_payload(summary_path, runs, comparison, failures, judge_tokens)
    print(f"[输出] 汇总已写入 {_display(summary_path)}")
    if judge_tokens:
        print(f"[成本] 判官另用 token {judge_tokens}（不计入上面的审查成本）")
    if failures:
        print("[未完成] 下列组合没有结果，横向比较时按缺失处理：")
        for dataset_name, protocol, error in failures:
            print(f"  {dataset_name} × {protocol}：{error}")

    invalid = [f"{run.dataset}/{run.protocol}" for run in runs if not run.valid]
    if invalid:
        print(f"[提示] 有方案未产出可用结论：{', '.join(invalid)}；对应的 F1 不可用于横向比较。")
    return 0 if not (invalid or failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
