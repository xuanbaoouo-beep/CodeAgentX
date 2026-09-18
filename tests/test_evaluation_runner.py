"""评估执行器测试：适配、落盘、以及"失败不能被伪装成好成绩"。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from codeagentx.agents.schemas import Finding, ReviewReport, parse_review_report
from codeagentx.core.agent import AgentResult
from codeagentx.evaluation.metrics import EvaluationDataset, load_dataset
from codeagentx.evaluation.runner import (
    ReviewOutcome,
    outcome_from_agent,
    outcome_from_workflow,
    run_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "data" / "evaluation" / "sample_repo" / "labels.json"


@pytest.fixture
def dataset() -> EvaluationDataset:
    return load_dataset(LABELS_PATH)


def _report(*findings: Finding, **metadata: object) -> ReviewReport:
    report = ReviewReport(target="data/sample_repo", findings=list(findings))
    report.metadata.update(metadata)
    return report


def test_run_evaluation_records_metrics_and_writes_json(
    dataset: EvaluationDataset, tmp_path: Path
) -> None:
    def review(_target: Path) -> ReviewOutcome:
        return ReviewOutcome(
            report=_report(
                Finding(title="SQL 注入", file="app/db/repository.py", line=24),
                Finding(title="代码风格不统一", file="app/auth/service.py", line=50),
            ),
            protocol="fake",
            usage={"calls": 3, "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            seconds=12.5,
        )

    run = run_evaluation(dataset, review, root=ROOT, out_dir=tmp_path)

    assert run.valid is True
    assert run.evaluation.true_positives == 1
    assert run.evaluation.false_positives == 1
    assert run.evaluation.false_negatives == 6
    assert run.seconds == 12.5
    assert run.total_tokens == 1200
    assert run.tokens_per_finding == 600.0  # 1200 / 2 条上报

    payload = json.loads(Path(run.output_path).read_text(encoding="utf-8"))
    assert payload["dataset"] == "sample_repo"
    assert payload["protocol"] == "fake"
    assert payload["evaluation"]["counts"]["true_positives"] == 1
    assert len(payload["findings"]) == 2
    assert payload["valid"] is True


def test_parse_error_makes_run_invalid(dataset: EvaluationDataset) -> None:
    """解析失败的结论不可用：绝不能算成"零问题、零误报"的完美结果。"""

    def review(_target: Path) -> ReviewOutcome:
        report = parse_review_report("这不是 JSON", target="data/sample_repo")
        return ReviewOutcome(report=report, protocol="fake")

    run = run_evaluation(dataset, review, root=ROOT)

    assert run.valid is False
    assert "parse_error" in run.flags
    assert run.evaluation.reported == 0
    assert run.evaluation.false_negatives == 7  # 全部标注都成了漏报


def test_signals_are_carried_into_flags(dataset: EvaluationDataset) -> None:
    def review(_target: Path) -> ReviewOutcome:
        return ReviewOutcome(
            report=_report(degraded=True, truncated=True, forced_convergence=True),
            protocol="fake",
        )

    run = run_evaluation(dataset, review, root=ROOT)

    assert run.flags == {"degraded": True, "truncated": True, "forced_convergence": True}
    assert run.valid is True  # 降级仍可用，只是要如实标注


def test_missing_target_raises(dataset: EvaluationDataset) -> None:
    with pytest.raises(FileNotFoundError, match="评估目标不存在"):
        run_evaluation(dataset, lambda _target: ReviewOutcome(report=_report()), root=ROOT / "nope")


def test_outcome_from_agent_parses_output_and_usage() -> None:
    output = json.dumps(
        {
            "summary": "查了登录路径",
            "findings": [
                {"title": "SQL 注入", "file": "app/db/repository.py", "line": 24, "severity": "high"}
            ],
        }
    )
    result = AgentResult(
        output=output,
        usage={"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420},
        metadata={"forced_convergence": True},
    )

    outcome = outcome_from_agent(result, protocol="react", seconds=42.0, target="data/sample_repo")

    assert outcome.usage["total_tokens"] == 420
    assert outcome.usage["calls"] == 0  # 缺失字段补齐，不丢原始键
    assert outcome.flags == {"forced_convergence": True}
    assert len(outcome.report.findings) == 1


def test_outcome_from_workflow_reads_state_metadata(dataset: EvaluationDataset) -> None:
    """编排结果按鸭子类型适配：只要求 .report 与 .state。"""
    result = SimpleNamespace(
        report=_report(Finding(title="SQL 注入", file="app/db/repository.py", line=24)),
        state=SimpleNamespace(
            metadata={
                "usage": {"calls": 18, "total_tokens": 98607},
                "llm_usage": {"total_tokens": 100000},
                "duration": 199.0,
            }
        ),
        success=False,
    )

    outcome = outcome_from_workflow(result, protocol="workflow")

    assert outcome.usage["total_tokens"] == 98607
    assert outcome.usage["llm_usage"] == {"total_tokens": 100000}
    assert outcome.seconds == 199.0
    # 阶段有失败必须留痕，否则评估会把一次坏运行当成好成绩
    assert outcome.flags.get("stage_failure") is True

    run = run_evaluation(dataset, lambda _target: outcome, root=ROOT)
    assert run.evaluation.true_positives == 1
    assert run.flags.get("stage_failure") is True
