"""评估层：数据集、指标、LLM Judge 与 Win Rate。"""

from codeagentx.evaluation.judge import (
    DIMENSIONS,
    JudgeScore,
    JudgeVerdict,
    LLMJudge,
    build_judge_prompt,
    parse_judge_output,
)
from codeagentx.evaluation.metrics import (
    DEFAULT_LINE_TOLERANCE,
    EvaluationDataset,
    EvaluationResult,
    LabeledDefect,
    evaluate,
    load_dataset,
)
from codeagentx.evaluation.runner import (
    FLAG_KEYS,
    EvaluationRun,
    ReviewOutcome,
    outcome_from_agent,
    outcome_from_workflow,
    run_evaluation,
)
from codeagentx.evaluation.win_rate import (
    METRICS,
    ComparisonResult,
    PairwiseResult,
    ProtocolSummary,
    compare_runs,
    pairwise_win_rate,
    summarize_runs,
)

__all__ = [
    "DEFAULT_LINE_TOLERANCE",
    "DIMENSIONS",
    "FLAG_KEYS",
    "METRICS",
    "ComparisonResult",
    "EvaluationDataset",
    "EvaluationResult",
    "EvaluationRun",
    "JudgeScore",
    "JudgeVerdict",
    "LLMJudge",
    "LabeledDefect",
    "PairwiseResult",
    "ProtocolSummary",
    "ReviewOutcome",
    "build_judge_prompt",
    "compare_runs",
    "evaluate",
    "load_dataset",
    "outcome_from_agent",
    "outcome_from_workflow",
    "pairwise_win_rate",
    "parse_judge_output",
    "run_evaluation",
    "summarize_runs",
]
