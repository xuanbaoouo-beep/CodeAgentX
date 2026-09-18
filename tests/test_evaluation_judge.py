"""LLM Judge 测试：解析容错、代码片段取证、失败必须显式标记。"""

from __future__ import annotations

import json
from pathlib import Path

from codeagentx.agents.schemas import Finding
from codeagentx.core.llm import MockLLM
from codeagentx.evaluation.judge import (
    DIMENSIONS,
    LLMJudge,
    build_judge_prompt,
    parse_judge_output,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REPO = ROOT / "data" / "sample_repo"


def _payload(**overrides: float) -> str:
    scores = {name: 4 for name in DIMENSIONS}
    scores.update(overrides)
    return json.dumps({"scores": scores, "comment": "整体可用"})


def test_parse_scores_dict_form() -> None:
    verdict = parse_judge_output(_payload(correctness=5, evidence=2))

    assert verdict.valid is True
    assert verdict.score_of("correctness") == 5.0
    assert verdict.score_of("evidence") == 2.0
    assert verdict.comment == "整体可用"
    assert verdict.overall == 3.8  # (5 + 2 + 4 + 4 + 4) / 5


def test_parse_scores_list_form() -> None:
    text = json.dumps(
        [
            {"dimension": "correctness", "score": 3, "reason": "有一处结论与代码不符"},
            {"dimension": "evidence", "score": 5},
        ]
    )
    verdict = parse_judge_output(text)

    assert verdict.score_of("correctness") == 3.0
    assert verdict.scores[0].reason == "有一处结论与代码不符"
    assert verdict.overall == 4.0


def test_scores_are_clamped_and_unknown_dimensions_dropped() -> None:
    text = json.dumps({"scores": {"correctness": 9, "evidence": 0, "vibes": 5}})
    verdict = parse_judge_output(text)

    assert verdict.score_of("correctness") == 5.0  # 上限夹到 5
    assert verdict.score_of("evidence") == 1.0  # 下限夹到 1
    assert all(item.dimension in DIMENSIONS for item in verdict.scores)
    assert len(verdict.scores) == 2


def test_parse_failure_is_flagged_not_scored() -> None:
    """判官没按契约输出是"这次评分不可用"，不能当成"质量差"拉低均分。"""
    verdict = parse_judge_output("我觉得写得还行，给 4 分")

    assert verdict.valid is False
    assert verdict.overall == 0.0
    assert "parse_error" in verdict.metadata


def test_json_without_recognizable_dimension_is_invalid() -> None:
    verdict = parse_judge_output(json.dumps({"scores": {"whatever": 5}}))

    assert verdict.valid is False
    assert "可识别" in verdict.metadata["parse_error"]


def test_prompt_includes_finding_and_code_excerpt() -> None:
    finding = Finding(
        title="SQL 字符串拼接",
        file="app/db/repository.py",
        line=24,
        description="username 直接拼进 SQL",
        suggestion="改用参数化查询",
    )
    prompt = build_judge_prompt([finding], target="data/sample_repo", target_root=SAMPLE_REPO)

    assert "SQL 字符串拼接" in prompt
    assert "app/db/repository.py:24" in prompt
    assert "SELECT name, password_hash FROM users" in prompt  # 真把代码贴进去了
    assert "  24|" in prompt  # 带行号，方便判官核对


def test_prompt_is_honest_when_excerpt_unavailable() -> None:
    missing = Finding(title="找不到的文件", file="app/ghost.py", line=3)
    no_line = Finding(title="没给行号", file="app/auth/service.py", line=0)
    prompt = build_judge_prompt(
        [missing, no_line], target="data/sample_repo", target_root=SAMPLE_REPO
    )

    assert "代码片段读取失败" in prompt
    assert "没有给出行号" in prompt


def test_prompt_handles_empty_report() -> None:
    prompt = build_judge_prompt([], target="data/sample_repo", target_root=SAMPLE_REPO)

    assert "一条结论都没有" in prompt


def test_judge_records_usage_and_sends_dimensions() -> None:
    llm = MockLLM(
        [
            {
                "content": _payload(correctness=5),
                "prompt_tokens": 900,
                "completion_tokens": 120,
            }
        ]
    )
    judge = LLMJudge(llm)
    finding = Finding(title="SQL 字符串拼接", file="app/db/repository.py", line=24)

    verdict = judge.judge([finding], target="data/sample_repo", target_root=SAMPLE_REPO)

    assert verdict.valid is True
    assert verdict.usage["total_tokens"] == 1020
    assert verdict.usage["calls"] == 1
    assert verdict.metadata["findings_judged"] == 1

    system_prompt = llm.calls[0]["messages"][0]["content"]
    user_prompt = llm.calls[0]["messages"][1]["content"]
    assert "severity_calibration" in system_prompt  # 维度说明进了系统提示
    assert "SQL 字符串拼接" in user_prompt


def test_judge_marks_invalid_when_model_ignores_contract() -> None:
    llm = MockLLM(["（我不想输出 JSON）"])
    verdict = LLMJudge(llm).judge(
        [Finding(title="x", file="app/auth/service.py", line=11)], target_root=SAMPLE_REPO
    )

    assert verdict.valid is False
    assert verdict.overall == 0.0


def test_max_findings_caps_prompt_size() -> None:
    findings = [
        Finding(title=f"结论 {index}", file="app/auth/service.py", line=22, description="d")
        for index in range(30)
    ]
    prompt = build_judge_prompt(
        findings, target="data/sample_repo", target_root=SAMPLE_REPO, max_findings=5
    )

    assert "报告共 30 条结论，下面展示 5 条" in prompt
    assert "结论 4" in prompt
    assert "结论 5" not in prompt
