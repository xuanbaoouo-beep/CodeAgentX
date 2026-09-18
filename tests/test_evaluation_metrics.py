"""评估指标测试：匹配规则、P/R/F1、误报率，以及标注集自检。

标注自检（最后两个用例）是刻意加上的：标注一旦写错行号或与源码漂移，
所有指标都会失真，而这种错**不会**让任何功能报错，只能靠断言盯住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeagentx.agents.schemas import Finding
from codeagentx.evaluation.metrics import (
    EvaluationDataset,
    LabeledDefect,
    evaluate,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "data" / "evaluation" / "sample_repo" / "labels.json"


@pytest.fixture
def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        name="demo",
        target="data/demo",
        defects=[
            LabeledDefect(
                id="sql-concat",
                title="SQL 拼接",
                file="app/db/repository.py",
                line=24,
                severity="high",
                category="security",
                keywords=("SQL", "拼接", "注入"),
            ),
            LabeledDefect(
                id="logout-noop",
                title="logout 空实现",
                file="app/auth/service.py",
                line=34,
                line_end=39,
                severity="high",
                category="security",
                keywords=("登出", "失效"),
            ),
        ],
    )


def test_exact_location_is_true_positive(dataset: EvaluationDataset) -> None:
    findings = [Finding(title="SQL 注入风险", file="app/db/repository.py", line=24)]
    result = evaluate(findings, dataset)

    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 0, 1)
    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.verdicts[0].defect_id == "sql-concat"
    assert result.verdicts[0].reason == "line"


def test_line_tolerance_boundary(dataset: EvaluationDataset) -> None:
    """行号差 3 行按行号命中；差 4 行不再算行号命中，但"文件对+问题说对"仍算找到。

    这是刻意的口径：位置偏差 ≠ 误报，所以另给 location_precision 反映定位质量，
    而不是把偏差藏进查准率里（否则会把"内容可用、只是要重新定位"的报告判死）。
    """
    near = evaluate(
        [Finding(title="SQL 注入风险", file="app/db/repository.py", line=27)], dataset
    )
    far = evaluate(
        [Finding(title="SQL 注入风险", file="app/db/repository.py", line=28)], dataset
    )

    assert near.verdicts[0].reason == "line"
    assert near.location_precision == 1.0
    # 差 4 行：仍在同一文件、标题说的仍是同一问题 → 记为关键词命中（位置偏差）
    assert far.verdicts[0].reason == "keyword"
    assert far.true_positives == 1
    assert far.location_precision == 0.0


def test_wrong_file_is_false_positive(dataset: EvaluationDataset) -> None:
    """文件对不上就是误报，关键词再像也不算。"""
    findings = [
        Finding(title="SQL 注入风险", file="app/auth/service.py", line=24),
        Finding(title="SQL 拼接", file="app/db/repository.py", line=24),  # 对照组：应命中
    ]
    result = evaluate(findings, dataset)

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.verdicts[0].defect_id == ""
    assert result.verdicts[1].defect_id == "sql-concat"


def test_target_prefixed_path_still_matches(dataset: EvaluationDataset) -> None:
    """模型把审查目标目录拼进路径属常见行为，应归一化后仍能命中。"""
    findings = [
        Finding(
            title="SQL 拼接",
            file="data/demo/app/db/repository.py",
            line=24,
        )
    ]
    assert evaluate(findings, dataset).true_positives == 1


def test_bare_basename_is_not_a_match(dataset: EvaluationDataset) -> None:
    """只写 file 名无法确定是哪个文件，不算命中（规则写进了 metrics 文档）。"""
    findings = [Finding(title="SQL 拼接", file="repository.py", line=24)]
    result = evaluate(findings, dataset)

    assert result.true_positives == 0
    assert result.false_positives == 1


def test_keyword_fallback_when_line_missing(dataset: EvaluationDataset) -> None:
    """没有行号时用关键词兜底，避免把所有"没给行号"的正确报告都算误报。"""
    findings = [Finding(title="logout 没有让 token 失效", file="app/auth/service.py", line=0)]
    result = evaluate(findings, dataset)

    assert result.true_positives == 1
    assert result.verdicts[0].reason == "keyword"


def test_line_match_wins_over_keyword(dataset: EvaluationDataset) -> None:
    """先精确后兜底：笼统报告不能把精确匹配挤掉。"""
    findings = [
        Finding(title="登出逻辑有问题", file="app/auth/service.py", line=0),
        Finding(title="logout 是空实现", file="app/auth/service.py", line=36),
    ]
    result = evaluate(findings, dataset)

    assert result.matched["logout-noop"] == "logout 是空实现"
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_positive_rate == 0.5


def test_same_defect_reported_twice_counts_once(dataset: EvaluationDataset) -> None:
    """同一缺陷报两次：只记一次命中，多出来的那条算误报（不虚增查全）。"""
    findings = [
        Finding(title="SQL 注入 A", file="app/db/repository.py", line=24),
        Finding(title="SQL 注入 B", file="app/db/repository.py", line=25),
    ]
    result = evaluate(findings, dataset)

    assert result.true_positives == 1
    assert result.false_positives == 1


def test_recall_by_severity(dataset: EvaluationDataset) -> None:
    result = evaluate(
        [Finding(title="SQL 注入", file="app/db/repository.py", line=24)], dataset
    )
    by_severity = result.recall_by_severity(dataset)

    assert by_severity["high"] == {"total": 2, "matched": 1, "recall": 0.5}


def test_empty_report_is_all_false_negative(dataset: EvaluationDataset) -> None:
    """空报告 ≠ 没有发现问题：必须是 FN，且 precision 不因分母为 0 变成 1。"""
    result = evaluate([], dataset)

    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 2)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.false_positive_rate == 0.0


# ---------------------------------------------------------------- 标注自检


def test_sample_repo_labels_are_inside_real_files() -> None:
    """每条标注的行号必须真的落在目标文件里（防手写行号越界）。"""
    dataset = load_dataset(LABELS_PATH)

    assert dataset.name == "sample_repo"
    assert dataset.total == 7  # README 植入缺陷表就是 7 条，多一条少一条都要显式改
    for defect in dataset.defects:
        path = ROOT / dataset.target / defect.file
        assert path.is_file(), f"{defect.id} 指向的文件不存在：{defect.file}"
        total_lines = len(path.read_text(encoding="utf-8").splitlines())
        start, end = defect.span
        assert 1 <= start <= end <= total_lines, f"{defect.id} 行号越界：{defect.span}"


def test_sample_repo_labeled_lines_contain_expected_code() -> None:
    """行号区间里必须真的是那段代码，否则说明标注与源码已经漂移。"""
    dataset = load_dataset(LABELS_PATH)
    expected = {
        "hardcoded-secret": "SECRET_KEY",
        "weak-password-hash": "sha256",
        "missing-input-validation": "def login",
        "logout-noop": "def logout",
        "exception-leak": "repr(exc)",
        "no-login-rate-limit": "def handle_login",
        "sql-string-concat": "SELECT name, password_hash FROM users",
    }

    for defect in dataset.defects:
        source = (ROOT / dataset.target / defect.file).read_text(encoding="utf-8")
        lines = source.splitlines()
        start, end = defect.span
        snippet = "\n".join(lines[start - 1 : end])
        assert expected[defect.id] in snippet, f"{defect.id} 的标注区间里找不到预期代码"


def test_load_dataset_rejects_labels_without_defects(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"dataset": "empty", "defects": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="没有任何标注缺陷"):
        load_dataset(path)


# ---------------------------------------------------------------- 全数据集自检


def _label_paths() -> list[Path]:
    """`data/evaluation/` 下所有数据集的标注文件（新增数据集会被自动纳入自检）。"""
    return sorted((ROOT / "data" / "evaluation").glob("*/labels.json"))


def _target_files(target: Path) -> list[Path]:
    return [
        path
        for path in target.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]


def test_at_least_one_dataset_exists() -> None:
    assert _label_paths(), "data/evaluation/ 下没有任何 labels.json"


@pytest.mark.parametrize("labels_path", _label_paths(), ids=lambda item: item.parent.name)
def test_dataset_labels_are_self_consistent(labels_path: Path) -> None:
    """逐数据集把关：标注必须指向真实文件、行号不越界、关键词非空、且能在 README 里对上。

    这些错**不会**让任何功能报错，只会让指标悄悄失真，所以必须由断言盯住。
    """
    dataset = load_dataset(labels_path)
    target = ROOT / dataset.target

    assert dataset.name == labels_path.parent.name
    assert target.is_dir(), f"{dataset.name} 的目标目录不存在：{dataset.target}"
    assert dataset.provenance.get("source"), f"{dataset.name} 没有声明标注出处"

    readme = target / "README.md"
    assert readme.is_file(), f"{dataset.name} 的目标仓库必须有 README（缺陷表是唯一的 ground truth 出处）"
    readme_text = readme.read_text(encoding="utf-8")

    actual_files = _target_files(target)
    assert dataset.file_count == len(actual_files), (
        f"{dataset.name} 声明的 file_count={dataset.file_count} 与实际 {len(actual_files)} 不符"
    )

    ids = [defect.id for defect in dataset.defects]
    assert len(ids) == len(set(ids)), f"{dataset.name} 的 id 有重复：{ids}"

    for defect in dataset.defects:
        path = target / defect.file
        assert path.is_file(), f"{defect.id} 指向的文件不存在：{defect.file}"
        total_lines = len(path.read_text(encoding="utf-8").splitlines())
        start, end = defect.span
        assert 1 <= start <= end <= total_lines, f"{defect.id} 行号越界：{defect.span}"
        assert defect.keywords, f"{defect.id} 没有关键词，第二轮兜底匹配会失效"
        # 标注只能来自 README 的缺陷表：表里没有 = 事后补标签 = 污染评测
        assert defect.id in readme_text, f"{defect.id} 在 {dataset.name} 的 README 缺陷表里找不到"
