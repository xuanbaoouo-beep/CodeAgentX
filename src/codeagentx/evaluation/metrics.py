"""评估指标：把一次审查的结论与人工标注比对，算出查准/查全/误报。

匹配规则的取舍（为什么这么定）
------------------------------
1. **文件必须对上**：路径归一化后按后缀比对，容忍模型写出
   ``data/sample_repo/app/auth/service.py`` 这类带目标前缀的路径；
   但只写 ``service.py`` 这种裸文件名不算命中（无法确定是哪个文件）。
2. **行号带容差**：模型常把缺陷定位到相邻语句（差 1~3 行很常见），
   因此用 ``[line - 容差, line_end + 容差]`` 区间判定；标注里的 ``line_end``
   让"缺失型缺陷"（该做而没做）能标整个函数体——它们本就没有唯一行号。
3. **两轮匹配**：先按行号精确匹配，再用关键词兜底。
   若只用一轮，一条笼统但正确的报告可能把精确匹配挤掉，把精确的那条记成误报。
   兜底也覆盖"给了行号但对不上"的情况——文件对了、问题说对了、只是指偏了几行，
   这属于**位置偏差**而不是误报。两类命中分别计数（``line`` / ``keyword``），
   ``location_precision`` 专门反映"定位准不准"，不会把位置偏差混进查准率里掩盖掉。
4. **分类与严重度不参与匹配**：模型对 category/severity 的口径与标注未必一致，
   因为口径差异就把整条判错会低估真实效果；它们只用于细分统计。

已知偏差（必须随结果一起披露，不能只报一个 F1）
-----------------------------------------------
ground truth 只包含"刻意植入且被文档记录"的缺陷，所以**真实存在但未标注**的问题
（本仓库 ``app/auth/models.py`` 的密码摘要明文暴露就是一例）会被记成误报。
因此 precision 是**下界**、误报率是**上界**；结果里会保留误报清单供人工复核。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagentx.agents.schemas import normalize_file, normalize_severity

__all__ = [
    "DEFAULT_LINE_TOLERANCE",
    "EvaluationDataset",
    "EvaluationResult",
    "LabeledDefect",
    "evaluate",
    "load_dataset",
]

#: 行号容差：模型把缺陷定位到相邻语句属正常偏差
DEFAULT_LINE_TOLERANCE = 3


@dataclass(frozen=True)
class LabeledDefect:
    """一条人工标注的缺陷。"""

    id: str
    title: str
    file: str
    line: int = 0
    line_end: int = 0
    severity: str = "medium"
    category: str = "other"
    keywords: tuple[str, ...] = ()
    readme_row: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", normalize_file(self.file))
        object.__setattr__(self, "severity", normalize_severity(self.severity))
        if self.line_end < self.line:
            object.__setattr__(self, "line_end", self.line)

    @property
    def span(self) -> tuple[int, int]:
        """行号区间；``line`` 为 0 表示只靠关键词匹配。"""
        if self.line <= 0:
            return (0, 0)
        return (self.line, max(self.line_end, self.line))

    @property
    def location(self) -> str:
        start, end = self.span
        if not start:
            return self.file
        return f"{self.file}:{start}" if start == end else f"{self.file}:{start}-{end}"

    @classmethod
    def from_dict(cls, payload: Any) -> LabeledDefect:
        data = dict(payload or {})
        keywords = data.get("keywords") or ()
        if isinstance(keywords, str):
            keywords = (keywords,)
        return cls(
            id=str(data.get("id") or ""),
            title=str(data.get("title") or ""),
            file=str(data.get("file") or ""),
            line=int(data.get("line") or 0),
            line_end=int(data.get("line_end") or 0),
            severity=str(data.get("severity") or "medium"),
            category=str(data.get("category") or "other"),
            keywords=tuple(str(item) for item in keywords),
            readme_row=int(data.get("readme_row") or 0),
        )


@dataclass
class EvaluationDataset:
    """一个数据集条目：被审目标 + 该目标的全部人工标注。"""

    name: str
    target: str = ""
    language: str = ""
    file_count: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    defects: list[LabeledDefect] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.defects)

    def by_id(self, defect_id: str) -> LabeledDefect | None:
        return next((item for item in self.defects if item.id == defect_id), None)

    @classmethod
    def from_dict(cls, payload: Any) -> EvaluationDataset:
        data = dict(payload or {})
        defects = [LabeledDefect.from_dict(item) for item in data.get("defects") or []]
        return cls(
            name=str(data.get("dataset") or ""),
            target=str(data.get("target") or ""),
            language=str(data.get("language") or ""),
            file_count=int(data.get("file_count") or 0),
            provenance=dict(data.get("provenance") or {}),
            defects=defects,
        )


@dataclass
class FindingVerdict:
    """单条上报结论的判定结果。"""

    index: int
    title: str
    file: str
    line: int
    defect_id: str = ""
    reason: str = ""  # line / keyword；空串表示误报

    @property
    def is_true_positive(self) -> bool:
        return bool(self.defect_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "file": self.file,
            "line": self.line,
            "defect_id": self.defect_id,
            "match": self.reason,
        }


@dataclass
class EvaluationResult:
    """一次评估的完整结果。"""

    dataset: str = ""
    target: str = ""
    protocol: str = ""
    verdicts: list[FindingVerdict] = field(default_factory=list)
    matched: dict[str, str] = field(default_factory=dict)  # 缺陷 id → 命中的报告标题
    missed: list[str] = field(default_factory=list)
    tolerance: int = DEFAULT_LINE_TOLERANCE

    # ------------------------------------------------------------ 计数
    @property
    def true_positives(self) -> int:
        return len(self.matched)

    @property
    def false_positives(self) -> int:
        return sum(1 for item in self.verdicts if not item.is_true_positive)

    @property
    def false_negatives(self) -> int:
        return len(self.missed)

    @property
    def reported(self) -> int:
        return len(self.verdicts)

    # ------------------------------------------------------------ 指标
    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return round(self.true_positives / denominator, 4) if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return round(self.true_positives / denominator, 4) if denominator else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return round(2 * self.precision * self.recall / total, 4) if total else 0.0

    @property
    def false_positive_rate(self) -> float:
        """误报率 = 误报数 / 上报总数（precision 的补，口径写在报告里）。"""
        return round(self.false_positives / self.reported, 4) if self.reported else 0.0

    @property
    def line_matched(self) -> int:
        """命中里靠行号精确对上的条数。"""
        return sum(1 for item in self.verdicts if item.reason == "line")

    @property
    def location_precision(self) -> float:
        """定位准确率 = 行号命中的 TP / 全部 TP。

        单独看这个数是为了不让"问题找对了但指错地方"被查准率掩盖：
        它偏低说明报告内容可用、但需要人工再定位一次。
        """
        return round(self.line_matched / self.true_positives, 4) if self.true_positives else 0.0

    def recall_by_severity(self, dataset: EvaluationDataset) -> dict[str, dict[str, float]]:
        """按严重度看查全率：高危漏报比低危漏报严重得多，必须能分别看。"""
        summary: dict[str, dict[str, float]] = {}
        for defect in dataset.defects:
            bucket = summary.setdefault(
                defect.severity, {"total": 0, "matched": 0, "recall": 0.0}
            )
            bucket["total"] += 1
            if defect.id in self.matched:
                bucket["matched"] += 1
        for bucket in summary.values():
            bucket["recall"] = round(bucket["matched"] / bucket["total"], 4) if bucket["total"] else 0.0
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "target": self.target,
            "protocol": self.protocol,
            "tolerance": self.tolerance,
            "counts": {
                "reported": self.reported,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "labeled_defects": self.true_positives + self.false_negatives,
            },
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "false_positive_rate": self.false_positive_rate,
                "location_precision": self.location_precision,
            },
            "matched": dict(self.matched),
            "missed": list(self.missed),
            "verdicts": [item.to_dict() for item in self.verdicts],
        }


# ------------------------------------------------------------------ 载入


def load_dataset(path: str | Path) -> EvaluationDataset:
    """读取 ``labels.json``。"""
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    dataset = EvaluationDataset.from_dict(payload)
    if not dataset.name:
        dataset.name = file_path.parent.name
    if not dataset.defects:
        raise ValueError(f"{file_path} 里没有任何标注缺陷")
    return dataset


# ------------------------------------------------------------------ 匹配


def _field(finding: Any, name: str, default: Any = "") -> Any:
    """同时支持 :class:`~codeagentx.agents.schemas.Finding` 与普通 dict。"""
    if isinstance(finding, dict):
        return finding.get(name, default)
    return getattr(finding, name, default)


def _file_matches(finding_file: str, label_file: str) -> bool:
    found = normalize_file(finding_file)
    if not found or not label_file:
        return False
    return found == label_file or found.endswith("/" + label_file)


def _line_in_span(finding: Any, defect: LabeledDefect, tolerance: int) -> bool:
    line = int(_field(finding, "line", 0) or 0)
    start, end = defect.span
    if line <= 0 or start <= 0:
        return False
    return start - tolerance <= line <= end + tolerance


def _keyword_hits(finding: Any, defect: LabeledDefect) -> bool:
    if not defect.keywords:
        return False
    text = " ".join(
        str(_field(finding, name, "") or "")
        for name in ("title", "description", "evidence", "suggestion")
    ).lower()
    return any(keyword.lower() in text for keyword in defect.keywords)


def evaluate(
    findings: list[Any],
    dataset: EvaluationDataset,
    *,
    protocol: str = "",
    tolerance: int = DEFAULT_LINE_TOLERANCE,
) -> EvaluationResult:
    """把上报的结论与标注比对，返回带指标的评估结果。"""
    remaining = list(dataset.defects)
    verdicts: list[FindingVerdict] = []
    claimed: list[tuple[int, LabeledDefect, str]] = []

    def claim(index: int, defect: LabeledDefect, reason: str) -> None:
        claimed.append((index, defect, reason))
        remaining.remove(defect)

    # 第一轮：文件 + 行号精确匹配
    for index, finding in enumerate(findings):
        for defect in list(remaining):
            if _file_matches(str(_field(finding, "file", "")), defect.file) and _line_in_span(
                finding, defect, tolerance
            ):
                claim(index, defect, "line")
                break

    # 第二轮：关键词兜底（含没有行号的报告）
    for index, finding in enumerate(findings):
        if any(item[0] == index for item in claimed):
            continue
        for defect in list(remaining):
            if _file_matches(str(_field(finding, "file", "")), defect.file) and _keyword_hits(
                finding, defect
            ):
                claim(index, defect, "keyword")
                break

    claimed_by_index = {index: (defect, reason) for index, defect, reason in claimed}
    for index, finding in enumerate(findings):
        defect, reason = claimed_by_index.get(index, (None, ""))
        verdicts.append(
            FindingVerdict(
                index=index,
                title=str(_field(finding, "title", "")),
                file=normalize_file(str(_field(finding, "file", ""))),
                line=int(_field(finding, "line", 0) or 0),
                defect_id=defect.id if defect else "",
                reason=reason,
            )
        )

    matched = {
        defect.id: str(_field(findings[index], "title", ""))
        for index, defect, _ in claimed
    }
    missed = [defect.id for defect in dataset.defects if defect.id not in matched]

    return EvaluationResult(
        dataset=dataset.name,
        target=dataset.target,
        protocol=protocol,
        verdicts=verdicts,
        matched=matched,
        missed=missed,
        tolerance=tolerance,
    )
