"""审查结论的数据结构：把 LLM 的自由文本收敛成可编程消费的结构。

设计要点
--------
1. **容错优先**：模型输出的字段名与取值常不规范（``critical`` / ``WARN`` /
   ``"42"`` / 百分比形式的置信度），构造函数一律做归一化，
   绝不因单个字段非法而丢弃整条问题——丢问题比丢格式严重得多。
2. **两级结构**：:class:`Finding` 是单条问题（含修复建议与证据），
   :class:`ReviewReport` 是一次审查的完整结论，可直接渲染 Markdown（W6/W10 复用）。
3. **解析与渲染分离**：:func:`parse_json_payload` 只负责"把模型输出变成 Python 对象"，
   结构组装交给 :func:`report_from_payload` / :func:`plan_from_payload`，便于单独测试。
4. **排序确定性**：:meth:`ReviewReport.sorted_findings` 固定按
   严重度 → 文件 → 行号 → 标题排序，保证同一份报告每次渲染顺序一致（AD-27）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from codeagentx.core.exceptions import AgentOutputError

#: 严重度，从高到低
SEVERITIES: tuple[str, ...] = ("high", "medium", "low")
#: 问题分类
CATEGORIES: tuple[str, ...] = (
    "bug",
    "security",
    "performance",
    "style",
    "maintainability",
    "test",
    "other",
)
#: 严重度排序权重
SEVERITY_ORDER: dict[str, int] = {name: index for index, name in enumerate(SEVERITIES)}

DEFAULT_SEVERITY = "medium"
DEFAULT_CATEGORY = "other"
DEFAULT_CONFIDENCE = 0.5

#: 模型可能给出的同义写法 → 标准取值
_SEVERITY_ALIASES: dict[str, str] = {
    "critical": "high",
    "blocker": "high",
    "fatal": "high",
    "error": "high",
    "severe": "high",
    "major": "medium",
    "warning": "medium",
    "warn": "medium",
    "moderate": "medium",
    "minor": "low",
    "info": "low",
    "note": "low",
    "trivial": "low",
    "严重": "high",
    "高": "high",
    "中等": "medium",
    "中": "medium",
    "一般": "medium",
    "轻微": "low",
    "低": "low",
}

_CATEGORY_ALIASES: dict[str, str] = {
    "correctness": "bug",
    "defect": "bug",
    "logic": "bug",
    "漏洞": "security",
    "安全": "security",
    "perf": "performance",
    "性能": "performance",
    "format": "style",
    "lint": "style",
    "pep8": "style",
    "风格": "style",
    "规范": "style",
    "refactor": "maintainability",
    "design": "maintainability",
    "smell": "maintainability",
    "可维护性": "maintainability",
    "重构": "maintainability",
    "testing": "test",
    "coverage": "test",
    "测试": "test",
    "错误": "bug",
    "缺陷": "bug",
}

#: 从报告级 payload 中寻找"问题列表"的候选键
_LIST_KEYS: tuple[str, ...] = ("findings", "issues", "problems", "results", "问题")
#: 从报告级 payload 中寻找"摘要"的候选键
_SUMMARY_KEYS: tuple[str, ...] = ("summary", "overview", "conclusion", "摘要", "总结")
#: 用于识别"这一条 dict 本身就是单个问题"
_TITLE_KEYS: tuple[str, ...] = ("title", "name", "issue", "message", "问题", "标题")

_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n?(.*?)```", re.DOTALL)
_SENTINEL = object()


# ---------------------------------------------------------------- 归一化工具
def normalize_severity(value: Any) -> str:
    """把任意写法的严重度归一化到 :data:`SEVERITIES`。"""
    text = str(value or "").strip().lower()
    if text in SEVERITY_ORDER:
        return text
    return _SEVERITY_ALIASES.get(text, DEFAULT_SEVERITY)


def normalize_category(value: Any) -> str:
    """把任意写法的分类归一化到 :data:`CATEGORIES`。"""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in CATEGORIES:
        return text
    return _CATEGORY_ALIASES.get(text, DEFAULT_CATEGORY)


def _to_int(value: Any, default: int = 0) -> int:
    """尽量转成整数（"42" / 42.0 / "L42" 都能兜住）。"""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(value).strip().lstrip("Ll")))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = DEFAULT_CONFIDENCE) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(str(value).strip().rstrip("%"))
        except (TypeError, ValueError):
            return default


def normalize_confidence(value: Any) -> float:
    """置信度归一化到 ``[0, 1]``：``80`` / ``"80%"`` 视为 0.8。"""
    number = _to_float(value)
    if number > 1.0:
        number = number / 100.0 if number <= 100.0 else 1.0
    return round(min(max(number, 0.0), 1.0), 4)


def normalize_file(value: Any) -> str:
    """统一路径分隔符，去掉 ``./`` 前缀，便于跨平台比对。"""
    text = str(value or "").strip().strip("`").replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def normalize_str_list(value: Any) -> list[str]:
    """把 ``"a, b"`` / ``["a", "b"]`` / 单值统一成字符串列表（容忍中英文分隔符）。"""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in re.split(r"[,\n;、]", value)]
        return [item for item in parts if item]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


# ---------------------------------------------------------------- Finding
@dataclass
class Finding:
    """单条审查问题。"""

    title: str
    description: str = ""
    file: str = ""
    line: int = 0
    severity: str = DEFAULT_SEVERITY
    category: str = DEFAULT_CATEGORY
    suggestion: str = ""
    evidence: str = ""
    confidence: float = DEFAULT_CONFIDENCE
    source: str = ""

    def __post_init__(self) -> None:
        self.title = _one_line(self.title) or "（未命名问题）"
        self.description = str(self.description or "").strip()
        self.suggestion = str(self.suggestion or "").strip()
        self.evidence = str(self.evidence or "").strip()
        self.source = str(self.source or "").strip()
        self.file = normalize_file(self.file)
        self.line = max(_to_int(self.line, 0), 0)
        self.severity = normalize_severity(self.severity)
        self.category = normalize_category(self.category)
        self.confidence = normalize_confidence(self.confidence)

    @property
    def location(self) -> str:
        """``文件:行号``；缺行号时只给文件名，都没有则空串。"""
        if self.file and self.line:
            return f"{self.file}:{self.line}"
        return self.file

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "file": self.file,
            "line": self.line,
            "severity": self.severity,
            "category": self.category,
            "suggestion": self.suggestion,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Any, *, source: str = "") -> Finding:
        """从任意 dict 构造；字段缺失/非法时取默认值而不抛异常。"""
        if not isinstance(data, dict):
            return cls(title=str(data), source=source)

        title = ""
        for key in _TITLE_KEYS:
            if data.get(key):
                title = str(data[key])
                break
        if not title:
            title = _one_line(data.get("description") or data.get("detail") or "")

        return cls(
            title=title,
            description=data.get("description") or data.get("detail") or data.get("impact") or "",
            file=data.get("file") or data.get("path") or data.get("filename") or "",
            line=data.get("line") or data.get("line_number") or data.get("lineno") or 0,
            severity=data.get("severity") or data.get("level") or data.get("priority") or "",
            category=data.get("category") or data.get("type") or data.get("kind") or "",
            suggestion=data.get("suggestion")
            or data.get("fix")
            or data.get("recommendation")
            or "",
            evidence=data.get("evidence") or data.get("snippet") or data.get("code") or "",
            confidence=data.get("confidence")
            if data.get("confidence") is not None
            else DEFAULT_CONFIDENCE,
            source=data.get("source") or source,
        )


# ---------------------------------------------------------------- ReviewReport
@dataclass
class ReviewReport:
    """一次代码审查的完整结论。"""

    target: str = ""
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.findings)

    def severity_counts(self) -> dict[str, int]:
        """按严重度计数，始终包含 high/medium/low 三个键（便于展示）。"""
        counts: dict[str, int] = dict.fromkeys(SEVERITIES, 0)
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts

    def sorted_findings(self) -> list[Finding]:
        """严重度 → 文件 → 行号 → 标题，保证渲染顺序稳定。"""
        return sorted(
            self.findings,
            key=lambda item: (
                SEVERITY_ORDER.get(item.severity, len(SEVERITIES)),
                item.file,
                item.line,
                item.title,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "summary": self.summary,
            "total": self.total,
            "severity_counts": self.severity_counts(),
            "findings": [finding.to_dict() for finding in self.sorted_findings()],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewReport:
        return cls(
            target=str(data.get("target") or ""),
            summary=str(data.get("summary") or ""),
            findings=[
                Finding.from_dict(item)
                for item in (data.get("findings") or [])
                if isinstance(item, (dict, str))
            ],
            metadata=dict(data.get("metadata") or {}),
        )

    # ------------------------------------------------------------ 渲染
    def to_markdown(self) -> str:
        """渲染成 Markdown 报告（不含时间戳，保证可复现）。"""
        counts = self.severity_counts()
        lines = [
            "# 代码审查报告",
            "",
            f"**审查目标**：{self.target or '（未指定）'}",
            "",
            f"**问题总数**：{self.total}"
            f"（high {counts['high']} / medium {counts['medium']} / low {counts['low']}）",
            "",
            "## 摘要",
            "",
            self.summary.strip() or "（无摘要）",
            "",
            "## 问题详情",
            "",
        ]
        if not self.findings:
            lines.append("未发现需要报告的问题。")
            return "\n".join(lines) + "\n"

        for index, finding in enumerate(self.sorted_findings(), start=1):
            meta = [f"**分类**：{finding.category}"]
            if finding.location:
                meta.append(f"**位置**：`{finding.location}`")
            meta.append(f"**置信度**：{finding.confidence:g}")
            if finding.source:
                meta.append(f"**来源**：{finding.source}")
            lines.append(f"### {index}. [{finding.severity}] {finding.title}")
            lines.append("")
            lines.append(" ｜ ".join(meta))
            if finding.description:
                lines.extend(["", "**说明**", "", finding.description])
            if finding.suggestion:
                lines.extend(["", "**修复建议**", "", finding.suggestion])
            if finding.evidence:
                lines.extend(["", "**证据**", ""])
                lines.extend(f"    {row}" for row in finding.evidence.splitlines())
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_text(self) -> str:
        """精简文本：适合打印到终端或喂回给 LLM。"""
        counts = self.severity_counts()
        lines = [
            f"审查目标：{self.target or '（未指定）'}",
            f"问题总数：{self.total}"
            f"（high {counts['high']} / medium {counts['medium']} / low {counts['low']}）",
        ]
        if self.summary.strip():
            lines.append(f"摘要：{_one_line(self.summary)}")
        for index, finding in enumerate(self.sorted_findings(), start=1):
            lines.append("")
            lines.append(
                f"[{index}] [{finding.severity}][{finding.category}] {finding.title}"
                f" @ {finding.location or '（未定位）'}"
            )
            if finding.description:
                lines.append(f"    说明：{_one_line(finding.description)}")
            if finding.suggestion:
                lines.append(f"    建议：{_one_line(finding.suggestion)}")
        if not self.findings:
            lines.append("未发现需要报告的问题。")
        return "\n".join(lines)


# ---------------------------------------------------------------- Plan-and-Solve
@dataclass
class PlanStep:
    """重构计划中的一步。"""

    description: str
    files: list[str] = field(default_factory=list)
    rationale: str = ""
    status: str = "pending"
    result: str = ""

    def __post_init__(self) -> None:
        self.description = _one_line(self.description) or "（未描述步骤）"
        self.rationale = str(self.rationale or "").strip()
        self.result = str(self.result or "").strip()
        self.files = [normalize_file(item) for item in normalize_str_list(self.files)]
        status = str(self.status or "pending").strip().lower()
        self.status = status if status in {"pending", "done", "skipped", "failed"} else "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "files": list(self.files),
            "rationale": self.rationale,
            "status": self.status,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Any) -> PlanStep:
        if isinstance(data, str):
            return cls(description=data)
        if not isinstance(data, dict):
            return cls(description=str(data))
        return cls(
            description=data.get("description") or data.get("step") or data.get("action") or "",
            files=data.get("files") or data.get("targets") or [],
            rationale=data.get("rationale") or data.get("reason") or data.get("why") or "",
            status=data.get("status") or "pending",
            result=data.get("result") or "",
        )


@dataclass
class RefactorPlan:
    """Plan-and-Solve 的目标与步骤集合。"""

    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def total(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "risks": list(self.risks),
            "verification": list(self.verification),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RefactorPlan:
        return cls(
            goal=str(data.get("goal") or ""),
            steps=[PlanStep.from_dict(item) for item in (data.get("steps") or [])],
            risks=normalize_str_list(data.get("risks")),
            verification=normalize_str_list(data.get("verification")),
            notes=str(data.get("notes") or ""),
        )

    def to_markdown(self) -> str:
        lines = ["# 重构计划", "", f"**目标**：{self.goal or '（未指明）'}", "", "## 步骤", ""]
        if not self.steps:
            lines.append("（无步骤）")
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"{index}. **[{step.status}]** {step.description}")
            if step.files:
                lines.append(f"   - 涉及文件：{', '.join(f'`{item}`' for item in step.files)}")
            if step.rationale:
                lines.append(f"   - 理由：{step.rationale}")
            if step.result:
                lines.append(f"   - 结论：{step.result}")
        if self.risks:
            lines.extend(["", "## 风险", ""])
            lines.extend(f"- {item}" for item in self.risks)
        if self.verification:
            lines.extend(["", "## 验证方式", ""])
            lines.extend(f"- {item}" for item in self.verification)
        if self.notes.strip():
            lines.extend(["", "## 补充说明", "", self.notes])
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- 多 Agent 协作
#: Planner 可以把子任务派给的角色
ASSIGNEES: tuple[str, ...] = ("reviewer", "security", "tester", "refactor")


@dataclass
class SubTask:
    """审查计划中的一个子任务（Planner 拆分，编排层据此派活）。"""

    description: str
    focus: str = DEFAULT_CATEGORY
    files: list[str] = field(default_factory=list)
    reason: str = ""
    assignee: str = ""
    status: str = "pending"

    def __post_init__(self) -> None:
        self.description = _one_line(self.description) or "（未描述子任务）"
        self.reason = str(self.reason or "").strip()
        self.focus = normalize_category(self.focus)
        self.files = [normalize_file(item) for item in normalize_str_list(self.files)]
        assignee = str(self.assignee or "").strip().lower()
        # 未识别的角色留空由编排层兜底，不硬塞一个可能错误的角色
        self.assignee = assignee if assignee in ASSIGNEES else ""
        status = str(self.status or "pending").strip().lower()
        self.status = status if status in {"pending", "done", "skipped", "failed"} else "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "focus": self.focus,
            "files": list(self.files),
            "reason": self.reason,
            "assignee": self.assignee,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Any) -> SubTask:
        if isinstance(data, str):
            return cls(description=data)
        if not isinstance(data, dict):
            return cls(description=str(data))
        return cls(
            description=data.get("description") or data.get("task") or data.get("step") or "",
            focus=data.get("focus") or data.get("category") or data.get("type") or "",
            files=data.get("files") or data.get("targets") or data.get("path") or [],
            reason=data.get("reason") or data.get("rationale") or data.get("why") or "",
            assignee=data.get("assignee") or data.get("agent") or data.get("role") or "",
            status=data.get("status") or "pending",
        )


@dataclass
class ReviewPlan:
    """Planner 产出的审查计划。"""

    target: str = ""
    tasks: list[SubTask] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "total": self.total,
            "tasks": [task.to_dict() for task in self.tasks],
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewPlan:
        return cls(
            target=str(data.get("target") or ""),
            tasks=[SubTask.from_dict(item) for item in (data.get("tasks") or [])],
            notes=str(data.get("notes") or ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_markdown(self) -> str:
        lines = ["# 审查计划", "", f"**目标**：{self.target or '（未指定）'}", "", "## 子任务", ""]
        if not self.tasks:
            lines.append("（无子任务）")
        for index, task in enumerate(self.tasks, start=1):
            owner = f" @{task.assignee}" if task.assignee else ""
            lines.append(f"{index}. [{task.status}] [{task.focus}]{owner} {task.description}")
            if task.files:
                lines.append(f"   - 涉及文件：{', '.join(f'`{item}`' for item in task.files)}")
            if task.reason:
                lines.append(f"   - 理由：{task.reason}")
        if self.notes.strip():
            lines.extend(["", "## 补充说明", "", self.notes])
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class Evidence:
    """一条检索证据：来自真实代码片段，带位置与命中来源，可回溯核对。"""

    path: str = ""
    start_line: int = 0
    end_line: int = 0
    kind: str = ""
    symbol: str = ""
    query: str = ""
    sources: list[str] = field(default_factory=list)
    score: float = 0.0
    snippet: str = ""

    def __post_init__(self) -> None:
        self.path = normalize_file(self.path)
        self.start_line = max(_to_int(self.start_line, 0), 0)
        self.end_line = max(_to_int(self.end_line, 0), 0)
        self.kind = str(self.kind or "").strip()
        self.symbol = str(self.symbol or "").strip()
        self.query = _one_line(self.query)
        self.sources = normalize_str_list(self.sources)
        self.score = round(_to_float(self.score, 0.0), 6)
        self.snippet = str(self.snippet or "").strip()

    @property
    def location(self) -> str:
        """``path:start-end``；缺行号时只给路径。"""
        if self.path and self.start_line:
            return f"{self.path}:{self.start_line}-{self.end_line or self.start_line}"
        return self.path

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "location": self.location,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind,
            "symbol": self.symbol,
            "query": self.query,
            "sources": list(self.sources),
            "score": self.score,
            "snippet": self.snippet,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Evidence:
        if not isinstance(data, dict):
            return cls(snippet=str(data))
        return cls(
            path=data.get("path") or data.get("file") or "",
            start_line=data.get("start_line") or 0,
            end_line=data.get("end_line") or 0,
            kind=data.get("kind") or "",
            symbol=data.get("symbol") or data.get("name") or "",
            query=data.get("query") or "",
            sources=data.get("sources") or [],
            score=data.get("score") or 0.0,
            snippet=data.get("snippet") or data.get("content") or "",
        )

    @classmethod
    def from_chunk(cls, chunk: Any, *, query: str = "") -> Evidence:
        """从检索片段构造。

        用鸭子类型而非直接 import ``RetrievedChunk``：``agents`` 只依赖
        "片段对象有哪些属性"这一最小契约，方便测试注入轻量替身。
        """
        if isinstance(chunk, dict):
            data = dict(chunk)
        else:
            data = {
                "path": getattr(chunk, "path", ""),
                "start_line": getattr(chunk, "start_line", 0),
                "end_line": getattr(chunk, "end_line", 0),
                "kind": getattr(chunk, "kind", ""),
                "symbol": getattr(chunk, "symbol", ""),
                "sources": list(getattr(chunk, "sources", ()) or ()),
                "score": getattr(chunk, "score", 0.0),
                "snippet": getattr(chunk, "content", ""),
            }
        data.setdefault("query", query)
        return cls.from_dict(data)

    def to_text(self) -> str:
        header = f"[{self.location}] {self.kind} {self.symbol}".rstrip()
        sources = "+".join(self.sources) or "unknown"
        return f"{header}  命中：{sources}\n{self.snippet}"


# ---------------------------------------------------------------- JSON 解析
def _candidate_slices(text: str) -> list[str]:
    """产出若干"可能是 JSON"的切片，从最严格到最宽松。"""
    candidates = [text]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    return candidates


def _try_loads(text: str) -> Any:
    """逐个候选切片试解析：先严格，再放宽对**裸控制字符**的容忍。

    真实模型写长 JSON（尤其带代码片段）时常在字符串里留下未转义的换行/制表符，
    严格模式下这算非法 JSON，但语义可以完整还原；``strict=False`` 只放宽这一点，
    缺括号、尾逗号、未转义引号这类**结构错误仍会被拒**，不会把坏 JSON 当好 JSON。
    """
    for chunk in _candidate_slices(text):
        for strict in (True, False):
            try:
                return json.loads(chunk, strict=strict)
            except (json.JSONDecodeError, ValueError):
                continue
    return _SENTINEL


def parse_json_payload(text: str) -> Any:
    """从模型输出中提取 JSON。

    依次尝试：整体解析 → 去掉 ``` 围栏后解析 → 截取最外层 ``{}`` / ``[]`` 解析。
    全部失败时抛 :class:`AgentOutputError`（附带输出片段便于定位）。
    """
    if not text or not text.strip():
        raise AgentOutputError("模型输出为空，无法解析 JSON")

    raw = text.strip()
    for candidate in [raw, *[block.strip() for block in _FENCE_RE.findall(raw)]]:
        parsed = _try_loads(candidate)
        if parsed is not _SENTINEL:
            return parsed

    raise AgentOutputError(
        "模型输出中未找到合法 JSON",
        detail=f"输出片段：{raw[:200]}",
    )


def _extract_finding_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if any(key in payload for key in _TITLE_KEYS):
            return [payload]
    return []


def findings_from_payload(payload: Any, *, source: str = "") -> list[Finding]:
    """把任意 payload 里能识别的问题项转成 :class:`Finding` 列表。"""
    findings: list[Finding] = []
    for item in _extract_finding_items(payload):
        if isinstance(item, (dict, str)):
            findings.append(Finding.from_dict(item, source=source))
    return findings


def report_from_payload(payload: Any, *, target: str = "", source: str = "") -> ReviewReport:
    """把模型输出组装成 :class:`ReviewReport`（识别不了的部分只丢摘要，不丢问题）。"""
    summary = ""
    if isinstance(payload, dict):
        for key in _SUMMARY_KEYS:
            if payload.get(key):
                summary = str(payload[key]).strip()
                break
    report = ReviewReport(
        target=target,
        summary=summary,
        findings=findings_from_payload(payload, source=source),
    )
    if source:
        # 记下"这份结论出自哪个角色"：汇总阶段据此统计各角色贡献，
        # 否则多角色报告里全是 ``unknown``，"来源可追溯"就成了一句空话
        report.metadata["agent"] = source
    return report


#: 解析失败时的摘要文案。刻意写得明确：解析失败 ≠ 没有问题。
UNPARSEABLE_SUMMARY = (
    "模型输出无法解析为结构化 JSON 报告，本轮未产出有效结论（原始输出保留在 metadata.raw_output）。"
)


def parse_review_report(text: str, *, target: str = "", source: str = "") -> ReviewReport:
    """把模型输出解析成 :class:`ReviewReport`，解析失败时降级而不抛异常。

    降级是**带标记的**：``metadata["parse_error"]`` 存在就表示本次结论不可用，
    调用方（编排层、评估层）必须据此认定该次审查无效，
    绝不能把"解析失败"当成"没有发现问题"。
    """
    try:
        payload = parse_json_payload(text)
    except AgentOutputError as exc:
        report = ReviewReport(target=target, summary=UNPARSEABLE_SUMMARY)
        report.metadata["parse_error"] = str(exc)
        report.metadata["raw_output"] = (text or "")[:2000]
        return report
    return report_from_payload(payload, target=target, source=source)


def plan_from_payload(payload: Any) -> RefactorPlan:
    """把模型输出组装成 :class:`RefactorPlan`。"""
    if isinstance(payload, list):
        return RefactorPlan(steps=[PlanStep.from_dict(item) for item in payload])
    if isinstance(payload, dict):
        if not any(key in payload for key in ("steps", "goal", "risks", "verification")):
            for key in _LIST_KEYS:
                value = payload.get(key)
                if isinstance(value, list):
                    return RefactorPlan(steps=[PlanStep.from_dict(item) for item in value])
        return RefactorPlan.from_dict(payload)
    raise AgentOutputError(f"无法从 {type(payload).__name__} 解析重构计划")


#: 审查计划解析失败时的说明文案（plan 没有 findings 可"空着"，必须显式说明）
UNPARSEABLE_PLAN_NOTE = (
    "模型输出无法解析为结构化审查计划，本计划不含任何子任务（原始输出保留在 metadata.raw_output）。"
)

#: 从计划级 payload 中寻找"子任务列表"的候选键
_TASK_KEYS: tuple[str, ...] = ("tasks", "steps", "subtasks", "计划", "子任务")


def _extract_task_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _TASK_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if any(key in payload for key in ("description", "task", "子任务")):
            return [payload]
    return []


def review_plan_from_payload(payload: Any, *, target: str = "") -> ReviewPlan:
    """把模型输出组装成 :class:`ReviewPlan`。"""
    notes = ""
    if isinstance(payload, dict):
        for key in ("notes", *_SUMMARY_KEYS):
            if payload.get(key):
                notes = str(payload[key]).strip()
                break
    return ReviewPlan(
        target=target or (str(payload.get("target") or "") if isinstance(payload, dict) else ""),
        tasks=[SubTask.from_dict(item) for item in _extract_task_items(payload)],
        notes=notes,
    )


def parse_review_plan(text: str, *, target: str = "") -> ReviewPlan:
    """把模型输出解析成 :class:`ReviewPlan`，解析失败时降级但保留标记。

    与 :func:`parse_review_report` 同理：``metadata["parse_error"]`` 存在
    就表示这份计划不可用，调用方必须显式降级（例如改用默认审查计划），
    绝不能让"空计划"静默通过，否则整条流水线会连着空转。
    """
    try:
        payload = parse_json_payload(text)
    except AgentOutputError as exc:
        plan = ReviewPlan(target=target, notes=UNPARSEABLE_PLAN_NOTE)
        plan.metadata["parse_error"] = str(exc)
        plan.metadata["raw_output"] = (text or "")[:2000]
        return plan
    return review_plan_from_payload(payload, target=target)


__all__ = [
    "ASSIGNEES",
    "CATEGORIES",
    "DEFAULT_CATEGORY",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_SEVERITY",
    "SEVERITIES",
    "SEVERITY_ORDER",
    "UNPARSEABLE_PLAN_NOTE",
    "UNPARSEABLE_SUMMARY",
    "Evidence",
    "Finding",
    "PlanStep",
    "RefactorPlan",
    "ReviewPlan",
    "ReviewReport",
    "SubTask",
    "findings_from_payload",
    "normalize_category",
    "normalize_confidence",
    "normalize_file",
    "normalize_severity",
    "normalize_str_list",
    "parse_json_payload",
    "parse_review_plan",
    "parse_review_report",
    "plan_from_payload",
    "report_from_payload",
    "review_plan_from_payload",
]
