"""上下文工程的数据结构：文档 → 分节 → 构建结果。

为什么集中定义：``builder`` 管"怎么收集、怎么筛、怎么排版"，
``compressor`` 管"怎么压"，二者围绕同一组数据打交道。
把结构放在一处，避免两边各定义一套而互不兼容
（典型事故：压缩器返回的对象排版器不认识，字段悄悄丢了）。

数据流::

    ContextDocument（一条证据）  →  ContextSection（一节，如"代码证据"）
                                  →  BuiltContext（可喂给 LLM 的最终文本 + 统计）

所有 token 数一律用 :func:`~codeagentx.core.message.estimate_tokens` 估算，
口径与预算控制保持一致；它只用于**预算**，不能当计费依据。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from codeagentx.core.message import estimate_tokens

__all__ = [
    "DEFAULT_SECTION",
    "SECTION_CONSTRAINTS",
    "SECTION_EVIDENCE",
    "SECTION_TASK",
    "SOURCES",
    "BuiltContext",
    "ContextDocument",
    "ContextSection",
    "ContextStats",
    "StageStat",
    "document_identity",
    "estimate_documents_tokens",
    "estimate_sections_tokens",
    "render_sections",
]

#: 文档来源：决定排版时的标注，也便于统计"上下文里各来源占多少"
SOURCES: tuple[str, ...] = ("retrieval", "file", "note", "memory", "mcp", "github")

SECTION_TASK = "任务"
SECTION_EVIDENCE = "代码证据"
SECTION_CONSTRAINTS = "约束与要求"
#: 未指定归节时的默认节
DEFAULT_SECTION = SECTION_EVIDENCE


@dataclass(frozen=True)
class ContextDocument:
    """进入上下文的一条证据（一个代码片段 / 一个文件 / 一条笔记）。

    ``path`` + ``start_line`` + ``end_line`` 是**可回溯性的载体**：
    即使后续被压缩截断，位置区间仍指向原始文件，报告里能据此回查。
    """

    doc_id: str
    content: str
    source: str = "retrieval"
    path: str = ""
    start_line: int = 0
    end_line: int = 0
    kind: str = ""
    name: str = ""
    score: float = 0.0
    section: str = DEFAULT_SECTION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ 展示
    @property
    def location(self) -> str:
        """``path:start-end``（无行号信息时退化为路径）。"""
        if not self.path:
            return ""
        if self.end_line and self.start_line:
            return f"{self.path}:{self.start_line}-{self.end_line}"
        return self.path

    @property
    def header(self) -> str:
        """证据头：让模型知道"这段代码在哪"。"""
        parts = [f"[{self.location}]" if self.location else "[无位置信息]"]
        if self.kind:
            parts.append(self.kind)
        if self.name:
            parts.append(self.name)
        if self.source != "retrieval":
            parts.append(f"来源={self.source}")
        return " ".join(parts)

    def to_text(self, *, with_header: bool = True) -> str:
        """带位置头的完整文本（喂 LLM 时用它）。"""
        if not with_header:
            return self.content
        return f"{self.header}\n{self.content}"

    def token_estimate(self, *, with_header: bool = True) -> int:
        return estimate_tokens(self.to_text(with_header=with_header))

    # ------------------------------------------------------------ 派生
    def with_content(self, content: str, **metadata: Any) -> ContextDocument:
        """换掉正文（压缩截断后调用），可选追加元数据说明改了什么。"""
        merged = dict(self.metadata)
        merged.update(metadata)
        return replace(self, content=content, metadata=merged)

    def with_score(self, score: float) -> ContextDocument:
        return replace(self, score=score)

    def with_section(self, section: str) -> ContextDocument:
        return replace(self, section=section)

    def to_dict(self, *, with_content: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "source": self.source,
            "path": self.path,
            "location": self.location,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind,
            "name": self.name,
            "score": round(self.score, 6),
            "section": self.section,
            "tokens": self.token_estimate(),
            "metadata": dict(self.metadata),
        }
        if with_content:
            payload["content"] = self.content
        return payload


@dataclass(frozen=True)
class ContextSection:
    """上下文中的一节（默认三节：任务 / 代码证据 / 约束与要求）。"""

    title: str
    documents: tuple[ContextDocument, ...] = ()
    note: str = ""

    def tokens(self) -> int:
        return estimate_sections_tokens([self])

    def to_text(self, *, headings: bool = True) -> str:
        """渲染成 Markdown 片段；``headings=False`` 时只输出正文。"""
        blocks: list[str] = []
        if headings:
            blocks.append(f"## {self.title}")
        if self.note:
            blocks.append(self.note)
        blocks.extend(document.to_text() for document in self.documents)
        return "\n\n".join(block for block in blocks if block)

    def with_documents(self, documents: Sequence[ContextDocument]) -> ContextSection:
        return replace(self, documents=tuple(documents))

    def to_dict(self, *, with_content: bool = False) -> dict[str, Any]:
        return {
            "title": self.title,
            "note": self.note,
            "documents": len(self.documents),
            "tokens": self.tokens(),
            "items": [document.to_dict(with_content=with_content) for document in self.documents],
        }


@dataclass
class StageStat:
    """GSSC 各阶段的统计（``stages`` 列表里的一项）。"""

    name: str
    documents_in: int = 0
    documents_out: int = 0
    tokens: int = 0
    duration: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "documents_in": self.documents_in,
            "documents_out": self.documents_out,
            "tokens": self.tokens,
            "duration": round(self.duration, 3),
            "detail": self.detail,
        }


@dataclass
class ContextStats:
    """一次上下文构建的汇总统计。"""

    budget: int
    tokens: int = 0
    documents: int = 0
    within_budget: bool = True
    dropped_documents: int = 0
    stages: list[StageStat] = field(default_factory=list)
    #: 压缩报告（:meth:`~codeagentx.context.compressor.CompressionReport.as_dict`）
    compression: dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str) -> StageStat | None:
        """按阶段名取统计（找不到返回 ``None``，便于调用方判分支）。"""
        for item in self.stages:
            if item.name == name:
                return item
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "tokens": self.tokens,
            "documents": self.documents,
            "within_budget": self.within_budget,
            "dropped_documents": self.dropped_documents,
            "stages": [item.as_dict() for item in self.stages],
            "compression": self.compression,
        }


@dataclass
class BuiltContext:
    """GSSC 的最终产物：可喂给 LLM 的文本 + 结构化分节 + 统计。

    ``text`` 由 ``sections`` 现场渲染，**不额外存一份**，
    避免"存下来的文本"和"分节内容"悄悄不一致。
    """

    task: str
    sections: list[ContextSection]
    stats: ContextStats

    @property
    def text(self) -> str:
        return render_sections(self.sections)

    @property
    def tokens(self) -> int:
        return self.stats.tokens

    @property
    def budget(self) -> int:
        return self.stats.budget

    @property
    def within_budget(self) -> bool:
        return self.stats.within_budget

    @property
    def documents(self) -> list[ContextDocument]:
        return [document for section in self.sections for document in section.documents]

    def to_dict(self, *, with_text: bool = True, with_content: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "tokens": self.tokens,
            "budget": self.budget,
            "within_budget": self.within_budget,
            "sections": [section.to_dict(with_content=with_content) for section in self.sections],
            "stats": self.stats.as_dict(),
        }
        if with_text:
            payload["text"] = self.text
        return payload


def render_sections(sections: Iterable[ContextSection]) -> str:
    """把多节渲染成完整文本（各节之间空一行）。"""
    blocks = [section.to_text() for section in sections if section.to_text()]
    return "\n\n".join(blocks)


def estimate_documents_tokens(documents: Iterable[ContextDocument]) -> int:
    """一批文档的 token 估算（含证据头）。"""
    return sum(document.token_estimate() for document in documents)


def estimate_sections_tokens(sections: Iterable[ContextSection]) -> int:
    """按**实际渲染结果**估算——标题与 note 也算进去，不低估开销。"""
    return estimate_tokens(render_sections(sections))


def document_identity(document: ContextDocument) -> tuple:
    """去重的稳定键：**位置优先**，没有位置信息才退回 ``doc_id``/正文。

    为什么位置优先：同一条代码可能被检索器与文件读取各贡献一份，
    它们的 ``doc_id`` 不同但内容重复；只看 ``doc_id`` 会漏掉这类重复，白占预算。
    """
    if document.path and (document.start_line or document.end_line):
        return ("loc", document.path, document.start_line, document.end_line)
    if document.doc_id:
        return ("id", document.doc_id)
    return ("content", document.content)
