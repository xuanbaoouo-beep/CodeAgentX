"""上下文压缩：把证据压进 token 预算，同时保住"可回溯"。

为什么不能简单地"截断字符串"：审查结论必须能指回 ``文件:行号``。
一旦为了省 token 把位置信息扔掉，报告就成了"模型说的"，
既不可复核也无法评估（W8 的 F1 需要它）。

分级策略（逐级施加，每级都留统计，便于消融与排障）::

    去重      同一位置只留分最高的一份
    合并      同一符号被切成多块且连续时合成一块（省掉重复的证据头）
    截断      单文档超长时保留头部 + 尾部，中间省略并标注省略行数
    装箱      按 score 降序在预算内装箱，装不下的低分文档整体丢弃

硬约束：``compress_documents(..., budget_tokens)`` 返回的文档集合，
其 token 估算**必定** ≤ 预算；连第一条都装不下时会把它硬截断到装下，
而不是返回空上下文（空上下文会让上层 Agent 凭空作答）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from codeagentx.context.schemas import (
    ContextDocument,
    ContextSection,
    document_identity,
    estimate_documents_tokens,
)
from codeagentx.core.logger import get_logger, log_event
from codeagentx.core.message import estimate_tokens

logger = get_logger("context.compressor")

__all__ = [
    "LEVEL_BUDGET",
    "LEVEL_DEDUPE",
    "LEVEL_MERGE",
    "LEVEL_TRUNCATE",
    "CompressionReport",
    "ContextCompressor",
    "LevelReport",
]

#: 各级名称（同时用作报告里的键，顺序即施加顺序）
LEVEL_DEDUPE = "去重"
LEVEL_MERGE = "合并相邻片段"
LEVEL_TRUNCATE = "长文截断"
LEVEL_BUDGET = "预算装箱"

#: 单文档默认最大行数（超过才触发截断）
DEFAULT_MAX_LINES_PER_DOCUMENT = 120
DEFAULT_HEAD_LINES = 80
DEFAULT_TAIL_LINES = 20
#: 硬截断时的最小可用预算：低于它就没有截断的意义，直接丢弃
MIN_USEFUL_TOKENS = 24
#: 省略标记模板（截断与硬截断共用同一文案，避免两处写法漂移）
ELISION_MARKER = "...（此处省略 {elided} 行，完整内容见原文件）"


@dataclass
class LevelReport:
    """单级压缩的统计。"""

    name: str
    applied: bool = False
    documents_in: int = 0
    documents_out: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    detail: str = ""

    @property
    def saved_tokens(self) -> int:
        return max(0, self.tokens_in - self.tokens_out)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "applied": self.applied,
            "documents_in": self.documents_in,
            "documents_out": self.documents_out,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "saved_tokens": self.saved_tokens,
            "detail": self.detail,
        }


@dataclass
class CompressionReport:
    """一次压缩的汇总结果。"""

    budget: int
    before_tokens: int = 0
    after_tokens: int = 0
    before_documents: int = 0
    after_documents: int = 0
    truncated_documents: int = 0
    dropped_documents: int = 0
    hard_trimmed: int = 0
    duration: float = 0.0
    levels: list[LevelReport] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """压缩后 / 压缩前（1.0 表示没压掉；越小越好）。"""
        if self.before_tokens <= 0:
            return 1.0
        return self.after_tokens / self.before_tokens

    @property
    def reduction(self) -> float:
        """压缩幅度（0.3 表示省了 30%）。"""
        return 1.0 - self.ratio

    @property
    def within_budget(self) -> bool:
        return self.after_tokens <= self.budget

    def level(self, name: str) -> LevelReport | None:
        for item in self.levels:
            if item.name == name:
                return item
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "before_documents": self.before_documents,
            "after_documents": self.after_documents,
            "ratio": round(self.ratio, 4),
            "reduction": round(self.reduction, 4),
            "within_budget": self.within_budget,
            "truncated_documents": self.truncated_documents,
            "dropped_documents": self.dropped_documents,
            "hard_trimmed": self.hard_trimmed,
            "duration": round(self.duration, 3),
            "levels": [item.as_dict() for item in self.levels],
        }


class ContextCompressor:
    """分级上下文压缩器（纯确定性，不调用 LLM）。

    刻意不调 LLM：压缩发生在"每一次构建上下文"的热路径上，
    用模型做摘要既慢又不稳定，还会引入新的幻觉面。
    真正需要语义摘要时，应由上层显式选择（后续可选增强），而不是这里悄悄做。
    """

    def __init__(
        self,
        *,
        max_lines_per_document: int = DEFAULT_MAX_LINES_PER_DOCUMENT,
        head_lines: int = DEFAULT_HEAD_LINES,
        tail_lines: int = DEFAULT_TAIL_LINES,
    ) -> None:
        if max_lines_per_document <= 0:
            raise ValueError("max_lines_per_document 必须为正整数")
        if head_lines < 0 or tail_lines < 0:
            raise ValueError("head_lines / tail_lines 不能为负")
        if head_lines + tail_lines >= max_lines_per_document:
            raise ValueError(
                "head_lines + tail_lines 必须小于 max_lines_per_document，否则截断不生效"
            )
        self.max_lines_per_document = max_lines_per_document
        self.head_lines = head_lines
        self.tail_lines = tail_lines

    # ------------------------------------------------------------ 对外入口
    def compress_documents(
        self,
        documents: Sequence[ContextDocument],
        *,
        budget_tokens: int,
    ) -> tuple[list[ContextDocument], CompressionReport]:
        """压缩一批文档到预算内，返回 ``(文档, 报告)``。"""
        if budget_tokens <= 0:
            raise ValueError("budget_tokens 必须为正整数")

        started = time.perf_counter()
        report = CompressionReport(
            budget=budget_tokens,
            before_documents=len(documents),
            before_tokens=estimate_documents_tokens(documents),
        )
        current = list(documents)

        current = self._apply(report, LEVEL_DEDUPE, current, self._dedupe)
        current = self._apply(report, LEVEL_MERGE, current, self._merge_adjacent)
        current = self._apply_truncate(report, current)

        # 装箱级必须记录"装箱前"的真实 token，否则看不出预算这一级省了多少
        tokens_before_fit = estimate_documents_tokens(current)
        count_before_fit = len(current)
        current, dropped, hard_trimmed = self._fit_budget(current, budget_tokens)
        report.dropped_documents = dropped
        report.hard_trimmed = hard_trimmed
        report.levels.append(
            LevelReport(
                name=LEVEL_BUDGET,
                applied=dropped > 0 or hard_trimmed > 0,
                documents_in=count_before_fit,
                documents_out=len(current),
                tokens_in=tokens_before_fit,
                tokens_out=estimate_documents_tokens(current),
                detail=f"按 score 装箱，丢弃 {dropped} 条，硬截断 {hard_trimmed} 条",
            )
        )

        report.after_documents = len(current)
        report.after_tokens = estimate_documents_tokens(current)
        report.duration = time.perf_counter() - started
        log_event(
            logger,
            "context.compress",
            before=report.before_tokens,
            after=report.after_tokens,
            ratio=round(report.ratio, 3),
            budget=budget_tokens,
            within_budget=report.within_budget,
        )
        return current, report

    def compress_sections(
        self,
        sections: Sequence[ContextSection],
        *,
        budget_tokens: int,
    ) -> tuple[list[ContextSection], CompressionReport]:
        """按节压缩：压完再按原节序还原，节标题与说明保持不变。

        为什么要还原节结构：排版（哪段是任务、哪段是证据）本身是信息，
        压缩只应删冗余内容，不应该顺手把结构也揉平。
        """
        flattened = [document for section in sections for document in section.documents]
        compressed, report = self.compress_documents(flattened, budget_tokens=budget_tokens)

        grouped: dict[str, list[ContextDocument]] = {section.title: [] for section in sections}
        for document in compressed:
            grouped.setdefault(document.section, []).append(document)

        notes: dict[str, str] = {}
        for section in sections:
            if section.note and section.title not in notes:
                notes[section.title] = section.note

        titles = [section.title for section in sections]
        for title in grouped:
            if title not in titles:
                titles.append(title)
        rebuilt = [
            ContextSection(
                title=title,
                documents=tuple(grouped.get(title, ())),
                note=notes.get(title, ""),
            )
            for title in titles
        ]
        return rebuilt, report

    # ------------------------------------------------------------ 各级实现
    def _apply(
        self,
        report: CompressionReport,
        name: str,
        documents: Sequence[ContextDocument],
        level: Callable[[Sequence[ContextDocument]], list[ContextDocument]],
    ) -> list[ContextDocument]:
        """跑一级"文档数可能变少"的压缩，并记录统计。"""
        tokens_in = estimate_documents_tokens(documents)
        result = level(documents)
        tokens_out = estimate_documents_tokens(result)
        report.levels.append(
            LevelReport(
                name=name,
                applied=len(result) != len(documents) or tokens_out != tokens_in,
                documents_in=len(documents),
                documents_out=len(result),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                detail=(f"文档 {len(documents)} → {len(result)}，token {tokens_in} → {tokens_out}"),
            )
        )
        return result

    def _apply_truncate(
        self, report: CompressionReport, documents: Sequence[ContextDocument]
    ) -> list[ContextDocument]:
        """截断级：条目数不变，但正文变短，需要单独记统计。"""
        tokens_in = estimate_documents_tokens(documents)
        result, truncated = self._truncate(documents)
        tokens_out = estimate_documents_tokens(result)
        report.truncated_documents = truncated
        report.levels.append(
            LevelReport(
                name=LEVEL_TRUNCATE,
                applied=truncated > 0,
                documents_in=len(documents),
                documents_out=len(result),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                detail=f"截断 {truncated} 条超长文档（阈值 {self.max_lines_per_document} 行）",
            )
        )
        return result

    def _dedupe(self, documents: Sequence[ContextDocument]) -> list[ContextDocument]:
        """同一位置/同一 doc_id 只保留分最高的一份；顺带丢掉空内容。"""
        winners: dict[tuple, ContextDocument] = {}
        order: list[tuple] = []
        for document in documents:
            if not document.content.strip():
                continue
            key = document_identity(document)
            existing = winners.get(key)
            if existing is None:
                winners[key] = document
                order.append(key)
            elif document.score > existing.score:
                winners[key] = document
        return [winners[key] for key in order]

    def _merge_adjacent(self, documents: Sequence[ContextDocument]) -> list[ContextDocument]:
        """同路径、同符号、行号连续的片段合成一块。

        只合并"同一个符号被切碎"的情况（例如超长函数被分块器切开），
        名字不同的片段绝不合并——那会把两个符号挤进一个证据头，损害可回溯性。
        """
        merged: list[ContextDocument] = []
        for document in documents:
            previous = merged[-1] if merged else None
            if (
                previous is not None
                and _mergeable(previous, document)
                and previous.end_line
                and document.start_line == previous.end_line + 1
            ):
                combined = previous.with_content(
                    f"{previous.content}\n{document.content}",
                    merged_count=int(previous.metadata.get("merged_count", 1)) + 1,
                )
                merged[-1] = ContextDocument(
                    doc_id=previous.doc_id,
                    content=combined.content,
                    source=previous.source,
                    path=previous.path,
                    start_line=previous.start_line,
                    end_line=document.end_line,
                    kind=previous.kind,
                    name=previous.name,
                    score=max(previous.score, document.score),
                    section=previous.section,
                    metadata=combined.metadata,
                )
                continue
            merged.append(document)
        return merged

    def _truncate(self, documents: Sequence[ContextDocument]) -> tuple[list[ContextDocument], int]:
        """超长文档保留头尾，中间标注省略行数。"""
        result: list[ContextDocument] = []
        truncated = 0
        for document in documents:
            lines = document.content.splitlines()
            if len(lines) <= self.max_lines_per_document:
                result.append(document)
                continue
            head = lines[: self.head_lines]
            tail = lines[-self.tail_lines :] if self.tail_lines else []
            elided = len(lines) - len(head) - len(tail)
            content = "\n".join([*head, ELISION_MARKER.format(elided=elided), *tail])
            result.append(document.with_content(content, elided_lines=elided))
            truncated += 1
        return result, truncated

    def _fit_budget(
        self, documents: Sequence[ContextDocument], budget_tokens: int
    ) -> tuple[list[ContextDocument], int, int]:
        """按 score 降序装箱；结果按原顺序还原，保证排版仍是"按文件阅读"。"""
        if not documents:
            return [], 0, 0

        ranked = sorted(
            enumerate(documents),
            key=lambda pair: (-pair[1].score, pair[1].path, pair[1].start_line, pair[0]),
        )
        kept: list[tuple[int, ContextDocument]] = []
        dropped = 0
        hard_trimmed = 0
        remaining = budget_tokens

        for index, document in ranked:
            cost = document.token_estimate()
            if cost <= remaining:
                kept.append((index, document))
                remaining -= cost
                continue
            if not kept:
                # 第一条就装不下：硬截断到装下，避免"空上下文"
                fitted = hard_fit(document, remaining)
                if fitted is not None:
                    kept.append((index, fitted))
                    remaining -= fitted.token_estimate()
                    hard_trimmed += 1
                    continue
            dropped += 1

        kept.sort(key=lambda pair: pair[0])
        return [document for _, document in kept], dropped, hard_trimmed

    # ------------------------------------------------------------ 展示
    def describe(self) -> dict[str, Any]:
        return {
            "max_lines_per_document": self.max_lines_per_document,
            "head_lines": self.head_lines,
            "tail_lines": self.tail_lines,
        }


# ------------------------------------------------------------------ 工具函数
def hard_fit(document: ContextDocument, max_tokens: int) -> ContextDocument | None:
    """把单条文档硬截断到 ``max_tokens`` 以内（截到行边界，二分找最长可行前缀）。

    注意：省略标记本身也要占 token，必须**一起**参与二分，
    否则会出现"截断后反而超预算"（初版就踩了这个坑）。
    返回 ``None`` 表示预算小到没有保留价值（连证据头都放不下）。
    """
    header_cost = estimate_tokens(document.header)
    available = max_tokens - header_cost
    if available < MIN_USEFUL_TOKENS:
        return None

    lines = document.content.splitlines()

    def cost(count: int) -> int:
        """保留前 ``count`` 行时的总开销（含证据头与省略标记）。"""
        elided = len(lines) - count
        return header_cost + estimate_tokens(_with_elision_marker("\n".join(lines[:count]), elided))

    low, high = 0, len(lines)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if cost(middle) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1

    # 省略标记的字符数会随省略行数的位数变化，二分前提不完全单调；
    # 这里再线性回退几步，确保"最终结果一定在预算内"这一硬约束成立。
    while best > 0 and cost(best) > max_tokens:
        best -= 1
    if best <= 0:
        return None

    elided = len(lines) - best
    content = _with_elision_marker("\n".join(lines[:best]), elided)
    return document.with_content(content, elided_lines=elided, hard_trimmed=True)


def _with_elision_marker(content: str, elided: int) -> str:
    """按需在结尾追加省略标记（``elided<=0`` 时原样返回）。"""
    if elided <= 0:
        return content
    return f"{content}\n{ELISION_MARKER.format(elided=elided)}"


def _mergeable(previous: ContextDocument, current: ContextDocument) -> bool:
    """能否合并：路径、来源、符号类型与名字都必须一致。"""
    return (
        previous.path == current.path
        and previous.source == current.source
        and previous.kind == current.kind
        and previous.name == current.name
        and previous.section == current.section
    )
