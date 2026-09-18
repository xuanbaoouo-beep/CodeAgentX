"""Retriever Agent：把"问题"变成"带位置的真实代码片段"。

为什么这个 Agent 不调用 LLM
---------------------------
检索本身是确定性操作（混合检索 + RRF 融合），让模型把检索结果转述一遍
只会引入失真、多花 Token，还会让"证据来自哪个文件"变得不可追溯。
所以这里只做三件事：

1. 把编排层给的查询词逐个交给 ``code_search``（必要时先建索引）；
2. 把工具返回的结构化片段收敛成 :class:`~codeagentx.agents.schemas.Evidence`
   （路径、起止行、符号名、命中来源、分数），并按 ``(-score, path, start_line)``
   稳定排序、按位置去重；
3. 用 :meth:`RetrieverAgent.evidence_digest` 生成一段文字线索交给审查角色。

仍返回 :class:`~codeagentx.core.agent.AgentResult`，是为了让编排层对所有角色
用同一套调用方式（``agent.run(...)``），不必为"有没有 LLM"分叉。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from codeagentx.agents.schemas import Evidence, normalize_str_list
from codeagentx.core.agent import ZERO_USAGE, AgentResult
from codeagentx.core.logger import get_logger, log_event
from codeagentx.tools.registry import ToolRegistry

logger = get_logger("agents.retriever")

#: 默认返回片段数
DEFAULT_TOP_K = 5
#: 单次请求片段数上限（与 RAGTool 的上限保持一致）
MAX_TOP_K = 20
#: 线索文本的总字符预算（超出按片段截断，避免把审查角色的上下文撑爆）
DEFAULT_DIGEST_CHARS = 6000

#: 检索工具名（rag 层实现）
SEARCH_TOOL = "code_search"


class RetrieverAgent:
    """检索 Agent：确定性取证，不做任何推测。"""

    name = "retriever"

    def __init__(
        self,
        tools: ToolRegistry,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_top_k: int = MAX_TOP_K,
        digest_chars: int = DEFAULT_DIGEST_CHARS,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        self.tools = tools
        self.top_k = top_k
        self.max_top_k = max_top_k
        self.digest_chars = digest_chars

    # ------------------------------------------------------------ 程序接口
    def ensure_index(self, root: str | Path) -> tuple[bool, str]:
        """为仓库建立（或重建）索引，返回 ``(是否成功, 说明)``。

        索引动作是幂等的：``chunk_id`` 由路径与位置决定，重复索引只覆盖不累积。
        """
        result = self.tools.execute(SEARCH_TOOL, {"action": "index", "path": str(root)})
        text = result.output if isinstance(result.output, str) else (result.error or "")
        if not result.success:
            log_event(
                logger,
                "retriever_index_failed",
                agent=self.name,
                error=result.error,
                error_type=result.error_type,
            )
        return result.success, text or ""

    def retrieve(self, queries: Sequence[str] | str, *, top_k: int | None = None) -> tuple[list[Evidence], list[str]]:
        """按查询词批量检索，返回 ``(证据列表, 错误列表)``。"""
        wanted = min(top_k or self.top_k, self.max_top_k)
        evidence: list[Evidence] = []
        errors: list[str] = []
        for query in normalize_str_list(queries):
            result = self.tools.execute(SEARCH_TOOL, {"query": query, "top_k": wanted})
            if not result.success:
                # 单个查询失败不终止整体检索：检索是"尽力而为"的取证，不是硬依赖
                errors.append(f"{query}：{result.error}")
                continue
            for item in result.metadata.get("results") or []:
                evidence.append(Evidence.from_chunk(item, query=query))
        return _dedupe(evidence), errors

    def run(
        self,
        queries: Sequence[str] | str,
        *,
        top_k: int | None = None,
        index_root: str | Path | None = None,
        **_: Any,
    ) -> AgentResult:
        """执行一次取证。

        Args:
            queries: 查询词列表（也可传单个字符串）。
            index_root: 需要先建索引的仓库根目录；``None`` 表示直接用已有索引。
        """
        notes: list[str] = []
        if index_root is not None:
            ok, detail = self.ensure_index(index_root)
            notes.append(detail if ok else f"索引失败：{detail}")
        evidence, errors = self.retrieve(queries, top_k=top_k)

        asked = normalize_str_list(queries)
        # 全部查询都失败才算这一步失败；"没命中"是正常结果，不是故障
        success = not errors or len(errors) < len(asked)
        result = AgentResult(
            output=self.evidence_digest(evidence),
            success=success,
            error=None if success else "；".join(errors),
            usage=dict(ZERO_USAGE),
            metadata={
                "agent": self.name,
                "queries": asked,
                "count": len(evidence),
                "evidence": [item.to_dict() for item in evidence],
                "errors": errors,
                "notes": notes,
            },
        )
        log_event(
            logger,
            "retriever_finished",
            agent=self.name,
            queries=len(asked),
            evidence=len(evidence),
            errors=len(errors),
        )
        return result

    # ------------------------------------------------------------ 内部
    def evidence_digest(self, evidence: Sequence[Evidence], *, limit_chars: int | None = None) -> str:
        """把证据拼成线索文本（每条带位置与命中来源，供复核）。"""
        if not evidence:
            return "（未检索到相关代码片段）"
        budget = limit_chars or self.digest_chars
        blocks: list[str] = []
        used = 0
        for index, item in enumerate(evidence, start=1):
            block = f"[证据 {index}] {item.to_text()}"
            if used + len(block) > budget:
                blocks.append(f"...（其余 {len(evidence) - index + 1} 条证据因篇幅省略）")
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)


def _dedupe(evidence: Sequence[Evidence]) -> list[Evidence]:
    """按位置去重（同一片段被多个查询命中时合并来源），并按分数稳定排序。"""
    merged: dict[tuple[str, int, int], Evidence] = {}
    for item in evidence:
        key = (item.path, item.start_line, item.end_line)
        current = merged.get(key)
        if current is None:
            merged[key] = item
            continue
        for source in item.sources:
            if source not in current.sources:
                current.sources.append(source)
        if item.score > current.score:
            current.score = item.score
    return sorted(
        merged.values(),
        key=lambda item: (-item.score, item.path, item.start_line, item.end_line),
    )


__all__ = [
    "DEFAULT_DIGEST_CHARS",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "SEARCH_TOOL",
    "RetrieverAgent",
]
