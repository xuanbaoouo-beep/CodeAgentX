"""RAGTool：把代码检索能力暴露给 Agent（放 rag 层以遵守依赖方向）。

Agent 需要的往往不是"整个仓库"，而是"和当前问题相关的那几段代码"。
本工具封装 :class:`~codeagentx.rag.retriever.HybridRetriever`，提供两个动作：

``search``（默认）
    用自然语言/关键词检索，返回片段内容 + ``path:start-end`` 位置 + 命中来源。
``index``
    为指定仓库建立（或重建）索引；路径必须落在 :class:`~codeagentx.tools.sandbox.SandboxPolicy`
    允许的根目录内——**路径来自模型，必须当成不可信输入校验**。

输出分两份（见 :class:`~codeagentx.tools.base.ToolResult`）：
``output`` 是给模型看的精简文本，``metadata["results"]`` 是给程序用的结构化结果。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from codeagentx.core.exceptions import ToolValidationError
from codeagentx.core.logger import get_logger
from codeagentx.rag.embedder import BaseEmbedder, build_embedder
from codeagentx.rag.indexer import (
    DEFAULT_INDEX_BATCH_SIZE,
    RepositoryIndexer,
    build_vector_store,
)
from codeagentx.rag.retriever import (
    DEFAULT_CANDIDATES,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    HybridRetriever,
    RetrievedChunk,
)
from codeagentx.rag.vector_store import BaseVectorStore
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.sandbox import SandboxPolicy

logger = get_logger("rag.rag_tool")

#: 支持的动作
ACTIONS: tuple[str, ...] = ("search", "index")
#: 单个片段在文本输出中的最大字符数（超出则截断，避免把上下文撑爆）
DEFAULT_MAX_CONTENT_CHARS = 1500
#: 单次检索返回片段数的上限
DEFAULT_MAX_TOP_K = 20


class RAGTool(BaseTool):
    """混合检索工具（语义 + 词法），带可选的索引动作。"""

    name = "code_search"
    description = (
        "在代码库中检索与问题相关的代码片段，返回片段内容与所在位置"
        "（文件路径:起止行、函数/类名）。适合回答"
        "“某功能在哪实现”“某函数的调用方是谁”这类需要定位代码的问题。"
        "若向量库为空，先用 action=\"index\" 并为 path 指定仓库根目录建立索引。"
    )
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="操作类型：search=检索（默认），index=为 path 建立/重建索引",
            required=False,
            default="search",
            enum=list(ACTIONS),
        ),
        ToolParameter(
            name="query",
            type="string",
            description="检索内容，可以是自然语言（“用户登录逻辑在哪”）或代码关键词（“login password”）",
            required=False,
        ),
        ToolParameter(
            name="path",
            type="string",
            description="要索引的仓库根目录，必须位于允许的根目录内（仅 action=index 时需要）",
            required=False,
        ),
        ToolParameter(
            name="top_k",
            type="integer",
            description=f"返回片段数，默认 {DEFAULT_TOP_K}，最大 {DEFAULT_MAX_TOP_K}",
            required=False,
        ),
        ToolParameter(
            name="language",
            type="string",
            description="只检索指定语言的片段，例如 python / markdown",
            required=False,
        ),
    ]

    def __init__(
        self,
        *,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        policy: SandboxPolicy | None = None,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        candidates: int = DEFAULT_CANDIDATES,
        rrf_k: int = DEFAULT_RRF_K,
        enable_lexical: bool = True,
        default_top_k: int = DEFAULT_TOP_K,
        max_top_k: int = DEFAULT_MAX_TOP_K,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.store = store
        self.policy = policy
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k
        self.max_content_chars = max_content_chars
        # 索引器与检索器共用同一个 store，避免"索引到一个库、查另一个库"
        self.indexer = RepositoryIndexer(embedder=embedder, store=store, batch_size=batch_size)
        self.retriever = HybridRetriever(
            embedder=embedder,
            store=store,
            candidates=candidates,
            rrf_k=rrf_k,
            enable_lexical=enable_lexical,
        )

    # ------------------------------------------------------------ 程序接口
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        language: str | None = None,
    ) -> list[RetrievedChunk]:
        """检索代码片段（供编排层直接调用，参数已校验）。"""
        filters = {"language": language} if language else None
        return self.retriever.retrieve(
            query, top_k=self._clamp_top_k(top_k), filters=filters
        )

    def index_repository(self, root: str | Path, *, reset: bool = True) -> dict[str, Any]:
        """为仓库建立索引；``root`` 必须先通过沙箱路径校验。"""
        resolved = self._resolve_root(root)
        stats = self.indexer.index_repository(resolved, reset=reset)
        self.retriever.refresh()  # 语料变了，BM25 必须重建
        logger.info("索引完成：%s → %s", resolved, stats.as_dict())
        return stats.as_dict()

    # ------------------------------------------------------------ 工具入口
    def _run(
        self,
        action: str = "search",
        query: str | None = None,
        path: str | None = None,
        top_k: int | None = None,
        language: str | None = None,
    ) -> ToolResult:
        if action not in ACTIONS:
            raise ToolValidationError(
                f"未知的 action：{action!r}",
                detail=f"可选动作：{list(ACTIONS)}",
            )
        if action == "index":
            return self._run_index(path)
        return self._run_search(query, top_k=top_k, language=language)

    def _run_index(self, path: str | None) -> ToolResult:
        if not path:
            raise ToolValidationError("action=index 时必须提供 path（仓库根目录）")
        if self.policy is None:
            return ToolResult.fail(
                "未配置沙箱策略，拒绝为任意路径建立索引",
                error_type="SecurityPolicyMissing",
                hint="构造 RAGTool 时传入 SandboxPolicy（build_rag_tool(..., root=...) 会自动构造）",
            )
        stats = self.index_repository(path)
        return ToolResult.ok(
            f"索引完成：{stats['files']} 个文件、{stats['chunks']} 个代码片段，"
            f"向量维度 {stats['dim']}，耗时 {stats['duration']}s"
            + ("（当前为离线降级向量化，无语义能力）" if not stats["semantic"] else ""),
            **stats,
        )

    def _run_search(
        self,
        query: str | None,
        *,
        top_k: int | None,
        language: str | None,
    ) -> ToolResult:
        if not query or not str(query).strip():
            raise ToolValidationError("action=search 时必须提供非空的 query")
        # 参数校验优先于环境状态判断：库空与否都不该放过非法参数
        effective_top_k = self._clamp_top_k(top_k)
        if not self.store.count():
            return ToolResult.ok(
                f"向量库为空，尚未索引任何仓库。请先调用 action=\"index\" 并指定 path。"
                f"（当前查询：{query}）",
                query=query,
                count=0,
                results=[],
                indexed=0,
            )
        results = self.search(str(query), top_k=effective_top_k, language=language)
        return ToolResult.ok(
            self._render(str(query), results),
            query=query,
            count=len(results),
            indexed=self.store.count(),
            results=[chunk.to_dict() for chunk in results],
        )

    # ------------------------------------------------------------ 内部
    def _clamp_top_k(self, top_k: int | None) -> int:
        if top_k is None:
            return self.default_top_k
        try:
            value = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"top_k 必须是整数，收到 {top_k!r}") from exc
        if value <= 0:
            raise ToolValidationError(f"top_k 必须为正整数，收到 {value}")
        return min(value, self.max_top_k)

    def _resolve_root(self, root: str | Path) -> Path:
        if self.policy is None:
            # 没有策略时不放行任何路径（由 _run_index 给出可操作的提示）
            raise ToolValidationError("未配置沙箱策略，无法解析索引目录")
        resolved = self.policy.path_guard().resolve(
            root, base=self.policy.default_workdir, must_exist=True
        )
        if not resolved.is_dir():
            raise ToolValidationError(f"索引目标不是目录：{resolved}")
        return resolved

    def _render(self, query: str, results: Sequence[RetrievedChunk]) -> str:
        if not results:
            return f"未检索到与「{query}」相关的代码片段。"
        lines = [f"检索「{query}」命中 {len(results)} 个代码片段："]
        for index, chunk in enumerate(results, start=1):
            symbol = f" {chunk.symbol}" if chunk.symbol else ""
            sources = "+".join(chunk.sources) or "unknown"
            lines.append("")
            lines.append(f"[{index}] {chunk.location}  {chunk.kind}{symbol}  命中：{sources}")
            lines.append(_truncate(chunk.content, self.max_content_chars))
        return "\n".join(lines)


def build_rag_tool(
    config: Any,
    *,
    root: str | Path | None = None,
    policy: SandboxPolicy | None = None,
    embedder: BaseEmbedder | None = None,
    store: BaseVectorStore | None = None,
    allow_fallback: bool = True,
    **kwargs: Any,
) -> RAGTool:
    """按配置组装 RAGTool（embedder 与 store 只建一次，索引与检索共用）。

    Args:
        root: 允许建立索引的仓库根目录；给了它就自动构造沙箱策略。
            不给则只能检索已有的向量库（更安全，适合只读场景）。
    """
    embedder = embedder or build_embedder(config, allow_fallback=allow_fallback)
    store = store or build_vector_store(config, dim=embedder.dim)
    if policy is None and root is not None:
        policy = SandboxPolicy.for_roots([root])
    return RAGTool(embedder=embedder, store=store, policy=policy, **kwargs)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n...（片段已截断，完整长度 {len(text)} 字符）"
