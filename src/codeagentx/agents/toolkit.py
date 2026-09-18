"""审查工具集工厂：一处装配 Agent 需要的工具，避免每个 Agent 各自拼装。

为什么需要它
------------
1. **作用域最小化**：Agent 只挂载自己需要的工具，既省 Token 又降低误调用概率
   （:class:`~codeagentx.tools.registry.ToolRegistry` 即一个工具作用域）。
2. **策略唯一**：所有工具共用同一个 :class:`~codeagentx.tools.sandbox.SandboxPolicy`，
   保证"路径能读"与"RAG 能索引"用的是同一套允许根目录，不会出现权限口径不一致。
3. **可测性**：embedder / store 可注入，测试时全部走内存实现，不触网也不写盘。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from codeagentx.config import Config, get_config
from codeagentx.rag.embedder import BaseEmbedder
from codeagentx.rag.rag_tool import build_rag_tool
from codeagentx.rag.vector_store import BaseVectorStore
from codeagentx.tools.registry import ToolRegistry
from codeagentx.tools.sandbox import SandboxPolicy
from codeagentx.tools.static_analyzer import StaticAnalyzer
from codeagentx.tools.terminal import TerminalTool
from codeagentx.tools.test_runner import TestRunner

#: 默认审查工具集：检索 + 静态分析 + 只读命令（不含 test_runner，跑测试较慢）
DEFAULT_REVIEW_TOOLS: tuple[str, ...] = ("code_search", "static_analyzer", "terminal")
#: 完整工具集：额外包含 pytest 运行器
FULL_REVIEW_TOOLS: tuple[str, ...] = (*DEFAULT_REVIEW_TOOLS, "test_runner")
#: 全部受支持的具名工具
SUPPORTED_TOOLS: frozenset[str] = frozenset(FULL_REVIEW_TOOLS)


def build_review_toolkit(
    config: Config | None = None,
    *,
    root: str | Path,
    tools: Sequence[str] = DEFAULT_REVIEW_TOOLS,
    policy: SandboxPolicy | None = None,
    embedder: BaseEmbedder | None = None,
    store: BaseVectorStore | None = None,
    allow_fallback: bool = True,
    registry_name: str = "review",
) -> ToolRegistry:
    """按需装配审查工具集。

    Args:
        root: 允许访问的仓库根目录（所有工具共用同一策略）。
        tools: 需要装配的工具名，可选 ``code_search`` / ``static_analyzer`` /
            ``terminal`` / ``test_runner``。
        policy: 显式传入沙箱策略；缺省按 ``root`` 构造。
        embedder / store: 检索后端，缺省按配置构建（测试可注入内存实现）。
        allow_fallback: 未配置 Embedding 密钥时是否降级为离线哈希向量。
    """
    unknown = sorted(set(tools) - SUPPORTED_TOOLS)
    if unknown:
        raise ValueError(f"不支持的工具名：{unknown}，可选 {sorted(SUPPORTED_TOOLS)}")

    config = config or get_config()
    policy = policy or SandboxPolicy.for_roots([root])
    registry = ToolRegistry(name=registry_name)

    if "code_search" in tools:
        registry.register(
            build_rag_tool(
                config,
                root=policy.default_workdir,
                policy=policy,
                embedder=embedder,
                store=store,
                allow_fallback=allow_fallback,
            )
        )
    if "static_analyzer" in tools:
        registry.register(StaticAnalyzer(policy))
    if "terminal" in tools:
        registry.register(TerminalTool(policy))
    if "test_runner" in tools:
        registry.register(TestRunner(policy))
    return registry


def describe_toolkit(registry: ToolRegistry) -> dict[str, Any]:
    """工具集摘要，便于日志与报告元信息。"""
    return {"registry": registry.name, "tools": registry.names(), "count": len(registry)}


__all__ = [
    "DEFAULT_REVIEW_TOOLS",
    "FULL_REVIEW_TOOLS",
    "SUPPORTED_TOOLS",
    "build_review_toolkit",
    "describe_toolkit",
]
