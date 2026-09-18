"""RAG 检索示例：索引一个仓库并检索相关代码片段（W4 验收项）。

用法::

    python examples/rag_search.py "用户登录逻辑在哪"
    python examples/rag_search.py --repo data/sample_repo --top-k 3 "SQL 注入风险"
    python examples/rag_search.py --interactive          # 交互式连续检索

验收标准：输入"用户登录逻辑在哪"能返回相关代码片段
（应命中 ``app/auth/service.py`` 中的 ``login``）。

无需 API Key：未配置 ``EMBEDDING_API_KEY`` 时自动降级为离线向量化（无语义能力）
+ BM25 词法检索，仍能靠标识符与中文 bigram 命中目标片段。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from codeagentx.config import get_config  # noqa: E402
from codeagentx.core.exceptions import CodeAgentXError  # noqa: E402
from codeagentx.rag.rag_tool import RAGTool, build_rag_tool  # noqa: E402

DEFAULT_REPO = Path(__file__).resolve().parents[1] / "data" / "sample_repo"
DEFAULT_QUERY = "用户登录逻辑在哪"


def _build_tool(repo: Path, *, top_k: int) -> RAGTool:
    config = get_config()
    tool = build_rag_tool(config, root=repo, default_top_k=top_k)
    embedder = tool.embedder
    mode = "语义向量" if embedder.is_semantic else "离线降级向量（无语义能力）+ BM25 词法"
    print(f"[配置] 向量化={embedder.model_id}（{mode}）| 向量维度={embedder.dim}")
    return tool


def _index(tool: RAGTool, repo: Path) -> None:
    result = tool.run(action="index", path=str(repo))
    if not result.success:
        print(f"[索引失败] {result.error}")
        return
    print(f"[索引] {result.output}")


def _search(tool: RAGTool, query: str, *, top_k: int, show_content: bool) -> None:
    result = tool.run(action="search", query=query, top_k=top_k)
    if not result.success:
        print(f"[检索失败] {result.error}")
        return
    if not show_content:
        # RAGTool 的 output 已含片段正文；这里只展示位置与命中来源时改用结构化结果
        print(f"检索「{query}」命中 {result.metadata['count']} 个片段：")
        for index, chunk in enumerate(result.metadata["results"], start=1):
            sources = "+".join(chunk["sources"])
            print(f"  [{index}] {chunk['location']}  {chunk['kind']} 命中：{sources}")
        return
    print(result.output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeAgentX RAG 检索示例")
    parser.add_argument("query", nargs="*", help=f"检索内容，缺省为「{DEFAULT_QUERY}」")
    parser.add_argument("--repo", default=str(DEFAULT_REPO), help="要索引的仓库根目录")
    parser.add_argument("--top-k", type=int, default=3, help="返回片段数，默认 3")
    parser.add_argument("--interactive", action="store_true", help="交互式连续检索")
    parser.add_argument("--brief", action="store_true", help="只打印片段位置，不打印正文")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"[参数错误] 仓库目录不存在：{repo}", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("[参数错误] --top-k 必须为正整数", file=sys.stderr)
        return 2

    try:
        tool = _build_tool(repo, top_k=args.top_k)
    except CodeAgentXError as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return 2

    _index(tool, repo)
    print()

    if args.interactive:
        print("输入检索内容，exit / quit 退出。\n")
        while True:
            try:
                query = input("检索: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit", ":q"}:
                break
            _search(tool, query, top_k=args.top_k, show_content=not args.brief)
            print()
        return 0

    _search(
        tool,
        " ".join(args.query) if args.query else DEFAULT_QUERY,
        top_k=args.top_k,
        show_content=not args.brief,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
