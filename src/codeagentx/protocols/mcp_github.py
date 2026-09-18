"""GitHub 的 MCP 服务端（stdio，只读）。

存在意义：W7 的总验收是"**能自动读取 GitHub 仓库并生成上下文**"。
:mod:`~codeagentx.protocols.github_client` 已经能读仓库，但 Agent 需要的是
"可被模型调用的工具"——所以这里把它按 MCP 的方式暴露出去：

    python -m codeagentx.protocols.mcp_github --repo owner/name

令牌怎么传
----------
**不放在命令行**（会出现在进程列表里），而是走环境变量 ``GITHUB_TOKEN``
（可由 ``--token-env`` 改名）；客户端的 :func:`~codeagentx.protocols.mcp_client.child_env`
会把父进程环境原样继承给子进程，所以不用额外做什么。

只读
----
全部工具都是读操作：仓库元信息、仓库树、文件内容、PR 列表/详情/变更/完整 diff。
没有任何写操作——Agent 可以"读世界的代码"，但不能替用户改远端仓库。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from codeagentx.core.exceptions import MCPError
from codeagentx.core.logger import get_logger
from codeagentx.protocols.github_client import (
    DEFAULT_API_BASE,
    DEFAULT_MAX_ENTRIES,
    DEFAULT_MAX_FILE_CHARS,
    DEFAULT_MAX_PR_FILES,
    GitHubClient,
    GitHubRepoRef,
)
from codeagentx.protocols.mcp_protocol import MCPErrorCode, MCPToolSpec
from codeagentx.protocols.mcp_server import MCPServerBase, ToolOutcome, serve_stdio

logger = get_logger("protocols.mcp_github")

__all__ = [
    "GitHubMCPServer",
    "github_server_command",
    "github_tools",
    "main",
]

SERVER_NAME = "codeagentx-github"
SERVER_VERSION = "0.1.0"
SERVER_CAPABILITIES: dict[str, Any] = {"tools": {"listChanged": False}}
SERVER_INSTRUCTIONS = (
    "只读 GitHub 服务：读取仓库元信息、仓库树、文件内容，以及 Pull Request 的"
    "列表/详情/文件变更/完整 diff。所有工具都是读操作，不会修改远端仓库。"
    "未配置 GITHUB_TOKEN 时只能访问公开仓库，且会更容易触发速率限制。"
)

TOOL_GET_REPOSITORY = "get_repository"
TOOL_LIST_TREE = "list_tree"
TOOL_READ_FILE = "read_file"
TOOL_LIST_PULL_REQUESTS = "list_pull_requests"
TOOL_GET_PULL_REQUEST = "get_pull_request"
TOOL_GET_PULL_REQUEST_FILES = "get_pull_request_files"
TOOL_GET_PULL_REQUEST_DIFF = "get_pull_request_diff"

#: 树清单在文本里最多渲染多少行（再多也没人看，真要看走 read_file）
TEXT_MAX_TREE_LINES = 400
#: PR 描述在文本里的截断长度
TEXT_MAX_PR_BODY = 600
#: 单个文件的 patch 在文本里的截断长度
TEXT_MAX_PATCH = 4000

_REPO_FIELD = {
    "type": "string",
    "description": "仓库标识，形如 owner/repo；也接受完整 URL 或 owner/repo@ref",
}
_REF_FIELD = {"type": "string", "description": "分支/标签/提交；缺省用仓库默认分支"}
_NUMBER_FIELD = {"type": "integer", "description": "Pull Request 编号"}


# ------------------------------------------------------------------ 工具定义
def github_tools() -> list[MCPToolSpec]:
    """服务端声明的工具列表（``tools/list`` 的返回内容）。"""
    return [
        MCPToolSpec(
            name=TOOL_GET_REPOSITORY,
            description="读取仓库元信息（默认分支、语言、星标、是否归档等）",
            input_schema={
                "type": "object",
                "properties": {"repo": _REPO_FIELD},
                "required": ["repo"],
            },
        ),
        MCPToolSpec(
            name=TOOL_LIST_TREE,
            description="读取仓库文件树（一次列全，可按目录前缀过滤）",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": _REPO_FIELD,
                    "ref": _REF_FIELD,
                    "recursive": {"type": "boolean", "description": "是否递归，默认 true"},
                    "path_prefix": {"type": "string", "description": "只保留该目录下的条目"},
                    "max_entries": {"type": "integer", "description": "最多返回条数"},
                },
                "required": ["repo"],
            },
        ),
        MCPToolSpec(
            name=TOOL_READ_FILE,
            description="读取仓库内一个文本文件的完整内容（超出上限会截断并标注）",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": _REPO_FIELD,
                    "path": {"type": "string", "description": "仓库内相对路径，如 app/auth/service.py"},
                    "ref": _REF_FIELD,
                    "max_chars": {"type": "integer", "description": "最多返回的字符数"},
                },
                "required": ["repo", "path"],
            },
        ),
        MCPToolSpec(
            name=TOOL_LIST_PULL_REQUESTS,
            description="列出 PR（可按 open/closed/all 过滤）",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": _REPO_FIELD,
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "默认 open",
                    },
                    "limit": {"type": "integer", "description": "最多返回条数，默认 20"},
                },
                "required": ["repo"],
            },
        ),
        MCPToolSpec(
            name=TOOL_GET_PULL_REQUEST,
            description="读取单个 PR 的详情（标题、描述、分支、增删行数）",
            input_schema={
                "type": "object",
                "properties": {"repo": _REPO_FIELD, "number": _NUMBER_FIELD},
                "required": ["repo", "number"],
            },
        ),
        MCPToolSpec(
            name=TOOL_GET_PULL_REQUEST_FILES,
            description="读取 PR 的文件变更列表，每个文件带 unified diff 片段",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": _REPO_FIELD,
                    "number": _NUMBER_FIELD,
                    "limit": {"type": "integer", "description": "最多返回多少个文件"},
                },
                "required": ["repo", "number"],
            },
        ),
        MCPToolSpec(
            name=TOOL_GET_PULL_REQUEST_DIFF,
            description="读取 PR 的完整 unified diff（纯文本）",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": _REPO_FIELD,
                    "number": _NUMBER_FIELD,
                    "max_chars": {"type": "integer", "description": "最多返回的字符数"},
                },
                "required": ["repo", "number"],
            },
        ),
    ]


def github_server_command(
    *,
    repos: Sequence[str] = (),
    python: str | None = None,
    base_url: str | None = None,
) -> tuple[str, list[str]]:
    """构造启动 GitHub MCP 服务端的命令（令牌走环境变量，不进命令行）。"""
    executable = python or sys.executable
    args = ["-m", "codeagentx.protocols.mcp_github"]
    for repo in repos:
        args.extend(["--repo", str(repo)])
    if base_url:
        args.extend(["--base-url", str(base_url)])
    return executable, args


# ------------------------------------------------------------------ 服务端
class GitHubMCPServer(MCPServerBase):
    """只读 GitHub MCP 服务端。

    Args:
        client: 注入的 :class:`GitHubClient`（测试用假 client / 离线 fixture 注入）。
        allowed_repos: 允许访问的仓库白名单；为空表示不限制。
    """

    server_name = SERVER_NAME
    server_version = SERVER_VERSION
    capabilities = SERVER_CAPABILITIES
    instructions = SERVER_INSTRUCTIONS

    def __init__(
        self,
        *,
        client: GitHubClient | None = None,
        allowed_repos: Sequence[str] = (),
        token: str = "",
        base_url: str = DEFAULT_API_BASE,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
        max_pr_files: int = DEFAULT_MAX_PR_FILES,
    ) -> None:
        self.client = client or GitHubClient(
            token=token,
            base_url=base_url,
            max_entries=max_entries,
            max_file_chars=max_file_chars,
            max_pr_files=max_pr_files,
        )
        self.allowed_repos = frozenset(
            GitHubRepoRef.parse(repo).slug.lower() for repo in allowed_repos if str(repo).strip()
        )
        self.max_entries = max_entries
        self.max_file_chars = max_file_chars
        self.max_pr_files = max_pr_files
        self._handlers = {
            TOOL_GET_REPOSITORY: self._get_repository,
            TOOL_LIST_TREE: self._list_tree,
            TOOL_READ_FILE: self._read_file,
            TOOL_LIST_PULL_REQUESTS: self._list_pull_requests,
            TOOL_GET_PULL_REQUEST: self._get_pull_request,
            TOOL_GET_PULL_REQUEST_FILES: self._get_pull_request_files,
            TOOL_GET_PULL_REQUEST_DIFF: self._get_pull_request_diff,
        }
        logger.debug(
            "GitHub MCP 服务端就绪：base_url=%s 已鉴权=%s 白名单=%s",
            self.client.base_url,
            self.client.is_authenticated,
            sorted(self.allowed_repos) or "（不限制）",
        )

    def close(self) -> None:
        self.client.close()

    # -------------------------------------------------------- 工具声明与调用
    def list_tools(self) -> list[MCPToolSpec]:
        return github_tools()

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        # 工具名合法性已由 MCPServerBase._tools_call 用 list_tools() 校验过
        return self._handlers[name](arguments)

    # -------------------------------------------------------- 工具实现
    def _get_repository(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        repository = self.client.get_repository(self._repo(arguments))
        return ToolOutcome(repository.to_text(), repository.to_dict())

    def _list_tree(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        tree = self.client.list_tree(
            self._repo(arguments),
            ref=_text_or_none(arguments.get("ref")),
            recursive=_bool_arg(arguments.get("recursive"), default=True),
            path_prefix=_text_or_none(arguments.get("path_prefix")),
            max_entries=_int_arg(arguments.get("max_entries"), self.max_entries, 1, self.max_entries),
        )
        lines = min(self.max_entries, TEXT_MAX_TREE_LINES)
        return ToolOutcome(tree.to_text(max_lines=lines), tree.to_dict(with_entries=False))

    def _read_file(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        path = str(arguments.get("path") or "").strip()
        if not path:
            raise MCPError("缺少必填参数：path", code=int(MCPErrorCode.INVALID_PARAMS))
        file = self.client.read_file(
            self._repo(arguments),
            path,
            ref=_text_or_none(arguments.get("ref")),
            max_chars=_int_arg(arguments.get("max_chars"), self.max_file_chars, 1, self.max_file_chars),
        )
        return ToolOutcome(file.text, file.to_dict())

    def _list_pull_requests(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        pulls = self.client.list_pull_requests(
            self._repo(arguments),
            state=_text_or_none(arguments.get("state")) or "open",
            limit=_int_arg(arguments.get("limit"), 20, 1, 100),
        )
        text = "\n\n".join(pull.to_text(body_chars=TEXT_MAX_PR_BODY) for pull in pulls)
        structured = {
            "count": len(pulls),
            "pull_requests": [pull.to_dict() for pull in pulls],
        }
        return ToolOutcome(text or "（没有匹配的 PR）", structured)

    def _get_pull_request(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        pull = self.client.get_pull_request(self._repo(arguments), self._number(arguments))
        return ToolOutcome(pull.to_text(), pull.to_dict())

    def _get_pull_request_files(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        files = self.client.get_pull_request_files(
            self._repo(arguments),
            self._number(arguments),
            limit=_int_arg(arguments.get("limit"), self.max_pr_files, 1, self.max_pr_files),
        )
        text = "\n\n".join(item.to_text(patch_chars=TEXT_MAX_PATCH) for item in files)
        structured = {"count": len(files), "files": [item.to_dict() for item in files]}
        return ToolOutcome(text or "（该 PR 没有文件变更）", structured)

    def _get_pull_request_diff(self, arguments: Mapping[str, Any]) -> ToolOutcome:
        diff = self.client.get_pull_request_diff(
            self._repo(arguments),
            self._number(arguments),
            max_chars=_int_arg(arguments.get("max_chars"), self.max_file_chars, 1, self.max_file_chars),
        )
        return ToolOutcome(diff or "（该 PR 没有 diff）")

    # -------------------------------------------------------- 入参工具
    def _repo(self, arguments: Mapping[str, Any]) -> GitHubRepoRef:
        raw = str(arguments.get("repo") or "").strip()
        if not raw:
            raise MCPError("缺少必填参数：repo", code=int(MCPErrorCode.INVALID_PARAMS))
        try:
            identifier = GitHubRepoRef.parse(raw)
        except ValueError as exc:
            raise MCPError(
                f"仓库标识非法：{raw}", detail=str(exc), code=int(MCPErrorCode.INVALID_PARAMS)
            ) from exc
        if self.allowed_repos and identifier.slug.lower() not in self.allowed_repos:
            raise MCPError(
                f"仓库不在允许列表：{identifier.slug}",
                detail=f"允许：{sorted(self.allowed_repos)}",
                code=int(MCPErrorCode.INVALID_PARAMS),
            )
        return identifier

    def _number(self, arguments: Mapping[str, Any]) -> int:
        value = arguments.get("number")
        if value is None:
            raise MCPError("缺少必填参数：number", code=int(MCPErrorCode.INVALID_PARAMS))
        return _int_arg(value, 0, 1, 10**9)


# ------------------------------------------------------------------ 入参辅助
def _int_arg(value: Any, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MCPError(
            f"参数必须是整数：{value!r}", code=int(MCPErrorCode.INVALID_PARAMS)
        ) from exc
    return max(minimum, min(number, maximum))


def _bool_arg(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise MCPError(f"参数必须是布尔值：{value!r}", code=int(MCPErrorCode.INVALID_PARAMS))


def _text_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


# ------------------------------------------------------------------ 入口
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codeagentx-github-mcp",
        description="CodeAgentX 内置 GitHub MCP 服务端（stdio，只读）",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="允许访问的仓库 owner/name（可多次指定；不指定则不限制）",
    )
    parser.add_argument("--base-url", default=DEFAULT_API_BASE, help="GitHub API 根地址")
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="从哪个环境变量读令牌（默认 GITHUB_TOKEN；**不要**写在命令行里）",
    )
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    parser.add_argument("--max-file-chars", type=int, default=DEFAULT_MAX_FILE_CHARS)
    parser.add_argument("--max-pr-files", type=int, default=DEFAULT_MAX_PR_FILES)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        logger.warning("未在环境变量 %s 中读到令牌，只能访问公开仓库", args.token_env)

    server = GitHubMCPServer(
        allowed_repos=args.repo,
        token=token,
        base_url=args.base_url,
        max_entries=args.max_entries,
        max_file_chars=args.max_file_chars,
        max_pr_files=args.max_pr_files,
    )
    try:
        handled = serve_stdio(server, log_level=args.log_level)
    finally:
        server.close()
    logger.debug("GitHub MCP 服务端退出，共处理 %d 条请求", handled)
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    raise SystemExit(main())
