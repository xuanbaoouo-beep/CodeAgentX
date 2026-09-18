"""内置的文件系统 MCP 服务端（stdio）。

为什么自备一个：MCP 生态里"文件系统 server"是最常见的入口，但第三方实现
依赖 Node/npx 或额外安装。本项目要求"零额外依赖也能演示协议打通"，
于是用 Python 写一个只读的等价实现——**只暴露读操作**，
并且所有路径都过 :class:`~codeagentx.tools.sandbox.PathGuard`，
把"模型给的路径不可信"这条约束落到协议层。

启动方式（由 :class:`~codeagentx.protocols.mcp_client.MCPClient` 负责）::

    python -m codeagentx.protocols.mcp_filesystem --root data/sample_repo

协议分发与 stdio 卫生（日志强制走 stderr、一行一条 JSON）由
:class:`~codeagentx.protocols.mcp_server.MCPServerBase` 统一负责，本模块只实现工具。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from codeagentx.core.exceptions import MCPError
from codeagentx.core.logger import get_logger
from codeagentx.protocols.mcp_protocol import (
    MCPContent,
    MCPErrorCode,
    MCPResourceSpec,
    MCPToolSpec,
)
from codeagentx.protocols.mcp_server import MCPServerBase, serve_stdio
from codeagentx.tools.sandbox import SandboxPolicy

logger = get_logger("protocols.mcp_filesystem")

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_FILE_CHARS",
    "DEFAULT_MAX_MATCHES",
    "FilesystemMCPServer",
    "filesystem_server_command",
    "filesystem_tools",
    "main",
]

SERVER_NAME = "codeagentx-filesystem"
SERVER_VERSION = "0.1.0"
SERVER_CAPABILITIES: dict[str, Any] = {
    "tools": {"listChanged": False},
    "resources": {"subscribe": False, "listChanged": False},
}
SERVER_INSTRUCTIONS = (
    "只读文件系统服务：可读取、列出、检索允许根目录内的文本文件。"
    "所有路径参数都会被沙箱校验，越界会被拒绝。"
)

TOOL_READ_TEXT_FILE = "read_text_file"
TOOL_LIST_DIRECTORY = "list_directory"
TOOL_DIRECTORY_TREE = "directory_tree"
TOOL_SEARCH_FILES = "search_files"
TOOL_GET_FILE_INFO = "get_file_info"

DEFAULT_MAX_FILE_CHARS = 60_000
DEFAULT_MAX_ENTRIES = 200
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_MATCHES = 50
#: 遍历时跳过的目录（与 RAG 分块保持同款口径，避免把依赖目录扫进来）
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        "dist",
        "build",
    }
)


# ------------------------------------------------------------------ 工具定义
def filesystem_tools() -> list[MCPToolSpec]:
    """服务端声明的工具列表（``tools/list`` 的返回内容）。"""
    return [
        MCPToolSpec(
            name=TOOL_READ_TEXT_FILE,
            description="读取一个文本文件的完整内容（超出上限会截断并标注）",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对根目录或绝对路径）"},
                    "max_chars": {
                        "type": "integer",
                        "description": f"最多返回的字符数，默认 {DEFAULT_MAX_FILE_CHARS}",
                    },
                },
                "required": ["path"],
            },
        ),
        MCPToolSpec(
            name=TOOL_LIST_DIRECTORY,
            description="列出目录下的条目（目录以 / 结尾，文件附字节数）",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，默认根目录"},
                    "max_entries": {
                        "type": "integer",
                        "description": f"最多返回条数，默认 {DEFAULT_MAX_ENTRIES}",
                    },
                },
            },
        ),
        MCPToolSpec(
            name=TOOL_DIRECTORY_TREE,
            description="输出目录树（深度可配，自动跳过 .git/__pycache__ 等目录）",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "起始目录，默认根目录"},
                    "depth": {"type": "integer", "description": f"最大深度，默认 {DEFAULT_MAX_DEPTH}"},
                },
            },
        ),
        MCPToolSpec(
            name=TOOL_SEARCH_FILES,
            description="按文件名通配符检索文件；给了 contains 则在文件内容里找子串",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "文件名通配符，如 *.py"},
                    "path": {"type": "string", "description": "检索起始目录，默认根目录"},
                    "contains": {"type": "string", "description": "可选：内容中必须包含的子串"},
                    "max_matches": {
                        "type": "integer",
                        "description": f"最多命中条数，默认 {DEFAULT_MAX_MATCHES}",
                    },
                },
                "required": ["pattern"],
            },
        ),
        MCPToolSpec(
            name=TOOL_GET_FILE_INFO,
            description="查看文件或目录的元信息（大小、行数、后缀等）",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件或目录路径"}},
                "required": ["path"],
            },
        ),
    ]


def filesystem_server_command(
    roots: Sequence[str | Path], *, python: str | None = None
) -> tuple[str, list[str]]:
    """构造启动内置文件系统服务端的命令（供 :class:`MCPClient` 直接使用）。"""
    if not roots:
        raise ValueError("至少需要一个根目录")
    executable = python or sys.executable
    args = ["-m", "codeagentx.protocols.mcp_filesystem"]
    for root in roots:
        args.extend(["--root", str(root)])
    return executable, args


# ------------------------------------------------------------------ 服务端
class FilesystemMCPServer(MCPServerBase):
    """只读文件系统 MCP 服务端（同步、按行处理）。

    协议分发交给 :class:`MCPServerBase`，本类只声明工具与实现工具。
    """

    server_name = SERVER_NAME
    server_version = SERVER_VERSION
    capabilities = SERVER_CAPABILITIES
    instructions = SERVER_INSTRUCTIONS

    def __init__(
        self,
        *,
        roots: Iterable[str | Path],
        max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_matches: int = DEFAULT_MAX_MATCHES,
    ) -> None:
        self.policy = SandboxPolicy.for_roots(roots)
        self.guard = self.policy.path_guard()
        self.max_file_chars = max_file_chars
        self.max_entries = max_entries
        self.max_depth = max_depth
        self.max_matches = max_matches
        self._handlers = {
            TOOL_READ_TEXT_FILE: self._read_text_file,
            TOOL_LIST_DIRECTORY: self._list_directory,
            TOOL_DIRECTORY_TREE: self._directory_tree,
            TOOL_SEARCH_FILES: self._search_files,
            TOOL_GET_FILE_INFO: self._get_file_info,
        }
        logger.debug(
            "文件系统 MCP 服务端就绪，根目录：%s", [str(root) for root in self.policy.allowed_roots]
        )

    # -------------------------------------------------------- 工具声明与调用
    def list_tools(self) -> list[MCPToolSpec]:
        return filesystem_tools()

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        # 工具名合法性已由 MCPServerBase._tools_call 用 list_tools() 校验过
        return self._handlers[name](arguments)

    # -------------------------------------------------------- 工具实现
    def _read_text_file(self, arguments: Mapping[str, Any]) -> str:
        path = self._require(arguments, "path")
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise MCPError(f"不是文件：{self._display(resolved)}", code=int(MCPErrorCode.INVALID_PARAMS))
        limit = self._clamp(arguments.get("max_chars"), self.max_file_chars, 1, self.max_file_chars)
        text = resolved.read_text(encoding="utf-8", errors="replace")
        if len(text) <= limit:
            return text
        clipped = text[:limit].rsplit("\n", 1)[0]
        return f"{clipped}\n...（已截断，仅返回前 {limit} 字符 / 共 {len(text)} 字符）"

    def _list_directory(self, arguments: Mapping[str, Any]) -> str:
        directory = self._resolve(arguments.get("path") or ".")
        if not directory.is_dir():
            raise MCPError(
                f"不是目录：{self._display(directory)}", code=int(MCPErrorCode.INVALID_PARAMS)
            )
        limit = self._clamp(arguments.get("max_entries"), self.max_entries, 1, self.max_entries)
        entries = sorted(
            directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
        )
        lines: list[str] = []
        for entry in entries[:limit]:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}  ({entry.stat().st_size} B)")
        if len(entries) > limit:
            lines.append(f"...（共 {len(entries)} 条，仅显示前 {limit} 条）")
        return "\n".join(lines) if lines else "（空目录）"

    def _directory_tree(self, arguments: Mapping[str, Any]) -> str:
        root = self._resolve(arguments.get("path") or ".")
        if not root.is_dir():
            raise MCPError(f"不是目录：{self._display(root)}", code=int(MCPErrorCode.INVALID_PARAMS))
        depth = self._clamp(arguments.get("depth"), self.max_depth, 1, 10)
        lines = [f"{self._display(root)}/"]
        counter = [0]
        self._append_children(root, "  ", depth, lines, counter)
        if counter[0] >= self.max_entries:
            lines.append(f"...（已达 {self.max_entries} 条上限）")
        return "\n".join(lines)

    def _append_children(
        self,
        directory: Path,
        prefix: str,
        depth: int,
        lines: list[str],
        counter: list[int],
    ) -> None:
        if depth <= 0 or counter[0] >= self.max_entries:
            return
        try:
            children = sorted(
                directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
            )
        except OSError as exc:
            lines.append(f"{prefix}<无法读取：{exc}>")
            return
        for child in children:
            if counter[0] >= self.max_entries:
                return
            if child.name in SKIP_DIRS or child.name.startswith("."):
                continue
            counter[0] += 1
            if child.is_dir():
                lines.append(f"{prefix}{child.name}/")
                self._append_children(child, prefix + "  ", depth - 1, lines, counter)
            else:
                lines.append(f"{prefix}{child.name}  ({child.stat().st_size} B)")

    def _search_files(self, arguments: Mapping[str, Any]) -> str:
        pattern = str(arguments.get("pattern") or "").strip()
        if not pattern:
            raise MCPError("search_files 需要 pattern", code=int(MCPErrorCode.INVALID_PARAMS))
        root = self._resolve(arguments.get("path") or ".")
        contains = arguments.get("contains")
        needle = str(contains) if contains else ""
        limit = self._clamp(arguments.get("max_matches"), self.max_matches, 1, self.max_matches)

        matches: list[str] = []
        for candidate in sorted(root.rglob("*")):
            if len(matches) >= limit:
                break
            if not candidate.is_file():
                continue
            relative_parts = self._relative_parts(candidate, root)
            if any(part in SKIP_DIRS or part.startswith(".") for part in relative_parts[:-1]):
                continue
            if not fnmatch.fnmatch(candidate.name, pattern):
                continue
            if not needle:
                matches.append(self._display(candidate))
                continue
            for number, line in enumerate(
                candidate.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                if needle in line:
                    matches.append(f"{self._display(candidate)}:{number}  {line.strip()[:200]}")
                    if len(matches) >= limit:
                        break
        if not matches:
            return f"未找到匹配 {pattern!r} 的文件" + (f"（要求包含 {needle!r}）" if needle else "")
        return "\n".join(matches)

    def _get_file_info(self, arguments: Mapping[str, Any]) -> str:
        target = self._resolve(self._require(arguments, "path"))
        info: dict[str, Any] = {
            "path": self._display(target),
            "name": target.name,
            "suffix": target.suffix,
            "is_dir": target.is_dir(),
            "size_bytes": target.stat().st_size,
        }
        if target.is_file():
            info["lines"] = len(
                target.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        return json.dumps(info, ensure_ascii=False, indent=2)

    # -------------------------------------------------------- 资源
    def list_resources(self) -> list[MCPResourceSpec]:
        """把允许的根目录声明为资源（客户端可据此"看到"能读哪些地方）。"""
        return [
            MCPResourceSpec(
                uri=root.as_uri(),
                name=root.name or str(root),
                description=f"只读根目录：{root}",
                mime_type="inode/directory",
            )
            for root in self.policy.allowed_roots
        ]

    def read_resource(self, uri: str) -> list[MCPContent]:
        """读取资源。``list_resources`` 声明的是**根目录**，所以"声明了就能读"：
        目录按列目录返回，文件才返回正文。
        """
        target = self._resolve(self._uri_to_path(uri))
        if target.is_dir():
            text = self._list_directory({"path": str(target)})
        else:
            text = self._read_text_file({"path": str(target)})
        return [MCPContent(type="text", text=text, uri=uri, mime_type="text/plain")]

    # -------------------------------------------------------- 内部工具
    def _resolve(self, raw: Any) -> Path:
        """路径校验：所有入参路径都必须落在允许的根目录内。"""
        value = str(raw or ".").strip() or "."
        return self.guard.resolve(value, base=self.policy.default_workdir, must_exist=True)

    def _display(self, path: Path) -> str:
        for root in self.policy.allowed_roots:
            if path == root:
                return root.name or str(root)  # 根目录自身显示名字，而不是 "."
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                continue
        return path.as_posix()

    def _relative_parts(self, path: Path, root: Path) -> tuple[str, ...]:
        try:
            return path.relative_to(root).parts
        except ValueError:
            return path.parts

    def _require(self, arguments: Mapping[str, Any], key: str) -> str:
        value = str(arguments.get(key) or "").strip()
        if not value:
            raise MCPError(f"缺少必填参数：{key}", code=int(MCPErrorCode.INVALID_PARAMS))
        return value

    def _clamp(self, value: Any, default: int, minimum: int, maximum: int) -> int:
        if value is None:
            return default
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise MCPError(
                f"参数必须是整数：{value!r}", code=int(MCPErrorCode.INVALID_PARAMS)
            ) from exc
        return max(minimum, min(number, maximum))

    def _uri_to_path(self, uri: str) -> str:
        """``file://`` URI → 平台路径（Windows 盘符要去掉多余的前导斜杠）。"""
        parsed = urlparse(uri)
        if parsed.scheme and parsed.scheme != "file":
            raise MCPError(
                f"只支持 file:// 资源：{uri}", code=int(MCPErrorCode.INVALID_PARAMS)
            )
        raw = unquote(parsed.path if parsed.scheme else uri)
        if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":  # /E:/x -> E:/x
            raw = raw[1:]
        return raw


# ------------------------------------------------------------------ 入口
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codeagentx-filesystem-mcp",
        description="CodeAgentX 内置文件系统 MCP 服务端（stdio，只读）",
    )
    parser.add_argument("--root", action="append", required=True, help="允许访问的根目录（可多次指定）")
    parser.add_argument("--max-file-chars", type=int, default=DEFAULT_MAX_FILE_CHARS)
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-matches", type=int, default=DEFAULT_MAX_MATCHES)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    server = FilesystemMCPServer(
        roots=args.root,
        max_file_chars=args.max_file_chars,
        max_entries=args.max_entries,
        max_depth=args.max_depth,
        max_matches=args.max_matches,
    )
    handled = serve_stdio(server, log_level=args.log_level)
    logger.debug("文件系统 MCP 服务端退出，共处理 %d 条请求", handled)
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    raise SystemExit(main())
