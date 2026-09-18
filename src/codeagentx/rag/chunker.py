"""AST 级代码分块。

为什么不用固定窗口切分：固定窗口会把一个函数劈成两半，检索到的上下文语义不完整；
AST 分块保留**函数/类边界**，每个块是一个"可理解的语义单元"。

分块粒度
--------
- ``module_header``：模块 docstring + 开头的连续 import（作为同文件其余块的公共上下文）
- ``class``：类头 + 类 docstring（成员方法单独成块，避免内容重复）
- ``method`` / ``function``：完整函数体（**含装饰器**）
- ``module``：模块级散装语句（常量赋值、``if __name__ == "__main__"`` 等）
- ``block``：非 Python 文件的滑动窗口（兜底）

超长块按 ``max_chunk_lines`` 与 ``max_chunk_chars`` 再切成多个 ``part``，
切分点尽量落在空行处，保证不把一个逻辑段落从中间截断。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeagentx.core.logger import get_logger

logger = get_logger("rag.chunker")

#: 单个块的最大行数
DEFAULT_MAX_CHUNK_LINES = 80
#: 单个块的最大字符数
DEFAULT_MAX_CHUNK_CHARS = 2000
#: 非 Python 文本的滑动窗口行数与重叠行数
DEFAULT_WINDOW_LINES = 40
DEFAULT_OVERLAP_LINES = 5
#: 单文件大小上限（超过则跳过，避免把大文件读进内存）
DEFAULT_MAX_FILE_BYTES = 200_000
#: 单次索引的文件数上限
DEFAULT_MAX_FILES = 500

PYTHON_SUFFIXES: frozenset[str] = frozenset({".py", ".pyi"})
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".rst"}
)
#: 默认参与索引的扩展名
DEFAULT_INCLUDE_SUFFIXES: frozenset[str] = PYTHON_SUFFIXES | TEXT_SUFFIXES

#: 默认跳过的目录名
DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "dist",
        "build",
        ".idea",
        ".vscode",
        "logs",
        "data",
    }
)


@dataclass(frozen=True)
class CodeChunk:
    """一个可独立检索的代码片段。"""

    chunk_id: str
    path: str
    content: str
    start_line: int
    end_line: int
    kind: str
    name: str = ""
    parent: str = ""
    part: int = 0
    language: str = "python"

    @property
    def line_span(self) -> str:
        return f"{self.start_line}-{self.end_line}"

    def to_text(self) -> str:
        """转成喂给 embedding / LLM 的文本，首行是位置与符号信息。"""
        header = format_chunk_header(
            self.path, self.start_line, self.end_line, self.kind, self.name, self.parent
        )
        return f"{header}\n{self.content}"

    def to_metadata(self) -> dict[str, Any]:
        """写入向量库的元数据（不含正文，避免重复存储）。"""
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "kind": self.kind,
            "name": self.name,
            "parent": self.parent,
            "part": self.part,
            "language": self.language,
        }


@dataclass(frozen=True)
class _Span:
    """源码中的一段行区间（1-based，闭区间）。"""

    kind: str
    name: str
    parent: str
    start: int
    end: int


def format_chunk_header(
    path: str,
    start_line: int,
    end_line: int,
    kind: str,
    name: str = "",
    parent: str = "",
) -> str:
    """块首行的统一格式（embedding 输入、BM25 语料、检索结果都用它）。

    例：``# src/a.py:12-20 function load_config``
    """
    header = f"# {path}:{start_line}-{end_line}"
    if name:
        header += f" {kind} {name}"
        if parent:
            header += f" (in class {parent})"
    else:
        header += f" {kind}"
    return header


# ------------------------------------------------------------------ 公开接口
def chunk_python_source(
    source: str,
    path: str,
    *,
    max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[CodeChunk]:
    """按 AST 结构切分 Python 源码；语法错误时回退到文本窗口切分。"""
    try:
        module = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        logger.debug("解析 %s 失败（%s），回退文本窗口切分", path, exc)
        return chunk_text_source(
            source,
            path,
            max_chunk_chars=max_chunk_chars,
        )

    lines = source.splitlines()
    chunks: list[CodeChunk] = []
    for span in _iter_spans(module):
        for part, (start, end) in enumerate(
            _split_span(lines, span.start, span.end, max_chunk_lines, max_chunk_chars)
        ):
            content = "\n".join(lines[start - 1 : end])
            if not content.strip():
                continue
            chunks.append(
                CodeChunk(
                    chunk_id=_make_chunk_id(path, span.kind, span.name, start, end, part),
                    path=path,
                    content=content,
                    start_line=start,
                    end_line=end,
                    kind=span.kind,
                    name=span.name,
                    parent=span.parent,
                    part=part,
                    language="python",
                )
            )
    return chunks


def chunk_text_source(
    source: str,
    path: str,
    *,
    window_lines: int = DEFAULT_WINDOW_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[CodeChunk]:
    """非 Python 文本的滑动窗口切分（带重叠，避免上下文在窗口边界断裂）。"""
    if window_lines <= 0:
        raise ValueError("window_lines 必须为正整数")
    if overlap_lines < 0 or overlap_lines >= window_lines:
        raise ValueError("overlap_lines 必须满足 0 <= overlap < window_lines")

    lines = source.splitlines()
    language = _language_of(path)
    step = window_lines - overlap_lines
    chunks: list[CodeChunk] = []
    part = 0
    start = 0
    while start < len(lines):
        end = min(start + window_lines, len(lines))
        content = "\n".join(lines[start:end])
        if content.strip():
            for piece_start, piece_end in _split_by_chars(lines, start + 1, end, max_chunk_chars):
                piece = "\n".join(lines[piece_start - 1 : piece_end])
                if not piece.strip():
                    continue
                chunks.append(
                    CodeChunk(
                        chunk_id=_make_chunk_id(
                            path, "block", "", piece_start, piece_end, part
                        ),
                        path=path,
                        content=piece,
                        start_line=piece_start,
                        end_line=piece_end,
                        kind="block",
                        part=part,
                        language=language,
                    )
                )
                part += 1
        if end >= len(lines):
            break
        start += step
    return chunks


def chunk_file(
    path: str | Path,
    *,
    root: str | Path | None = None,
    max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[CodeChunk]:
    """读取并切分单个文件；``root`` 用于把路径转成相对路径。"""
    file_path = Path(path)
    relative = _relative_posix(file_path, Path(root) if root else None)
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("读取文件失败：%s（%s）", file_path, exc)
        return []

    if file_path.suffix.lower() in PYTHON_SUFFIXES:
        return chunk_python_source(
            source, relative, max_chunk_lines=max_chunk_lines, max_chunk_chars=max_chunk_chars
        )
    return chunk_text_source(source, relative, max_chunk_chars=max_chunk_chars)


def chunk_repository(
    root: str | Path,
    *,
    include_suffixes: frozenset[str] | None = None,
    exclude_dirs: frozenset[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_chunk_lines: int = DEFAULT_MAX_CHUNK_LINES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[CodeChunk]:
    """遍历目录并切分所有受支持的文件，结果按路径排序（保证可复现）。"""
    root_path = Path(root)
    if not root_path.is_dir():
        raise ValueError(f"仓库根目录不存在或不是目录：{root_path}")

    suffixes = include_suffixes if include_suffixes is not None else DEFAULT_INCLUDE_SUFFIXES
    excluded = exclude_dirs if exclude_dirs is not None else DEFAULT_EXCLUDE_DIRS

    files = sorted(
        (
            item
            for item in root_path.rglob("*")
            if item.is_file()
            and item.suffix.lower() in suffixes
            and not _is_excluded(item, root_path, excluded)
        ),
        key=lambda item: item.as_posix(),
    )

    chunks: list[CodeChunk] = []
    skipped_large = 0
    for file_path in files[:max_files]:
        try:
            if file_path.stat().st_size > max_file_bytes:
                skipped_large += 1
                logger.debug("跳过过大的文件：%s", file_path)
                continue
        except OSError:
            continue
        chunks.extend(
            chunk_file(
                file_path,
                root=root_path,
                max_chunk_lines=max_chunk_lines,
                max_chunk_chars=max_chunk_chars,
            )
        )
    if skipped_large:
        logger.info("索引时跳过了 %d 个超过 %d 字节的文件", skipped_large, max_file_bytes)
    return chunks


# ------------------------------------------------------------------ AST 遍历
def _iter_spans(module: ast.Module) -> list[_Span]:
    body = list(module.body)
    spans: list[_Span] = []

    # 1. 模块 docstring + 开头连续的 import
    index = 0
    header: list[ast.stmt] = []
    if body and _is_docstring(body[0]):
        header.append(body[0])
        index = 1
    while index < len(body) and isinstance(body[index], (ast.Import, ast.ImportFrom)):
        header.append(body[index])
        index += 1
    if header:
        spans.append(
            _Span("module_header", "", "", _start_line(header[0]), _end_line(header[-1]))
        )

    # 2. 顶层定义与散装语句
    pending: list[ast.stmt] = []
    for node in body[index:]:
        if isinstance(node, ast.ClassDef):
            _flush_module_statements(pending, spans)
            spans.extend(_class_spans(node, parent=""))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _flush_module_statements(pending, spans)
            spans.append(_Span("function", node.name, "", _start_line(node), _end_line(node)))
        else:
            pending.append(node)
    _flush_module_statements(pending, spans)
    return spans


def _class_spans(node: ast.ClassDef, *, parent: str) -> list[_Span]:
    """类头单独成块（到第一个成员之前），成员方法各自成块。"""
    class_start = _start_line(node)
    members = [
        child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    header_end = _start_line(members[0]) - 1 if members else _end_line(node)
    # 同行定义（如 class A: pass）时保证区间不为空
    header_end = max(header_end, class_start)

    spans = [_Span("class", node.name, parent, class_start, header_end)]
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append(
                _Span("method", child.name, node.name, _start_line(child), _end_line(child))
            )
        elif isinstance(child, ast.ClassDef):
            spans.extend(_class_spans(child, parent=node.name))
    return spans


def _flush_module_statements(pending: list[ast.stmt], spans: list[_Span]) -> None:
    if not pending:
        return
    spans.append(_Span("module", "", "", pending[0].lineno, _end_line(pending[-1])))
    pending.clear()


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _start_line(node: ast.stmt) -> int:
    """节点起始行；带装饰器时取最靠上的装饰器行。"""
    candidates = [node.lineno]
    candidates.extend(
        decorator.lineno for decorator in getattr(node, "decorator_list", [])
    )
    return min(candidates)


def _end_line(node: ast.stmt) -> int:
    return node.end_lineno or node.lineno


# ------------------------------------------------------------------ 区间切分
def _split_span(
    lines: list[str], start: int, end: int, max_lines: int, max_chars: int
) -> list[tuple[int, int]]:
    """把 ``[start, end]`` 切成若干不超过行数/字符数上限的片段。"""
    if end < start:
        return []
    parts: list[tuple[int, int]] = []
    cursor = start
    while cursor <= end:
        limit = min(cursor + max_lines - 1, end)
        if limit < end:
            # 尽量在空行处断开，避免把逻辑段落从中间截断
            blank = _find_blank_line(lines, cursor + max(1, max_lines // 2), limit)
            if blank is not None:
                limit = blank
        parts.extend(_split_by_chars(lines, cursor, limit, max_chars))
        cursor = limit + 1
    return parts


def _split_by_chars(
    lines: list[str], start: int, end: int, max_chars: int
) -> list[tuple[int, int]]:
    """在行边界上按字符数上限再切一层（单行超限时该行独占一个片段）。"""
    if max_chars <= 0:
        return [(start, end)]
    parts: list[tuple[int, int]] = []
    part_start = start
    length = 0
    for line_no in range(start, end + 1):
        line_length = len(lines[line_no - 1]) + 1
        if length and length + line_length > max_chars:
            parts.append((part_start, line_no - 1))
            part_start = line_no
            length = 0
        length += line_length
    parts.append((part_start, end))
    return parts


def _find_blank_line(lines: list[str], low: int, high: int) -> int | None:
    """在 ``[low, high]`` 中从后往前找空行，返回可作为片段末行的行号。"""
    for line_no in range(high, low - 1, -1):
        if line_no - 1 < len(lines) and not lines[line_no - 1].strip():
            return line_no
    return None


# ------------------------------------------------------------------ 小工具
def _make_chunk_id(path: str, kind: str, name: str, start: int, end: int, part: int) -> str:
    symbol = f"{kind}:{name}" if name else kind
    return f"{path}::{symbol}::{start}-{end}#{part}"


def _relative_posix(file_path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return file_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return file_path.as_posix()


def _language_of(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in PYTHON_SUFFIXES:
        return "python"
    if suffix in {".md", ".rst"}:
        return "markdown"
    return "text"


def _is_excluded(file_path: Path, root: Path, excluded: frozenset[str]) -> bool:
    try:
        relative_parts = file_path.relative_to(root).parts[:-1]
    except ValueError:  # pragma: no cover - rglob 结果必然在 root 下
        return True
    return any(part in excluded for part in relative_parts)
