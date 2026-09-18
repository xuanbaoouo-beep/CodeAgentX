"""NoteTool：让 Agent 在审查过程中随手记笔记。

为什么需要它
------------
多 Agent 流水线里，Reviewer 发现的疑点、Security 的结论需要跨步骤传递。
把这些中间结论写进笔记，比塞进对话历史更省 Token，
也便于最后由 Reporter 一次性汇总成报告。

动作
----
``add``     记一条笔记（``content`` 必填，可带 ``tags``）
``list``    按时间倒序列出笔记（可用 ``tags`` 过滤）
``search``  按关键词 + 标签检索笔记
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codeagentx.core.exceptions import ToolValidationError
from codeagentx.core.logger import get_logger, log_event
from codeagentx.memory.persistence import append_jsonl, read_jsonl
from codeagentx.rag.bm25 import BM25Index
from codeagentx.rag.vector_store import StoredDocument
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult

logger = get_logger("memory.note_tool")

#: 支持的动作
ACTIONS: tuple[str, ...] = ("add", "list", "search")
#: 默认最多保留多少条笔记
DEFAULT_MAX_NOTES = 200
#: list 默认返回条数
DEFAULT_LIST_LIMIT = 20
#: search 默认返回条数
DEFAULT_SEARCH_LIMIT = 5
#: 单次返回条数上限
DEFAULT_MAX_LIMIT = 50
#: 单条笔记在文本输出中的最大字符数
DEFAULT_MAX_CONTENT_CHARS = 1200


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n...（已截断，完整长度 {len(text)} 字符）"


@dataclass
class Note:
    """一条审查笔记。"""

    content: str
    note_id: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """用于检索的文本：正文 + 标签 + 来源。"""
        parts = [self.content, *self.tags, self.source]
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "content": self.content,
            "tags": list(self.tags),
            "source": self.source,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Note:
        return cls(
            content=str(data.get("content", "")),
            note_id=str(data.get("note_id", "")),
            tags=[str(item) for item in (data.get("tags") or [])],
            source=str(data.get("source", "")),
            created_at=str(data.get("created_at", "")),
            metadata=dict(data.get("metadata") or {}),
        )


class NoteBook:
    """笔记存储（内存 + 可选 JSONL 落盘）。"""

    def __init__(
        self,
        *,
        limit: int = DEFAULT_MAX_NOTES,
        path: str | Path | None = None,
        load_existing: bool = True,
    ) -> None:
        if limit <= 0:
            raise ValueError(f"limit 必须为正整数，收到 {limit}")
        self.limit = limit
        self._path = Path(path) if path else None
        self._notes: list[Note] = []
        self._index: BM25Index | None = None
        self._dropped = 0
        if self._path is not None and load_existing:
            self._load(self._path)

    # ------------------------------------------------------------ 写入
    def add(
        self,
        content: str,
        *,
        tags: Sequence[str] | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        note_id: str | None = None,
    ) -> Note:
        """记录一条笔记并返回它；``content`` 必填。"""
        if not content or not str(content).strip():
            raise ValueError("笔记内容不能为空")
        note = Note(
            content=str(content).strip(),
            note_id=note_id or f"note-{uuid.uuid4().hex[:8]}",
            tags=[str(tag) for tag in (tags or [])],
            source=str(source or ""),
            created_at=_now(),
            metadata=dict(metadata or {}),
        )
        self._notes.append(note)
        self._invalidate()
        self._enforce_limit()
        if self._path is not None:
            append_jsonl(self._path, note.to_dict())
        log_event(logger, "memory.note_added", note=note.note_id, tags=len(note.tags))
        return note

    def _enforce_limit(self) -> None:
        if len(self._notes) <= self.limit:
            return
        excess = len(self._notes) - self.limit
        del self._notes[:excess]
        self._dropped += excess
        self._invalidate()

    # ------------------------------------------------------------ 查询
    def list(self, *, limit: int | None = None, tags: Sequence[str] | None = None) -> list[Note]:
        """按时间倒序（新 → 旧）列出笔记。"""
        wanted = {str(tag) for tag in (tags or [])}
        ordered = [
            note for note in reversed(self._notes) if not wanted or wanted.intersection(note.tags)
        ]
        if limit is not None:
            return ordered[:limit]
        return ordered

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        tags: Sequence[str] | None = None,
    ) -> list[Note]:
        """按关键词检索笔记；``query`` 不能为空。"""
        if not query or not str(query).strip():
            raise ValueError("检索笔记时 query 不能为空")
        if limit <= 0:
            raise ValueError(f"limit 必须为正整数，收到 {limit}")
        wanted = {str(tag) for tag in (tags or [])}
        candidates = [
            note for note in self._notes if not wanted or wanted.intersection(note.tags)
        ]
        if not candidates:
            return []
        by_id = {note.note_id: note for note in candidates}
        hits = self._ensure_index().search(
            str(query), limit=max(limit * 4, len(candidates))
        )
        ranked = [hit.vector_id for hit in hits if hit.vector_id in by_id]
        return [by_id[note_id] for note_id in ranked[:limit]]

    def get(self, note_id: str) -> Note | None:
        for note in self._notes:
            if note.note_id == note_id:
                return note
        return None

    def _ensure_index(self) -> BM25Index:
        if self._index is None or self._index.size != len(self._notes):
            self._index = BM25Index(
                [
                    StoredDocument(vector_id=note.note_id, text=note.text)
                    for note in self._notes
                ]
            )
        return self._index

    def _invalidate(self) -> None:
        self._index = None

    def __len__(self) -> int:
        return len(self._notes)

    def __iter__(self) -> Iterator[Note]:
        return iter(list(self._notes))

    def describe(self) -> dict[str, Any]:
        return {
            "notes": len(self._notes),
            "dropped": self._dropped,
            "limit": self.limit,
            "path": str(self._path) if self._path else None,
        }

    def _load(self, path: Path) -> None:
        for record in read_jsonl(path):
            note = Note.from_dict(record)
            if note.content:
                self._notes.append(note)
        if len(self._notes) > self.limit:
            self._dropped += len(self._notes) - self.limit
            self._notes = self._notes[-self.limit :]
        self._invalidate()
        logger.info("载入历史笔记 %d 条（%s）", len(self._notes), path)


class NoteTool(BaseTool):
    """记录与检索审查笔记的工具。"""

    name = "note"
    description = (
        "记录与检索审查笔记：把发现的问题、可疑点、待确认事项写下来，"
        "供后续步骤与最终报告使用。action=add 记一条，list 按时间倒序列出，"
        "search 按关键词检索。适合在分析过程中留存中间结论，避免遗忘或重复分析。"
    )
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="操作类型：add=记笔记（默认），list=列出，search=检索",
            required=False,
            default="add",
            enum=list(ACTIONS),
        ),
        ToolParameter(
            name="content",
            type="string",
            description="笔记内容，例如“login 接口未校验 password 是否为 None”（action=add 必填）",
            required=False,
        ),
        ToolParameter(
            name="query",
            type="string",
            description="检索关键词，例如“password 校验”（action=search 必填）",
            required=False,
        ),
        ToolParameter(
            name="tags",
            type="array",
            description='标签，用于分类与过滤，例如 ["security", "p0"]',
            required=False,
            items={"type": "string"},
        ),
        ToolParameter(
            name="source",
            type="string",
            description="笔记来源，便于汇总时区分角色，例如 reviewer / security（action=add 可用）",
            required=False,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description=f"返回条数上限，默认 list={DEFAULT_LIST_LIMIT}、search={DEFAULT_SEARCH_LIMIT}",
            required=False,
        ),
    ]

    def __init__(
        self,
        *,
        notebook: NoteBook | None = None,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        default_list_limit: int = DEFAULT_LIST_LIMIT,
        default_search_limit: int = DEFAULT_SEARCH_LIMIT,
        max_limit: int = DEFAULT_MAX_LIMIT,
    ) -> None:
        super().__init__()
        self.notebook = notebook or NoteBook()
        self.max_content_chars = max_content_chars
        self.default_list_limit = default_list_limit
        self.default_search_limit = default_search_limit
        self.max_limit = max_limit

    # ------------------------------------------------------------ 工具入口
    def _run(
        self,
        action: str = "add",
        content: str | None = None,
        query: str | None = None,
        tags: Any = None,
        source: str = "",
        limit: int | None = None,
    ) -> ToolResult:
        if action not in ACTIONS:
            raise ToolValidationError(
                f"未知的 action：{action!r}", detail=f"可选动作：{list(ACTIONS)}"
            )
        normalized_tags = _normalize_tags(tags)
        if action == "add":
            return self._run_add(content, tags=normalized_tags, source=source)
        if action == "list":
            return self._run_list(tags=normalized_tags, limit=limit)
        return self._run_search(query, tags=normalized_tags, limit=limit)

    def _run_add(
        self,
        content: str | None,
        *,
        tags: list[str] | None,
        source: str,
    ) -> ToolResult:
        if not content or not str(content).strip():
            raise ToolValidationError("action=add 时必须提供非空的 content")
        note = self.notebook.add(str(content), tags=tags, source=source)
        suffix = f"（标签：{', '.join(note.tags)}）" if note.tags else ""
        return ToolResult.ok(
            f"已记录笔记 {note.note_id}{suffix}\n当前共 {len(self.notebook)} 条笔记",
            note=note.to_dict(),
            count=len(self.notebook),
        )

    def _run_list(self, *, tags: list[str] | None, limit: int | None) -> ToolResult:
        notes = self.notebook.list(limit=self._clamp_limit(limit, default=self.default_list_limit), tags=tags)
        if not notes:
            return ToolResult.ok("暂无笔记。", count=0, notes=[])
        header = f"共 {len(self.notebook)} 条笔记，最近的 {len(notes)} 条（新 → 旧）："
        return ToolResult.ok(
            self._render(header, notes),
            count=len(notes),
            total=len(self.notebook),
            notes=[note.to_dict() for note in notes],
        )

    def _run_search(
        self,
        query: str | None,
        *,
        tags: list[str] | None,
        limit: int | None,
    ) -> ToolResult:
        if not query or not str(query).strip():
            raise ToolValidationError("action=search 时必须提供非空的 query")
        queried = str(query).strip()
        notes = self.notebook.search(
            queried,
            limit=self._clamp_limit(limit, default=self.default_search_limit),
            tags=tags,
        )
        if not notes:
            return ToolResult.ok(
                f"未找到与「{queried}」相关的笔记。", query=queried, count=0, notes=[]
            )
        return ToolResult.ok(
            self._render(f"检索「{queried}」命中 {len(notes)} 条笔记：", notes),
            query=queried,
            count=len(notes),
            notes=[note.to_dict() for note in notes],
        )

    # ------------------------------------------------------------ 内部
    def _clamp_limit(self, limit: int | None, *, default: int) -> int:
        if limit is None:
            return default
        try:
            value = int(limit)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"limit 必须是整数，收到 {limit!r}") from exc
        if value <= 0:
            raise ToolValidationError(f"limit 必须为正整数，收到 {value}")
        return min(value, self.max_limit)

    def _render(self, header: str, notes: Sequence[Note]) -> str:
        lines = [header]
        for index, note in enumerate(notes, start=1):
            tags = f"  [{', '.join(note.tags)}]" if note.tags else ""
            origin = f"  @{note.source}" if note.source else ""
            lines.append("")
            lines.append(f"[{index}] {note.note_id}{tags}{origin}  {note.created_at}")
            lines.append(_truncate(note.content, self.max_content_chars))
        return "\n".join(lines)


def _normalize_tags(tags: Any) -> list[str] | None:
    """把 tags 归一成字符串列表；容错地接受 ``"a,b"`` 这种逗号串。"""
    if tags is None or tags == "":
        return None
    if isinstance(tags, str):
        items: list[Any] = tags.split(",")
    elif isinstance(tags, (list, tuple, set)):
        items = list(tags)
    else:
        raise ToolValidationError(f"tags 必须是字符串数组，收到 {type(tags).__name__}")
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return cleaned or None
