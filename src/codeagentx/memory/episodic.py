"""EpisodicMemory：跨任务的经验库。

与 :class:`~codeagentx.memory.working.WorkingMemory` 的分工
-----------------------------------------------------------
- 工作记忆服务**当前任务**，是短期上下文，任务结束即丢弃；
- 情景记忆服务**未来任务**，记录"做过什么任务、结论是什么、哪些文件出过问题"，
  下次遇到相似任务先查这里，避免重复劳动——这正是多 Agent 审查的复用价值。

召回用 BM25（复用 rag 层实现）而不是向量检索：
经验条目是短文本、关键词密度高，词法检索更快也更准；且中文会走 bigram 切分，
"登录接口缺少校验" 这类中文查询同样能命中。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codeagentx.core.logger import get_logger, log_event
from codeagentx.memory.persistence import append_jsonl, read_jsonl
from codeagentx.rag.bm25 import BM25Index
from codeagentx.rag.vector_store import StoredDocument

logger = get_logger("memory.episodic")

#: 任务结果的合法取值
OUTCOMES: tuple[str, ...] = ("success", "failure", "partial", "unknown")
#: 默认最多保留多少条经验（防止无限增长）
DEFAULT_LIMIT = 200
#: 默认召回条数
DEFAULT_RECALL_LIMIT = 5
#: 带标签过滤时的候选放大倍数（先多召回一些，过滤后再收敛到 limit）
_RECALL_CANDIDATES_FACTOR = 4


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class Episode:
    """一条任务经验。"""

    task: str
    summary: str = ""
    outcome: str = "unknown"
    findings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    episode_id: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """用于检索的文本：任务 + 结论 + 发现 + 涉及文件 + 标签。"""
        parts = [self.task, self.summary, *self.findings, *self.files, *self.tags]
        return "\n".join(part for part in parts if part)

    def to_metadata(self) -> dict[str, Any]:
        """给 BM25 索引用的元数据（只放可精确过滤的字段）。"""
        return {"episode_id": self.episode_id, "outcome": self.outcome}

    def to_text(self) -> str:
        lines = [f"[{self.created_at}] {self.outcome} | {self.task}"]
        if self.summary:
            lines.append(f"结论：{self.summary}")
        if self.findings:
            lines.append("发现：" + "；".join(self.findings))
        if self.files:
            lines.append("涉及文件：" + ", ".join(self.files))
        if self.tags:
            lines.append("标签：" + ", ".join(self.tags))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "created_at": self.created_at,
            "task": self.task,
            "summary": self.summary,
            "outcome": self.outcome,
            "findings": list(self.findings),
            "files": list(self.files),
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        return cls(
            task=str(data.get("task", "")),
            summary=str(data.get("summary", "")),
            outcome=str(data.get("outcome", "unknown")),
            findings=[str(item) for item in (data.get("findings") or [])],
            files=[str(item) for item in (data.get("files") or [])],
            tags=[str(item) for item in (data.get("tags") or [])],
            episode_id=str(data.get("episode_id", "")),
            created_at=str(data.get("created_at", "")),
            metadata=dict(data.get("metadata") or {}),
        )


class EpisodicMemory:
    """任务经验库（内存 + 可选 JSONL 落盘）。"""

    def __init__(
        self,
        *,
        limit: int = DEFAULT_LIMIT,
        path: str | Path | None = None,
        load_existing: bool = True,
    ) -> None:
        if limit <= 0:
            raise ValueError(f"limit 必须为正整数，收到 {limit}")
        self.limit = limit
        self._path = Path(path) if path else None
        self._episodes: list[Episode] = []
        self._index: BM25Index | None = None
        self._dropped = 0
        if self._path is not None and load_existing:
            self._load(self._path)

    # ------------------------------------------------------------ 写入
    def record(
        self,
        task: str,
        *,
        summary: str = "",
        outcome: str = "unknown",
        findings: Sequence[str] | None = None,
        files: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        episode_id: str | None = None,
    ) -> Episode:
        """记录一条经验并返回它；``task`` 必填。"""
        if not task or not str(task).strip():
            raise ValueError("记录情景记忆时 task 不能为空")
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome 必须是 {list(OUTCOMES)} 之一，收到 {outcome!r}")

        episode = Episode(
            task=str(task).strip(),
            summary=str(summary).strip(),
            outcome=outcome,
            findings=[str(item) for item in (findings or [])],
            files=[str(item) for item in (files or [])],
            tags=[str(item) for item in (tags or [])],
            episode_id=episode_id or f"ep-{uuid.uuid4().hex[:12]}",
            created_at=_now(),
            metadata=dict(metadata or {}),
        )
        self._episodes.append(episode)
        self._invalidate()
        self._enforce_limit()
        if self._path is not None:
            append_jsonl(self._path, episode.to_dict())
        log_event(
            logger,
            "memory.episode_recorded",
            episode=episode.episode_id,
            outcome=episode.outcome,
            findings=len(episode.findings),
            tags=len(episode.tags),
        )
        return episode

    def _enforce_limit(self) -> None:
        if len(self._episodes) <= self.limit:
            return
        excess = len(self._episodes) - self.limit
        del self._episodes[:excess]
        self._dropped += excess
        self._invalidate()
        logger.info("情景记忆超过上限 %d，淘汰最早 %d 条", self.limit, excess)

    # ------------------------------------------------------------ 召回
    def recall(
        self,
        query: str = "",
        *,
        limit: int = DEFAULT_RECALL_LIMIT,
        outcome: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[Episode]:
        """召回相关经验。

        Args:
            query: 查询文本；为空时退化为"返回最近的若干条"。
            limit: 返回条数上限。
            outcome: 只召回指定结果（success/failure/partial/unknown）。
            tags: 命中**任意一个**标签即可。
        """
        if limit <= 0:
            raise ValueError(f"limit 必须为正整数，收到 {limit}")
        candidates = self._filtered(outcome=outcome, tags=tags)
        if not query or not str(query).strip():
            return list(reversed(candidates))[:limit]
        if not candidates:
            return []
        return self._ranked(str(query), candidates, limit=limit)

    def _filtered(self, *, outcome: str | None, tags: Sequence[str] | None) -> list[Episode]:
        wanted = {str(tag) for tag in (tags or [])}
        result: list[Episode] = []
        for episode in self._episodes:
            if outcome and episode.outcome != outcome:
                continue
            if wanted and not wanted.intersection(episode.tags):
                continue
            result.append(episode)
        return result

    def _ranked(self, query: str, candidates: Sequence[Episode], *, limit: int) -> list[Episode]:
        index = self._ensure_index()
        wanted_ids = {episode.episode_id for episode in candidates}
        # 标签/结果过滤发生在打分之后，故先放大候选，避免过滤后剩不下几条
        hits = index.search(query, limit=max(limit * _RECALL_CANDIDATES_FACTOR, len(candidates)))
        ranked_ids = [hit.vector_id for hit in hits if hit.vector_id in wanted_ids]
        by_id = {episode.episode_id: episode for episode in candidates}
        return [by_id[episode_id] for episode_id in ranked_ids[:limit]]

    def _ensure_index(self) -> BM25Index:
        if self._index is None or self._index.size != len(self._episodes):
            self._index = BM25Index(
                [
                    StoredDocument(
                        vector_id=episode.episode_id,
                        text=episode.text,
                        metadata=episode.to_metadata(),
                    )
                    for episode in self._episodes
                ]
            )
        return self._index

    def _invalidate(self) -> None:
        self._index = None

    # ------------------------------------------------------------ 查询
    def get(self, episode_id: str) -> Episode | None:
        for episode in self._episodes:
            if episode.episode_id == episode_id:
                return episode
        return None

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[Episode]:
        return iter(list(self._episodes))

    def describe(self) -> dict[str, Any]:
        outcomes: dict[str, int] = {}
        for episode in self._episodes:
            outcomes[episode.outcome] = outcomes.get(episode.outcome, 0) + 1
        return {
            "episodes": len(self._episodes),
            "dropped": self._dropped,
            "limit": self.limit,
            "outcomes": outcomes,
            "path": str(self._path) if self._path else None,
        }

    # ------------------------------------------------------------ 载入
    def _load(self, path: Path) -> None:
        for record in read_jsonl(path):
            episode = Episode.from_dict(record)
            if episode.task:
                self._episodes.append(episode)
        if len(self._episodes) > self.limit:
            self._dropped += len(self._episodes) - self.limit
            self._episodes = self._episodes[-self.limit :]
        self._invalidate()
        logger.info("载入历史经验 %d 条（%s）", len(self._episodes), path)
