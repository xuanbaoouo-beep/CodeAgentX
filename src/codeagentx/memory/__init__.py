"""记忆层：工作记忆、情景记忆与任务笔记。

- :class:`WorkingMemory`  —— 当前任务的对话窗口（含 Token 预算与安全裁剪）
- :class:`EpisodicMemory` —— 跨任务的经验库（BM25 召回历史结论）
- :class:`NoteTool`       —— 把笔记能力暴露给 Agent（含 :class:`NoteBook` 存储）
"""

from codeagentx.memory.episodic import (
    DEFAULT_LIMIT,
    DEFAULT_RECALL_LIMIT,
    OUTCOMES,
    Episode,
    EpisodicMemory,
)
from codeagentx.memory.note_tool import (
    ACTIONS,
    Note,
    NoteBook,
    NoteTool,
)
from codeagentx.memory.persistence import append_jsonl, read_jsonl
from codeagentx.memory.working import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_RESERVE_TOKENS,
    WorkingMemory,
)

__all__ = [
    "ACTIONS",
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_RECALL_LIMIT",
    "DEFAULT_RESERVE_TOKENS",
    "OUTCOMES",
    "Episode",
    "EpisodicMemory",
    "Note",
    "NoteBook",
    "NoteTool",
    "WorkingMemory",
    "append_jsonl",
    "read_jsonl",
]
