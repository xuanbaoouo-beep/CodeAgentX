"""WorkingMemory：当前任务的对话窗口 + Token 预算。

为什么不能简单地"超长就截断最早的几条"
----------------------------------------
``assistant`` 消息里的 ``tool_calls`` 与紧随其后的 ``tool`` 结果必须成对出现，
拆散它们会形成**非法消息序列**（OpenAI 接口直接报 400）。
因此裁剪的最小单位是**消息单元**：
``assistant(tool_calls)`` 连同紧随其后、``tool_call_id`` 匹配的 ``tool`` 结果
算作一个单元，要么整体保留、要么整体丢弃。

固定（pin）机制
---------------
被标记为 pinned 的消息永不丢弃，典型用途是把系统提示与任务描述钉在窗口里。
``tool`` 消息**不允许** pinned——它必须跟随自己的 ``assistant`` 消息，单独留下就是脏数据。

最新的一条消息同样永不裁剪：用户刚贴进来的一大段代码要是被自己挤掉，
就会出现"窗口空了但问题还在"的怪状。若固定内容 + 最新消息本身就超出预算，
本类不强行裁剪，只发一条警告（真正的解法是缩小提示或调大 ``max_tokens``）。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from codeagentx.core.logger import get_logger, log_event
from codeagentx.core.message import (
    Message,
    Role,
    ToolCall,
    estimate_messages_tokens,
    to_openai_messages,
)

logger = get_logger("memory.working")

#: 默认上下文窗口（token），可通过构造参数覆盖
DEFAULT_MAX_TOKENS = 8000
#: 默认留给模型输出的 token 数（不计入历史预算）
DEFAULT_RESERVE_TOKENS = 1000
#: pinned 标记存放位置（随 ``Message.to_dict()`` 一起持久化，恢复后依然有效）
_PINNED_KEY = "_pinned"


class WorkingMemory:
    """带 Token 预算的对话窗口。"""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reserve_tokens: int | None = None,
        auto_trim: bool = True,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正整数，收到 {max_tokens}")
        # 预留量缺省取窗口的 1/5：显式写死默认值会让"小窗口"（如 max_tokens=1000）
        # 一构造就报错，而这类小窗口在单测与本地小模型场景里很常见。
        if reserve_tokens is None:
            reserve_tokens = min(DEFAULT_RESERVE_TOKENS, max_tokens // 5)
        if not 0 <= reserve_tokens < max_tokens:
            raise ValueError(f"reserve_tokens 必须落在 [0, max_tokens) 区间，收到 {reserve_tokens}")
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.auto_trim = auto_trim
        self._messages: list[Message] = []
        self._dropped = 0
        self._overflow_warned = False
        if system_prompt:
            self.set_system_prompt(system_prompt)

    # ------------------------------------------------------------ 只读视图
    @property
    def messages(self) -> list[Message]:
        """当前窗口内的消息副本（外部修改不影响内部状态）。"""
        return list(self._messages)

    @property
    def system_prompt(self) -> str | None:
        """首条 system 消息的正文；没有则返回 ``None``。"""
        if self._messages and self._messages[0].role is Role.SYSTEM:
            return self._messages[0].text
        return None

    @property
    def budget(self) -> int:
        """可用于历史消息的 token 上限（已扣除输出预留）。"""
        return self.max_tokens - self.reserve_tokens

    @property
    def tokens(self) -> int:
        """当前窗口的估算 token 数。"""
        return estimate_messages_tokens(self._messages)

    @property
    def dropped_count(self) -> int:
        """累计被裁掉的（非固定）消息条数。"""
        return self._dropped

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(list(self._messages))

    # ------------------------------------------------------------ 写入
    def set_system_prompt(self, content: str) -> Message:
        """设置/替换系统提示，并把它钉在窗口最前面。"""
        message = Message.system(content)
        message.metadata[_PINNED_KEY] = True
        if self._messages and self._messages[0].role is Role.SYSTEM:
            self._messages[0] = message
        else:
            self._messages.insert(0, message)
        return message

    def add(self, message: Message, *, pinned: bool = False) -> Message:
        """加入一条消息（``auto_trim`` 为真时自动裁剪到预算内）。"""
        if pinned:
            if message.role is Role.TOOL:
                raise ValueError("tool 消息不能单独 pinned：它必须跟随对应的 assistant 消息")
            message.metadata[_PINNED_KEY] = True
        self._messages.append(message)
        if self.auto_trim:
            self.trim()
        return message

    def add_user(self, content: str, *, pinned: bool = False, **metadata: Any) -> Message:
        return self.add(Message.user(content, **metadata), pinned=pinned)

    def add_assistant(
        self,
        content: str | None = None,
        *,
        tool_calls: Sequence[ToolCall] | None = None,
        pinned: bool = False,
        **metadata: Any,
    ) -> Message:
        return self.add(Message.assistant(content, list(tool_calls or []), **metadata), pinned=pinned)

    def add_tool(
        self,
        content: str,
        tool_call_id: str,
        *,
        name: str | None = None,
        **metadata: Any,
    ) -> Message:
        return self.add(Message.tool(content, tool_call_id, name, **metadata))

    # ------------------------------------------------------------ 裁剪
    def trim(self) -> int:
        """按单元丢弃最旧的、未被固定且非最新的消息，直到满足预算；返回丢弃条数。"""
        if self.tokens <= self.budget:
            self._overflow_warned = False
            return 0

        dropped = 0
        while self.tokens > self.budget:
            unit = self._first_droppable_unit()
            if unit is None:
                if not self._overflow_warned:
                    logger.warning(
                        "工作记忆无法再裁剪：固定消息与最新消息合计 %d tokens 超出预算 %d，"
                        "请精简系统提示/输入或调大 MAX_TOKENS。",
                        self.tokens,
                        self.budget,
                    )
                    self._overflow_warned = True
                break
            start, size = unit
            del self._messages[start : start + size]
            dropped += size

        if dropped:
            self._dropped += dropped
            log_event(
                logger,
                "memory.trim",
                dropped=dropped,
                messages=len(self._messages),
                tokens=self.tokens,
                budget=self.budget,
            )
        return dropped

    def _units(self) -> list[tuple[int, int]]:
        """把消息切成不可拆分的单元 ``(起始下标, 长度)``。"""
        units: list[tuple[int, int]] = []
        index = 0
        total = len(self._messages)
        while index < total:
            size = self._unit_size(index)
            units.append((index, size))
            index += size
        return units

    def _unit_size(self, index: int) -> int:
        """``assistant(tool_calls)`` 及其配对 ``tool`` 结果视为一个单元。"""
        message = self._messages[index]
        if message.role is not Role.ASSISTANT or not message.tool_calls:
            return 1
        pending = {call.id for call in message.tool_calls}
        size = 1
        total = len(self._messages)
        while index + size < total:
            following = self._messages[index + size]
            if following.role is not Role.TOOL or following.tool_call_id not in pending:
                break
            size += 1
        return size

    def _first_droppable_unit(self) -> tuple[int, int] | None:
        newest = len(self._messages) - 1
        for start, size in self._units():
            if start + size - 1 >= newest:  # 最新消息永不裁剪
                continue
            pinned = any(
                self._messages[start + offset].metadata.get(_PINNED_KEY) for offset in range(size)
            )
            if not pinned:
                return start, size
        return None

    # ------------------------------------------------------------ 序列化
    def to_openai(self) -> list[dict[str, Any]]:
        """转换为 OpenAI Chat Completions 的 messages 列表。"""
        return to_openai_messages(self._messages)

    def snapshot(self) -> list[dict[str, Any]]:
        """导出全部消息（含 pinned 标记），用于工作流中断后恢复（见 W6）。"""
        return [message.to_dict() for message in self._messages]

    def restore(self, snapshot: Sequence[dict[str, Any]]) -> None:
        """从 :meth:`snapshot` 的结果恢复；恢复后若超预算会立即裁剪。"""
        self._messages = [Message.from_dict(item) for item in snapshot]
        self._dropped = 0
        self._overflow_warned = False
        if self.auto_trim:
            self.trim()

    def clear(self, *, keep_system: bool = True) -> None:
        """清空窗口；``keep_system`` 为真时保留系统提示。"""
        if keep_system and self.system_prompt is not None:
            self._messages = [self._messages[0]]
        else:
            self._messages = []
        self._dropped = 0
        self._overflow_warned = False

    def describe(self) -> dict[str, Any]:
        roles: dict[str, int] = {}
        for message in self._messages:
            roles[message.role.value] = roles.get(message.role.value, 0) + 1
        return {
            "messages": len(self._messages),
            "tokens": self.tokens,
            "budget": self.budget,
            "max_tokens": self.max_tokens,
            "dropped": self._dropped,
            "roles": roles,
        }
