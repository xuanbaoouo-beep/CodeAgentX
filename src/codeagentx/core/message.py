"""消息抽象：统一内部表示，并可无损转换为 OpenAI Chat Completions 格式。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """对话角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:  # pragma: no cover - 便于日志打印
        return self.value


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 与 SDK 对象两种形态的取值。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class ToolCall:
    """模型发起的一次工具调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    @classmethod
    def from_openai(cls, data: Any) -> ToolCall:
        """从 OpenAI 响应对象/字典构造。

        ``arguments`` 在 OpenAI 协议中是 JSON 字符串；解析失败时不抛异常，
        而是保留原始串（``raw_arguments``）并把 ``arguments`` 置空，
        交由工具层做参数校验并给出可读错误。
        """
        function = _get(data, "function", {}) or {}
        name = _get(function, "name", "") or ""
        raw = _get(function, "arguments", "") or ""

        if isinstance(raw, dict):
            return cls(id=_get(data, "id", "") or "", name=name, arguments=raw)

        raw_str = str(raw)
        if not raw_str.strip():
            return cls(
                id=_get(data, "id", "") or "", name=name, arguments={}, raw_arguments=raw_str
            )
        try:
            parsed = json.loads(raw_str)
        except json.JSONDecodeError:
            return cls(
                id=_get(data, "id", "") or "", name=name, arguments={}, raw_arguments=raw_str
            )
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        return cls(
            id=_get(data, "id", "") or "", name=name, arguments=parsed, raw_arguments=raw_str
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "raw_arguments": self.raw_arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments") or {},
            raw_arguments=data.get("raw_arguments"),
        )


@dataclass
class Message:
    """一条对话消息。"""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.role, Role):
            self.role = Role(self.role)

    # ------------------------------------------------------------ 快捷构造
    @classmethod
    def system(cls, content: str, **metadata: Any) -> Message:
        return cls(role=Role.SYSTEM, content=content, metadata=metadata)

    @classmethod
    def user(cls, content: str, **metadata: Any) -> Message:
        return cls(role=Role.USER, content=content, metadata=metadata)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None, **metadata: Any
    ) -> Message:
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=list(tool_calls or []),
            metadata=metadata,
        )

    @classmethod
    def tool(
        cls, content: str, tool_call_id: str, name: str | None = None, **metadata: Any
    ) -> Message:
        return cls(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            metadata=metadata,
        )

    # ------------------------------------------------------------ 序列化
    def to_openai(self) -> dict[str, Any]:
        """转换为 OpenAI Chat Completions 的 message 结构。"""
        payload: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content if self.content is not None else "",
        }
        if self.tool_calls:
            payload["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_openai(cls, data: Any) -> Message:
        tool_calls = [ToolCall.from_openai(item) for item in (_get(data, "tool_calls", None) or [])]
        return cls(
            role=Role(_get(data, "role", "user")),
            content=_get(data, "content", None),
            tool_calls=tool_calls,
            tool_call_id=_get(data, "tool_call_id", None),
            name=_get(data, "name", None),
        )

    def to_dict(self) -> dict[str, Any]:
        """用于持久化（工作流状态恢复）。"""
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=Role(data.get("role", "user")),
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(item) for item in (data.get("tool_calls") or [])],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            metadata=data.get("metadata") or {},
        )

    # ------------------------------------------------------------ 辅助
    @property
    def text(self) -> str:
        """正文内容，``None`` 归一化为空串。"""
        return self.content or ""

    def __str__(self) -> str:  # pragma: no cover - 便于调试
        suffix = f" tool_calls={len(self.tool_calls)}" if self.tool_calls else ""
        preview = self.text[:60].replace("\n", " ")
        return f"<Message {self.role.value}{suffix}: {preview}>"


def to_openai_messages(messages: Any) -> list[dict[str, Any]]:
    """把 ``Message`` / ``dict`` / ``str`` 混合序列统一转成 OpenAI 格式列表。"""
    if isinstance(messages, (str, Message, dict)):
        messages = [messages]
    result: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, Message):
            result.append(item.to_openai())
        elif isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str):
            result.append({"role": Role.USER.value, "content": item})
        else:
            raise TypeError(f"不支持的消息类型：{type(item)!r}")
    return result


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：CJK 字符按 1 token，其余按 4 字符 1 token。

    仅用于上下文预算控制（W7），不作为计费依据；真实用量以 API 返回为准。
    """
    if not text:
        return 0
    cjk = 0
    for char in text:
        if "\u3000" <= char <= "\u9fff" or "\uff00" <= char <= "\uffef":
            cjk += 1
    other = len(text) - cjk
    if other <= 0:
        return cjk
    return cjk + max(1, other // 4)


def estimate_messages_tokens(messages: Any) -> int:
    """估算一组消息的总 token 数（含少量角色开销）。"""
    total = 0
    for item in to_openai_messages(messages):
        total += 4
        total += estimate_tokens(str(item.get("content") or ""))
        for call in item.get("tool_calls") or []:
            function = call.get("function") or {}
            total += estimate_tokens(str(function.get("name") or ""))
            total += estimate_tokens(str(function.get("arguments") or ""))
    return total
