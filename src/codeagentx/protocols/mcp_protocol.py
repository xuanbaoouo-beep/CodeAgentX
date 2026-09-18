"""MCP（Model Context Protocol）报文与数据结构。

为什么自研而不装官方 SDK：与 AD-01 同款理由——本项目要覆盖 Python 3.10~3.13，
依赖越少越稳；而 MCP 的线路格式本质就是 **JSON-RPC 2.0 + 一组约定方法**，
自己实现可控、可测、可离线复现。本项目只实现用到的那一层：
``initialize`` / ``tools/list`` / ``tools/call`` / ``resources/list`` / ``resources/read``。

线路约定（stdio 传输）：一行一条 JSON 报文，UTF-8，``\\n`` 结尾；
请求带自增 ``id``，通知不带 ``id``，响应必须带回同样的 ``id``。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from codeagentx.core.exceptions import MCPError

__all__ = [
    "CLIENT_CAPABILITIES",
    "JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "METHOD_INITIALIZE",
    "METHOD_INITIALIZED",
    "METHOD_PING",
    "METHOD_RESOURCES_LIST",
    "METHOD_RESOURCES_READ",
    "METHOD_TOOLS_CALL",
    "METHOD_TOOLS_LIST",
    "MCPContent",
    "MCPErrorCode",
    "MCPResourceSpec",
    "MCPServerInfo",
    "MCPToolResult",
    "MCPToolSpec",
    "build_error",
    "build_notification",
    "build_request",
    "build_result",
    "decode_message",
    "encode_message",
    "is_notification",
    "parse_result",
]

JSONRPC_VERSION = "2.0"
#: 本项目实现的 MCP 协议版本
MCP_PROTOCOL_VERSION = "2024-11-05"

METHOD_INITIALIZE = "initialize"
METHOD_INITIALIZED = "notifications/initialized"
METHOD_PING = "ping"
METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"
METHOD_RESOURCES_LIST = "resources/list"
METHOD_RESOURCES_READ = "resources/read"

#: 客户端声明自己支持的能力（本实现只用工具与资源，不需要 sampling/roots）
CLIENT_CAPABILITIES: dict[str, Any] = {"tools": {}, "resources": {}}


class MCPErrorCode(IntEnum):
    """JSON-RPC 2.0 标准错误码。"""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# ------------------------------------------------------------------ 数据结构
@dataclass(frozen=True)
class MCPToolSpec:
    """服务端声明的工具定义（``tools/list`` 返回项）。"""

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    annotations: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MCPToolSpec:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise MCPError("MCP 工具定义缺少 name", detail=str(payload)[:200])
        schema = payload.get("inputSchema") or payload.get("input_schema") or {}
        annotations = payload.get("annotations") or {}
        return cls(
            name=name,
            description=str(payload.get("description") or ""),
            input_schema=dict(schema) if isinstance(schema, Mapping) else {},
            annotations=dict(annotations) if isinstance(annotations, Mapping) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema) or {"type": "object", "properties": {}},
        }
        if self.annotations:
            payload["annotations"] = dict(self.annotations)
        return payload


@dataclass(frozen=True)
class MCPResourceSpec:
    """服务端声明的资源（``resources/list`` 返回项）。"""

    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MCPResourceSpec:
        uri = str(payload.get("uri") or "").strip()
        if not uri:
            raise MCPError("MCP 资源定义缺少 uri", detail=str(payload)[:200])
        return cls(
            uri=uri,
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            mime_type=str(payload.get("mimeType") or payload.get("mime_type") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"uri": self.uri}
        if self.name:
            payload["name"] = self.name
        if self.description:
            payload["description"] = self.description
        if self.mime_type:
            payload["mimeType"] = self.mime_type
        return payload


@dataclass(frozen=True)
class MCPContent:
    """工具/资源返回的内容块（目前支持 text 与 image）。"""

    type: str = "text"
    text: str = ""
    data: str = ""
    mime_type: str = ""
    uri: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MCPContent:
        return cls(
            type=str(payload.get("type") or "text"),
            text=str(payload.get("text") or ""),
            data=str(payload.get("data") or ""),
            mime_type=str(payload.get("mimeType") or payload.get("mime_type") or ""),
            uri=str(payload.get("uri") or ""),
        )

    @classmethod
    def text_content(cls, text: str) -> MCPContent:
        return cls(type="text", text=text)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.text:
            payload["text"] = self.text
        if self.data:
            payload["data"] = self.data
        if self.mime_type:
            payload["mimeType"] = self.mime_type
        if self.uri:
            payload["uri"] = self.uri
        return payload


@dataclass
class MCPToolResult:
    """一次 ``tools/call`` 的结果。

    注意区分两类失败：
    - **协议级**：JSON-RPC 的 ``error``（方法不存在、参数非法）→ ``error_code`` 有值；
    - **工具级**：正常响应但 ``isError=true``（工具自己执行失败）→ ``ok=False``。
    两者都不应让调用方崩溃，但排查时含义完全不同，所以分开记录。
    """

    tool: str
    ok: bool = True
    content: list[MCPContent] = field(default_factory=list)
    structured: Any = None
    error: str | None = None
    error_code: int | None = None
    duration: float = 0.0

    @property
    def text(self) -> str:
        """把文本内容块拼成一段文本（忽略非文本块）。"""
        return "\n".join(block.text for block in self.content if block.text)

    def to_dict(self, *, with_content: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "ok": self.ok,
            "error": self.error,
            "error_code": self.error_code,
            "duration": round(self.duration, 4),
            "content_types": [block.type for block in self.content],
        }
        if with_content:
            payload["text"] = self.text
            payload["structured"] = self.structured
        return payload

    @classmethod
    def from_payload(
        cls, tool: str, payload: Mapping[str, Any], *, duration: float = 0.0
    ) -> MCPToolResult:
        raw_content = payload.get("content")
        blocks: list[MCPContent] = []
        if isinstance(raw_content, Sequence) and not isinstance(raw_content, (str, bytes)):
            blocks = [
                MCPContent.from_dict(item)
                for item in raw_content
                if isinstance(item, Mapping)
            ]
        is_error = bool(payload.get("isError") or payload.get("is_error"))
        result = cls(
            tool=tool,
            ok=not is_error,
            content=blocks,
            structured=payload.get("structuredContent") or payload.get("structured_content"),
            duration=duration,
        )
        if is_error:
            result.error = result.text or "工具返回 isError=true"
        return result


@dataclass(frozen=True)
class MCPServerInfo:
    """``initialize`` 的握手结果。"""

    name: str
    version: str = ""
    protocol_version: str = ""
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    instructions: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MCPServerInfo:
        server = payload.get("serverInfo") or payload.get("server_info") or {}
        if not isinstance(server, Mapping):
            server = {}
        capabilities = payload.get("capabilities") or {}
        return cls(
            name=str(server.get("name") or "unknown"),
            version=str(server.get("version") or ""),
            protocol_version=str(payload.get("protocolVersion") or payload.get("protocol_version") or ""),
            capabilities=dict(capabilities) if isinstance(capabilities, Mapping) else {},
            instructions=str(payload.get("instructions") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "capabilities": dict(self.capabilities),
            "instructions": self.instructions,
        }


# ------------------------------------------------------------------ 报文构造
def build_request(request_id: int | str, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """构造请求报文（带 ``id``，需要响应）。"""
    payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": method}
    if params is not None:
        payload["params"] = dict(params)
    return payload


def build_notification(method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """构造通知报文（不带 ``id``，服务端不应回复）。"""
    payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        payload["params"] = dict(params)
    return payload


def build_result(request_id: int | str | None, result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """构造成功响应。"""
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": dict(result or {})}


def build_error(
    request_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """构造错误响应（``id`` 为 ``None`` 表示连请求都解析不出来）。"""
    error: dict[str, Any] = {"code": int(code), "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


def encode_message(payload: Mapping[str, Any], *, newline: bool = True) -> str:
    """编码成一行报文（默认带换行，便于 stdio 传输）。"""
    text = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    return f"{text}\n" if newline else text


def decode_message(raw: str) -> dict[str, Any]:
    """解析一行报文。

    Raises:
        MCPError: 不是合法 JSON 或不是 JSON 对象（码为 ``PARSE_ERROR``）。
    """
    text = (raw or "").strip()
    if not text:
        raise MCPError("收到空报文", code=int(MCPErrorCode.PARSE_ERROR))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MCPError(
            "报文不是合法 JSON", detail=f"{exc}: {text[:200]}", code=int(MCPErrorCode.PARSE_ERROR)
        ) from exc
    if not isinstance(payload, dict):
        raise MCPError(
            "报文必须是 JSON 对象",
            detail=text[:200],
            code=int(MCPErrorCode.INVALID_REQUEST),
        )
    return payload


def is_notification(payload: Mapping[str, Any]) -> bool:
    """没有 ``id`` 的报文即通知（无需响应）。"""
    return "id" not in payload or payload.get("id") is None


def parse_result(
    payload: Mapping[str, Any],
    *,
    request_id: int | str | None = None,
    method: str = "",
) -> Any:
    """从响应报文中取出 ``result``；是错误对象则抛 :class:`MCPError`。

    Args:
        request_id: 期望的响应 ``id``；给了就会校验（防止把上一条的响应错配给本次请求）。
    """
    if payload.get("jsonrpc") != JSONRPC_VERSION:
        raise MCPError(
            "响应的 jsonrpc 版本不是 2.0",
            detail=str(payload)[:200],
            code=int(MCPErrorCode.INVALID_REQUEST),
        )
    error = payload.get("error")
    if isinstance(error, Mapping):
        raise MCPError(
            f"MCP 服务端返回错误：{error.get('message') or '未知错误'}",
            detail=str(error.get("data") or "")[:500] or f"method={method}",
            code=_as_int(error.get("code")),
        )
    if request_id is not None and payload.get("id") != request_id:
        raise MCPError(
            "响应 id 与请求不匹配",
            detail=f"期望 {request_id!r}，收到 {payload.get('id')!r}",
            code=int(MCPErrorCode.INVALID_REQUEST),
        )
    return payload.get("result")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
