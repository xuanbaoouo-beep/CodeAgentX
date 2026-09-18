"""MCP 服务端公共骨架（stdio）。

为什么单独抽一层：``mcp_filesystem`` 与 ``mcp_github`` 的差异只在"有哪些工具"，
而下面这些**容易写错且必须只写一次**的东西完全一样：

1. stdio 两条硬规矩——**stdout 只跑协议**（日志强制改到 stderr）、
   每条报文一行 JSON + UTF-8 + ``\\n``；
2. 协议分发骨架（``initialize`` / ``ping`` / ``tools/list`` / ``tools/call`` /
   ``resources/list`` / ``resources/read``）与错误码映射；
3. **两类失败分开**：工具自身执行失败 → ``isError=true`` 的正常响应（连接不炸，
   Agent 能"看到失败继续走"）；协议级问题（方法不存在、参数非法）→ JSON-RPC ``error``。

子类只需要实现 :meth:`MCPServerBase.list_tools` / :meth:`MCPServerBase.call_tool`
（以及可选的资源钩子），并填好 ``server_name`` 等类属性。
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from codeagentx.core.exceptions import (
    MCPError,
    ProtocolError,
    SecurityViolationError,
    ToolExecutionError,
)
from codeagentx.core.logger import get_logger
from codeagentx.protocols.mcp_protocol import (
    MCP_PROTOCOL_VERSION,
    METHOD_INITIALIZE,
    METHOD_PING,
    METHOD_RESOURCES_LIST,
    METHOD_RESOURCES_READ,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    MCPContent,
    MCPErrorCode,
    MCPResourceSpec,
    MCPToolSpec,
    build_error,
    build_result,
    decode_message,
    encode_message,
    is_notification,
)

logger = get_logger("protocols.mcp_server")

__all__ = [
    "MCPServerBase",
    "TOOL_FAILURES",
    "ToolOutcome",
    "configure_stderr_logging",
    "reconfigure_stdio_streams",
    "serve_stdio",
]

#: 工具**自身**执行失败时按"正常响应 + isError"返回的异常集合（连接不该因此断掉）。
#: 用 ``ProtocolError`` 而不仅是 ``MCPError``：工具内部再去调外部服务（GitHub 等）
#: 失败，同样属于"这次调用没成"，要如实回给 Agent 让它看到失败继续走；
#: 若只列 ``MCPError``，``GitHubNotFoundError`` 这类错误会逃到 ``handle()`` 的兜底分支，
#: 被误升级成 ``-32603`` 协议级错误——调用方就分不清"协议不会用"和"上游挂了"。
TOOL_FAILURES: tuple[type[BaseException], ...] = (
    ProtocolError,
    SecurityViolationError,
    ToolExecutionError,
    OSError,
    ValueError,
)


@dataclass
class ToolOutcome:
    """工具执行结果：给模型看的文本 + 可选的机器可读结构。"""

    text: str
    structured: Mapping[str, Any] | None = None


class MCPServerBase:
    """MCP 服务端骨架：负责协议与 IO，子类只负责工具。"""

    #: 子类覆盖：服务端标识（出现在 ``initialize`` 的 ``serverInfo``）
    server_name: str = "codeagentx-mcp"
    server_version: str = "0.1.0"
    capabilities: Mapping[str, Any] = {"tools": {}}
    instructions: str = ""

    # -------------------------------------------------------- 子类钩子
    def list_tools(self) -> list[MCPToolSpec]:
        """声明本服务端提供的工具。"""
        return []

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome | str:
        """执行一个工具。抛异常即视为**工具级失败**（会被转成 ``isError``）。"""
        raise MCPError(
            f"未知工具：{name}",
            detail=f"可用工具：{[tool.name for tool in self.list_tools()]}",
            code=int(MCPErrorCode.INVALID_PARAMS),
        )

    def list_resources(self) -> list[MCPResourceSpec]:
        """声明本服务端暴露的资源（默认没有）。"""
        return []

    def read_resource(self, uri: str) -> list[MCPContent]:
        """读取一个资源；未实现时按"方法不存在"回复。"""
        raise MCPError(
            f"本服务端不提供资源读取：{uri}", code=int(MCPErrorCode.METHOD_NOT_FOUND)
        )

    # -------------------------------------------------------- 协议分发
    def server_info_payload(self) -> dict[str, Any]:
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": dict(self.capabilities),
            "serverInfo": {"name": self.server_name, "version": self.server_version},
            "instructions": self.instructions,
        }

    def handle(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """处理一条报文，返回响应；若为通知则返回 ``None``。

        这是纯函数式的（不碰 IO），因此单测可以直接喂报文对象。
        """
        if not isinstance(payload, Mapping):  # pragma: no cover - decode 已保证是对象
            return build_error(None, int(MCPErrorCode.INVALID_REQUEST), "报文必须是 JSON 对象")

        method = str(payload.get("method") or "")
        request_id = payload.get("id")
        if is_notification(payload):
            logger.debug("收到通知：%s", method)
            return None

        try:
            result = self.dispatch(method, payload.get("params"))
        except MCPError as exc:
            return build_error(
                request_id,
                exc.code if exc.code is not None else int(MCPErrorCode.INVALID_PARAMS),
                exc.message,
                exc.detail,
            )
        except SecurityViolationError as exc:
            return build_error(request_id, int(MCPErrorCode.INVALID_PARAMS), f"安全策略拒绝：{exc}")
        except ToolExecutionError as exc:
            return build_error(request_id, int(MCPErrorCode.INTERNAL_ERROR), str(exc))
        except OSError as exc:
            return build_error(
                request_id, int(MCPErrorCode.INTERNAL_ERROR), f"{type(exc).__name__}: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - 服务端不能因单条请求崩溃
            logger.exception("处理 %s 时发生未预期错误", method)
            return build_error(
                request_id, int(MCPErrorCode.INTERNAL_ERROR), f"{type(exc).__name__}: {exc}"
            )
        return build_result(request_id, result)

    def dispatch(self, method: str, params: Any) -> dict[str, Any]:
        """按方法名分发；未知方法抛 :class:`MCPError`（码 ``-32601``）。"""
        if method == METHOD_INITIALIZE:
            return self.server_info_payload()
        if method == METHOD_PING:
            return {}
        if method == METHOD_TOOLS_LIST:
            return {"tools": [tool.to_dict() for tool in self.list_tools()]}
        if method == METHOD_TOOLS_CALL:
            return self._tools_call(params)
        if method == METHOD_RESOURCES_LIST:
            return {"resources": [item.to_dict() for item in self.list_resources()]}
        if method == METHOD_RESOURCES_READ:
            return self._resources_read(params)
        raise MCPError(
            f"未实现的方法：{method or '<空>'}",
            detail="支持：initialize / tools/list / tools/call / resources/list / resources/read / ping",
            code=int(MCPErrorCode.METHOD_NOT_FOUND),
        )

    def _tools_call(self, params: Any) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            raise MCPError("tools/call 缺少 params", code=int(MCPErrorCode.INVALID_PARAMS))
        name = str(params.get("name") or "").strip()
        arguments = params.get("arguments") or {}
        if not name:
            raise MCPError("tools/call 缺少工具名", code=int(MCPErrorCode.INVALID_PARAMS))
        if not isinstance(arguments, Mapping):
            raise MCPError(
                "tools/call 的 arguments 必须是对象", code=int(MCPErrorCode.INVALID_PARAMS)
            )
        # 工具名先校验：调用方（模型）报了个不存在的工具属于**协议级**的入参错误（-32602），
        # 必须在下面被 isError 兜住之前就拒掉，否则调用方分不清"工具没写对"与"工具跑挂了"。
        available = [tool.name for tool in self.list_tools()]
        if name not in available:
            raise MCPError(
                f"未知工具：{name}",
                detail=f"可用工具：{sorted(available)}",
                code=int(MCPErrorCode.INVALID_PARAMS),
            )
        try:
            outcome = self.call_tool(name, arguments)
        except TOOL_FAILURES as exc:
            # 工具级失败：协议是通的，只是这次调用没成——如实返回 isError，
            # 调用方（Agent）需要看到失败继续走，而不是整个连接炸掉
            logger.warning("工具 %s 执行失败：%s", name, exc)
            return {
                "content": [MCPContent.text_content(f"{type(exc).__name__}: {exc}").to_dict()],
                "isError": True,
            }
        text, structured = (
            (outcome.text, outcome.structured)
            if isinstance(outcome, ToolOutcome)
            else (str(outcome), None)
        )
        payload: dict[str, Any] = {
            "content": [MCPContent.text_content(text).to_dict()],
            "isError": False,
        }
        if structured is not None:
            payload["structuredContent"] = dict(structured)
        return payload

    def _resources_read(self, params: Any) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            raise MCPError("resources/read 缺少 params", code=int(MCPErrorCode.INVALID_PARAMS))
        uri = str(params.get("uri") or "").strip()
        if not uri:
            raise MCPError("resources/read 缺少 uri", code=int(MCPErrorCode.INVALID_PARAMS))
        return {"contents": [block.to_dict() for block in self.read_resource(uri)]}

    # -------------------------------------------------------- IO 循环
    def serve(self, stdin: TextIO, stdout: TextIO) -> int:
        """逐行处理请求，返回处理过的请求条数（通知不计）。"""
        handled = 0
        for raw in stdin:
            try:
                payload = decode_message(raw)
            except MCPError as exc:
                stdout.write(
                    encode_message(
                        build_error(
                            None,
                            exc.code if exc.code is not None else int(MCPErrorCode.PARSE_ERROR),
                            exc.message,
                            exc.detail,
                        )
                    )
                )
                stdout.flush()
                continue
            response = self.handle(payload)
            if response is None:
                continue
            stdout.write(encode_message(response))
            stdout.flush()
            handled += 1
        return handled

    def run(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
        """在真实标准流上运行（被 :class:`MCPClient` 以子进程方式启动时走这里）。"""
        return self.serve(stdin or sys.stdin, stdout or sys.stdout)


# ------------------------------------------------------------------ stdio 卫生
def configure_stderr_logging(level: str = "WARNING") -> None:
    """把日志全部改到 stderr：stdout 是协议通道，绝不能被日志污染。

    **必须无条件清空所有 handler**，不能只挑 ``StreamHandler`` 删：
    ``rich`` 的 ``RichHandler`` 直接继承 ``logging.Handler``（不是 ``StreamHandler``），
    且默认写 stdout——按类型过滤会漏掉它，日志就会混进协议报文。
    文件 handler 同样移除：作为子进程运行时不需要另外落盘，
    诊断信息由客户端的 stderr 收集（见 ``StdioTransport.stderr_tail``）。
    """
    root_logger = logging.getLogger("codeagentx")
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - 关闭失败不影响主流程
            pass
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, str(level).upper(), logging.WARNING))


def reconfigure_stdio_streams() -> None:
    """把标准流切成 UTF-8 与 ``\\n`` 行尾（Windows 默认编码会毁掉协议报文）。"""
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover - 非标准流
            continue
        try:
            reconfigure(encoding="utf-8", newline="\n")
        except (ValueError, OSError):  # pragma: no cover - 已重定向的流
            pass


def serve_stdio(server: MCPServerBase, *, log_level: str = "WARNING") -> int:
    """准备好 stdio 卫生后跑服务端主循环，返回处理过的请求条数。"""
    configure_stderr_logging(log_level)
    reconfigure_stdio_streams()
    return server.run()
