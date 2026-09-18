"""MCP 客户端：以子进程 stdio 方式连接 MCP 服务端。

分工
----
:mod:`~codeagentx.protocols.mcp_protocol`
    报文长什么样（JSON-RPC 2.0 与 MCP 方法）。
:mod:`~codeagentx.protocols.mcp_client`（本模块）
    怎么把它送出去、怎么把响应收回来、出错怎么归类。

两处刻意的设计
--------------
1. **传输层可替换**（:class:`MCPTransport`）：真实场景用 :class:`StdioTransport`
   起子进程；测试可以注入进程内假传输，不需要真的 fork 进程，
   协议解析与超时逻辑照样被测到。
2. **两类失败分开对待**：服务端明确返回的协议错误（方法不存在、参数非法）
   不抛异常，而是返回 ``ok=False`` 的 :class:`~codeagentx.protocols.mcp_protocol.MCPToolResult`，
   让上层的 Agent 循环能"看到失败并继续"；而连接断开、等待超时这类
   **基础设施故障**抛 :class:`~codeagentx.core.exceptions.MCPConnectionError`，
   因为它们意味着后续所有调用都不会成功，硬撑没有意义。
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import codeagentx
from codeagentx.core.exceptions import MCPConnectionError, MCPError
from codeagentx.core.logger import get_logger, log_event
from codeagentx.protocols.mcp_protocol import (
    CLIENT_CAPABILITIES,
    MCP_PROTOCOL_VERSION,
    METHOD_INITIALIZE,
    METHOD_INITIALIZED,
    METHOD_PING,
    METHOD_RESOURCES_LIST,
    METHOD_RESOURCES_READ,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    MCPContent,
    MCPResourceSpec,
    MCPServerInfo,
    MCPToolResult,
    MCPToolSpec,
    build_notification,
    build_request,
    decode_message,
    encode_message,
    is_notification,
    parse_result,
)

logger = get_logger("protocols.mcp_client")

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT",
    "MCPClient",
    "MCPClientStats",
    "MCPTransport",
    "StdioTransport",
    "child_env",
    "default_python",
]

DEFAULT_REQUEST_TIMEOUT = 20.0
#: 单次请求最多跳过多少条无关报文（服务端通知等）
MAX_SKIPPED_MESSAGES = 20
#: 关闭时等待子进程退出的秒数
CLOSE_GRACE_SECONDS = 3.0
#: stderr 最多保留的诊断行数
STDERR_TAIL_LINES = 40


# ------------------------------------------------------------------ 传输层
@runtime_checkable
class MCPTransport(Protocol):
    """传输层接口：一行一条报文（含换行）。"""

    def start(self) -> None: ...

    def send(self, line: str) -> None: ...

    def receive(self, timeout: float) -> str: ...

    def close(self) -> None: ...

    @property
    def closed(self) -> bool: ...

    def describe(self) -> dict[str, Any]: ...


class StdioTransport:
    """子进程 stdio 传输（MCP 的 stdio 约定）。

    stdout 由后台线程**逐行**读入队列，``receive`` 从队列取——
    这样超时才可控（管道读取本身在 Windows 上不便设超时）。
    stderr 另开线程收集末尾若干行，用于"服务端崩了"时给出可读的诊断。
    """

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        stderr_tail_lines: int = STDERR_TAIL_LINES,
        label: str = "",
    ) -> None:
        self.command = command
        self.args = [str(arg) for arg in args]
        self.cwd = None if cwd is None else str(cwd)
        self.env = dict(env) if env is not None else None
        self.label = label or Path(command).name
        self.stderr_tail_lines = stderr_tail_lines
        self._process: subprocess.Popen[str] | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=stderr_tail_lines)
        self._threads: list[threading.Thread] = []

    # -------------------------------------------------------- 生命周期
    def start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(  # noqa: S603 - 命令由本进程构造，非用户输入拼接
                [self.command, *self.args],
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise MCPConnectionError(
                f"MCP 服务端进程启动失败：{self.command}", detail=str(exc)
            ) from exc

        self._spawn(self._pump_stdout, "mcp-stdout")
        self._spawn(self._pump_stderr, "mcp-stderr")
        logger.debug("MCP 服务端已启动：%s %s", self.command, " ".join(self.args))

    def send(self, line: str) -> None:
        process = self._require_process()
        assert process.stdin is not None  # Popen 已指定 stdin=PIPE
        try:
            process.stdin.write(line)
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise MCPConnectionError(
                "MCP 服务端连接已断开，无法发送请求", detail=self._diagnostics()
            ) from exc

    def receive(self, timeout: float) -> str:
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise MCPConnectionError(
                f"等待 MCP 响应超时（{timeout}s）", detail=self._diagnostics()
            ) from exc
        if line is None:
            raise MCPConnectionError("MCP 服务端已结束输出", detail=self._diagnostics())
        return line

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()  # 关掉 stdin，服务端主循环读到 EOF 会自然退出
        except OSError:  # pragma: no cover - 已断开
            pass
        try:
            process.wait(timeout=CLOSE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=CLOSE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况
                process.kill()
        self._process = None

    @property
    def closed(self) -> bool:
        return self._process is None or self._process.poll() is not None

    @property
    def returncode(self) -> int | None:
        return None if self._process is None else self._process.returncode

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def describe(self) -> dict[str, Any]:
        return {
            "transport": "stdio",
            "command": " ".join([self.command, *self.args]),
            "cwd": self.cwd,
            "closed": self.closed,
            "returncode": self.returncode,
        }

    # -------------------------------------------------------- 内部
    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=target, daemon=True, name=f"{name}-{self.label}")
        thread.start()
        self._threads.append(thread)

    def _pump_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:  # pragma: no cover - 不会发生
            return
        try:
            for line in process.stdout:
                self._queue.put(line)
        except (OSError, ValueError):  # pragma: no cover - 管道异常关闭
            pass
        finally:
            self._queue.put(None)  # 哨兵：告诉接收方"不会再有数据了"

    def _pump_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:  # pragma: no cover - 不会发生
            return
        try:
            for line in process.stderr:
                self._stderr_tail.append(line.rstrip("\n"))
        except (OSError, ValueError):  # pragma: no cover
            pass

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise MCPConnectionError("MCP 传输尚未启动")
        return self._process

    def _diagnostics(self) -> str:
        tail = self.stderr_tail
        if tail:
            return f"服务端 stderr 末尾：\n{tail}"
        return f"服务端无 stderr 输出（returncode={self.returncode}）"


# ------------------------------------------------------------------ 统计
@dataclass
class MCPClientStats:
    """客户端调用统计（可观测性 + 后续评估的输入）。"""

    requests: int = 0
    tool_calls: int = 0
    notifications: int = 0
    skipped_messages: int = 0
    invalid_messages: int = 0
    errors: int = 0
    duration: float = 0.0
    methods: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "notifications": self.notifications,
            "skipped_messages": self.skipped_messages,
            "invalid_messages": self.invalid_messages,
            "errors": self.errors,
            "duration": round(self.duration, 3),
            "methods": dict(self.methods),
        }


# ------------------------------------------------------------------ 客户端
class MCPClient:
    """MCP 客户端（同步、单连接）。可作为上下文管理器使用。"""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        client_name: str = "CodeAgentX",
        client_version: str = "0.1.0",
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        server_label: str = "",
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout 必须为正数")
        self.transport = transport
        self.client_name = client_name
        self.client_version = client_version
        self.request_timeout = request_timeout
        self.server_label = server_label or getattr(transport, "label", "")
        self.stats = MCPClientStats()
        self._next_id = 0
        self._server_info: MCPServerInfo | None = None
        self._tools: list[MCPToolSpec] | None = None
        self._resources: list[MCPResourceSpec] | None = None

    # -------------------------------------------------------- 构造
    @classmethod
    def from_command(
        cls,
        command: str,
        args: Sequence[str] = (),
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        label: str = "",
        **kwargs: Any,
    ) -> MCPClient:
        """用一条命令构造客户端（会额外注入 PYTHONPATH，保证子进程能 import 本项目）。"""
        transport = StdioTransport(
            command,
            args,
            cwd=cwd,
            env=dict(env) if env is not None else child_env(),
            label=label,
        )
        return cls(transport, server_label=label, **kwargs)

    @classmethod
    def from_transport(cls, transport: MCPTransport, **kwargs: Any) -> MCPClient:
        """注入自定义传输（测试或进程内服务端）。"""
        return cls(transport, **kwargs)

    @classmethod
    def from_roots(cls, roots: Sequence[str | Path], **kwargs: Any) -> MCPClient:
        """直接连接**内置**文件系统 MCP 服务端（最常用的用法）。"""
        from codeagentx.protocols.mcp_filesystem import filesystem_server_command

        command, args = filesystem_server_command(roots)
        return cls.from_command(command, args, label="filesystem", **kwargs)

    @classmethod
    def from_github(
        cls,
        *,
        repos: Sequence[str] = (),
        python: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> MCPClient:
        """连接**内置** GitHub MCP 服务端（只读）。

        令牌通过环境变量传递（``child_env()`` 会继承父进程环境），
        因此**不要**把令牌写进 ``repos`` 之类的命令行参数里。
        """
        from codeagentx.protocols.mcp_github import github_server_command

        command, args = github_server_command(repos=repos, python=python, base_url=base_url)
        return cls.from_command(command, args, label="github", **kwargs)

    # -------------------------------------------------------- 生命周期
    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> MCPServerInfo:
        """启动传输并完成 ``initialize`` 握手。"""
        self.transport.start()
        return self.initialize()

    def initialize(self) -> MCPServerInfo:
        """握手：交换协议版本与能力，随后发送 ``initialized`` 通知。"""
        params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": CLIENT_CAPABILITIES,
            "clientInfo": {"name": self.client_name, "version": self.client_version},
        }
        result = self._request(METHOD_INITIALIZE, params)
        payload = result if isinstance(result, Mapping) else {}
        self._server_info = MCPServerInfo.from_payload(payload)
        self._notify(METHOD_INITIALIZED)
        log_event(
            logger,
            "mcp.initialize",
            server=self._server_info.name,
            version=self._server_info.version,
            protocol=self._server_info.protocol_version,
        )
        return self._server_info

    def close(self) -> None:
        self.transport.close()

    def ping(self) -> bool:
        """探活：服务端返回任意结果即视为存活。"""
        try:
            self._request(METHOD_PING, {})
        except MCPError:
            return False
        return True

    # -------------------------------------------------------- 工具
    def list_tools(self, *, refresh: bool = False) -> list[MCPToolSpec]:
        """列出服务端工具（默认带缓存）。"""
        if self._tools is not None and not refresh:
            return list(self._tools)
        result = self._request(METHOD_TOOLS_LIST, {})
        payload = result if isinstance(result, Mapping) else {}
        raw_tools = payload.get("tools") or []
        tools: list[MCPToolSpec] = []
        for item in raw_tools:
            if isinstance(item, Mapping):
                tools.append(MCPToolSpec.from_dict(item))
        self._tools = tools
        return list(tools)

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> MCPToolResult:
        """调用服务端工具。

        协议级错误（服务端明确报错）会转成 ``ok=False`` 的结果返回；
        连接类故障则抛 :class:`~codeagentx.core.exceptions.MCPConnectionError`。
        """
        if not name or not str(name).strip():
            raise ValueError("工具名不能为空")
        tool = str(name).strip()
        params: dict[str, Any] = {"name": tool}
        if arguments:
            params["arguments"] = dict(arguments)

        started = time.perf_counter()
        self.stats.tool_calls += 1
        try:
            result = self._request(METHOD_TOOLS_CALL, params)
        except MCPError as exc:
            self.stats.errors += 1
            logger.warning("MCP 工具 %s 调用失败：%s", tool, exc)
            return MCPToolResult(
                tool=tool,
                ok=False,
                error=str(exc),
                error_code=exc.code,
                duration=time.perf_counter() - started,
            )

        payload = result if isinstance(result, Mapping) else {}
        tool_result = MCPToolResult.from_payload(
            tool, payload, duration=time.perf_counter() - started
        )
        if not tool_result.ok:
            self.stats.errors += 1
        log_event(
            logger,
            "mcp.tool_call",
            tool=tool,
            ok=tool_result.ok,
            duration=round(tool_result.duration, 4),
            error=tool_result.error,
        )
        return tool_result

    # -------------------------------------------------------- 资源
    def list_resources(self, *, refresh: bool = False) -> list[MCPResourceSpec]:
        """列出服务端资源（默认带缓存）。"""
        if self._resources is not None and not refresh:
            return list(self._resources)
        result = self._request(METHOD_RESOURCES_LIST, {})
        payload = result if isinstance(result, Mapping) else {}
        resources: list[MCPResourceSpec] = []
        for item in payload.get("resources") or []:
            if isinstance(item, Mapping):
                resources.append(MCPResourceSpec.from_dict(item))
        self._resources = resources
        return list(resources)

    def read_resource(self, uri: str) -> list[MCPContent]:
        """读取资源内容。"""
        result = self._request(METHOD_RESOURCES_READ, {"uri": uri})
        payload = result if isinstance(result, Mapping) else {}
        contents = payload.get("contents") or []
        return [
            MCPContent.from_dict(item) for item in contents if isinstance(item, Mapping)
        ]

    # -------------------------------------------------------- 展示
    @property
    def server_info(self) -> MCPServerInfo | None:
        return self._server_info

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools or []]

    def describe(self) -> dict[str, Any]:
        return {
            "server_label": self.server_label,
            "server_info": None if self._server_info is None else self._server_info.to_dict(),
            "tools": self.tool_names,
            "transport": self.transport.describe(),
            "stats": self.stats.as_dict(),
        }

    # -------------------------------------------------------- 内部
    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        self.transport.send(encode_message(build_notification(method, params)))
        self.stats.notifications += 1

    def _request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """发一条请求并等待**对应 id** 的响应，期间忽略服务端通知。"""
        if self.transport.closed and self._server_info is not None:
            raise MCPConnectionError("MCP 连接已关闭", detail=self.transport.describe())

        request_id = self._new_id()
        started = time.perf_counter()
        self.stats.requests += 1
        self.stats.methods[method] = self.stats.methods.get(method, 0) + 1
        self.transport.send(encode_message(build_request(request_id, method, params)))

        wait = self.request_timeout if timeout is None else timeout
        for _ in range(MAX_SKIPPED_MESSAGES + 1):
            raw = self.transport.receive(wait)
            try:
                message = decode_message(raw)
            except MCPError as exc:
                # 服务端写了非协议内容（例如误把日志打到 stdout）：记下来，继续等
                self.stats.invalid_messages += 1
                logger.warning("忽略非法 MCP 报文：%s", exc)
                continue

            if is_notification(message):
                self.stats.notifications += 1
                logger.debug("收到 MCP 通知：%s", message.get("method"))
                continue
            if message.get("id") != request_id:
                self.stats.skipped_messages += 1
                logger.debug("忽略 id 不匹配的 MCP 报文：%s", message.get("id"))
                continue

            result = parse_result(message, request_id=request_id, method=method)
            self.stats.duration += time.perf_counter() - started
            return result

        raise MCPConnectionError(
            "连续收到无法匹配的报文，已放弃本次请求",
            detail=f"method={method} id={request_id} max_skipped={MAX_SKIPPED_MESSAGES}",
        )


# ------------------------------------------------------------------ 环境
def child_env(**overrides: str) -> dict[str, str]:
    """构造子进程环境变量：把本项目 ``src`` 目录塞进 ``PYTHONPATH``。

    为什么需要：子进程用 ``python -m codeagentx.protocols.mcp_filesystem`` 启动时，
    若本项目不是以可编辑模式安装的，子进程根本 import 不到自己。
    显式注入 ``PYTHONPATH`` 让"源码布局"与"安装布局"都能跑。
    """
    env = dict(os.environ)
    package_parent = Path(codeagentx.__file__).resolve().parents[1]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{package_parent}{os.pathsep}{existing}" if existing else str(package_parent)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(overrides)
    return env


def default_python() -> str:
    """当前解释器路径（MCP 服务端默认用同一个解释器，环境才不会错配）。"""
    return sys.executable
