"""MCP 协议层单元测试。

覆盖三块：
1. **报文与数据结构**：``build_*`` / ``decode_message`` / ``parse_result`` 的构造、
   编解码与校验，以及各 ``from_dict`` 的解析与容错；
2. **客户端协议边界**（注入进程内假传输，不起子进程）：跳过非法报文与通知、
   统计计数、服务端 ``error`` 转 :class:`MCPError`、噪声过多放弃请求、关闭后拒绝调用；
3. **服务端**：``FilesystemMCPServer`` 的纯函数分发，以及真子进程 stdio 联调
   （握手、工具调用、路径越界被拒但连接不炸、日志只走 stderr）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from codeagentx.core.exceptions import MCPConnectionError, MCPError
from codeagentx.protocols.mcp_client import MCPClient, MCPClientStats, child_env
from codeagentx.protocols.mcp_filesystem import (
    FilesystemMCPServer,
    filesystem_server_command,
)
from codeagentx.protocols.mcp_protocol import (
    JSONRPC_VERSION,
    MCP_PROTOCOL_VERSION,
    MCPContent,
    MCPErrorCode,
    MCPResourceSpec,
    MCPServerInfo,
    MCPToolResult,
    MCPToolSpec,
    build_error,
    build_notification,
    build_request,
    build_result,
    decode_message,
    encode_message,
    is_notification,
    parse_result,
)

SAMPLE_REPO = Path(__file__).resolve().parents[1] / "data" / "sample_repo"


# ------------------------------------------------------------------ 工具
def line(payload: dict[str, Any]) -> str:
    """把报文对象编码成一行（假传输的脚本条目）。"""
    return json.dumps(payload, ensure_ascii=False) + "\n"


def init_result(name: str = "fake-server", version: str = "9.9") -> dict[str, Any]:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {}, "resources": {}},
        "serverInfo": {"name": name, "version": version},
    }


class FakeTransport:
    """进程内假传输：按预设脚本逐条吐报文，记录所有发出的内容。

    有了它，客户端"跳过噪声、按 id 匹配、错误归类"这些逻辑不用真起子进程就能测，
    而且可以把服务端的畸形行为（非法 JSON、id 错配、噪声轰炸）精确复现出来。
    """

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.sent: list[str] = []
        self.started = False
        self.label = "fake"
        self._closed = False

    def start(self) -> None:
        self.started = True

    def send(self, text: str) -> None:
        self.sent.append(text)

    def receive(self, timeout: float) -> str:
        if not self.script:
            raise MCPConnectionError("脚本已耗尽")
        return self.script.pop(0)

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def describe(self) -> dict[str, Any]:
        return {"transport": "fake", "closed": self._closed}


def make_client(script: list[str], **kwargs: Any) -> tuple[MCPClient, FakeTransport]:
    transport = FakeTransport(script)
    return MCPClient.from_transport(transport, request_timeout=1.0, **kwargs), transport


def make_server(**kwargs: Any) -> FilesystemMCPServer:
    return FilesystemMCPServer(roots=[SAMPLE_REPO], **kwargs)


def call(server: FilesystemMCPServer, request_id: int, method: str, params: Any = None):
    """喂一条请求给服务端纯函数分发，返回响应报文。"""
    payload: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return server.handle(payload)


def tool_text(server: FilesystemMCPServer, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    """调用一个服务端工具，返回（文本内容，是否 isError）。"""
    response = call(server, 1, "tools/call", {"name": name, "arguments": arguments})
    assert response is not None
    result = response["result"]
    text = "\n".join(block.get("text", "") for block in result["content"])
    return text, bool(result.get("isError"))


# ================================================================== 1. 报文层
def test_build_request_and_notification_shapes() -> None:
    request = build_request(7, "tools/call", {"name": "echo"})
    assert request["jsonrpc"] == JSONRPC_VERSION
    assert request["id"] == 7
    assert request["params"] == {"name": "echo"}
    assert not is_notification(request)

    notification = build_notification("notifications/initialized")
    assert "id" not in notification
    assert "params" not in notification
    assert is_notification(notification)

    # id 显式为 None 也算通知
    assert is_notification({"jsonrpc": JSONRPC_VERSION, "id": None, "method": "x"})


def test_encode_decode_round_trip_keeps_non_ascii() -> None:
    encoded = encode_message(build_result(1, {"text": "中文内容"}))
    assert encoded.endswith("\n")
    assert "中文内容" in encoded  # ensure_ascii=False，报文里不出现转义
    assert decode_message(encoded)["result"]["text"] == "中文内容"

    assert encode_message({"a": 1}, newline=False) == '{"a":1}'


def test_decode_rejects_empty_and_malformed_payloads() -> None:
    with pytest.raises(MCPError) as empty:
        decode_message("")
    assert empty.value.code == int(MCPErrorCode.PARSE_ERROR)

    with pytest.raises(MCPError) as broken:
        decode_message("{坏 JSON}")
    assert broken.value.code == int(MCPErrorCode.PARSE_ERROR)

    with pytest.raises(MCPError) as not_object:
        decode_message("[1, 2, 3]")
    assert not_object.value.code == int(MCPErrorCode.INVALID_REQUEST)


def test_parse_result_validates_version_id_and_error() -> None:
    assert parse_result(build_result(3, {"ok": True}), request_id=3) == {"ok": True}

    with pytest.raises(MCPError) as bad_version:
        parse_result({"jsonrpc": "1.0", "id": 1, "result": {}}, request_id=1)
    assert bad_version.value.code == int(MCPErrorCode.INVALID_REQUEST)

    with pytest.raises(MCPError) as bad_id:
        parse_result(build_result(7, {}), request_id=8)
    assert "不匹配" in str(bad_id.value)
    assert bad_id.value.code == int(MCPErrorCode.INVALID_REQUEST)

    with pytest.raises(MCPError) as error_object:
        parse_result(
            build_error(1, int(MCPErrorCode.METHOD_NOT_FOUND), "Method not found", "tools/x"),
            request_id=1,
        )
    assert error_object.value.code == int(MCPErrorCode.METHOD_NOT_FOUND)
    assert "Method not found" in str(error_object.value)


def test_build_error_omits_empty_data() -> None:
    assert "data" not in build_error(1, -32601, "nope")["error"]
    assert build_error(1, -32601, "nope", "detail")["error"]["data"] == "detail"


def test_tool_spec_from_dict_accepts_both_schema_keys() -> None:
    camel = MCPToolSpec.from_dict({"name": "echo", "inputSchema": {"type": "object"}})
    snake = MCPToolSpec.from_dict({"name": "echo", "input_schema": {"type": "object"}})
    assert camel.input_schema == snake.input_schema == {"type": "object"}

    # 缺 schema 时补一个合法的空对象，避免下游拿到 None
    assert MCPToolSpec.from_dict({"name": "echo"}).to_dict()["inputSchema"] == {
        "type": "object",
        "properties": {},
    }

    with pytest.raises(MCPError):
        MCPToolSpec.from_dict({"description": "没有名字"})


def test_resource_spec_from_dict_requires_uri_and_reads_mime_type() -> None:
    spec = MCPResourceSpec.from_dict({"uri": "file:///x", "mimeType": "text/plain"})
    assert spec.mime_type == "text/plain"
    assert spec.to_dict()["mimeType"] == "text/plain"

    with pytest.raises(MCPError):
        MCPResourceSpec.from_dict({"name": "没有 uri"})


def test_tool_result_from_payload_marks_tool_level_failure() -> None:
    ok = MCPToolResult.from_payload(
        "echo",
        {"content": [{"type": "text", "text": "hi"}, {"type": "data", "data": "AA=="}]},
    )
    assert ok.ok is True
    assert ok.text == "hi"  # 非文本块不进 text

    failed = MCPToolResult.from_payload(
        "read", {"content": [{"type": "text", "text": "boom"}], "isError": True}
    )
    assert failed.ok is False
    assert failed.error == "boom"
    assert failed.error_code is None  # 工具级失败没有 JSON-RPC 错误码

    empty = MCPToolResult.from_payload("read", {"isError": True})
    assert empty.error == "工具返回 isError=true"


def test_tool_result_reads_structured_content() -> None:
    result = MCPToolResult.from_payload(
        "tree", {"content": [], "structuredContent": {"total": 5}}
    )
    assert result.structured == {"total": 5}
    assert result.to_dict(with_content=False)["content_types"] == []


def test_content_and_server_info_round_trip() -> None:
    block = MCPContent.text_content("正文")
    assert MCPContent.from_dict(block.to_dict()).text == "正文"
    assert "mimeType" not in MCPContent(type="text", text="x").to_dict()

    info = MCPServerInfo.from_payload(init_result("srv", "1.2"))
    assert info.name == "srv"
    assert info.version == "1.2"
    assert info.protocol_version == MCP_PROTOCOL_VERSION
    assert info.to_dict()["capabilities"] == {"tools": {}, "resources": {}}

    # 服务端没按规范填 serverInfo 时，退化成 unknown 而不是崩掉
    assert MCPServerInfo.from_payload({}).name == "unknown"


def test_client_stats_serialization() -> None:
    stats = MCPClientStats(requests=3, tool_calls=1, duration=0.123456)
    stats.methods["tools/call"] = 1
    payload = stats.as_dict()
    assert payload["requests"] == 3
    assert payload["duration"] == 0.123
    assert payload["methods"] == {"tools/call": 1}


# ================================================================== 2. 客户端边界
def test_start_skips_noise_and_mismatched_ids() -> None:
    client, transport = make_client(
        [
            "这不是 JSON\n",  # 非法报文
            "\n",  # 空行
            line(build_notification("notifications/message", {})),  # 通知
            line(build_result(999, {})),  # id 不匹配
            "[1,2,3]\n",  # 非对象报文
            line(build_result(1, init_result())),
        ]
    )
    info = client.start()

    assert info.name == "fake-server"
    assert client.stats.invalid_messages == 3
    assert client.stats.skipped_messages == 1
    assert client.stats.notifications >= 2  # 发出 initialized + 收到一条通知
    assert any("notifications/initialized" in item for item in transport.sent)
    assert client.server_info is info


def test_server_error_response_becomes_mcp_error() -> None:
    client, _ = make_client(
        [
            line(build_result(1, init_result())),
            line(build_error(2, int(MCPErrorCode.METHOD_NOT_FOUND), "Method not found")),
        ]
    )
    client.start()
    with pytest.raises(MCPError) as excinfo:
        client.list_tools()
    assert excinfo.value.code == int(MCPErrorCode.METHOD_NOT_FOUND)


def test_gives_up_when_only_unmatched_messages_arrive() -> None:
    client, _ = make_client([line(build_result(index, {})) for index in range(100, 140)])
    with pytest.raises(MCPConnectionError) as excinfo:
        client.initialize()
    assert "无法匹配" in str(excinfo.value)


def test_call_tool_turns_protocol_error_into_failed_result() -> None:
    client, _ = make_client(
        [
            line(build_result(1, init_result())),
            line(build_error(2, int(MCPErrorCode.INVALID_PARAMS), "未知工具：nope")),
            line(build_result(3, {"content": [{"type": "text", "text": "ok"}], "isError": False})),
        ]
    )
    client.start()

    failed = client.call_tool("nope", {})
    assert failed.ok is False
    assert failed.error_code == int(MCPErrorCode.INVALID_PARAMS)
    assert failed.duration >= 0.0

    succeeded = client.call_tool("echo", {"text": "hi"})
    assert succeeded.ok is True
    assert succeeded.text == "ok"
    assert client.stats.tool_calls == 2
    assert client.stats.errors == 1


def test_ping_returns_false_on_server_error() -> None:
    client, _ = make_client(
        [
            line(build_result(1, init_result())),
            line(build_error(2, int(MCPErrorCode.INTERNAL_ERROR), "服务端内部错误")),
        ]
    )
    client.start()
    assert client.ping() is False


def test_call_tool_rejects_blank_name() -> None:
    client, _ = make_client([])
    with pytest.raises(ValueError, match="工具名"):
        client.call_tool("   ")


def test_request_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="request_timeout"):
        MCPClient(FakeTransport([]), request_timeout=0)


def test_requests_are_refused_after_close() -> None:
    client, _ = make_client([line(build_result(1, init_result()))])
    client.start()
    client.close()
    with pytest.raises(MCPConnectionError, match="已关闭"):
        client.ping()


def test_tool_and_resource_lists_are_cached_until_refreshed() -> None:
    client, _ = make_client(
        [
            line(build_result(1, init_result())),
            line(build_result(2, {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]})),
            line(build_result(3, {"tools": [{"name": "echo"}, {"name": "extra"}]})),
            line(build_result(4, {"resources": [{"uri": "file:///a"}]})),
        ]
    )
    client.start()

    assert [tool.name for tool in client.list_tools()] == ["echo"]
    assert client.tool_names == ["echo"]
    assert client.stats.methods["tools/list"] == 1  # 第二次走缓存，没有再发请求

    assert [tool.name for tool in client.list_tools(refresh=True)] == ["echo", "extra"]
    assert client.stats.methods["tools/list"] == 2

    assert [item.uri for item in client.list_resources()] == ["file:///a"]
    assert client.stats.methods["resources/list"] == 1


def test_read_resource_parses_contents() -> None:
    client, _ = make_client(
        [
            line(build_result(1, init_result())),
            line(
                build_result(
                    2, {"contents": [{"type": "text", "text": "hi", "uri": "file:///x"}]}
                )
            ),
        ]
    )
    client.start()
    contents = client.read_resource("file:///x")
    assert len(contents) == 1
    assert contents[0].text == "hi"
    assert contents[0].uri == "file:///x"


def test_describe_reports_transport_and_stats() -> None:
    client, transport = make_client([line(build_result(1, init_result()))])
    client.start()
    payload = client.describe()
    assert payload["transport"]["transport"] == "fake"
    assert payload["server_info"]["name"] == "fake-server"
    assert payload["stats"]["requests"] == 1
    assert transport.started is True


def test_child_env_puts_project_src_on_pythonpath() -> None:
    env = child_env(EXTRA="1")
    assert env["EXTRA"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONPATH"].split(os.pathsep)[0].endswith("src")


# ================================================================== 3. 服务端（纯函数）
def test_server_notification_returns_none() -> None:
    server = make_server()
    assert (
        server.handle(
            {
                "jsonrpc": JSONRPC_VERSION,
                "method": "notifications/initialized",
                "params": {},
            }
        )
        is None
    )


def test_server_initialize_reports_name_and_capabilities() -> None:
    server = make_server()
    response = call(server, 5, "initialize", {})
    assert response is not None
    assert response["id"] == 5
    result = response["result"]
    assert result["serverInfo"]["name"] == "codeagentx-filesystem"
    assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert set(result["capabilities"]) == {"tools", "resources"}


def test_server_lists_five_tools_with_schemas() -> None:
    server = make_server()
    response = call(server, 1, "tools/list", {})
    assert response is not None
    tools = response["result"]["tools"]
    assert [tool["name"] for tool in tools] == [
        "read_text_file",
        "list_directory",
        "directory_tree",
        "search_files",
        "get_file_info",
    ]
    assert all(tool["inputSchema"]["type"] == "object" for tool in tools)


def test_server_unknown_method_and_tool_error_codes() -> None:
    server = make_server()

    unknown_method = call(server, 6, "no/such", {})
    assert unknown_method is not None
    assert unknown_method["error"]["code"] == int(MCPErrorCode.METHOD_NOT_FOUND)

    unknown_tool = call(server, 7, "tools/call", {"name": "nope", "arguments": {}})
    assert unknown_tool is not None
    assert unknown_tool["error"]["code"] == int(MCPErrorCode.INVALID_PARAMS)

    missing_params = call(server, 8, "tools/call", {})
    assert missing_params is not None
    assert missing_params["error"]["code"] == int(MCPErrorCode.INVALID_PARAMS)


def test_server_tool_failures_stay_on_the_protocol() -> None:
    """工具级失败必须是 ``isError`` 的正常响应，而不是 JSON-RPC ``error``。"""
    server = make_server()

    escaped = call(server, 1, "tools/call", {"name": "read_text_file", "arguments": {"path": "../../.env"}})
    assert escaped is not None and "error" not in escaped
    text = escaped["result"]["content"][0]["text"]
    assert escaped["result"]["isError"] is True
    assert "越界" in text

    missing_argument = call(server, 2, "tools/call", {"name": "read_text_file", "arguments": {}})
    assert missing_argument is not None and "error" not in missing_argument
    assert missing_argument["result"]["isError"] is True
    assert "缺少必填参数" in missing_argument["result"]["content"][0]["text"]


def test_server_reads_lists_searches_and_inspects() -> None:
    server = make_server()

    readme, is_error = tool_text(server, "read_text_file", {"path": "README.md"})
    assert is_error is False
    assert "示例仓库" in readme

    listing, _ = tool_text(server, "list_directory", {"path": "."})
    assert "app/" in listing

    tree, _ = tool_text(server, "directory_tree", {"depth": 3})
    assert "app/" in tree

    found, _ = tool_text(server, "search_files", {"pattern": "*.py", "contains": "def "})
    assert "auth" in found and "def " in found

    info, _ = tool_text(server, "get_file_info", {"path": "README.md"})
    assert json.loads(info)["size_bytes"] > 0

    nothing, _ = tool_text(server, "search_files", {"pattern": "*.nope"})
    assert "未找到匹配" in nothing


def test_server_truncates_oversized_file() -> None:
    server = make_server(max_file_chars=10)
    text, is_error = tool_text(server, "read_text_file", {"path": "README.md"})
    assert is_error is False
    assert "已截断" in text


def test_server_rejects_non_file_and_directory_mismatch() -> None:
    server = make_server()

    directory_as_file, is_error = tool_text(server, "read_text_file", {"path": "app"})
    assert is_error is True
    assert "不是文件" in directory_as_file

    file_as_directory, is_error = tool_text(server, "list_directory", {"path": "README.md"})
    assert is_error is True
    assert "不是目录" in file_as_directory


def test_server_exposes_roots_as_resources() -> None:
    server = make_server()
    response = call(server, 1, "resources/list", {})
    assert response is not None
    resources = response["result"]["resources"]
    assert len(resources) == 1
    assert resources[0]["uri"].startswith("file://")

    root_uri = resources[0]["uri"]
    contents = call(server, 2, "resources/read", {"uri": root_uri})
    assert contents is not None
    assert "app/" in contents["result"]["contents"][0]["text"]

    # 资源读取走的是协议级错误（不是工具级 isError）：uri 用错属于"不会用 API"
    bad_scheme = call(server, 3, "resources/read", {"uri": "http://example.com"})
    assert bad_scheme is not None
    assert bad_scheme["error"]["code"] == int(MCPErrorCode.INVALID_PARAMS)


def test_filesystem_server_command_shape() -> None:
    command, args = filesystem_server_command([SAMPLE_REPO])
    assert args[:2] == ["-m", "codeagentx.protocols.mcp_filesystem"]
    assert args[2] == "--root"
    assert str(SAMPLE_REPO) in args

    with pytest.raises(ValueError, match="根目录"):
        filesystem_server_command([])


# ================================================================== 4. 真子进程联调
def test_stdio_client_handshake_and_tool_calls() -> None:
    with MCPClient.from_roots([SAMPLE_REPO], request_timeout=30.0) as client:
        info = client.server_info
        assert info is not None
        assert info.name == "codeagentx-filesystem"
        assert info.protocol_version == MCP_PROTOCOL_VERSION
        assert "tools" in info.capabilities

        tools = client.list_tools()
        assert len(tools) == 5
        assert all(tool.input_schema for tool in tools)

        readme = client.call_tool("read_text_file", {"path": "README.md"})
        assert readme.ok is True and "示例仓库" in readme.text

        listing = client.call_tool("list_directory", {"path": "."})
        assert listing.ok is True and "app/" in listing.text

        grep = client.call_tool("search_files", {"pattern": "*.py", "contains": "def "})
        assert grep.ok is True and "def " in grep.text

        info_payload = client.call_tool("get_file_info", {"path": "README.md"})
        assert info_payload.ok is True
        assert json.loads(info_payload.text)["size_bytes"] > 0

        resources = client.list_resources()
        assert len(resources) == 1 and resources[0].uri.startswith("file://")
        contents = client.read_resource(resources[0].uri)
        assert contents and "app/" in contents[0].text

        assert client.ping() is True

        # 关键回归点：服务端日志一旦写进 stdout，就会被客户端计成"非法报文"；
        # 同时"没有多余报文被跳过"保证响应与请求的 id 一一对应。
        assert client.stats.invalid_messages == 0
        assert client.stats.skipped_messages == 0
        assert client.stats.tool_calls >= 4


def test_stdio_client_survives_rejected_calls_and_exits_cleanly() -> None:
    client = MCPClient.from_roots([SAMPLE_REPO], request_timeout=30.0)
    client.start()

    escaped = client.call_tool("read_text_file", {"path": "../../.env"})
    assert escaped.ok is False
    assert "越界" in (escaped.error or "")
    assert client.ping() is True  # 越界不该把连接炸掉

    missing = client.call_tool("read_text_file", {"path": "不存在.txt"})
    assert missing.ok is False and "不存在" in (missing.error or "")

    unknown = client.call_tool("nope", {})
    assert unknown.ok is False
    assert unknown.error_code == int(MCPErrorCode.INVALID_PARAMS)

    bad_params = client.call_tool("read_text_file", {})
    assert bad_params.ok is False and "缺少必填参数" in (bad_params.error or "")

    # 日志确实在 stderr：既是"没污染 stdout"的正面证据，也是崩溃时的诊断来源
    assert "路径越界" in client.transport.stderr_tail

    process = client.transport._process  # 只为验证子进程真的退出了
    assert process is not None and process.poll() is None
    client.close()
    assert process.poll() is not None
    assert client.transport.closed is True

    with pytest.raises(MCPConnectionError):
        client.ping()
