"""协议层：MCP 客户端/服务端与 A2A 通信。

依赖方向
--------
本包只依赖 ``core`` 与 ``tools``（沙箱路径守卫），**不依赖** ``agents`` /
``orchestrator`` / ``rag`` / ``context``——协议是通道，不该知道上层业务。
A2A 路由只用到 ``core.agent`` 的 ``AgentResult``（返回值归一化），
并不认识 ``agents`` 里那七个具体角色。
"""

from codeagentx.protocols.a2a import (
    A2A_PROTOCOL_VERSION,
    A2A_ROLE_AGENT,
    A2A_ROLE_USER,
    A2AErrorCode,
    A2AMessage,
    A2ANetwork,
    A2ANetworkStats,
    A2APart,
    A2AResponse,
    A2ATask,
    A2ATaskState,
    AgentCard,
    AgentSkill,
    agent_handler,
    card_from_agent,
)
from codeagentx.protocols.github_archive import (
    ArchiveExtractResult,
    extract_zipball,
)
from codeagentx.protocols.github_client import (
    GitHubClient,
    GitHubClientStats,
    GitHubFile,
    GitHubPullRequest,
    GitHubPullRequestFile,
    GitHubRepoRef,
    GitHubRepository,
    GitHubTree,
    GitHubTreeEntry,
    normalize_repo_path,
)
from codeagentx.protocols.mcp_client import (
    DEFAULT_REQUEST_TIMEOUT,
    MCPClient,
    MCPClientStats,
    MCPTransport,
    StdioTransport,
    child_env,
    default_python,
)
from codeagentx.protocols.mcp_filesystem import (
    FilesystemMCPServer,
    filesystem_server_command,
    filesystem_tools,
)
from codeagentx.protocols.mcp_github import (
    GitHubMCPServer,
    github_server_command,
    github_tools,
)
from codeagentx.protocols.mcp_protocol import (
    CLIENT_CAPABILITIES,
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
from codeagentx.protocols.mcp_server import MCPServerBase, ToolOutcome

__all__ = [
    "A2A_PROTOCOL_VERSION",
    "A2A_ROLE_AGENT",
    "A2A_ROLE_USER",
    "A2AErrorCode",
    "A2AMessage",
    "A2ANetwork",
    "A2ANetworkStats",
    "A2APart",
    "A2AResponse",
    "A2ATask",
    "A2ATaskState",
    "AgentCard",
    "AgentSkill",
    "ArchiveExtractResult",
    "CLIENT_CAPABILITIES",
    "DEFAULT_REQUEST_TIMEOUT",
    "JSONRPC_VERSION",
    "MCP_PROTOCOL_VERSION",
    "FilesystemMCPServer",
    "GitHubClient",
    "GitHubClientStats",
    "GitHubFile",
    "GitHubMCPServer",
    "GitHubPullRequest",
    "GitHubPullRequestFile",
    "GitHubRepoRef",
    "GitHubRepository",
    "GitHubTree",
    "GitHubTreeEntry",
    "MCPClient",
    "MCPClientStats",
    "MCPContent",
    "MCPErrorCode",
    "MCPResourceSpec",
    "MCPServerBase",
    "MCPServerInfo",
    "MCPToolResult",
    "MCPToolSpec",
    "MCPTransport",
    "StdioTransport",
    "ToolOutcome",
    "agent_handler",
    "build_error",
    "build_notification",
    "build_request",
    "build_result",
    "card_from_agent",
    "child_env",
    "decode_message",
    "default_python",
    "encode_message",
    "extract_zipball",
    "filesystem_server_command",
    "filesystem_tools",
    "github_server_command",
    "github_tools",
    "is_notification",
    "normalize_repo_path",
    "parse_result",
]
