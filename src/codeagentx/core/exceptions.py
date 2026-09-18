"""CodeAgentX 异常体系。

所有自定义异常均继承 ``CodeAgentXError``，上层可统一捕获。
"""

from __future__ import annotations


class CodeAgentXError(Exception):
    """项目所有异常的基类。"""

    def __init__(self, message: str = "", *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} | detail={self.detail}"
        return self.message


# ---------------------------------------------------------------- 配置
class ConfigError(CodeAgentXError):
    """配置缺失或非法。"""


# ---------------------------------------------------------------- 审查目标
class TargetError(CodeAgentXError):
    """审查目标无法解析成可审查的工作区（既不是已存在的路径，也不像 ``owner/repo``）。"""


# ---------------------------------------------------------------- LLM
class LLMError(CodeAgentXError):
    """LLM 调用相关错误基类。"""


class LLMAuthError(LLMError):
    """鉴权失败或未配置密钥（不可重试）。"""


class LLMRateLimitError(LLMError):
    """触发限流（可重试）。"""


class LLMTimeoutError(LLMError):
    """请求超时（可重试）。"""


class LLMResponseError(LLMError):
    """响应结构异常（不可重试）。"""


# ---------------------------------------------------------------- 工具
class ToolError(CodeAgentXError):
    """工具相关错误基类。"""


class ToolNotFoundError(ToolError):
    """工具未注册。"""


class ToolValidationError(ToolError):
    """工具入参校验失败。"""


class ToolExecutionError(ToolError):
    """工具执行过程中抛错。"""


class SecurityViolationError(ToolError):
    """触碰安全沙箱策略。"""


# ---------------------------------------------------------------- Agent
class AgentError(CodeAgentXError):
    """Agent 相关错误基类。"""


class MaxIterationsExceeded(AgentError):
    """超过最大迭代轮次。"""


class AgentOutputError(AgentError):
    """Agent 输出无法解析成约定结构（例如不符合 JSON 契约）。"""


# ---------------------------------------------------------------- 协议
class ProtocolError(CodeAgentXError):
    """外部协议交互错误基类（MCP / A2A）。"""


class MCPError(ProtocolError):
    """MCP 协议错误：报文非法、方法不存在，或服务端返回了错误对象。"""

    def __init__(self, message: str = "", *, detail: str | None = None, code: int | None = None) -> None:
        super().__init__(message, detail=detail)
        #: JSON-RPC 错误码（协议级错误才有）
        self.code = code

    def __str__(self) -> str:
        if self.code is None:
            return super().__str__()
        prefix = f"[{self.code}] "
        return f"{prefix}{super().__str__()}"


class MCPConnectionError(ProtocolError):
    """MCP 连接不可用：进程启动失败、管道断开、等待响应超时。"""


class A2AError(ProtocolError):
    """A2A 协议级错误：消息/名片非法，或 API 使用方式不对。

    "运行时失败"（目标 Agent 不存在、处理器自己抛异常）**不走异常**，
    而是返回 ``ok=False`` 的响应——与 MCP 的 ``isError`` 同款口径。
    """


class GitHubError(ProtocolError):
    """GitHub REST API 交互错误基类。"""


class GitHubAuthError(GitHubError):
    """令牌缺失、无效或权限不足（401，或 403 但不是限流）。"""


class GitHubNotFoundError(GitHubError):
    """仓库、路径或 PR 不存在（404）。"""


class GitHubRateLimitError(GitHubError):
    """触发 GitHub 速率限制（403 且 ``x-ratelimit-remaining=0``，或 429）。"""


class GitHubResponseError(GitHubError):
    """响应不符合预期：非 2xx 且不属于上述情形，或 JSON 结构异常。"""


class GitHubArchiveError(GitHubError):
    """仓库归档下载/解压失败：不是合法 zip、解压超出上限。"""


# ---------------------------------------------------------------- 其他
class RAGError(CodeAgentXError):
    """RAG 检索相关错误。"""


class EvaluationError(CodeAgentXError):
    """评估流程相关错误。"""
