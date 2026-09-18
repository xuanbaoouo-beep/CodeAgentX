"""Agent 基类：定义统一的 ``run`` 契约，并提供可复用的工具调用循环。

设计说明
--------
- 子类只需实现 :meth:`Agent.run`；工具调用循环（多轮 reasoning + acting）
  由基类 :meth:`Agent.tool_loop` 提供，避免每个 Agent 重复实现。
- :attr:`Agent._history` 只保存**不含 system prompt** 的对话，
  以便多次 ``run`` 复用上下文而不重复注入系统提示。
- 循环有硬性最大轮次，绝不无限执行（对应架构约束：多 Agent 必须有终止条件）。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from codeagentx.core.llm import BaseLLM, UsageStats
from codeagentx.core.logger import get_logger, log_event
from codeagentx.core.message import Message, Role
from codeagentx.tools.base import ToolResult
from codeagentx.tools.registry import ToolRegistry

logger = get_logger("core.agent")

#: 工具轮次耗尽时的收尾指令。
#: 放在核心层而不是 ``prompts`` 层，是因为它约束的是**循环协议**（不要再请求工具），
#: 与具体任务的输出契约无关；各 Agent 的 JSON 契约由自己的 system prompt 负责。
CONVERGENCE_PROMPT = (
    "工具调用轮次已用尽，本轮不再提供工具。请立即基于上文已经取到的工具结果，"
    "按你的输出契约给出最终结果；不要再请求工具，也不要解释为什么停止调用工具。"
)


@dataclass
class AgentResult:
    """一次 Agent 执行的完整结果。"""

    output: str
    success: bool = True
    error: str | None = None
    iterations: int = 0
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "iterations": self.iterations,
            "messages": [message.to_dict() for message in self.messages],
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "metadata": self.metadata,
        }


class Agent(ABC):
    """所有智能体的基类。"""

    #: 默认名称，子类可覆盖
    name: str = "agent"

    def __init__(
        self,
        llm: BaseLLM,
        *,
        name: str | None = None,
        system_prompt: str | None = None,
        tools: ToolRegistry | None = None,
        max_iterations: int = 8,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations 必须 >= 1")
        self.llm = llm
        if name:
            self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_iterations = max_iterations
        self._history: list[Message] = []

    # ------------------------------------------------------------ 生命周期
    @property
    def history(self) -> list[Message]:
        """历史对话副本（不含 system prompt）。"""
        return list(self._history)

    def reset(self) -> None:
        """清空对话历史，便于复用同一个 Agent 实例处理新任务。"""
        self._history.clear()

    @abstractmethod
    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """执行一次任务。"""

    # ------------------------------------------------------------ 供子类复用
    def _seed_messages(self, task: str) -> list[Message]:
        """构造本轮起始消息：system + 历史 + 当前任务。"""
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message.system(self.system_prompt))
        messages.extend(self._history)
        messages.append(Message.user(task))
        return messages

    def _tool_schemas(self, allowed_tools: Sequence[str] | None = None) -> list[dict[str, Any]] | None:
        if self.tools is None or len(self.tools) == 0:
            return None
        schemas = self.tools.to_openai_tools(allowed_tools)
        return schemas or None

    def tool_loop(
        self,
        messages: list[Message],
        *,
        allowed_tools: Sequence[str] | None = None,
        max_iterations: int | None = None,
    ) -> AgentResult:
        """通用工具调用循环：LLM 推理 → 执行工具 → 回填结果 → 再推理。

        终止条件：模型不再请求工具，或达到最大迭代轮次。
        """
        limit = self.max_iterations if max_iterations is None else max_iterations
        working = list(messages)
        schemas = self._tool_schemas(allowed_tools)
        usage_before = self.llm.stats.snapshot()
        tool_records: list[dict[str, Any]] = []
        last_content = ""
        last_finish_reason = ""
        iterations = 0

        while iterations < limit:
            iterations += 1
            response = self.llm.chat(working, tools=schemas)
            last_content = response.content
            last_finish_reason = response.finish_reason
            working.append(Message.assistant(response.content, tool_calls=response.tool_calls))

            if not response.tool_calls:
                return self._finish(
                    working,
                    last_content,
                    usage_before,
                    tool_records,
                    iterations,
                    success=True,
                    finish_reason=last_finish_reason,
                )

            for call in response.tool_calls:
                result = self._invoke_tool(call.name, call.arguments)
                tool_records.append({"name": call.name, "arguments": call.arguments, **result.to_dict()})
                working.append(Message.tool(result.to_text(), tool_call_id=call.id, name=call.name))

        error = f"达到最大迭代轮次（{limit}），任务未收敛"
        log_event(
            logger,
            "agent_max_iterations",
            level=logging.WARNING,
            agent=self.name,
            iterations=iterations,
        )
        # 走到这里说明**最后一轮仍在请求工具**：刚执行完的工具结果模型还没读到，
        # 直接结束等于白扔一轮证据、交出一个空结果（真实模型上很容易复现）。
        # 因此无条件补一次**不挂工具**的调用强制收敛——此时模型手里已有全部工具结果，
        # 只能产出正文；正文非空即视为收敛（并在 metadata 里如实标记是被强制的），
        # 仍为空才判未收敛。注意不能只判"这一轮正文为空"：真实模型常常一边说
        # "我再看看 X" 一边继续调工具，那种过程话不是结论。
        working.append(Message.user(CONVERGENCE_PROMPT))
        forced = self.llm.chat(working, tools=None)
        # 不把这一轮的 tool_calls 写进历史：没有对应的工具结果，
        # 留着会让下次复用历史的请求违反 OpenAI 的消息配对要求。
        working.append(Message.assistant(forced.content))
        if (forced.content or "").strip():
            log_event(
                logger,
                "agent_forced_convergence",
                level=logging.WARNING,
                agent=self.name,
                iterations=iterations,
            )
            result = self._finish(
                working,
                forced.content or "",
                usage_before,
                tool_records,
                iterations,
                success=True,
                finish_reason=forced.finish_reason,
            )
            result.metadata["forced_convergence"] = True
            return result
        return self._finish(
            working,
            forced.content or "",
            usage_before,
            tool_records,
            iterations,
            success=False,
            error=error,
            finish_reason=forced.finish_reason,
        )

    def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self.tools is None:
            return ToolResult.fail(
                f"工具 {name} 不可用：当前 Agent 未挂载任何工具",
                error_type="ToolNotFoundError",
            )
        return self.tools.execute(name, arguments)

    def _finish(
        self,
        working: list[Message],
        output: str,
        usage_before: dict[str, Any],
        tool_records: list[dict[str, Any]],
        iterations: int,
        *,
        success: bool,
        error: str | None = None,
        finish_reason: str = "",
    ) -> AgentResult:
        """收尾：落历史、算增量用量、组装结果。"""
        self._history = [message for message in working if message.role != Role.SYSTEM]
        usage = usage_delta(usage_before, self.llm.stats.snapshot())
        # 「输出被 max_tokens 截断」必须显式暴露：否则半个 JSON 只会表现为"解析失败"，
        # 让人误以为模型不会按契约输出（实际是预算不够），排查方向完全相反。
        truncated = finish_reason == "length"
        if truncated:
            log_event(
                logger,
                "llm_output_truncated",
                level=logging.WARNING,
                agent=self.name,
                iterations=iterations,
            )
        log_event(
            logger,
            "agent_finished",
            agent=self.name,
            success=success,
            iterations=iterations,
            tool_calls=len(tool_records),
            total_tokens=usage["total_tokens"],
        )
        result = AgentResult(
            output=output,
            success=success,
            error=error,
            iterations=iterations,
            messages=working,
            tool_calls=tool_records,
            usage=usage,
        )
        if truncated:
            result.metadata["truncated"] = True
        return result


def usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """计算两次用量快照的差值。

    多阶段 Agent（如 Plan-and-Solve 的规划 + 逐步执行）需要把若干次调用
    合并成一份增量用量，因此这里作为公开工具函数导出。
    """
    keys = ("calls", "failed_calls", "prompt_tokens", "completion_tokens", "total_tokens")
    delta: dict[str, Any] = {key: after.get(key, 0) - before.get(key, 0) for key in keys}
    delta["latency"] = round(after.get("total_latency", 0.0) - before.get("total_latency", 0.0), 3)
    return delta


#: 不产生 Token 消耗的角色（确定性 Agent）使用的零用量，字段与 :func:`usage_delta` 一致
ZERO_USAGE: dict[str, Any] = {
    "calls": 0,
    "failed_calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "latency": 0.0,
}


__all__ = ["ZERO_USAGE", "Agent", "AgentResult", "CONVERGENCE_PROMPT", "UsageStats", "usage_delta"]
