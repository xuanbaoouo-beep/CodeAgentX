"""Agent 基类与工具调用循环单元测试。"""

from __future__ import annotations

import pytest

from codeagentx.core.agent import Agent, AgentResult
from codeagentx.core.llm import MockLLM
from codeagentx.core.message import Role
from codeagentx.tools.registry import ToolRegistry


class RecordingAgent(Agent):
    """最小可测 Agent：直接把 tool_loop 暴露出来。"""

    name = "recording"

    def __init__(
        self,
        llm,
        *,
        system_prompt: str | None = "系统提示",
        tools: ToolRegistry | None = None,
        max_iterations: int = 4,
    ) -> None:
        super().__init__(
            llm,
            system_prompt=system_prompt,
            tools=tools,
            max_iterations=max_iterations,
        )

    def run(self, task: str, **kwargs) -> AgentResult:
        return self.tool_loop(self._seed_messages(task))


def _tool_call(name: str = "echo", arguments: dict | None = None, call_id: str = "call_1") -> dict:
    return {
        "content": "",
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments or {"text": "hi"}}],
    }


# ------------------------------------------------------------------ 构造
def test_max_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        RecordingAgent(MockLLM(), max_iterations=0)


def test_agent_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Agent(MockLLM())  # type: ignore[abstract]


# ------------------------------------------------------------------ 基本对话
def test_run_without_tools_single_iteration() -> None:
    llm = MockLLM(["你好"])
    agent = RecordingAgent(llm)

    result = agent.run("hi")

    assert result.success is True
    assert result.output == "你好"
    assert result.iterations == 1
    assert result.tool_calls == []
    assert result.error is None
    assert llm.calls[0]["tools"] is None


def test_system_prompt_is_first_message() -> None:
    llm = MockLLM(["ok"])
    agent = RecordingAgent(llm, system_prompt="你是审查员")

    agent.run("hi")

    messages = llm.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "你是审查员"}
    assert messages[1] == {"role": "user", "content": "hi"}


def test_agent_without_system_prompt_skips_system_message() -> None:
    llm = MockLLM(["ok"])
    agent = RecordingAgent(llm, system_prompt=None)

    agent.run("hi")

    assert [item["role"] for item in llm.calls[0]["messages"]] == ["user"]


def test_usage_delta_only_counts_current_run() -> None:
    llm = MockLLM(["a", "b"])
    agent = RecordingAgent(llm)

    first = agent.run("one")
    second = agent.run("two")

    assert first.usage["calls"] == 1
    assert second.usage["calls"] == 1
    assert llm.stats.calls == 2


# ------------------------------------------------------------------ 工具调用
def test_tool_call_then_final_answer(registry: ToolRegistry) -> None:
    llm = MockLLM([_tool_call("echo", {"text": "hi"}), "最终答案"])
    agent = RecordingAgent(llm, tools=registry)

    result = agent.run("请回显 hi")

    assert result.success is True
    assert result.output == "最终答案"
    assert result.iterations == 2

    assert len(result.tool_calls) == 1
    record = result.tool_calls[0]
    assert record["name"] == "echo"
    assert record["success"] is True
    assert record["output"] == "hi"

    # 工具结果以 tool 消息回填给模型
    second_messages = llm.calls[1]["messages"]
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["content"] == "hi"
    assert second_messages[-1]["tool_call_id"] == "call_1"

    # 工具 Schema 透传给了模型
    assert {item["function"]["name"] for item in llm.calls[0]["tools"]} == {
        "boom",
        "echo",
        "guarded",
    }


def test_tool_failure_is_fed_back(registry: ToolRegistry) -> None:
    llm = MockLLM([_tool_call("boom", {}), "已处理失败"])
    agent = RecordingAgent(llm, tools=registry)

    result = agent.run("触发失败")

    assert result.success is True
    assert result.tool_calls[0]["success"] is False
    assert llm.calls[1]["messages"][-1]["content"].startswith("[工具执行失败]")


def test_unknown_tool_is_fed_back(registry: ToolRegistry) -> None:
    llm = MockLLM([_tool_call("not-exist", {}), "好的"])
    agent = RecordingAgent(llm, tools=registry)

    result = agent.run("调用不存在的工具")

    assert result.tool_calls[0]["success"] is False
    assert result.tool_calls[0]["error_type"] == "ToolNotFoundError"


def test_agent_without_registry_reports_tool_unavailable() -> None:
    llm = MockLLM([_tool_call("echo"), "好的"])
    agent = RecordingAgent(llm, tools=None)

    result = agent.run("调用工具")

    assert result.tool_calls[0]["success"] is False
    assert result.tool_calls[0]["error_type"] == "ToolNotFoundError"


def test_allowed_tools_filters_schema(registry: ToolRegistry) -> None:
    llm = MockLLM(["ok"])
    agent = RecordingAgent(llm, tools=registry)

    agent.tool_loop(agent._seed_messages("hi"), allowed_tools=["echo"])  # noqa: SLF001

    schemas = llm.calls[0]["tools"]
    assert [item["function"]["name"] for item in schemas] == ["echo"]


# ------------------------------------------------------------------ 终止条件
def test_max_iterations_terminates_loop(registry: ToolRegistry) -> None:
    looping = _tool_call("echo")
    llm = MockLLM([looping], default_response=looping)
    agent = RecordingAgent(llm, tools=registry, max_iterations=2)

    result = agent.run("陷入循环")

    assert result.success is False
    assert result.iterations == 2
    assert "最大迭代轮次" in (result.error or "")
    # 2 轮工具调用 + 1 次「预算已尽，请收尾」的强制收敛调用（不给工具）
    assert len(llm.calls) == 3
    assert llm.calls[-1]["tools"] is None
    # 强制收敛这一轮仍只回工具调用、没有正文 → 判为未收敛，且不再执行工具
    assert len(result.tool_calls) == 2


def test_forced_convergence_uses_last_round_evidence(registry: ToolRegistry) -> None:
    """最后一轮只调了工具时，要用一次不挂工具的调用把已有证据收成结论。"""
    looping = _tool_call("echo")
    llm = MockLLM([looping, looping, "最终结论"])
    agent = RecordingAgent(llm, tools=registry, max_iterations=2)

    result = agent.run("给我结论")

    assert result.success is True
    assert result.output == "最终结论"
    assert result.metadata["forced_convergence"] is True
    assert llm.calls[-1]["tools"] is None
    # 强制收敛轮不执行工具，且历史里不能留"有 tool_calls 却没有工具结果"的助手消息
    assert len(result.tool_calls) == 2
    assert [message.role for message in result.messages][-2:] == [Role.USER, Role.ASSISTANT]
    assert result.messages[-1].tool_calls == []
    assert "工具调用轮次已用尽" in (result.messages[-2].content or "")


def test_truncated_output_is_flagged(registry: ToolRegistry) -> None:
    """被 max_tokens 截断要显式标记，否则只会表现为"JSON 解析失败"。"""
    llm = MockLLM([{"content": '{"summary": "被截', "finish_reason": "length"}])
    agent = RecordingAgent(llm, tools=registry)

    result = agent.run("给我报告")

    assert result.success is True
    assert result.metadata["truncated"] is True


# ------------------------------------------------------------------ 历史管理
def test_history_excludes_system_prompt(registry: ToolRegistry) -> None:
    llm = MockLLM([_tool_call("echo"), "完成"])
    agent = RecordingAgent(llm, tools=registry)

    result = agent.run("hi")

    assert all(message.role is not Role.SYSTEM for message in agent.history)
    assert len(agent.history) == len(result.messages) - 1


def test_history_reused_on_second_run() -> None:
    llm = MockLLM(["a", "b"])
    agent = RecordingAgent(llm)

    agent.run("first")
    agent.run("second")

    second_messages = llm.calls[1]["messages"]
    assert [item["role"] for item in second_messages] == ["system", "user", "assistant", "user"]
    assert second_messages[-1]["content"] == "second"


def test_reset_clears_history() -> None:
    llm = MockLLM(["a", "b"])
    agent = RecordingAgent(llm)

    agent.run("first")
    agent.reset()
    agent.run("second")

    assert [item["role"] for item in llm.calls[1]["messages"]] == ["system", "user"]


# ------------------------------------------------------------------ 结果序列化
def test_agent_result_to_dict(registry: ToolRegistry) -> None:
    llm = MockLLM([_tool_call("echo"), "完成"])
    agent = RecordingAgent(llm, tools=registry)

    payload = agent.run("hi").to_dict()

    assert payload["success"] is True
    assert payload["iterations"] == 2
    assert isinstance(payload["messages"], list)
    assert payload["messages"][0]["role"] in {"system", "user", "assistant", "tool"}
