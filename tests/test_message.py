"""Message / ToolCall 单元测试。"""

from __future__ import annotations

import json

import pytest

from codeagentx.core.message import (
    Message,
    Role,
    ToolCall,
    estimate_messages_tokens,
    estimate_tokens,
    to_openai_messages,
)


def test_user_message_to_openai() -> None:
    payload = Message.user("你好").to_openai()
    assert payload == {"role": "user", "content": "你好"}


def test_role_accepts_plain_string() -> None:
    assert Message(role="assistant", content="hi").role is Role.ASSISTANT


def test_assistant_without_content_normalized_to_empty_string() -> None:
    payload = Message.assistant().to_openai()
    assert payload["content"] == ""
    assert "tool_calls" not in payload


def test_assistant_with_tool_calls_to_openai() -> None:
    call = ToolCall(id="call_1", name="echo", arguments={"text": "hi"})
    payload = Message.assistant(tool_calls=[call]).to_openai()

    assert payload["role"] == "assistant"
    assert payload["content"] == ""
    assert payload["tool_calls"][0]["id"] == "call_1"
    assert payload["tool_calls"][0]["type"] == "function"
    assert payload["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(payload["tool_calls"][0]["function"]["arguments"]) == {"text": "hi"}


def test_tool_message_to_openai() -> None:
    payload = Message.tool("结果", tool_call_id="call_1", name="echo").to_openai()
    assert payload == {
        "role": "tool",
        "content": "结果",
        "tool_call_id": "call_1",
        "name": "echo",
    }


def test_tool_call_from_openai_parses_json_string() -> None:
    raw = {
        "id": "call_9",
        "type": "function",
        "function": {"name": "echo", "arguments": '{"text": "hi", "repeat": 2}'},
    }
    call = ToolCall.from_openai(raw)
    assert call.id == "call_9"
    assert call.name == "echo"
    assert call.arguments == {"text": "hi", "repeat": 2}


def test_tool_call_from_openai_handles_object_form() -> None:
    class _Function:
        name = "echo"
        arguments = '{"text": "obj"}'

    class _Call:
        id = "call_obj"
        function = _Function()

    call = ToolCall.from_openai(_Call())
    assert call.name == "echo"
    assert call.arguments == {"text": "obj"}


def test_tool_call_invalid_json_keeps_raw_arguments() -> None:
    raw = {"id": "c", "function": {"name": "echo", "arguments": "{not-json"}}
    call = ToolCall.from_openai(raw)
    assert call.arguments == {}
    assert call.raw_arguments == "{not-json"


def test_tool_call_empty_arguments() -> None:
    call = ToolCall.from_openai({"id": "c", "function": {"name": "echo", "arguments": ""}})
    assert call.arguments == {}
    assert call.raw_arguments == ""


def test_tool_call_non_object_arguments_wrapped() -> None:
    call = ToolCall.from_openai({"id": "c", "function": {"name": "echo", "arguments": "[1, 2]"}})
    assert call.arguments == {"value": [1, 2]}


def test_message_from_openai() -> None:
    message = Message.from_openai(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "echo", "arguments": "{}"}}],
        }
    )
    assert message.role is Role.ASSISTANT
    assert message.text == ""
    assert message.tool_calls[0].name == "echo"


def test_message_dict_roundtrip() -> None:
    original = Message.assistant(
        "内容",
        tool_calls=[ToolCall(id="c1", name="echo", arguments={"a": 1})],
    )
    restored = Message.from_dict(original.to_dict())

    assert restored.role == original.role
    assert restored.content == original.content
    assert restored.tool_calls[0].arguments == {"a": 1}


def test_to_openai_messages_accepts_mixed_input() -> None:
    messages = to_openai_messages([Message.user("a"), {"role": "assistant", "content": "b"}, "c"])
    assert [item["role"] for item in messages] == ["user", "assistant", "user"]
    assert messages[2]["content"] == "c"


def test_to_openai_messages_wraps_single_string() -> None:
    assert to_openai_messages("hi") == [{"role": "user", "content": "hi"}]


def test_to_openai_messages_rejects_unknown_type() -> None:
    with pytest.raises(TypeError):
        to_openai_messages([123])


def test_estimate_tokens_monotonic_and_nonzero() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world" * 20)
    assert 0 < short < long


def test_estimate_tokens_counts_cjk_per_char() -> None:
    assert estimate_tokens("中文测试") == 4


def test_estimate_messages_tokens_includes_tool_calls() -> None:
    plain = estimate_messages_tokens([Message.user("hi")])
    with_call = estimate_messages_tokens(
        [Message.assistant(tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "hi"})])]
    )
    assert with_call > plain
