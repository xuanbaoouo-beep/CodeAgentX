"""LLM 封装单元测试（全部离线，不产生网络调用）。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from codeagentx.config import Config
from codeagentx.core.exceptions import (
    ConfigError,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from codeagentx.core.llm import CodeAgentXLLM, MockLLM, build_llm
from codeagentx.core.message import Message


class RateLimitError(Exception):
    """模拟 openai.RateLimitError。"""


class AuthenticationError(Exception):
    """模拟 openai.AuthenticationError。"""


class APITimeoutError(Exception):
    """模拟 openai.APITimeoutError。"""


class _FakeCompletions:
    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._results:
            raise AssertionError("脚本化响应已耗尽")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _fake_response(
    content: str = "hello",
    *,
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    model: str = "fake-model",
    finish_reason: str = "stop",
) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _fake_tool_call(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class _FakeStreamingChat:
    """最小化模拟 openai 流式客户端（可作为 client 直接注入）。"""

    def __init__(self, text: str) -> None:
        self._text = text
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        assert kwargs["stream"] is True
        return [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=char))])
            for char in self._text
        ]


def _make_llm(results: list[Any], **kwargs: Any) -> tuple[CodeAgentXLLM, _FakeCompletions]:
    completions = _FakeCompletions(results)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = Config.from_env(load_dotenv_file=False, overrides={"llm_api_key": "sk-test"})
    llm = CodeAgentXLLM(config, client=client, **kwargs)
    return llm, completions


# ------------------------------------------------------------------ CodeAgentXLLM
def test_chat_returns_content_and_usage() -> None:
    llm, completions = _make_llm([_fake_response("你好")])

    response = llm.chat([Message.user("hi")])

    assert response.content == "你好"
    assert response.model == "fake-model"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7
    assert response.total_tokens == 18
    assert response.latency >= 0
    assert completions.calls[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_accumulates_stats() -> None:
    llm, _ = _make_llm([_fake_response(), _fake_response()])

    llm.chat("a")
    llm.chat("b")

    stats = llm.stats
    assert stats.calls == 2
    assert stats.total_tokens == 36
    assert stats.prompt_tokens == 22

    llm.reset_stats()
    assert llm.stats.calls == 0


def test_chat_forwards_tools_and_forces_auto_choice() -> None:
    llm, completions = _make_llm([_fake_response()])
    schema = [{"type": "function", "function": {"name": "echo", "parameters": {}}}]

    llm.chat("hi", tools=schema)

    assert completions.calls[0]["tools"] == schema
    assert completions.calls[0]["tool_choice"] == "auto"


def test_chat_omits_tools_when_empty() -> None:
    llm, completions = _make_llm([_fake_response()])
    llm.chat("hi", tools=[])
    assert "tools" not in completions.calls[0]


def test_chat_parses_tool_calls() -> None:
    raw = _fake_response(
        "",
        tool_calls=[_fake_tool_call("call_1", "echo", '{"text": "hi"}')],
        finish_reason="tool_calls",
    )
    llm, _ = _make_llm([raw])

    response = llm.chat("hi")

    assert response.has_tool_calls is True
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "echo"
    assert response.tool_calls[0].arguments == {"text": "hi"}


def test_chat_forwards_temperature_and_max_tokens_override() -> None:
    llm, completions = _make_llm([_fake_response()], temperature=0.1, max_tokens=100)

    llm.chat("hi", temperature=0.9, max_tokens=2048)

    assert completions.calls[0]["temperature"] == 0.9
    assert completions.calls[0]["max_tokens"] == 2048


def test_chat_forwards_response_format() -> None:
    llm, completions = _make_llm([_fake_response()])
    llm.chat("hi", response_format={"type": "json_object"})
    assert completions.calls[0]["response_format"] == {"type": "json_object"}


def test_chat_retries_on_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr("codeagentx.core.llm.time.sleep", lambda *_: None)
    llm, completions = _make_llm(
        [RateLimitError("429 Too Many Requests"), _fake_response("ok")],
        max_retries=2,
    )

    response = llm.chat("hi")

    assert response.content == "ok"
    assert len(completions.calls) == 2
    assert llm.stats.failed_calls == 1


def test_chat_does_not_retry_auth_error() -> None:
    llm, completions = _make_llm([AuthenticationError("invalid api key")] * 3, max_retries=3)

    with pytest.raises(LLMAuthError):
        llm.chat("hi")

    assert len(completions.calls) == 1
    assert llm.stats.failed_calls == 1


def test_chat_raises_after_exhausting_retries(monkeypatch) -> None:
    monkeypatch.setattr("codeagentx.core.llm.time.sleep", lambda *_: None)
    llm, completions = _make_llm([APITimeoutError("timeout")] * 5, max_retries=2)

    with pytest.raises(LLMTimeoutError):
        llm.chat("hi")

    assert len(completions.calls) == 3  # 首次 + 2 次重试
    assert llm.stats.failed_calls == 3
    assert llm.stats.calls == 0


def test_chat_empty_choices_raises_response_error() -> None:
    empty = SimpleNamespace(choices=[], usage=None, model="fake")
    llm, _ = _make_llm([empty])

    with pytest.raises(LLMResponseError):
        llm.chat("hi")


def test_chat_without_api_key_raises_auth_error(config: Config) -> None:
    llm = CodeAgentXLLM(config)
    with pytest.raises(LLMAuthError):
        llm.chat("hi")


def test_translate_error_falls_back_to_generic() -> None:
    translated = CodeAgentXLLM._translate_error(ValueError("未知错误"))
    assert isinstance(translated, LLMError)
    assert not isinstance(translated, (LLMAuthError, LLMRateLimitError, LLMTimeoutError))


def test_stream_chat_yields_incremental_text() -> None:
    config = Config.from_env(load_dotenv_file=False, overrides={"llm_api_key": "sk-test"})
    llm = CodeAgentXLLM(config, client=_FakeStreamingChat("abcd"))

    pieces = list(llm.stream_chat("hi"))

    assert "".join(pieces) == "abcd"
    assert len(pieces) == 4
    assert llm.stats.calls == 1


def test_stream_chat_wraps_errors(monkeypatch) -> None:
    class _Broken:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs: Any) -> Any:
            raise AuthenticationError("invalid api key")

    config = Config.from_env(load_dotenv_file=False, overrides={"llm_api_key": "sk-test"})
    llm = CodeAgentXLLM(config, client=_Broken())

    with pytest.raises(LLMAuthError):
        list(llm.stream_chat("hi"))

    assert llm.stats.failed_calls == 1


# ------------------------------------------------------------------ MockLLM
def test_mock_llm_scripted_then_default() -> None:
    llm = MockLLM(["第一", "第二"], default_response="兜底")

    assert llm.chat("a").content == "第一"
    assert llm.chat("b").content == "第二"
    assert llm.chat("c").content == "兜底"
    assert len(llm.calls) == 3


def test_mock_llm_accepts_dict_and_callable() -> None:
    llm = MockLLM(
        [
            {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "echo", "arguments": {"text": "x"}}],
            },
            lambda: "动态回复",
        ]
    )

    first = llm.chat("a")
    second = llm.chat("b")

    assert first.tool_calls[0].name == "echo"
    assert second.content == "动态回复"


def test_mock_llm_rejects_unknown_type() -> None:
    llm = MockLLM([123])
    with pytest.raises(LLMResponseError):
        llm.chat("a")


def test_mock_llm_stream_chunks() -> None:
    llm = MockLLM(["0123456789abcdef"])
    assert "".join(llm.stream_chat("hi")) == "0123456789abcdef"


# ------------------------------------------------------------------ build_llm
def test_build_llm_requires_key(config: Config) -> None:
    with pytest.raises(ConfigError):
        build_llm(config)


def test_build_llm_falls_back_to_mock(config: Config) -> None:
    llm = build_llm(config, allow_mock=True)
    assert isinstance(llm, MockLLM)


def test_build_llm_returns_real_impl_when_configured() -> None:
    config = Config.from_env(load_dotenv_file=False, overrides={"llm_api_key": "sk-test"})
    llm = build_llm(config)
    assert isinstance(llm, CodeAgentXLLM)
    assert llm.model_id == config.llm_model_id
