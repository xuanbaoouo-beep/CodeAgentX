"""LLM 封装：统一入口，屏蔽 provider 差异，内置重试、超时与用量统计。

业务层禁止直接调用 ``openai`` SDK，一律通过 :class:`BaseLLM` 的实例，
以保证 Token 统计、重试策略、Mock 切换三件事只有一个实现点。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from codeagentx.config import Config, get_config
from codeagentx.core.exceptions import (
    ConfigError,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from codeagentx.core.logger import get_logger, log_event
from codeagentx.core.message import Message, ToolCall, estimate_tokens, to_openai_messages

logger = get_logger("core.llm")


@dataclass
class LLMResponse:
    """一次 LLM 调用的结果。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class UsageStats:
    """累计用量统计。

    ``failed_calls`` 统计的是**失败尝试次数**（含最终被重试成功的那些），
    因此 ``calls + failed_calls`` 等于实际发起的请求总数。
    """

    calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_latency": round(self.total_latency, 3),
        }


class BaseLLM(ABC):
    """LLM 抽象基类。

    子类必须实现 :meth:`chat` 与 :meth:`stream_chat`；
    用量统计由基类统一维护。
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._stats = UsageStats()

    @property
    def stats(self) -> UsageStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = UsageStats()

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]] | str,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """单轮对话（非流式）。"""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]] | str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式对话，逐段产出文本增量。"""

    # ------------------------------------------------------------ 内部工具
    @staticmethod
    def _to_openai_messages(messages: Any) -> list[dict[str, Any]]:
        return to_openai_messages(messages)

    def _record(self, response: LLMResponse, *, attempt: int = 1) -> None:
        self._stats.calls += 1
        self._stats.prompt_tokens += response.prompt_tokens
        self._stats.completion_tokens += response.completion_tokens
        self._stats.total_tokens += response.total_tokens
        self._stats.total_latency += response.latency
        log_event(
            logger,
            "llm_call",
            model=response.model or self.model_id,
            attempt=attempt,
            latency=round(response.latency, 3),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            tool_calls=len(response.tool_calls),
            finish_reason=response.finish_reason,
        )


class CodeAgentXLLM(BaseLLM):
    """基于 OpenAI 兼容协议的 LLM 实现（ModelScope / DashScope / OpenAI / 本地 vLLM 均可）。"""

    def __init__(
        self,
        config: Config | None = None,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        client: Any | None = None,
    ) -> None:
        self._config = config or get_config()
        super().__init__(model_id or self._config.llm_model_id)

        self.api_key = self._config.llm_api_key if api_key is None else api_key
        self.base_url = base_url or self._config.llm_base_url
        self.temperature = self._config.llm_temperature if temperature is None else temperature
        self.max_tokens = self._config.llm_max_tokens if max_tokens is None else max_tokens
        self.timeout = self._config.llm_timeout if timeout is None else timeout
        self.max_retries = self._config.llm_max_retries if max_retries is None else max_retries
        self._client = client

    # ------------------------------------------------------------ 客户端
    @property
    def client(self) -> Any:
        """惰性创建 OpenAI 客户端（缺少密钥时才报错，便于离线测试）。"""
        if self._client is None:
            if not self.api_key.strip():
                raise LLMAuthError(
                    "未配置 LLM_API_KEY",
                    detail="请复制 .env.example 为 .env 并填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID",
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    # ------------------------------------------------------------ 对话
    def chat(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]] | str,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._to_openai_messages(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format:
            kwargs["response_format"] = response_format
        return self._call_with_retry(**kwargs)

    def stream_chat(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]] | str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._to_openai_messages(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        start = time.perf_counter()
        collected: list[str] = []
        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    collected.append(piece)
                    yield piece
        except Exception as exc:  # noqa: BLE001 - 统一转换为项目内异常
            self._stats.failed_calls += 1
            raise self._translate_error(exc) from exc

        latency = time.perf_counter() - start
        content = "".join(collected)
        # 流式响应默认不返回 usage，按字符估算 completion tokens（真实用量以非流式为准）
        self._stats.calls += 1
        estimated = estimate_tokens(content)
        self._stats.completion_tokens += estimated
        self._stats.total_tokens += estimated
        self._stats.total_latency += latency
        log_event(
            logger,
            "llm_stream_call",
            model=self.model_id,
            latency=round(latency, 3),
            completion_tokens_estimated=estimated,
            chars=len(content),
        )

    # ------------------------------------------------------------ 重试
    def _call_with_retry(self, **kwargs: Any) -> LLMResponse:
        last_error: LLMError | None = None
        for attempt in range(1, self.max_retries + 2):
            start = time.perf_counter()
            try:
                raw = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - 统一转换为项目内异常
                last_error = self._translate_error(exc)
                self._stats.failed_calls += 1
                retryable = isinstance(last_error, (LLMRateLimitError, LLMTimeoutError))
                log_event(
                    logger,
                    "llm_call_failed",
                    level=logging.WARNING,
                    attempt=attempt,
                    error_type=type(last_error).__name__,
                    error=str(last_error),
                    retryable=retryable,
                )
                if not retryable or attempt > self.max_retries:
                    raise last_error from exc
                time.sleep(min(2 ** (attempt - 1), 8))
                continue

            response = self._parse_response(raw, time.perf_counter() - start)
            self._record(response, attempt=attempt)
            return response

        raise last_error or LLMError("LLM 调用失败")  # pragma: no cover - 理论不可达

    @staticmethod
    def _parse_response(raw: Any, latency: float) -> LLMResponse:
        choices = getattr(raw, "choices", None) or []
        if not choices:
            raise LLMResponseError("LLM 返回内容为空（缺少 choices）")

        choice = choices[0]
        message = getattr(choice, "message", None)
        content = (getattr(message, "content", None) or "") if message is not None else ""
        tool_calls = [
            ToolCall.from_openai(item) for item in (getattr(message, "tool_calls", None) or [])
        ]

        usage = getattr(raw, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion_tokens = (
            int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        )
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=getattr(raw, "model", "") or "",
            finish_reason=getattr(choice, "finish_reason", "") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
        )

    @staticmethod
    def _translate_error(exc: Exception) -> LLMError:
        """把 SDK 异常翻译成项目内异常，并标记是否可重试。"""
        if isinstance(exc, LLMError):  # 已是项目内异常，直接透传，避免语义丢失
            return exc

        name = type(exc).__name__
        text = str(exc)

        if name in {"AuthenticationError", "PermissionDeniedError"}:
            return LLMAuthError("LLM 鉴权失败", detail=text)
        if name == "RateLimitError":
            return LLMRateLimitError("LLM 触发限流", detail=text)
        if name == "APIStatusError":
            if "429" in text:
                return LLMRateLimitError("LLM 触发限流", detail=text)
            if "401" in text or "403" in text:
                return LLMAuthError("LLM 鉴权失败", detail=text)
            return LLMError("LLM 返回异常状态码", detail=text)
        if name in {
            "APITimeoutError",
            "APIConnectionTimeoutError",
            "Timeout",
            "ReadTimeout",
            "ConnectTimeout",
        }:
            return LLMTimeoutError("LLM 请求超时", detail=text)
        if name in {"APIConnectionError", "ConnectError"}:
            return LLMError("LLM 网络连接失败", detail=text)

        lowered = text.lower()
        if "429" in text or "rate limit" in lowered:
            return LLMRateLimitError("LLM 触发限流", detail=text)
        if "timeout" in lowered or "timed out" in lowered:
            return LLMTimeoutError("LLM 请求超时", detail=text)
        if (
            "401" in text
            or "403" in text
            or "unauthorized" in lowered
            or "invalid api key" in lowered
        ):
            return LLMAuthError("LLM 鉴权失败", detail=text)
        return LLMError(f"LLM 调用失败：{name}", detail=text)


class MockLLM(BaseLLM):
    """离线可用的脚本化 LLM，用于单元测试、CI 与无密钥演示。

    不产生任何网络调用。行为：
    - 按构造时传入的 ``responses`` 顺序依次返回，超出后返回 ``default_response``；
    - 元素可以是 ``str``、``LLMResponse``、``dict`` 或零参可调用对象；
    - 每次调用都会把入参与 kwargs 记入 ``calls``，供断言使用。
    """

    def __init__(
        self,
        responses: Sequence[Any] | None = None,
        *,
        model_id: str = "mock-llm",
        default_response: Any = "（MockLLM 默认回复）",
    ) -> None:
        super().__init__(model_id)
        self._responses = list(responses or [])
        self._cursor = 0
        self.default_response = default_response
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]] | str,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": self._to_openai_messages(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        item: Any = self.default_response
        if self._cursor < len(self._responses):
            item = self._responses[self._cursor]
            self._cursor += 1
        response = self._coerce(item)
        self._record(response)
        return response

    def stream_chat(
        self,
        messages: Sequence[Message] | Sequence[dict[str, Any]] | str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        response = self.chat(messages)
        for index in range(0, len(response.content), 8):
            yield response.content[index : index + 8]

    @staticmethod
    def _coerce(item: Any) -> LLMResponse:
        if callable(item) and not isinstance(item, (str, dict, LLMResponse)):
            return MockLLM._coerce(item())
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, str):
            completion = estimate_tokens(item)
            return LLMResponse(
                content=item,
                model="mock-llm",
                finish_reason="stop",
                completion_tokens=completion,
                total_tokens=completion,
            )
        if isinstance(item, dict):
            prompt = int(item.get("prompt_tokens", 0) or 0)
            completion = int(item.get("completion_tokens", 0) or 0)
            total = int(item.get("total_tokens", 0) or 0) or (prompt + completion)
            return LLMResponse(
                content=item.get("content", "") or "",
                tool_calls=[ToolCall(**call) for call in (item.get("tool_calls") or [])],
                model=item.get("model", "mock-llm"),
                finish_reason=item.get("finish_reason", "stop"),
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=total,
            )
        raise LLMResponseError(f"MockLLM 无法识别的响应类型：{type(item)!r}")


def build_llm(config: Config | None = None, *, allow_mock: bool = False) -> BaseLLM:
    """按配置构建 LLM。

    Args:
        config: 配置对象，缺省使用全局单例。
        allow_mock: 未配置密钥时是否降级为 :class:`MockLLM`（离线演示/CI 用）。
    """
    config = config or get_config()
    if not config.is_llm_configured:
        if allow_mock:
            logger.warning("未配置 LLM_API_KEY，已降级为 MockLLM（离线模式）")
            return MockLLM()
        raise ConfigError(
            "未配置 LLM_API_KEY",
            detail="请复制 .env.example 为 .env 并填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID",
        )
    return CodeAgentXLLM(config)
