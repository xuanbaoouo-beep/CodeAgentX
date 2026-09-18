"""工具基类与数据结构。

约定：所有工具必须返回结构化结果（:class:`ToolResult`），
禁止返回裸字符串，以便上层判定成败、统计耗时与拦截安全违规。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from codeagentx.core.exceptions import SecurityViolationError, ToolValidationError
from codeagentx.core.logger import get_logger, log_event

logger = get_logger("tools.base")

# JSON Schema 基础类型
_JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


@dataclass
class ToolParameter:
    """工具入参描述，用于生成 OpenAI function calling 的 JSON Schema。"""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: list[Any] | None = None
    default: Any = None
    items: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type not in _JSON_TYPES:
            raise ValueError(f"不支持的参数类型：{self.type!r}，可选 {sorted(_JSON_TYPES)}")

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.items:
            schema["items"] = self.items
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolResult:
    """工具执行结果。"""

    success: bool
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: Any = None, **metadata: Any) -> ToolResult:
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, *, error_type: str | None = None, **metadata: Any) -> ToolResult:
        return cls(success=False, error=error, error_type=error_type, metadata=metadata)

    def to_text(self, max_length: int = 8000) -> str:
        """转成喂回给 LLM 的文本，过长时截断并标注。"""
        if not self.success:
            return f"[工具执行失败] {self.error}"
        text = self.output if isinstance(self.output, str) else _json_dumps(self.output)
        if len(text) > max_length:
            return f"{text[:max_length]}\n...（输出已截断，原始长度 {len(text)} 字符）"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type,
            "latency": round(self.latency, 4),
            "metadata": self.metadata,
        }


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str, indent=2)


class BaseTool(ABC):
    """工具抽象基类。

    子类需声明 ``name`` / ``description`` / ``parameters``，并实现 :meth:`_run`。
    """

    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = []
    #: 是否为高危工具（供 Orchestrator 与权限策略使用）
    dangerous: bool = False

    def __init__(self) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__} 必须定义类属性 name")
        if not self.description:
            raise ValueError(f"{type(self).__name__} 必须定义类属性 description")

    # ------------------------------------------------------------ 参数校验
    def validate(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """校验并补全默认值，返回规范化后的参数字典。"""
        arguments = dict(arguments or {})
        known = {param.name: param for param in self.parameters}

        unknown = set(arguments) - set(known)
        if unknown:
            raise ToolValidationError(
                f"工具 {self.name} 收到未知参数：{sorted(unknown)}",
                detail=f"可用参数：{sorted(known)}",
            )

        normalized: dict[str, Any] = {}
        missing: list[str] = []
        for param in self.parameters:
            if param.name in arguments:
                normalized[param.name] = arguments[param.name]
            elif param.default is not None:
                normalized[param.name] = param.default
            elif param.required:
                missing.append(param.name)
        if missing:
            raise ToolValidationError(
                f"工具 {self.name} 缺少必填参数：{missing}",
                detail=f"参数说明：{self.parameters_schema()['properties']}",
            )
        return normalized

    # ------------------------------------------------------------ 执行
    def run(self, **arguments: Any) -> ToolResult:
        """统一执行入口：校验 -> 执行 -> 计时 -> 异常归一化。"""
        start = time.perf_counter()
        try:
            normalized = self.validate(arguments)
        except ToolValidationError as exc:
            result = ToolResult.fail(str(exc), error_type=type(exc).__name__)
        except SecurityViolationError as exc:  # 校验阶段的安全拦截
            result = ToolResult.fail(str(exc), error_type=type(exc).__name__)
        else:
            try:
                output = self._run(**normalized)
                result = output if isinstance(output, ToolResult) else ToolResult.ok(output)
            except SecurityViolationError as exc:
                result = ToolResult.fail(str(exc), error_type=type(exc).__name__)
            except Exception as exc:  # noqa: BLE001 - 工具异常不应击穿编排层
                result = ToolResult.fail(
                    f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                )
                logger.debug("工具 %s 执行异常", self.name, exc_info=True)
        result.latency = time.perf_counter() - start
        log_event(
            logger,
            "tool_call",
            tool=self.name,
            success=result.success,
            latency=round(result.latency, 4),
            error=result.error,
        )
        return result

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """子类实现的实际逻辑，可直接返回 ``ToolResult`` 或任意可序列化对象。"""

    # ------------------------------------------------------------ Schema
    def parameters_schema(self) -> dict[str, Any]:
        properties = {param.name: param.to_schema() for param in self.parameters}
        required = [
            param.name for param in self.parameters if param.required and param.default is None
        ]
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} name={self.name!r}>"
