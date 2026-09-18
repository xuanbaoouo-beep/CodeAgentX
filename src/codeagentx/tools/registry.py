"""工具注册中心：负责注册、发现、批量导出 Schema 与调用分发。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from codeagentx.core.exceptions import ToolNotFoundError, ToolValidationError
from codeagentx.core.logger import get_logger, log_event
from codeagentx.tools.base import BaseTool, ToolResult

logger = get_logger("tools.registry")


class ToolRegistry:
    """工具注册表。

    一个注册表即一个工具作用域：Agent 只持有自己需要的注册表，
    避免把全部工具暴露给模型（既省 Token 又降低误调用概率）。
    """

    def __init__(self, tools: Iterable[BaseTool] | None = None, *, name: str = "default") -> None:
        self.name = name
        self._tools: dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    # ------------------------------------------------------------ 注册
    def register(self, tool: BaseTool, *, override: bool = False) -> BaseTool:
        """注册一个工具。重名时默认报错，除非 ``override=True``。"""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"只能注册 BaseTool 实例，收到 {type(tool)!r}")
        if tool.name in self._tools and not override:
            raise ToolValidationError(
                f"工具名冲突：{tool.name!r}",
                detail="如确认要覆盖，请传 override=True",
            )
        self._tools[tool.name] = tool
        log_event(logger, "tool_registered", registry=self.name, tool=tool.name)
        return tool

    def register_many(self, *tools: BaseTool, override: bool = False) -> ToolRegistry:
        for tool in tools:
            self.register(tool, override=override)
        return self

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ------------------------------------------------------------ 查询
    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"工具未注册：{name!r}",
                detail=f"可用工具：{sorted(self._tools)}",
            )
        return tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[BaseTool]:
        return [self._tools[name] for name in self.names()]

    # ------------------------------------------------------------ 导出
    def to_openai_tools(self, names: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """导出 OpenAI function calling 的 tools 参数。"""
        selected = self.list_tools() if names is None else [self.get(name) for name in names]
        return [tool.to_openai_schema() for tool in selected]

    def describe(self) -> str:
        """人类可读的工具清单，可注入提示词。"""
        if not self._tools:
            return "（当前无可用工具）"
        lines = []
        for tool in self.list_tools():
            params = ", ".join(
                f"{param.name}:{param.type}" + ("" if param.required else "（可选）")
                for param in tool.parameters
            )
            lines.append(f"- {tool.name}({params})：{tool.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------ 调用
    def execute(self, name: str, arguments: dict[str, Any] | str | None = None) -> ToolResult:
        """执行工具。

        ``arguments`` 允许是 JSON 字符串（模型时常返回字符串），
        解析失败时返回失败结果而非抛异常，由上层决定是否重试。
        """
        try:
            tool = self.get(name)
        except ToolNotFoundError as exc:
            return ToolResult.fail(str(exc), error_type=type(exc).__name__)

        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                parsed: dict[str, Any] = {}
            else:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    return ToolResult.fail(
                        f"工具 {name} 的 arguments 不是合法 JSON：{exc}",
                        error_type="JSONDecodeError",
                    )
                if not isinstance(parsed, dict):
                    return ToolResult.fail(
                        f"工具 {name} 的 arguments 必须是 JSON 对象，收到 {type(parsed).__name__}",
                        error_type="TypeError",
                    )
        else:
            parsed = dict(arguments or {})

        return tool.run(**parsed)

    # ------------------------------------------------------------ 魔术方法
    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self.list_tools())

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<ToolRegistry {self.name} tools={self.names()}>"
