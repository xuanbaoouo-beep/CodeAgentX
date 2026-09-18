"""ToolRegistry / BaseTool 单元测试。"""

from __future__ import annotations

import pytest

from codeagentx.core.exceptions import ToolNotFoundError, ToolValidationError
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.registry import ToolRegistry


def test_tool_requires_name_and_description() -> None:
    class Nameless(BaseTool):
        description = "缺少 name"

        def _run(self, **kwargs):
            return None

    with pytest.raises(ValueError, match="name"):
        Nameless()


def test_tool_requires_description() -> None:
    class NoDescription(BaseTool):
        name = "no-desc"

        def _run(self, **kwargs):
            return None

    with pytest.raises(ValueError, match="description"):
        NoDescription()


def test_tool_rejects_invalid_parameter_type() -> None:
    with pytest.raises(ValueError, match="不支持的参数类型"):
        ToolParameter(name="x", type="not-a-type")


# ------------------------------------------------------------------ BaseTool
def test_parameters_schema_shape(echo_tool: BaseTool) -> None:
    schema = echo_tool.parameters_schema()

    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"text", "repeat"}
    assert schema["required"] == ["text"]  # repeat 有默认值，不进 required
    assert schema["properties"]["repeat"]["default"] == 1


def test_to_openai_schema(echo_tool: BaseTool) -> None:
    schema = echo_tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert "parameters" in schema["function"]


def test_validate_fills_defaults(echo_tool: BaseTool) -> None:
    assert echo_tool.validate({"text": "hi"}) == {"text": "hi", "repeat": 1}


def test_validate_rejects_unknown_argument(echo_tool: BaseTool) -> None:
    with pytest.raises(ToolValidationError, match="未知参数"):
        echo_tool.validate({"text": "hi", "nope": 1})


def test_validate_rejects_missing_required(echo_tool: BaseTool) -> None:
    with pytest.raises(ToolValidationError, match="缺少必填参数"):
        echo_tool.validate({})


def test_run_success_measures_latency(echo_tool: BaseTool) -> None:
    result = echo_tool.run(text="ab", repeat=3)

    assert result.success is True
    assert result.output == "ababab"
    assert result.latency >= 0


def test_run_exception_is_normalized(registry: ToolRegistry) -> None:
    result = registry.get("boom").run()

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert "预期内的失败" in (result.error or "")


def test_run_security_violation_is_captured(registry: ToolRegistry) -> None:
    result = registry.get("guarded").run()

    assert result.success is False
    assert result.error_type == "SecurityViolationError"


def test_tool_result_to_text() -> None:
    assert ToolResult.ok("内容").to_text() == "内容"
    assert ToolResult.fail("炸了").to_text() == "[工具执行失败] 炸了"
    assert "输出已截断" in ToolResult.ok("x" * 100).to_text(max_length=10)

    structured = ToolResult.ok({"a": 1}).to_text()
    assert '"a": 1' in structured


# ------------------------------------------------------------------ ToolRegistry
def test_registry_basic_operations(registry: ToolRegistry) -> None:
    assert len(registry) == 3
    assert "echo" in registry
    assert "missing" not in registry
    assert registry.names() == ["boom", "echo", "guarded"]
    assert registry.has("echo") is True


def test_registry_get_unknown_raises(registry: ToolRegistry) -> None:
    with pytest.raises(ToolNotFoundError):
        registry.get("not-exist")


def test_registry_rejects_duplicate(registry: ToolRegistry, echo_tool: BaseTool) -> None:
    with pytest.raises(ToolValidationError, match="工具名冲突"):
        registry.register(echo_tool)


def test_registry_override(registry: ToolRegistry) -> None:
    class OverrideEcho(BaseTool):
        name = "echo"
        description = "覆盖版"
        parameters: list[ToolParameter] = []

        def _run(self, **kwargs):
            return "override"

    registry.register(OverrideEcho(), override=True)

    assert registry.get("echo").description == "覆盖版"
    assert len(registry) == 3


def test_registry_register_rejects_non_tool() -> None:
    with pytest.raises(TypeError):
        ToolRegistry().register("not-a-tool")  # type: ignore[arg-type]


def test_registry_unregister(registry: ToolRegistry) -> None:
    assert registry.unregister("echo") is True
    assert registry.unregister("echo") is False
    assert "echo" not in registry


def test_registry_to_openai_tools_all(registry: ToolRegistry) -> None:
    schemas = registry.to_openai_tools()
    assert len(schemas) == 3
    assert {item["function"]["name"] for item in schemas} == {"boom", "echo", "guarded"}


def test_registry_to_openai_tools_filtered(registry: ToolRegistry) -> None:
    schemas = registry.to_openai_tools(["echo"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "echo"


def test_registry_to_openai_tools_unknown_name_raises(registry: ToolRegistry) -> None:
    with pytest.raises(ToolNotFoundError):
        registry.to_openai_tools(["missing"])


def test_registry_execute_with_dict(registry: ToolRegistry) -> None:
    result = registry.execute("echo", {"text": "hi"})
    assert result.success is True
    assert result.output == "hi"


def test_registry_execute_with_json_string(registry: ToolRegistry) -> None:
    result = registry.execute("echo", '{"text": "json", "repeat": 2}')
    assert result.output == "jsonjson"


def test_registry_execute_with_empty_string_arguments(registry: ToolRegistry) -> None:
    result = registry.execute("boom", "")
    assert result.success is False
    assert result.error_type == "RuntimeError"


def test_registry_execute_with_invalid_json(registry: ToolRegistry) -> None:
    result = registry.execute("echo", "{not-json")
    assert result.success is False
    assert result.error_type == "JSONDecodeError"


def test_registry_execute_with_non_object_json(registry: ToolRegistry) -> None:
    result = registry.execute("echo", "[1, 2]")
    assert result.success is False
    assert result.error_type == "TypeError"


def test_registry_execute_unknown_tool(registry: ToolRegistry) -> None:
    result = registry.execute("missing", {})
    assert result.success is False
    assert result.error_type == "ToolNotFoundError"


def test_registry_execute_missing_required_argument(registry: ToolRegistry) -> None:
    result = registry.execute("echo", {})
    assert result.success is False
    assert result.error_type == "ToolValidationError"


def test_registry_execute_rejects_unknown_argument(registry: ToolRegistry) -> None:
    result = registry.execute("echo", {"text": "hi", "extra": 1})
    assert result.success is False
    assert result.error_type == "ToolValidationError"


def test_registry_describe(registry: ToolRegistry) -> None:
    description = registry.describe()
    assert "- echo(" in description
    assert "- boom()" in description

    assert ToolRegistry().describe() == "（当前无可用工具）"


def test_registry_iteration(registry: ToolRegistry) -> None:
    assert {tool.name for tool in registry} == {"boom", "echo", "guarded"}


def test_registry_register_many_returns_self() -> None:
    class Simple(BaseTool):
        name = "simple"
        description = "简单工具"
        parameters: list[ToolParameter] = []

        def _run(self, **kwargs):
            return "ok"

    registry = ToolRegistry()
    assert registry.register_many(Simple()) is registry
    assert len(registry) == 1
