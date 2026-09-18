"""Tester 角色测试。

关键区分（本项目的能力边界）：
- 沙箱没有写文件工具，所以 Tester **只产出测试代码文本**，不写入仓库；
- 它可以顺带跑一遍仓库现有的测试套件，此时"我写了测试"与"我跑了测试"必须能被区分：
  没挂载 ``test_runner`` 时返回 ``available=False``，调用方必须当成"未验证"而非"通过"。
"""

from __future__ import annotations

from codeagentx.agents.tester import TesterAgent, extract_test_code
from codeagentx.core.llm import MockLLM
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult
from codeagentx.tools.registry import ToolRegistry

CODE_BLOCK = """```python
import pytest

from app.auth import service


def test_login_rejects_empty_password():
    with pytest.raises(ValueError):
        service.login("u", "")
```

这个测试断言空口令会被拒绝；当实现改成返回 None 时它会失败。"""

RAW_CODE = """def test_something():
    assert 1 + 1 == 2
"""


class FakeTestRunner(BaseTool):
    """``test_runner`` 测试替身。"""

    name = "test_runner"
    description = "测试替身：返回预设的 pytest 结果。"
    parameters = [
        ToolParameter(name="target", description="测试目标", required=False, default="."),
        ToolParameter(name="args", description="额外参数", type="array", required=False),
        ToolParameter(name="timeout", description="超时秒数", type="number", required=False),
    ]

    def __init__(self, *, passed: bool = True):
        super().__init__()
        self.passed = passed
        self.calls: list[dict] = []

    def _run(self, target: str = ".", args=None, timeout=None) -> ToolResult:
        self.calls.append({"target": target, "args": args, "timeout": timeout})
        return ToolResult.ok(
            {
                "target": target,
                "exit_code": 0 if self.passed else 1,
                "passed": self.passed,
                "counts": {"passed": 3, "failed": 0 if self.passed else 1},
                "summary_line": "3 passed" if self.passed else "1 failed, 2 passed",
                "failures": [] if self.passed else ["test_x"],
                "output_tail": "",
            }
        )


def registry_with(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry(name="tester-test")
    registry.register_many(*tools)
    return registry


class TestExtractTestCode:
    def test_fenced_python_block_is_preferred(self):
        assert extract_test_code(CODE_BLOCK).startswith("import pytest")
        assert "service.login" in extract_test_code(CODE_BLOCK)

    def test_block_language_tag_is_optional(self):
        assert extract_test_code("```\ndef test_a():\n    pass\n```") == "def test_a():\n    pass"

    def test_raw_code_without_fence_is_accepted(self):
        assert extract_test_code(RAW_CODE) == RAW_CODE.strip()

    def test_plain_prose_yields_nothing(self):
        assert extract_test_code("我觉得这个问题很难写测试。") == ""

    def test_empty_input(self):
        assert extract_test_code("") == ""


class TestRun:
    def test_generates_test_code(self):
        agent = TesterAgent(MockLLM([CODE_BLOCK]))
        result = agent.run("app/auth/service.py", finding="[high][security] 硬编码密钥 @ a.py:8")

        assert result.success is True
        assert result.metadata["agent"] == "tester"
        assert "def test_login_rejects_empty_password" in result.metadata["test_code"]
        # 没要求跑现有测试时不产生测试结果，避免"没跑"被误读成"通过"
        assert result.metadata["test_result"] is None

    def test_finding_and_target_reach_the_prompt(self):
        llm = MockLLM([CODE_BLOCK])
        TesterAgent(llm).run("app/auth/service.py", finding="空口令仍可登录", context="def login(): ...")

        user = llm.calls[0]["messages"][1]["content"]
        assert "app/auth/service.py" in user
        assert "空口令仍可登录" in user
        assert "def login(): ..." in user
        assert "pytest" in llm.calls[0]["messages"][0]["content"]

    def test_missing_test_code_marks_failure(self):
        result = TesterAgent(MockLLM(["这段逻辑太简单了，不需要测试。"])).run("a.py")

        assert result.success is False
        assert result.metadata["test_code"] == ""
        assert "提取到测试代码" in result.error

    def test_default_target_when_omitted(self):
        result = TesterAgent(MockLLM([CODE_BLOCK])).run()
        assert "未指定" in result.metadata["target"]


class TestExistingSuiteVerification:
    def test_existing_suite_is_run_when_requested(self):
        runner = FakeTestRunner(passed=True)
        agent = TesterAgent(MockLLM([CODE_BLOCK]), tools=registry_with(runner))
        result = agent.run("a.py", run_existing=True, test_path="tests")

        assert result.success is True
        payload = result.metadata["test_result"]
        assert payload["available"] is True
        assert payload["passed"] is True
        assert runner.calls[0]["target"] == "tests"

    def test_runner_result_is_available_in_verify(self):
        runner = FakeTestRunner(passed=False)
        payload = TesterAgent(MockLLM([]), tools=registry_with(runner)).verify("tests")

        assert payload["available"] is True
        assert payload["passed"] is False
        assert payload["summary_line"] == "1 failed, 2 passed"

    def test_without_runner_it_is_unverified_not_passed(self):
        """未挂载 test_runner：必须报 available=False，绝不能表现为"测试通过"。"""
        payload = TesterAgent(MockLLM([])).verify("tests")

        assert payload["available"] is False
        assert payload["success"] is False
        assert "未挂载" in payload["error"]

    def test_run_without_runner_keeps_code_but_marks_unverified(self):
        result = TesterAgent(MockLLM([CODE_BLOCK])).run("a.py", run_existing=True)

        assert result.metadata["test_code"]
        assert result.metadata["test_result"]["available"] is False


class TestReuse:
    def test_history_is_reset_between_runs(self):
        agent = TesterAgent(MockLLM([CODE_BLOCK, CODE_BLOCK]))
        agent.run("a.py")
        agent.run("b.py")
        assert len(agent.history) == 2
