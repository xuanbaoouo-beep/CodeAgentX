"""Plan-and-Solve 重构 Agent 测试。

重点验证"规划"与"执行"两阶段的契约：
计划能解析与截断、每步状态被正确写回、失败步骤不能伪装成完成、
规划输出不可解析时降级而不是崩溃。
"""

from __future__ import annotations

import json

import pytest

from codeagentx.agents.plan_solve_refactor import PlanSolveRefactor
from codeagentx.agents.schemas import RefactorPlan
from codeagentx.core.llm import MockLLM

PLAN_JSON = json.dumps(
    {
        "goal": "消除登录逻辑中的硬编码密钥",
        "steps": [
            {
                "description": "把 SECRET_KEY 改为从环境变量读取",
                "files": ["app/auth/service.py"],
                "rationale": "避免密钥进入版本库",
            },
            {
                "description": "补充缺失密钥时的启动校验",
                "files": ["app/auth/service.py"],
            },
        ],
        "risks": ["本地未设置环境变量时启动直接失败"],
        "verification": ["python -m pytest tests/"],
    },
    ensure_ascii=False,
)

TOOL_CALL = {"tool_calls": [{"id": "c1", "name": "echo", "arguments": {"text": "x"}}]}


def _many_steps(count: int) -> str:
    return json.dumps(
        {"goal": "g", "steps": [{"description": f"第 {index} 步"} for index in range(count)]},
        ensure_ascii=False,
    )


class TestPlan:
    def test_plan_only_mode(self):
        agent = PlanSolveRefactor(MockLLM([PLAN_JSON]), tools=None)
        result = agent.run("重构登录逻辑", execute=False)

        assert result.success is True
        assert result.iterations == 1
        plan = result.metadata["plan"]
        assert plan["goal"] == "消除登录逻辑中的硬编码密钥"
        assert [step["status"] for step in plan["steps"]] == ["pending", "pending"]
        assert "# 重构计划" in result.output
        assert result.metadata["executed"] is False

    def test_refactor_returns_structured_plan(self):
        agent = PlanSolveRefactor(MockLLM([PLAN_JSON]), tools=None)
        plan = agent.refactor("g", execute=False)

        assert isinstance(plan, RefactorPlan)
        assert plan.total == 2
        assert plan.steps[0].files == ["app/auth/service.py"]
        assert plan.risks == ["本地未设置环境变量时启动直接失败"]

    def test_plan_is_truncated_by_max_steps(self):
        agent = PlanSolveRefactor(MockLLM([_many_steps(10)]), tools=None, max_steps=3)
        plan = agent.refactor("g", execute=False)

        assert plan.total == 3
        assert "截断" in plan.notes
        assert "10 步" in plan.notes

    def test_unparsable_plan_degrades_with_note(self):
        agent = PlanSolveRefactor(MockLLM(["我先想想，不打算输出 JSON。"]), tools=None)
        plan = agent.refactor("重构目标", execute=False)

        assert plan.total == 0
        assert plan.goal == "重构目标"
        assert "无法解析" in plan.notes

    def test_goal_falls_back_to_argument(self):
        agent = PlanSolveRefactor(MockLLM(['{"steps": [{"description": "改一下"}]}']), tools=None)
        assert agent.refactor("原始目标", execute=False).goal == "原始目标"

    def test_max_steps_must_be_positive(self):
        with pytest.raises(ValueError):
            PlanSolveRefactor(MockLLM([]), tools=None, max_steps=0)


class TestExecute:
    def test_each_step_records_status_and_result(self):
        llm = MockLLM([PLAN_JSON, "已改为 os.getenv 读取密钥。", "已补充启动校验。"])
        agent = PlanSolveRefactor(llm, tools=None)
        result = agent.run("重构登录逻辑")

        assert result.success is True
        steps = result.metadata["plan"]["steps"]
        assert [step["status"] for step in steps] == ["done", "done"]
        assert "os.getenv" in steps[0]["result"]
        assert "已补充启动校验。" in result.output
        assert result.iterations == 3  # 1 次规划 + 2 次执行

    def test_step_failure_is_reported_honestly(self, registry):
        """步骤达到最大轮次未收敛时，必须标记 failed，而不是当成已完成。"""
        agent = PlanSolveRefactor(
            MockLLM([PLAN_JSON], default_response=TOOL_CALL),
            tools=registry,
            max_iterations=1,
        )
        result = agent.run("重构登录逻辑")

        assert result.success is False
        assert "个步骤未成功完成" in result.error
        assert [step["status"] for step in result.metadata["plan"]["steps"]] == ["failed", "failed"]

    def test_execute_step_index_is_validated(self):
        agent = PlanSolveRefactor(MockLLM([PLAN_JSON]), tools=None)
        plan = agent.refactor("g", execute=False)

        with pytest.raises(IndexError):
            agent.execute_step(plan, 5)

    def test_tools_are_available_during_execution(self, registry):
        llm = MockLLM(
            [
                PLAN_JSON,
                TOOL_CALL,
                "第一步：已确认代码位置。",
                "第二步：已补充校验。",
            ]
        )
        agent = PlanSolveRefactor(llm, tools=registry)
        result = agent.run("重构登录逻辑")

        assert result.success is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "echo"
        assert result.metadata["plan"]["steps"][0]["status"] == "done"

    def test_planning_stage_has_no_tools(self, registry):
        """规划阶段不该动代码，因此不向模型暴露任何工具。"""
        llm = MockLLM([PLAN_JSON, "done", "done"])
        agent = PlanSolveRefactor(llm, tools=registry)
        agent.run("重构登录逻辑")

        assert llm.calls[0]["tools"] is None

    def test_usage_covers_all_phases(self):
        agent = PlanSolveRefactor(MockLLM([PLAN_JSON, "a", "b"]), tools=None)
        result = agent.run("重构登录逻辑")

        assert result.usage["calls"] == 3
        assert result.usage["total_tokens"] > 0
