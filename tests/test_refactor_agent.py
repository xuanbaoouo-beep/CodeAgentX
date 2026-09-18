"""Refactor 角色测试。

W5 的 Plan-and-Solve 已经覆盖了"计划 + 逐步执行"，W6 的 Refactor 角色只补一件事：
**把上一步产出的 Finding 列表转成重构目标**，并且默认只出计划、不改代码。
本文件验证这两点，以及 W5 的规划约束（步数上限、解析失败不抛异常）在角色上仍然成立。
"""

from __future__ import annotations

import json

from codeagentx.agents.plan_solve_refactor import DEFAULT_GOAL
from codeagentx.agents.refactor import MAX_FINDINGS_IN_GOAL, RefactorAgent, build_goal_from_findings
from codeagentx.agents.schemas import Finding, RefactorPlan
from codeagentx.core.llm import MockLLM

PLAN_JSON = json.dumps(
    {
        "goal": "修复硬编码密钥与 SQL 拼接",
        "steps": [
            {"description": "把密钥改为读环境变量", "files": ["app/config.py"], "rationale": "消除泄露"},
            {"description": "改用参数化查询", "files": ["app/db/query.py"], "rationale": "消除注入"},
            {"description": "补充回归测试", "files": ["tests/test_auth.py"], "rationale": "防止回退"},
        ],
        "risks": ["改动配置读取会影响启动流程"],
        "verification": ["运行 pytest tests/test_auth.py"],
    },
    ensure_ascii=False,
)


def finding(
    title: str,
    *,
    file: str = "app/config.py",
    line: int = 8,
    severity: str = "high",
    category: str = "security",
) -> Finding:
    return Finding(
        title=title,
        file=file,
        line=line,
        severity=severity,
        category=category,
        description=f"{title} 的影响说明",
    )


class TestBuildGoal:
    def test_no_findings_uses_default_goal(self):
        assert build_goal_from_findings([]) == DEFAULT_GOAL

    def test_findings_become_a_goal_with_locations(self):
        goal = build_goal_from_findings([finding("硬编码密钥")])

        assert "保持外部行为不变" in goal
        assert "硬编码密钥" in goal
        assert "app/config.py:8" in goal

    def test_findings_are_sorted_by_severity(self):
        goal = build_goal_from_findings(
            [
                finding("低危", severity="low", line=1),
                finding("高危", severity="high", line=2),
            ]
        )
        assert goal.index("高危") < goal.index("低危")

    def test_extra_findings_are_summarized_not_listed(self):
        findings = [finding(f"问题 {index}", line=index + 1) for index in range(8)]
        goal = build_goal_from_findings(findings, limit=3)

        assert "另有 5 条同类问题" in goal
        assert goal.count("@") == 3

    def test_default_limit_is_bounded(self):
        assert MAX_FINDINGS_IN_GOAL == 5


class TestPlanFromFindings:
    def test_plan_is_built_from_findings(self):
        llm = MockLLM([PLAN_JSON])
        agent = RefactorAgent(llm)
        plan = agent.plan_from_findings([finding("硬编码密钥")], scope="data/sample_repo")

        assert isinstance(plan, RefactorPlan)
        assert plan.total == 3
        assert plan.steps[0].files == ["app/config.py"]
        assert plan.risks and plan.verification

        prompt = llm.calls[0]["messages"][1]["content"]
        assert "硬编码密钥" in prompt
        assert "data/sample_repo" in prompt

    def test_explicit_goal_overrides_the_generated_one(self):
        llm = MockLLM([PLAN_JSON])
        RefactorAgent(llm).plan_from_findings([finding("硬编码密钥")], goal="统一配置读取方式")
        prompt = llm.calls[0]["messages"][1]["content"]

        assert "统一配置读取方式" in prompt
        assert "硬编码密钥" not in prompt

    def test_default_goal_without_findings(self):
        llm = MockLLM([PLAN_JSON])
        RefactorAgent(llm).plan_from_findings([])
        assert DEFAULT_GOAL in llm.calls[0]["messages"][1]["content"]

    def test_only_plans_and_never_executes(self):
        """重构是写操作的前置，默认只出计划：一次调用、步骤全部 pending。"""
        llm = MockLLM([PLAN_JSON])
        plan = RefactorAgent(llm).plan_from_findings([finding("硬编码密钥")])

        assert len(llm.calls) == 1
        assert {step.status for step in plan.steps} == {"pending"}
        assert all(not step.result for step in plan.steps)

    def test_step_limit_is_enforced(self):
        llm = MockLLM([PLAN_JSON])
        plan = RefactorAgent(llm, max_steps=2).plan_from_findings([finding("硬编码密钥")])

        assert plan.total == 2
        assert "截断" in plan.notes

    def test_unparsable_plan_does_not_raise(self):
        llm = MockLLM(["我建议先把配置收拢一下，具体步骤回头再说。"])
        plan = RefactorAgent(llm).plan_from_findings([finding("硬编码密钥")])

        assert plan.total == 0
        assert "无法解析" in plan.notes


class TestIdentity:
    def test_name_and_plan_system_prompt(self):
        llm = MockLLM([PLAN_JSON])
        agent = RefactorAgent(llm)
        agent.plan_from_findings([finding("硬编码密钥")])

        assert agent.name == "refactor"
        assert "重构规划专家" in llm.calls[0]["messages"][0]["content"]
