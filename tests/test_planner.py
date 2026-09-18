"""Planner 角色测试。

三条必须守住的行为：
1. 正常规划：子任务被结构化解析，角色名归一化到合法枚举；
2. 解析失败/空计划：**降级必须显式**（``success=False`` + ``metadata.fallback``），
   绝不能把兜底计划伪装成模型的规划成果；
3. 不越界：Planner 不挂工具，只产出计划。
"""

from __future__ import annotations

import json

import pytest

from codeagentx.agents.planner import DEFAULT_MAX_TASKS, FALLBACK_TASKS, PlannerAgent
from codeagentx.agents.schemas import ReviewPlan
from codeagentx.core.llm import MockLLM

PLAN_JSON = json.dumps(
    {
        "summary": "先覆盖认证与数据层，再查可维护性。",
        "tasks": [
            {
                "description": "审查登录与会话逻辑",
                "focus": "bug",
                "files": ["app/auth/service.py"],
                "reason": "认证是高风险面",
                "assignee": "reviewer",
            },
            {
                "description": "排查 SQL 拼接与密钥管理",
                "focus": "security",
                "files": [],
                "reason": "外部输入直达数据层",
                "assignee": "Security",
            },
            {
                "description": "为登录失败路径补测试",
                "focus": "test",
                "files": [],
                "reason": "缺乏回归保护",
                "assignee": "tester",
            },
        ],
    },
    ensure_ascii=False,
)

EMPTY_PLAN_JSON = json.dumps({"summary": "没什么可查的", "tasks": []}, ensure_ascii=False)


class TestPlanParsing:
    def test_structured_tasks_are_parsed(self):
        agent = PlannerAgent(MockLLM([PLAN_JSON]))
        result = agent.run("data/sample_repo", file_list=["app/auth/service.py"])

        assert result.success is True
        assert result.metadata["agent"] == "planner"
        plan = ReviewPlan.from_dict(result.metadata["plan"])
        assert plan.total == 3
        assert plan.target == "data/sample_repo"
        assert plan.tasks[0].description == "审查登录与会话逻辑"
        assert plan.tasks[0].files == ["app/auth/service.py"]
        assert plan.notes  # summary 被当作 notes 保留下来

    def test_assignee_is_normalized(self):
        agent = PlannerAgent(MockLLM([PLAN_JSON]))
        plan = agent.plan("repo")

        assert [task.assignee for task in plan.tasks] == ["reviewer", "security", "tester"]

    def test_unknown_assignee_is_left_empty(self):
        """无法识别的角色留空，由编排层兜底，而不是硬塞一个错角色。"""
        payload = json.dumps(
            {
                "summary": "x",
                "tasks": [{"description": "随手看看", "assignee": "wizard"}],
            },
            ensure_ascii=False,
        )
        plan = PlannerAgent(MockLLM([payload])).plan("repo")

        assert plan.tasks[0].assignee == ""

    def test_plan_helper_returns_structured_object(self):
        plan = PlannerAgent(MockLLM([PLAN_JSON])).plan("repo")
        assert isinstance(plan, ReviewPlan)
        assert plan.total == 3

    def test_task_aliases_are_accepted(self):
        """模型有时用 task/step、agent/role 等别名，解析要容忍。"""
        payload = json.dumps(
            {
                "summary": "x",
                "steps": [
                    {"task": "看看认证", "category": "security", "role": "security", "targets": ["a.py"]}
                ],
            },
            ensure_ascii=False,
        )
        plan = PlannerAgent(MockLLM([payload])).plan("repo")

        task = plan.tasks[0]
        assert task.description == "看看认证"
        assert task.focus == "security"
        assert task.assignee == "security"
        assert task.files == ["a.py"]


class TestFallback:
    """降级路径：可用但不可信。"""

    def test_unparsable_output_falls_back_and_marks_failure(self):
        agent = PlannerAgent(MockLLM(["我先聊两句，JSON 一会儿再给。"]))
        result = agent.run("repo")

        plan = ReviewPlan.from_dict(result.metadata["plan"])
        assert plan.total == len(FALLBACK_TASKS)
        assert plan.metadata["fallback"] is True
        assert "parse_error" in plan.metadata
        assert result.success is False
        assert "无法解析" in result.error

    def test_empty_task_list_falls_back(self):
        agent = PlannerAgent(MockLLM([EMPTY_PLAN_JSON]))
        result = agent.run("repo")

        plan = ReviewPlan.from_dict(result.metadata["plan"])
        assert plan.metadata["fallback"] is True
        assert plan.metadata["empty_plan"] is True
        assert result.success is False
        assert "未产出任何子任务" in result.error

    def test_fallback_tasks_cover_high_risk_faces(self):
        """兜底计划不能是空壳：必须覆盖整体通读、安全与正确性三类面。"""
        assignees = {task.assignee for task in FALLBACK_TASKS}
        assert assignees == {"reviewer", "security"}
        assert any(task.focus == "security" for task in FALLBACK_TASKS)

    def test_fallback_result_is_a_copy_not_the_shared_object(self):
        agent = PlannerAgent(MockLLM(["no json here"]))
        plan = agent.plan("repo")
        plan.tasks[0].description = "被改过了"
        assert FALLBACK_TASKS[0].description != "被改过了"


class TestTruncation:
    def test_too_many_tasks_are_truncated(self):
        payload = json.dumps(
            {
                "summary": "x",
                "tasks": [{"description": f"子任务 {index}"} for index in range(1, 6)],
            },
            ensure_ascii=False,
        )
        agent = PlannerAgent(MockLLM([payload]), max_tasks=2)
        result = agent.run("repo")

        plan = ReviewPlan.from_dict(result.metadata["plan"])
        assert plan.total == 2
        assert plan.metadata["truncated"] == 3
        # 截断是容量限制，不是规划的失败
        assert result.success is True

    def test_max_tasks_must_be_positive(self):
        with pytest.raises(ValueError):
            PlannerAgent(MockLLM([PLAN_JSON]), max_tasks=0)

    def test_default_limit_matches_prompt_contract(self):
        assert DEFAULT_MAX_TASKS == 5


class TestPromptAndTools:
    def test_planner_has_no_tools(self):
        llm = MockLLM([PLAN_JSON])
        PlannerAgent(llm).run("repo", file_list=["a.py"])
        assert llm.calls[0]["tools"] is None
        assert "规划" in llm.calls[0]["messages"][0]["content"]

    def test_target_files_and_context_reach_the_prompt(self):
        llm = MockLLM([PLAN_JSON])
        PlannerAgent(llm).run("data/sample_repo", file_list=["app/auth/service.py"], context="只看认证")

        user_content = llm.calls[0]["messages"][1]["content"]
        assert "data/sample_repo" in user_content
        assert "app/auth/service.py" in user_content
        assert "只看认证" in user_content
        assert "不超过 5 条" in user_content
