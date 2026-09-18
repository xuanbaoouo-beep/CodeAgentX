"""提示词模板：集中管理 Agent 的系统提示与任务模板，保证输出契约稳定。"""

from codeagentx.prompts.agents import (
    PLANNER_CONTRACT,
    PLANNER_SYSTEM,
    REVIEWER_SYSTEM,
    SECURITY_SYSTEM,
    TESTER_SYSTEM,
    build_focused_review_task,
    build_planner_task,
    build_test_task,
)
from codeagentx.prompts.refactor import (
    PLAN_SYSTEM,
    SOLVE_SYSTEM,
    build_plan_task,
    build_solve_task,
)
from codeagentx.prompts.review import (
    CRITIQUE_SYSTEM,
    DIRECT_REVIEW_SYSTEM,
    JSON_REPAIR_PROMPT,
    REACT_REVIEW_SYSTEM,
    REPORT_CONTRACT,
    REVISE_SYSTEM,
    build_critique_task,
    build_direct_review_task,
    build_react_review_task,
    build_revise_task,
)

__all__ = [
    "CRITIQUE_SYSTEM",
    "DIRECT_REVIEW_SYSTEM",
    "JSON_REPAIR_PROMPT",
    "PLANNER_CONTRACT",
    "PLANNER_SYSTEM",
    "PLAN_SYSTEM",
    "REACT_REVIEW_SYSTEM",
    "REPORT_CONTRACT",
    "REVIEWER_SYSTEM",
    "REVISE_SYSTEM",
    "SECURITY_SYSTEM",
    "SOLVE_SYSTEM",
    "TESTER_SYSTEM",
    "build_critique_task",
    "build_direct_review_task",
    "build_focused_review_task",
    "build_plan_task",
    "build_planner_task",
    "build_react_review_task",
    "build_revise_task",
    "build_solve_task",
    "build_test_task",
]
