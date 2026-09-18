"""Agent 层：七类审查角色 + 三种经典范式。

角色与范式的分工
----------------
- **范式**（W5）：:class:`ReActReviewer` / :class:`PlanSolveRefactor` / :class:`Reflection`，
  描述"怎么思考"；
- **角色**（W6）：Planner / Retriever / Reviewer / Security / Tester / Refactor / Reporter，
  描述"负责哪一块"，其中 Reviewer 与 Security 直接复用 ReAct 范式的取证与解析流程。

所有角色都能被 :class:`~codeagentx.orchestrator.workflow.CodeReviewWorkflow` 以
``agent.run(...)`` 的同一方式调用，返回 :class:`~codeagentx.core.agent.AgentResult`。
"""

from codeagentx.agents.plan_solve_refactor import (
    DEFAULT_GOAL,
    DEFAULT_MAX_STEPS,
    PlanSolveRefactor,
)
from codeagentx.agents.planner import (
    FALLBACK_TASKS,
    PlannerAgent,
)
from codeagentx.agents.react_reviewer import DEFAULT_TASK, ReActReviewer
from codeagentx.agents.refactor import (
    MAX_FINDINGS_IN_GOAL,
    RefactorAgent,
    build_goal_from_findings,
)
from codeagentx.agents.reflection import (
    DEFAULT_ACCEPT_SCORE,
    DEFAULT_MAX_ROUNDS,
    Critique,
    Reflection,
    ReflectionResult,
    ReflectionReviewer,
    ReflectionRound,
    parse_critique,
)
from codeagentx.agents.reporter import DEGRADED_WARNING, ReporterAgent
from codeagentx.agents.retriever import RetrieverAgent
from codeagentx.agents.reviewer import (
    DEFAULT_FOCUSED_TASK,
    FOCUSED_REVIEW_TOOLS,
    FocusedReviewAgent,
    ReviewerAgent,
)
from codeagentx.agents.schemas import (
    ASSIGNEES,
    CATEGORIES,
    SEVERITIES,
    Evidence,
    Finding,
    PlanStep,
    RefactorPlan,
    ReviewPlan,
    ReviewReport,
    SubTask,
    parse_review_plan,
    parse_review_report,
)
from codeagentx.agents.security import SecurityAgent
from codeagentx.agents.tester import TesterAgent, extract_test_code
from codeagentx.agents.toolkit import (
    DEFAULT_REVIEW_TOOLS,
    FULL_REVIEW_TOOLS,
    SUPPORTED_TOOLS,
    build_review_toolkit,
    describe_toolkit,
)

__all__ = [
    "ASSIGNEES",
    "CATEGORIES",
    "DEFAULT_ACCEPT_SCORE",
    "DEFAULT_FOCUSED_TASK",
    "DEFAULT_GOAL",
    "DEFAULT_MAX_ROUNDS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_REVIEW_TOOLS",
    "DEFAULT_TASK",
    "DEGRADED_WARNING",
    "FALLBACK_TASKS",
    "FOCUSED_REVIEW_TOOLS",
    "FULL_REVIEW_TOOLS",
    "MAX_FINDINGS_IN_GOAL",
    "SEVERITIES",
    "SUPPORTED_TOOLS",
    "Critique",
    "Evidence",
    "Finding",
    "FocusedReviewAgent",
    "PlanSolveRefactor",
    "PlanStep",
    "PlannerAgent",
    "ReActReviewer",
    "RefactorAgent",
    "RefactorPlan",
    "Reflection",
    "ReflectionResult",
    "ReflectionReviewer",
    "ReflectionRound",
    "ReporterAgent",
    "RetrieverAgent",
    "ReviewPlan",
    "ReviewReport",
    "ReviewerAgent",
    "SecurityAgent",
    "SubTask",
    "TesterAgent",
    "build_goal_from_findings",
    "build_review_toolkit",
    "describe_toolkit",
    "extract_test_code",
    "parse_critique",
    "parse_review_plan",
    "parse_review_report",
]
