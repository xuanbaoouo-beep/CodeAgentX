"""多 Agent 审查流水线：把七个角色串成一条可观测、可恢复的链路。

阶段顺序
--------
``plan → retrieve → review → security →（test）→（refactor）→ report``

四条硬约束
----------
1. **每步留痕**：每个阶段都写进 :class:`~codeagentx.orchestrator.state.WorkflowState`，
   并区分 ``done / failed / skipped``——"跳过"和"失败"不是一回事，
   报告读者必须能看出哪一步没跑、为什么没跑。
2. **失败不静默**：单阶段异常只影响该阶段（记 ``failed`` 后继续），
   ``report`` 阶段恒执行，最终报告的 ``metadata.degraded`` 会置真、摘要带警告。
   唯一不允许的是"某一步没跑却没人知道"。
3. **恢复要真恢复**：各阶段产物（计划 / 证据 / 结论）写入 ``state.artifacts`` 并落盘，
   ``resume=True`` 时复用已完成的阶段，不重复烧 Token。
4. **用量可查**：每个角色返回的用量累加进 ``state.metadata["usage"]``；
   若外部注入了 LLM，还会额外记录 ``state.metadata["llm_usage"]``
   （两者差异来自不经 :class:`AgentResult` 的直连调用，例如重构规划）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from codeagentx.agents.planner import PlannerAgent
from codeagentx.agents.refactor import RefactorAgent
from codeagentx.agents.reflection import ReflectionReviewer
from codeagentx.agents.reporter import ReporterAgent
from codeagentx.agents.retriever import RetrieverAgent
from codeagentx.agents.reviewer import ReviewerAgent
from codeagentx.agents.schemas import (
    Evidence,
    RefactorPlan,
    ReviewPlan,
    ReviewReport,
    parse_review_report,
)
from codeagentx.agents.security import SecurityAgent
from codeagentx.agents.tester import TesterAgent
from codeagentx.agents.toolkit import (
    DEFAULT_REVIEW_TOOLS,
    FULL_REVIEW_TOOLS,
    build_review_toolkit,
)
from codeagentx.config import Config, get_config
from codeagentx.context import ContextDocument, GitHubSource, documents_from_github
from codeagentx.core.agent import ZERO_USAGE, usage_delta
from codeagentx.core.exceptions import GitHubError
from codeagentx.core.llm import BaseLLM, build_llm
from codeagentx.core.logger import get_logger, log_event
from codeagentx.orchestrator.state import WorkflowState
from codeagentx.tools.registry import ToolRegistry

logger = get_logger("orchestrator.workflow")

#: 可注入的角色名（与 :data:`~codeagentx.orchestrator.state.STAGES` 一一对应）
ROLES: tuple[str, ...] = ("plan", "retrieve", "review", "security", "test", "refactor", "report")
#: 扫描代码文件时忽略的目录
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
        ".idea",
        ".vscode",
        "logs",
    }
)
#: 默认最多列出多少个文件给 Planner（避免上下文被文件清单撑爆）
DEFAULT_MAX_FILES = 200
#: 单次流水线最多发起的检索查询数（每个查询各取 top_k 条）
MAX_QUERIES = 6


@dataclass
class StageOutcome:
    """阶段的业务结果：``error`` 非空即表示这一步不可信。"""

    detail: str = ""
    error: str = ""


@dataclass
class WorkflowResult:
    """流水线产出：最终报告 + 状态。"""

    report: ReviewReport
    state: WorkflowState

    @property
    def success(self) -> bool:
        """是否所有阶段都成功，且报告未被标记降级。"""
        return not self.state.failed_stages() and not bool(self.report.metadata.get("degraded"))

    def to_markdown(self) -> str:
        return self.report.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "report": self.report.to_dict(),
            "state": self.state.to_dict(),
        }


def collect_python_files(root: str | Path, *, limit: int = DEFAULT_MAX_FILES) -> list[str]:
    """列出仓库内的 Python 文件（相对路径、排序、忽略虚拟环境与缓存目录）。"""
    base = Path(root)
    if not base.is_dir():
        return [base.as_posix()] if base.suffix == ".py" else []
    files: list[str] = []
    for path in sorted(base.rglob("*.py")):
        parts = path.relative_to(base).parts[:-1]
        if any(part in IGNORED_DIRS for part in parts):
            continue
        files.append(path.relative_to(base).as_posix())
        if len(files) >= limit:
            break
    return files


def _evidence_from_document(document: ContextDocument, *, repo: str) -> Evidence:
    """把上下文文档转成证据（W7：远端仓库文件进入审查证据链的最后一步）。

    ``source`` 固定标成 ``github``：报告里必须一眼看出"这条证据来自远端仓库"，
    而不是被当成"检索器从本地索引里找出来的"。
    """
    query = f"github:{repo}" if repo else "github"
    return Evidence(
        path=document.path,
        start_line=document.start_line,
        end_line=document.end_line,
        kind=document.kind or "file",
        symbol=document.name,
        query=query,
        sources=["github"],
        score=document.score,
        snippet=document.content,
    )


class CodeReviewWorkflow:
    """七个角色协作的代码审查流水线。"""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        *,
        root: str | Path,
        target: str | None = None,
        paths: Sequence[str] | None = None,
        tools: ToolRegistry | None = None,
        config: Config | None = None,
        state_path: str | Path | None = None,
        enable_test: bool = False,
        enable_refactor: bool = False,
        reflect: bool = False,
        max_iterations: int = 6,
        max_files: int = DEFAULT_MAX_FILES,
        agents: dict[str, Any] | None = None,
        allow_mock: bool = False,
        github: GitHubSource | None = None,
    ) -> None:
        """
        Args:
            llm: LLM 实例；``None`` 时按配置延迟构建（``allow_mock`` 决定可否降级为 MockLLM）。
            root: 待审查仓库的根目录（所有工具共用它作为沙箱允许根）。
            target: 报告里显示的审查目标，缺省用 ``root`` 相对当前目录的路径。
            paths: 审查范围（相对 ``root`` 的路径或绝对路径）；给了它**只审这些文件**。
                ``None`` 表示整个 ``root``。沙箱根仍然是目录（RAG 索引与路径守卫都
                要求目录），但规划、取证与最终报告都被限制在范围内——给单文件审查
                走的就是这条路。
            tools: 工具集；``None`` 时按 ``enable_test`` 自动装配。
            state_path: 状态文件路径；给了它才具备"中断恢复"能力。
            enable_test: 是否启用测试阶段（生成复现测试 + 跑现有用例）。
            enable_refactor: 是否启用重构规划阶段（只出计划，不改代码）。
            reflect: 是否让主审查角色使用 Reflection 范式（W9 消融实验用）。
            agents: 注入自定义角色实例（键为 :data:`ROLES` 之一），测试用。
            github: 额外从远端 GitHub 仓库取证据（见
                :class:`~codeagentx.context.builder.GitHubSource`，只读）。
                取不到时**只降级为"仅本地证据"并留痕**——远端不可用不该让整条流水线失败。
        """
        overrides = dict(agents or {})
        unknown = sorted(set(overrides) - set(ROLES))
        if unknown:
            raise ValueError(f"未知的角色名：{unknown}，可选 {list(ROLES)}")

        self.config = config or get_config()
        self.root = Path(root).resolve()
        #: 本次审查的范围（相对 root 的 posix 路径；空元组表示整个仓库）
        self.paths = self._normalize_paths(paths)
        self.llm = llm
        self.allow_mock = allow_mock
        self._tools = tools
        self.state_path = Path(state_path) if state_path else None
        self.enable_test = enable_test
        self.enable_refactor = enable_refactor
        self.reflect = reflect
        self.max_iterations = max_iterations
        self.max_files = max_files
        self.github = github

        self._overrides = overrides
        self._instances: dict[str, Any] = {}
        self._resume = False

        self.state = WorkflowState.create(target or self._display_target())
        self._plan: ReviewPlan | None = None
        self._evidence: list[Evidence] = []
        self._reports: list[ReviewReport] = []
        self._test_payload: dict[str, Any] | None = None
        self._refactor_plan: RefactorPlan | None = None
        self._final_report: ReviewReport | None = None

    # ------------------------------------------------------------ 主入口
    def run(
        self,
        target: str | None = None,
        *,
        resume: bool = False,
        reset: bool = False,
    ) -> WorkflowResult:
        """执行（或恢复）一次完整审查。

        Args:
            target: 覆盖报告中的审查目标。
            resume: 从 ``state_path`` 读取状态，跳过已完成的阶段。
            reset: 先丢弃已有状态再从头执行（与 ``resume`` 互斥使用）。
        """
        if target:
            self.state.target = target
        if reset:
            self._discard_state()
        self._resume = resume
        if resume:
            self._load_state()
        self._restore_artifacts()

        usage_before = self._usage_snapshot()
        started = time.monotonic()
        self._stage_plan()
        self._stage_retrieve()
        self._stage_review()
        self._stage_security()
        self._stage_test()
        self._stage_refactor()
        report = self._stage_report()
        usage_after = self._usage_snapshot()

        if usage_before and usage_after:
            self.state.metadata["llm_usage"] = usage_delta(usage_before, usage_after)
        self.state.metadata["duration"] = round(time.monotonic() - started, 3)
        self.state.metadata["success"] = not self.state.failed_stages()
        self._persist()

        log_event(
            logger,
            "workflow_finished",
            target=self.state.target,
            stages=self.state.describe(),
            findings=report.total,
            success=self.state.metadata["success"],
        )
        return WorkflowResult(report=report, state=self.state)

    @property
    def plan(self) -> ReviewPlan | None:
        """本次流水线使用的审查计划（未运行时为 ``None``）。"""
        return self._plan

    @property
    def evidence(self) -> list[Evidence]:
        """本次流水线收集到的检索证据。"""
        return list(self._evidence)

    @property
    def reports(self) -> list[ReviewReport]:
        """各审查角色单独产出的报告（合并前的原始结论）。"""
        return list(self._reports)

    # ------------------------------------------------------------ 各阶段
    def _stage_plan(self) -> None:
        def action() -> StageOutcome:
            # 设了审查范围时直接用范围本身：既避免被 max_files 截断，
            # 也保证"给单文件就只审单文件"
            files = (
                list(self.paths)
                if self.paths
                else collect_python_files(self.root, limit=self.max_files)
            )
            result = self._agent("plan").run(
                self.state.target, file_list=files, context=self._scope_hint()
            )
            self._record_usage(result.usage)
            plan = ReviewPlan.from_dict(result.metadata.get("plan") or {})
            self._plan = plan
            self.state.artifacts["plan"] = plan.to_dict()
            if plan.metadata.get("fallback"):
                return StageOutcome(
                    detail=f"{plan.total} 条子任务（默认兜底计划）",
                    error=result.error or "规划降级为默认计划",
                )
            return StageOutcome(detail=f"{plan.total} 条子任务")

        self._stage("plan", action)

    def _stage_retrieve(self) -> None:
        def action() -> StageOutcome:
            queries = self._retrieval_queries()
            # 索引只建一次：chunk_id 由路径与位置决定，重复索引只覆盖不累积
            index_root = None if self.state.artifacts.get("indexed") else self.root
            result = self._agent("retrieve").run(queries, index_root=index_root)
            if index_root is not None and result.success:
                self.state.artifacts["indexed"] = True
            self._evidence = [
                Evidence.from_dict(item) for item in (result.metadata.get("evidence") or [])
            ]
            remote, note = self._github_evidence()
            self._evidence.extend(remote)
            # 远端证据也一并受范围约束：本次只审哪些文件，证据就只来自哪些文件
            self._evidence = self._scope_evidence(self._evidence)
            self.state.artifacts["evidence"] = [item.to_dict() for item in self._evidence]
            detail = f"{len(self._evidence)} 条证据 / {len(queries)} 次查询"
            if remote:
                detail += f"（含 GitHub {len(remote)} 条）"
            if note:
                detail += f"；{note}"
            if not result.success:
                return StageOutcome(detail=detail, error=result.error or "检索失败")
            return StageOutcome(detail=detail)

        self._stage("retrieve", action)

    def _github_evidence(self) -> tuple[list[Evidence], str]:
        """W7：从远端 GitHub 仓库取文件并转成证据（只读）。

        为什么这次转换放在编排层：``context`` 层只认"有位置、有内容"的文档，
        ``agents`` 层只认 :class:`Evidence`，把两者接起来是编排层的职责——
        这样 ``context`` 与 ``agents`` 谁都不用反向依赖对方。

        远端不可用时**只降级**：返回空列表 + 一句说明写进阶段详情，
        让报告摘要里看得见"少了远端证据"，而不是悄悄当成"远端没有内容"。
        """
        if self.github is None:
            return [], ""
        try:
            documents = documents_from_github(
                self.github.client,
                self.github.repo,
                paths=self.github.paths,
                path_prefix=self.github.path_prefix,
                ref=self.github.ref,
                max_files=self.github.max_files,
            )
        except GitHubError as exc:
            logger.warning("GitHub 证据获取失败，本次只用本地证据：%s", exc)
            return [], f"GitHub 证据获取失败（{exc}）"

        seen = {(item.path, item.start_line, item.end_line) for item in self._evidence}
        evidence: list[Evidence] = []
        for document in documents:
            converted = _evidence_from_document(document, repo=self.github.repo)
            key = (converted.path, converted.start_line, converted.end_line)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(converted)
        return evidence, ""

    def _stage_review(self) -> None:
        def action() -> StageOutcome:
            result = self._agent("review").run(
                target=self.state.target, hints=self._scope_hint(), evidence=self._evidence_digest()
            )
            self._record_usage(result.usage)
            report = self._report_from(result)
            self._reports.append(report)
            self.state.artifacts["review"] = report.to_dict()
            return self._outcome_for(report, result)

        self._stage("review", action)

    def _stage_security(self) -> None:
        def action() -> StageOutcome:
            result = self._agent("security").run(
                target=self.state.target, hints=self._scope_hint(), evidence=self._evidence_digest()
            )
            self._record_usage(result.usage)
            report = self._report_from(result)
            self._reports.append(report)
            self.state.artifacts["security"] = report.to_dict()
            return self._outcome_for(report, result)

        self._stage("security", action)

    def _stage_test(self) -> None:
        if self._should_skip("test"):
            return
        if not self.enable_test:
            self.state.set_stage("test", "skipped", detail="未启用（enable_test=False）")
            self._persist()
            return

        def action() -> StageOutcome:
            result = self._agent("test").run(
                self.state.target, finding=self._top_finding_text(), run_existing=True
            )
            self._record_usage(result.usage)
            test_result = result.metadata.get("test_result")
            self._test_payload = {
                "test_code": result.metadata.get("test_code") or "",
                "test_result": test_result,
                "success": result.success,
            }
            self.state.artifacts["test"] = self._test_payload
            # "写了测试"与"跑了测试"是两件事：只有真的跑了才在详情里这么说
            ran_existing = bool(isinstance(test_result, dict) and test_result.get("available"))
            detail = "已生成复现测试" + ("，并运行现有用例" if ran_existing else "（未运行现有用例）")
            if not result.success:
                return StageOutcome(detail=detail, error=result.error or "测试阶段未完成")
            return StageOutcome(detail=detail)

        self._stage("test", action)

    def _stage_refactor(self) -> None:
        if self._should_skip("refactor"):
            return
        if not self.enable_refactor:
            self.state.set_stage("refactor", "skipped", detail="未启用（enable_refactor=False）")
            self._persist()
            return

        def action() -> StageOutcome:
            findings = [
                item for report in self._scope_reports(self._reports) for item in report.findings
            ]
            plan = self._agent("refactor").plan_from_findings(findings, scope=self.state.target)
            self._refactor_plan = plan
            self.state.artifacts["refactor"] = plan.to_dict()
            # 只出计划不改代码：重构是写操作的前置，必须在人确认后手动触发
            return StageOutcome(detail=f"{plan.total} 步（仅规划）")

        self._stage("refactor", action)

    def _stage_report(self) -> ReviewReport:
        if self._should_skip("report") and self._final_report is not None:
            return self._final_report

        def action() -> StageOutcome:
            result = self._agent("report").run(
                self.state.target,
                reports=self._scope_reports(self._reports),
                plan=self._plan,
                evidence=self._evidence,
                test_result=(self._test_payload or {}).get("test_result"),
                workflow=self.state.snapshot(),
                extra_metadata={
                    "usage": self.state.metadata.get("usage", dict(ZERO_USAGE)),
                    "stages": self.state.describe(),
                },
            )
            report = self._report_from(result)
            self._final_report = report
            self.state.artifacts["report"] = report.to_dict()
            detail = f"{report.total} 条问题"
            if report.metadata.get("degraded"):
                # 降级不等于本阶段失败：报告确实产出了，降级信息由 metadata.degraded 与
                # failed_stages 表达（见 Reporter 的摘要警告），阶段本身仍记 done
                return StageOutcome(detail=f"{detail}（含降级环节）")
            if not result.success:
                return StageOutcome(detail=detail, error=result.error or "汇总未完成")
            return StageOutcome(detail=detail)

        self._stage("report", action)
        if self._final_report is None:
            # 汇总阶段自身失败：仍然返回一份"说明失败"的报告，
            # 绝不能用空报告冒充"没有发现问题"
            failure = self.state.stage("report").error or "未知原因"
            self._final_report = ReviewReport(
                target=self.state.target,
                summary=f"汇总阶段失败（{failure}），本次未能产出可用报告。",
            )
        else:
            # 阶段状态在 report 完成后才最终确定：补一次快照，
            # 否则报告里会写着 report=pending，看起来像这一步没跑
            self._final_report.metadata["workflow"] = self.state.snapshot()
            self.state.artifacts["report"] = self._final_report.to_dict()
        return self._final_report

    # ------------------------------------------------------------ 阶段脚手架
    def _stage(self, name: str, action: Callable[[], StageOutcome]) -> bool:
        """执行一个阶段：成功记 ``done``，业务失败记 ``failed``，异常也记 ``failed``。

        异常只影响本阶段：流水线继续往下走（``report`` 恒执行），
        最终报告会带上 ``degraded`` 标记。这样"某一步没跑成"永远可见。
        """
        if self._should_skip(name):
            return True
        started = time.monotonic()
        try:
            outcome = action()
        except Exception as exc:  # noqa: BLE001 - 单阶段失败不应终止整条流水线
            logger.debug("阶段 %s 异常", name, exc_info=True)
            log_event(
                logger,
                "stage_crashed",
                level=logging.ERROR,
                stage=name,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.state.set_stage(
                name,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                duration=time.monotonic() - started,
            )
            self._persist()
            return False
        self.state.set_stage(
            name,
            "failed" if outcome.error else "done",
            detail=outcome.detail,
            error=outcome.error,
            duration=time.monotonic() - started,
        )
        self._persist()
        return not outcome.error

    def _should_skip(self, name: str) -> bool:
        """恢复模式下，已有最终结果的阶段直接跳过（不重复烧 Token）。"""
        return self._resume and self.state.is_settled(name)

    def _persist(self) -> None:
        if self.state_path is not None:
            self.state.save(self.state_path)

    # ------------------------------------------------------------ 角色与资源
    def _agent(self, role: str) -> Any:
        if role in self._instances:
            return self._instances[role]
        agent = self._overrides.get(role) or self._create_agent(role)
        self._instances[role] = agent
        return agent

    def _create_agent(self, role: str) -> Any:
        if role == "retrieve":
            return RetrieverAgent(self._tools_instance())
        if role == "report":
            return ReporterAgent()
        llm = self._llm_instance()
        if role == "plan":
            return PlannerAgent(llm)
        if role == "review":
            review_cls = ReflectionReviewer if self.reflect else ReviewerAgent
            return review_cls(llm, tools=self._tools_instance(), max_iterations=self.max_iterations)
        if role == "security":
            return SecurityAgent(llm, tools=self._tools_instance(), max_iterations=self.max_iterations)
        if role == "test":
            return TesterAgent(llm, tools=self._tools_instance())
        if role == "refactor":
            return RefactorAgent(llm, tools=self._tools_instance())
        raise ValueError(f"未知的角色名：{role}")  # pragma: no cover - 构造时已校验

    def _llm_instance(self) -> BaseLLM:
        if self.llm is None:
            self.llm = build_llm(self.config, allow_mock=self.allow_mock)
        return self.llm

    def _tools_instance(self) -> ToolRegistry:
        if self._tools is None:
            names = FULL_REVIEW_TOOLS if self.enable_test else DEFAULT_REVIEW_TOOLS
            self._tools = build_review_toolkit(self.config, root=self.root, tools=names)
        return self._tools

    # ------------------------------------------------------------ 状态与产物
    def _discard_state(self) -> None:
        if self.state_path is not None:
            try:
                self.state_path.unlink(missing_ok=True)
            except OSError as exc:  # 删不掉不影响重新开始
                log_event(
                    logger,
                    "state_discard_failed",
                    level=logging.WARNING,
                    path=str(self.state_path),
                    error=str(exc),
                )
        self.state = WorkflowState.create(self._display_target())
        self._plan = None
        self._evidence = []
        self._reports = []
        self._test_payload = None
        self._refactor_plan = None
        self._final_report = None

    def _load_state(self) -> None:
        if self.state_path is None:
            return
        loaded = WorkflowState.load(self.state_path)
        if loaded is None:
            log_event(
                logger,
                "workflow_resume_unavailable",
                level=logging.WARNING,
                path=str(self.state_path),
            )
            return
        self.state = loaded
        if not self.state.target:
            self.state.target = self._display_target()

    def _restore_artifacts(self) -> None:
        """从状态里恢复各阶段产物（恢复模式的关键：不重复烧 Token）。"""
        artifacts = self.state.artifacts
        payload = artifacts.get("plan")
        if isinstance(payload, dict):
            self._plan = ReviewPlan.from_dict(payload)
        items = artifacts.get("evidence")
        if isinstance(items, list):
            self._evidence = [Evidence.from_dict(item) for item in items]
        self._reports = []
        for key in ("review", "security"):
            payload = artifacts.get(key)
            if isinstance(payload, dict):
                self._reports.append(ReviewReport.from_dict(payload))
        payload = artifacts.get("test")
        if isinstance(payload, dict):
            self._test_payload = payload
        payload = artifacts.get("refactor")
        if isinstance(payload, dict):
            self._refactor_plan = RefactorPlan.from_dict(payload)
        payload = artifacts.get("report")
        if isinstance(payload, dict):
            self._final_report = ReviewReport.from_dict(payload)

    def _record_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        totals = self.state.metadata.setdefault("usage", dict(ZERO_USAGE))
        for key in ZERO_USAGE:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] = totals[key] + value
        totals["latency"] = round(totals["latency"], 3)

    def _usage_snapshot(self) -> dict[str, Any] | None:
        return self.llm.stats.snapshot() if self.llm is not None else None

    # ------------------------------------------------------------ 小工具
    def _display_target(self) -> str:
        try:
            return self.root.relative_to(Path.cwd()).as_posix() or "."
        except ValueError:
            return self.root.as_posix()

    # ------------------------------------------------------------ 审查范围
    def _normalize_paths(self, paths: Sequence[str] | None) -> tuple[str, ...]:
        """把"只审这几个文件"规整成相对 ``root`` 的 posix 路径。

        路径跳出 ``root`` 或根本不存在时**直接报错**：静默退化成"审整个仓库"
        会让用户以为只审了一个文件，这种误解比报错更贵。
        """
        if not paths:
            return ()
        normalized: list[str] = []
        for item in paths:
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            candidate = candidate.resolve()
            if not candidate.is_relative_to(self.root):
                raise ValueError(f"审查范围超出仓库根目录：{item}")
            if not candidate.is_file():
                raise ValueError(f"审查范围内的文件不存在：{item}")
            relative = candidate.relative_to(self.root).as_posix()
            if relative not in normalized:
                normalized.append(relative)
        return tuple(normalized)

    @staticmethod
    def _same_file(left: str, right: str) -> bool:
        """两份"文件标识"指的是不是同一个文件。

        模型写的路径未必与仓库内的相对路径完全一致（可能带仓库名前缀、
        反斜杠或大小写差异），所以按归一化后"相等或收尾匹配"判断：
        宁可少剔除一条，也不要静默丢掉一条真问题。
        """

        def normalize(value: str) -> str:
            return value.replace("\\", "/").strip("/").lstrip("./").lower()

        first, second = normalize(left), normalize(right)
        if not first or not second:
            return False
        return first == second or first.endswith(f"/{second}") or second.endswith(f"/{first}")

    def _in_scope(self, path: str) -> bool:
        """``path`` 是否落在本次审查范围内（未设范围时一律为真）。"""
        if not self.paths:
            return True
        return any(self._same_file(path, item) for item in self.paths)

    def _scope_hint(self) -> str:
        """写进提示词的"本次只看这些文件"说明（未设范围时为空串）。"""
        if not self.paths:
            return ""
        return (
            f"本次审查范围**仅限**以下文件：{'、'.join(self.paths)}。"
            "范围之外的代码只当作理解上下文，不作为上报对象。"
        )

    def _scope_evidence(self, evidence: Sequence[Evidence]) -> list[Evidence]:
        """剔除检索到的范围外证据，把取证限制在本次审查的对象上。"""
        if not self.paths:
            return list(evidence)
        kept = [item for item in evidence if self._in_scope(item.path)]
        if len(kept) != len(evidence):
            logger.info(
                "剔除 %d 条审查范围之外的证据（本次只审 %s）",
                len(evidence) - len(kept),
                "、".join(self.paths),
            )
        return kept

    def _scope_reports(self, reports: Sequence[ReviewReport]) -> list[ReviewReport]:
        """剔除报告里范围之外的问题（未设范围时原样返回）。"""
        if not self.paths:
            return list(reports)
        scoped: list[ReviewReport] = []
        dropped = 0
        for report in reports:
            kept = [item for item in report.findings if self._in_scope(item.file)]
            dropped += len(report.findings) - len(kept)
            scoped.append(
                report if len(kept) == len(report.findings) else replace(report, findings=kept)
            )
        if dropped:
            logger.info(
                "剔除 %d 条审查范围之外的问题（本次只审 %s）", dropped, "、".join(self.paths)
            )
        return scoped

    def _retrieval_queries(self) -> list[str]:
        queries: list[str] = []
        if self._plan is not None:
            for task in self._plan.tasks:
                if task.description and task.description not in queries:
                    queries.append(task.description)
        if not queries:
            queries.append(self.state.target or self.root.name)
        return queries[:MAX_QUERIES]

    def _evidence_digest(self) -> str:
        if not self._evidence:
            return ""
        digest = getattr(self._agent("retrieve"), "evidence_digest", None)
        if callable(digest):
            return digest(self._evidence)
        return "\n\n".join(item.to_text() for item in self._evidence)

    def _report_from(self, result: Any) -> ReviewReport:
        payload = result.metadata.get("report")
        if isinstance(payload, dict):
            return ReviewReport.from_dict(payload)
        return parse_review_report(result.output, target=self.state.target, source=result.metadata.get("agent") or "")

    def _outcome_for(self, report: ReviewReport, result: Any) -> StageOutcome:
        detail = f"{report.total} 条问题"
        parse_error = report.metadata.get("parse_error")
        if parse_error:
            # 解析失败 ≠ 没有问题：阶段必须记为失败，否则报告会"显得很干净"
            return StageOutcome(detail=detail, error=str(parse_error))
        if not result.success:
            return StageOutcome(detail=detail, error=result.error or "角色未收敛")
        return StageOutcome(detail=detail)

    def _top_finding_text(self) -> str:
        findings = [
            item for report in self._scope_reports(self._reports) for item in report.findings
        ]
        if not findings:
            return ""
        top = ReviewReport(findings=findings).sorted_findings()[0]
        location = top.location or "未定位"
        return f"[{top.severity}][{top.category}] {top.title} @ {location}\n{top.description}"


__all__ = [
    "DEFAULT_MAX_FILES",
    "IGNORED_DIRS",
    "MAX_QUERIES",
    "ROLES",
    "CodeReviewWorkflow",
    "StageOutcome",
    "WorkflowResult",
    "collect_python_files",
]
