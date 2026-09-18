"""七角色流水线测试（编排层）。

这里用替身角色（``agents=`` 注入）把 LLM 与工具全部隔离，只验证编排契约：
- 阶段顺序与状态：``done / failed / skipped`` 三者不能混为一谈；
- 单阶段失败**不终止**整条流水线：``report`` 恒执行，报告带 ``degraded``；
- 「跳过」与「失败」都要能在最终报告里看到；
- 状态落盘后可恢复，且恢复时不再重复调用已完成阶段的角色。

唯一不替身的是 Reporter（真实实现），因为"报告是否如实反映降级"正是要验证的东西。
"""

from __future__ import annotations

import pytest

from codeagentx.agents.reporter import DEGRADED_WARNING, ReporterAgent
from codeagentx.agents.schemas import (
    Evidence,
    Finding,
    PlanStep,
    RefactorPlan,
    ReviewPlan,
    ReviewReport,
    SubTask,
)
from codeagentx.agents.toolkit import build_review_toolkit
from codeagentx.context import GitHubSource
from codeagentx.core.agent import AgentResult
from codeagentx.core.exceptions import GitHubAuthError
from codeagentx.core.llm import MockLLM
from codeagentx.orchestrator.state import STAGES, WorkflowState
from codeagentx.orchestrator.workflow import CodeReviewWorkflow, collect_python_files
from codeagentx.protocols.github_client import GitHubFile

#: 编排层注入的 LLM 只用于统计用量：所有角色都被替身接管，正常不会真的调用它
WORKFLOW_LLM_REPLY = '{"summary": "替身未接管时的兜底回复", "findings": []}'


# ---------------------------------------------------------------- 替身与夹具
class FakeAgent:
    """固定返回同一个 ``AgentResult``，并记录被调用的参数。"""

    def __init__(self, result: AgentResult, *, name: str = "fake"):
        self.result = result
        self.name = name
        self.calls: list[dict] = []

    def run(self, *args, **kwargs) -> AgentResult:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.result


class FakeRefactor:
    """重构角色的替身（只实现编排层用到的方法）。"""

    name = "refactor"

    def __init__(self, steps: int = 2):
        self.steps = steps
        self.calls: list[dict] = []

    def plan_from_findings(self, findings, *, scope: str = "", goal: str = "", context: str = ""):
        self.calls.append({"findings": list(findings), "scope": scope, "goal": goal})
        return RefactorPlan(
            goal=goal or "修复审查发现的问题",
            steps=[PlanStep(description=f"第 {index} 步") for index in range(1, self.steps + 1)],
        )


class ExplodingAgent:
    """调用即抛异常，用于验证单阶段异常不会击穿整条流水线。"""

    name = "boom"

    def __init__(self):
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("阶段内部炸了")


def finding(
    title: str = "硬编码密钥",
    *,
    file: str = "app/config.py",
    line: int = 8,
    severity: str = "high",
    category: str = "security",
    source: str = "reviewer",
) -> Finding:
    return Finding(
        title=title,
        file=file,
        line=line,
        severity=severity,
        category=category,
        source=source,
        description="说明",
        suggestion="建议",
    )


def plan_result(*tasks: SubTask, fallback: bool = False) -> AgentResult:
    plan = ReviewPlan(
        target="repo",
        tasks=list(tasks) or [SubTask(description="审查登录逻辑", assignee="reviewer")],
    )
    if fallback:
        plan.metadata["fallback"] = True
        plan.metadata["fallback_reason"] = "模型输出无法解析为审查计划"
    return AgentResult(output="计划", metadata={"agent": "planner", "plan": plan.to_dict()})


def retrieve_result(
    *evidence: Evidence, success: bool = True, error: str | None = None
) -> AgentResult:
    if evidence:
        items = list(evidence)
    elif success:
        items = [
            Evidence(path="app/config.py", start_line=1, end_line=5, score=0.9, snippet="SECRET = 'x'")
        ]
    else:
        items = []  # 检索整体失败时不会带回证据
    return AgentResult(
        output="线索",
        success=success,
        error=error,
        metadata={
            "agent": "retriever",
            "queries": ["审查登录逻辑"],
            "count": len(items),
            "evidence": [item.to_dict() for item in items],
            "errors": [] if success else ["检索失败"],
            "notes": [],
        },
    )


def review_result(
    *findings: Finding,
    agent: str = "reviewer",
    success: bool = True,
    error: str | None = None,
    parse_error: str | None = None,
    usage: dict | None = None,
) -> AgentResult:
    report = ReviewReport(
        target="repo",
        summary=f"{agent} 的结论",
        findings=list(findings),
        metadata={"agent": agent},
    )
    if parse_error:
        report.metadata["parse_error"] = parse_error
    return AgentResult(
        output="报告",
        success=success,
        error=error,
        usage=dict(usage or {}),
        metadata={"agent": agent, "report": report.to_dict()},
    )


def fake_tester_result(*, available: bool = True) -> AgentResult:
    return AgentResult(
        output="测试代码",
        metadata={
            "agent": "tester",
            "test_code": "def test_x():\n    assert True",
            "test_result": {"available": available, "passed": available},
            "target": "repo",
        },
    )


def build(
    tmp_path,
    config,
    *,
    overrides: dict | None = None,
    tools=None,
    **kwargs,
) -> CodeReviewWorkflow:
    agents = {
        "plan": FakeAgent(plan_result()),
        "retrieve": FakeAgent(retrieve_result()),
        "review": FakeAgent(review_result(finding())),
        "security": FakeAgent(review_result(finding(source="security"), agent="security")),
        "test": FakeAgent(fake_tester_result()),
        "refactor": FakeRefactor(),
        "report": ReporterAgent(),
    }
    agents.update(overrides or {})
    return CodeReviewWorkflow(
        llm=MockLLM([WORKFLOW_LLM_REPLY]),
        root=tmp_path,
        target="repo",
        config=config,
        tools=tools,
        agents=agents,
        **kwargs,
    )


@pytest.fixture
def sample_root(tmp_path):
    """一个最小仓库：一个源码文件 + 一个被忽略目录。"""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "service.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.py").write_text("y = 1\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------- 阶段顺序与状态
class TestStageFlow:
    def test_stages_run_in_the_documented_order(self, tmp_path, config):
        workflow = build(tmp_path, config)
        result = workflow.run()

        assert [stage.name for stage in result.state.stages] == list(STAGES)
        assert result.state.status_of("plan") == "done"
        assert result.state.status_of("retrieve") == "done"
        assert result.state.status_of("review") == "done"
        assert result.state.status_of("security") == "done"
        assert result.state.status_of("report") == "done"

    def test_optional_stages_are_skipped_not_failed(self, tmp_path, config):
        workflow = build(tmp_path, config)
        result = workflow.run()

        assert result.state.status_of("test") == "skipped"
        assert result.state.status_of("refactor") == "skipped"
        assert "未启用（enable_test=False）" in result.state.stage("test").detail
        assert "未启用（enable_refactor=False）" in result.state.stage("refactor").detail
        # 跳过是设计如此，不能让整条流水线判失败
        assert result.state.failed_stages() == []
        assert result.success is True

    def test_plan_tasks_become_retrieval_queries(self, tmp_path, config):
        tasks = [SubTask(description=f"任务 {index}") for index in range(1, 4)]
        retrieve = FakeAgent(retrieve_result())
        workflow = build(tmp_path, config, overrides={"plan": FakeAgent(plan_result(*tasks)), "retrieve": retrieve})
        workflow.run()

        queries = retrieve.calls[0]["args"][0]
        assert queries == ["任务 1", "任务 2", "任务 3"]

    def test_index_is_built_only_once(self, tmp_path, config):
        retrieve = FakeAgent(retrieve_result())
        workflow = build(tmp_path, config, overrides={"retrieve": retrieve})
        workflow.run()
        workflow.run()

        assert str(retrieve.calls[0]["kwargs"]["index_root"]) == str(workflow.root)
        assert retrieve.calls[1]["kwargs"]["index_root"] is None  # 已有索引，不再重建

    def test_evidence_digest_is_handed_to_review_roles(self, tmp_path, config):
        review = FakeAgent(review_result(finding()))
        workflow = build(tmp_path, config, overrides={"review": review})
        workflow.run()

        assert "app/config.py" in review.calls[0]["kwargs"]["evidence"]

    def test_report_stage_covers_both_review_roles(self, tmp_path, config):
        workflow = build(tmp_path, config)
        result = workflow.run()

        assert len(workflow.reports) == 2
        assert result.state.stage("report").detail == "1 条问题"  # 同一问题被两个角色报出，合并成一条

    def test_usage_is_accumulated(self, tmp_path, config):
        review = FakeAgent(
            review_result(finding(), usage={"calls": 2, "total_tokens": 120, "latency": 0.5})
        )
        workflow = build(tmp_path, config, overrides={"review": review})
        result = workflow.run()

        assert result.state.metadata["usage"]["calls"] == 2
        assert result.state.metadata["usage"]["total_tokens"] == 120
        assert result.report.metadata["usage"]["calls"] == 2

    def test_artifacts_are_recorded_for_every_stage(self, tmp_path, config):
        workflow = build(tmp_path, config)
        workflow.run()

        for key in ("plan", "evidence", "review", "security", "report"):
            assert key in workflow.state.artifacts


class TestOptionalStages:
    def test_test_stage_runs_when_enabled(self, tmp_path, config):
        workflow = build(tmp_path, config, enable_test=True)
        result = workflow.run()

        assert result.state.status_of("test") == "done"
        assert "已生成复现测试" in result.state.stage("test").detail
        assert workflow.state.artifacts["test"]["test_code"].startswith("def test_x")
        assert result.report.metadata["test_result"]["passed"] is True

    def test_test_stage_does_not_claim_it_ran_existing_suite(self, tmp_path, config):
        """未真实运行现有用例时，阶段详情不能写"并运行现有用例"。"""
        workflow = build(
            tmp_path, config, enable_test=True, overrides={"test": FakeAgent(fake_tester_result(available=False))}
        )
        result = workflow.run()

        assert "未运行现有用例" in result.state.stage("test").detail

    def test_test_stage_failure_is_flagged(self, tmp_path, config):
        failed = AgentResult(output="", success=False, error="未能从模型输出中提取到测试代码")
        failed.metadata.update({"agent": "tester", "test_code": "", "test_result": None})
        workflow = build(tmp_path, config, enable_test=True, overrides={"test": FakeAgent(failed)})
        result = workflow.run()

        assert result.state.status_of("test") == "failed"
        assert result.state.stage("test").error == "未能从模型输出中提取到测试代码"

    def test_refactor_stage_only_plans(self, tmp_path, config):
        refactor = FakeRefactor(steps=3)
        workflow = build(tmp_path, config, enable_refactor=True, overrides={"refactor": refactor})
        result = workflow.run()

        assert result.state.status_of("refactor") == "done"
        assert "3 步（仅规划）" in result.state.stage("refactor").detail
        assert workflow.state.artifacts["refactor"]["steps"][0]["status"] == "pending"
        # 只做规划：不执行任何一步
        assert all(not step["result"] for step in workflow.state.artifacts["refactor"]["steps"])
        assert refactor.calls[0]["scope"] == "repo"


# ---------------------------------------------------------------- 失败与降级
class TestFailureHandling:
    def test_role_failure_marks_stage_failed_but_pipeline_continues(self, tmp_path, config):
        failing = FakeAgent(review_result(finding(), success=False, error="达到最大迭代轮次"))
        workflow = build(tmp_path, config, overrides={"review": failing})
        result = workflow.run()

        assert result.state.status_of("review") == "failed"
        assert result.state.stage("review").error == "达到最大迭代轮次"
        # 后面三个阶段照常执行
        assert result.state.status_of("security") == "done"
        assert result.state.status_of("report") == "done"

    def test_degraded_report_says_so(self, tmp_path, config):
        failing = FakeAgent(review_result(finding(), success=False, error="达到最大迭代轮次"))
        workflow = build(tmp_path, config, overrides={"review": failing})
        result = workflow.run()

        assert result.report.metadata["degraded"] is True
        assert result.report.metadata["failed_stages"] == ["review"]
        assert DEGRADED_WARNING in result.report.summary
        assert result.success is False
        # 报告确实产出了：降级不是"汇总这一步失败"
        assert result.state.status_of("report") == "done"
        assert "含降级环节" in result.state.stage("report").detail
        assert DEGRADED_WARNING in result.to_markdown()

    def test_report_stage_records_its_own_completion(self, tmp_path, config):
        """报告里的阶段快照不能停在 report=pending，否则看起来像这一步没跑。"""
        result = build(tmp_path, config).run()

        workflow = result.report.metadata["workflow"]
        statuses = {stage["name"]: stage["status"] for stage in workflow["stages"]}
        assert statuses["report"] == "done"
        assert statuses["test"] == "skipped"

    def test_unparsable_review_is_treated_as_failure(self, tmp_path, config):
        """解析失败 ≠ 没有问题：阶段必须记失败，否则报告会"显得很干净"。"""
        broken = FakeAgent(review_result(finding(), parse_error="输出不是 JSON"))
        workflow = build(tmp_path, config, overrides={"review": broken})
        result = workflow.run()

        assert result.state.status_of("review") == "failed"
        assert result.report.metadata["parse_errors"] == [{"agent": "reviewer", "error": "输出不是 JSON"}]
        assert result.report.metadata["degraded"] is True

    def test_stage_exception_is_contained(self, tmp_path, config):
        workflow = build(tmp_path, config, overrides={"review": ExplodingAgent()})
        result = workflow.run()

        assert result.state.status_of("review") == "failed"
        assert "RuntimeError" in result.state.stage("review").error
        assert result.state.status_of("report") == "done"
        assert DEGRADED_WARNING in result.report.summary

    def test_retrieval_failure_degrades_but_keeps_reviewing(self, tmp_path, config):
        workflow = build(
            tmp_path,
            config,
            overrides={"retrieve": FakeAgent(retrieve_result(success=False, error="索引不可用"))},
        )
        result = workflow.run()

        assert result.state.status_of("retrieve") == "failed"
        assert result.state.status_of("review") == "done"
        assert result.report.metadata["evidence_count"] == 0

    def test_fallback_plan_marks_the_plan_stage_failed(self, tmp_path, config):
        workflow = build(tmp_path, config, overrides={"plan": FakeAgent(plan_result(fallback=True))})
        result = workflow.run()

        assert result.state.status_of("plan") == "failed"
        assert "默认兜底计划" in result.state.stage("plan").detail
        assert result.report.metadata["degraded"] is True

    def test_missing_whitelisted_tool_becomes_a_failed_stage(self, tmp_path, config):
        """工具集缺少角色白名单里的工具时，阶段记 failed（可发现），而不是整条流水线崩掉。"""
        registry = build_review_toolkit(config, root=tmp_path, tools=("code_search",))
        workflow = build(tmp_path, config, tools=registry, overrides={"review": None})
        result = workflow.run()

        assert result.state.status_of("review") == "failed"
        assert "ToolNotFoundError" in result.state.stage("review").error
        assert result.state.status_of("report") == "done"

    def test_report_stage_failure_still_returns_an_explaining_report(self, tmp_path, config):
        """汇总阶段自己失败时，返回"本次未能产出可用报告"，绝不用空报告冒充没问题。"""
        workflow = build(tmp_path, config, overrides={"report": ExplodingAgent()})
        result = workflow.run()

        assert result.state.status_of("report") == "failed"
        assert "未能产出可用报告" in result.report.summary
        assert result.success is False

    def test_unknown_role_is_rejected(self, tmp_path, config):
        with pytest.raises(ValueError):
            build(tmp_path, config, overrides={"wizard": FakeAgent(retrieve_result())})


# ---------------------------------------------------------------- 落盘与恢复
class TestPersistence:
    def test_state_is_written_after_each_stage(self, tmp_path, config):
        state_path = tmp_path / "state" / "run.json"
        workflow = build(tmp_path, config, state_path=state_path)
        workflow.run()

        loaded = WorkflowState.load(state_path)
        assert loaded is not None
        assert loaded.status_of("report") == "done"
        assert loaded.artifacts["report"]["total"] == 1
        assert loaded.metadata["success"] is True
        assert loaded.metadata["duration"] >= 0

    def test_resume_skips_settled_stages(self, tmp_path, config):
        state_path = tmp_path / "run.json"
        first = build(tmp_path, config, state_path=state_path)
        first_report = first.run().report

        plan = FakeAgent(plan_result())
        review = FakeAgent(review_result(finding()))
        resumed = build(
            tmp_path, config, state_path=state_path, overrides={"plan": plan, "review": review}
        )
        result = resumed.run(resume=True)

        # 已完成阶段不再调用角色：不重复烧 Token
        assert plan.calls == []
        assert review.calls == []
        assert result.report.total == first_report.total
        assert result.report.summary == first_report.summary
        assert result.state.failed_stages() == []

    def test_resume_without_state_file_starts_over(self, tmp_path, config):
        plan = FakeAgent(plan_result())
        workflow = build(
            tmp_path, config, state_path=tmp_path / "missing.json", overrides={"plan": plan}
        )
        workflow.run(resume=True)

        assert len(plan.calls) == 1

    def test_reset_discards_previous_state(self, tmp_path, config):
        state_path = tmp_path / "run.json"
        plan = FakeAgent(plan_result())
        workflow = build(tmp_path, config, state_path=state_path, overrides={"plan": plan})
        workflow.run()
        workflow.run(reset=True)

        assert len(plan.calls) == 2
        assert state_path.exists()

    def test_report_metadata_carries_the_stage_summary(self, tmp_path, config):
        workflow = build(tmp_path, config)
        result = workflow.run()

        assert result.report.metadata["stages"].startswith("plan=done")
        assert result.report.metadata["workflow"]["target"] == "repo"

    def test_result_serialization(self, tmp_path, config):
        payload = build(tmp_path, config).run().to_dict()

        assert payload["success"] is True
        assert payload["report"]["target"] == "repo"
        assert payload["state"]["version"] == 1


class TestCollectPythonFiles:
    def test_ignores_caches_and_returns_relative_sorted_paths(self, sample_root):
        files = collect_python_files(sample_root)

        assert files == ["app/service.py"]

    def test_limit_is_respected(self, sample_root):
        assert len(collect_python_files(sample_root, limit=1)) == 1

    def test_single_file_is_returned_as_is(self, sample_root):
        target = sample_root / "app" / "service.py"
        assert collect_python_files(target) == [target.as_posix()]

    def test_non_python_file_yields_nothing(self, tmp_path):
        other = tmp_path / "notes.txt"
        other.write_text("x", encoding="utf-8")

        assert collect_python_files(other) == []

    def test_missing_path_yields_nothing(self, tmp_path):
        assert collect_python_files(tmp_path / "nope") == []


# ---------------------------------------------------------------- 审查范围（paths）
class TestReviewScope:
    """给单文件做审查：范围外的代码只当上下文，不进规划、不进证据、不上报。"""

    def test_plan_only_sees_files_in_scope(self, sample_root, config):
        plan = FakeAgent(plan_result())
        workflow = build(sample_root, config, overrides={"plan": plan}, paths=["app/service.py"])
        workflow.run()

        assert plan.calls[0]["kwargs"]["file_list"] == ["app/service.py"]
        assert "app/service.py" in plan.calls[0]["kwargs"]["context"]

    def test_scope_beats_file_limit(self, sample_root, config):
        """范围内的文件不能被 max_files 截断掉（否则"只审这个文件"会变成"审不到"）。"""
        plan = FakeAgent(plan_result())
        workflow = build(
            sample_root, config, overrides={"plan": plan}, paths=["app/service.py"], max_files=0
        )
        workflow.run()

        assert plan.calls[0]["kwargs"]["file_list"] == ["app/service.py"]

    def test_out_of_scope_evidence_is_dropped(self, sample_root, config):
        workflow = build(sample_root, config, paths=["app/service.py"])
        workflow.run()

        assert workflow.evidence == []  # 替身检索回的 app/config.py 在范围外

    def test_out_of_scope_findings_are_dropped(self, sample_root, config):
        workflow = build(sample_root, config, paths=["app/service.py"])
        result = workflow.run()

        assert result.report.total == 0
        assert workflow.reports[0].total == 1  # 角色确实报了一条，只是被范围挡在报告之外

    def test_findings_with_repo_prefix_stay_in_scope(self, sample_root, config):
        """模型写的路径可能带仓库名前缀，不能因此被误判成范围外。"""
        prefixed = finding(file="sample_root/app/service.py")
        workflow = build(
            sample_root,
            config,
            overrides={"review": FakeAgent(review_result(prefixed))},
            paths=["app/service.py"],
        )
        result = workflow.run()

        assert result.report.total == 1

    def test_scope_outside_root_is_rejected(self, sample_root, config):
        with pytest.raises(ValueError, match="超出仓库根目录"):
            build(sample_root, config, paths=["../evil.py"])

    def test_missing_file_in_scope_is_rejected(self, sample_root, config):
        with pytest.raises(ValueError, match="不存在"):
            build(sample_root, config, paths=["app/nope.py"])

    def test_scope_hint_is_not_passed_when_reviewing_whole_repo(self, sample_root, config):
        plan = FakeAgent(plan_result())
        workflow = build(sample_root, config, overrides={"plan": plan})
        workflow.run()

        assert plan.calls[0]["kwargs"]["file_list"] == ["app/service.py"]  # 整仓库时照旧扫描
        assert plan.calls[0]["kwargs"]["context"] == ""


# ---------------------------------------------------------------- W7：远端仓库证据
class FakeGitHub:
    """只实现 ``documents_from_github`` 依赖的两个方法的替身。

    ``list_tree`` 直接断言失败：这些用例都显式给了 ``paths``，
    再去拉整棵仓库树就说明调用链走错了。
    """

    def __init__(self, files: dict[str, GitHubFile], *, error: Exception | None = None) -> None:
        self.files = files
        self.error = error
        self.calls: list[str] = []

    def list_tree(self, repo, **kwargs):  # pragma: no cover - 走错分支才会触发
        raise AssertionError("给了 paths 就不该再拉仓库树")

    def read_file(self, repo, path, *, ref=None):
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        return self.files[path]


def github_file(path: str, text: str = "SECRET = 'x'\n") -> GitHubFile:
    return GitHubFile(path=path, text=text, sha="sha-1", size=len(text), ref="main")


class TestGitHubEvidence:
    def test_remote_files_join_the_evidence_chain(self, tmp_path, config):
        client = FakeGitHub({"vendor/auth.py": github_file("vendor/auth.py")})
        workflow = build(
            tmp_path,
            config,
            github=GitHubSource(client=client, repo="acme/demo", paths=("vendor/auth.py",)),
        )
        result = workflow.run()

        evidence = result.state.artifacts["evidence"]
        assert client.calls == ["vendor/auth.py"]
        assert [item["path"] for item in evidence] == ["app/config.py", "vendor/auth.py"]
        remote = evidence[-1]
        assert remote["sources"] == ["github"]
        assert remote["query"] == "github:acme/demo"
        assert remote["snippet"] == "SECRET = 'x'"  # Evidence 会 strip 首尾空白
        assert "含 GitHub 1 条" in result.state.stage("retrieve").detail
        # 远端证据同样要进"证据摘要"，否则审查角色看不到它
        assert "vendor/auth.py" in workflow._evidence_digest()

    def test_remote_failure_degrades_instead_of_failing_the_stage(self, tmp_path, config):
        client = FakeGitHub({}, error=GitHubAuthError("令牌无效"))
        workflow = build(
            tmp_path,
            config,
            github=GitHubSource(client=client, repo="acme/demo", paths=("vendor/auth.py",)),
        )
        result = workflow.run()

        # 远端拿不到 ≠ 整次审查失败：阶段照常 done，但降级原因必须写进详情
        assert result.state.status_of("retrieve") == "done"
        assert result.state.failed_stages() == []
        detail = result.state.stage("retrieve").detail
        assert "GitHub 证据获取失败" in detail and "令牌无效" in detail
        assert [item["path"] for item in result.state.artifacts["evidence"]] == ["app/config.py"]

    def test_duplicate_remote_evidence_is_dropped(self, tmp_path, config):
        # 远端文件与本地证据位置完全相同（app/config.py:1-5）→ 只保留本地那条
        client = FakeGitHub({"app/config.py": github_file("app/config.py", "SECRET = 'x'\n" * 5)})
        workflow = build(
            tmp_path,
            config,
            github=GitHubSource(client=client, repo="acme/demo", paths=("app/config.py",)),
        )
        result = workflow.run()

        evidence = result.state.artifacts["evidence"]
        assert [item["query"] for item in evidence] == [""]  # 留下的是检索那条（无 query）
