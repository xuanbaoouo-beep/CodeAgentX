"""Reporter 角色测试。

Reporter 是唯一直接交付给人的角色，因此测试聚焦两件事：
1. **合并确定性**：去重、来源合并、取更严重等级、稳定排序，同样的输入必须给出同样的输出；
2. **降级显式**：任一角色解析失败或流水线某阶段失败时，摘要必须带警告且 ``success=False``，
   绝不能让读者把"报告里没写"理解成"没有问题"。
"""

from __future__ import annotations

from codeagentx.agents.reporter import DEGRADED_WARNING, ReporterAgent
from codeagentx.agents.schemas import Evidence, Finding, ReviewPlan, ReviewReport
from codeagentx.core.agent import ZERO_USAGE


def finding(
    title: str,
    *,
    file: str = "a.py",
    line: int = 1,
    severity: str = "medium",
    category: str = "bug",
    source: str = "reviewer",
    description: str = "",
    suggestion: str = "",
    confidence: float = 0.5,
) -> Finding:
    return Finding(
        title=title,
        file=file,
        line=line,
        severity=severity,
        category=category,
        source=source,
        description=description,
        suggestion=suggestion,
        confidence=confidence,
    )


def report(*findings: Finding, summary: str = "结论", agent: str = "reviewer") -> ReviewReport:
    return ReviewReport(
        target="repo",
        summary=summary,
        findings=list(findings),
        metadata={"agent": agent},
    )


class TestMergeFindings:
    def test_same_location_is_reported_once(self):
        merged = ReporterAgent().merge_findings(
            [
                report(finding("硬编码密钥", source="reviewer")),
                report(finding("硬编码密钥", source="security"), agent="security"),
            ]
        )

        assert len(merged) == 1
        assert merged[0].source == "reviewer+security"

    def test_title_case_and_spacing_do_not_create_duplicates(self):
        merged = ReporterAgent().merge_findings(
            [
                report(finding("硬编码密钥")),
                report(finding("  硬编码密钥  ".upper().lower())),
            ]
        )
        assert len(merged) == 1

    def test_more_severe_level_wins(self):
        merged = ReporterAgent().merge_findings(
            [
                report(finding("问题 A", severity="low", source="reviewer")),
                report(finding("问题 A", severity="high", source="security"), agent="security"),
            ]
        )

        assert merged[0].severity == "high"

    def test_confidence_takes_the_higher_value(self):
        merged = ReporterAgent().merge_findings(
            [
                report(finding("问题 A", confidence=0.3)),
                report(finding("问题 A", confidence=0.9), agent="security"),
            ]
        )
        assert merged[0].confidence == 0.9

    def test_empty_fields_are_filled_from_the_other_report(self):
        merged = ReporterAgent().merge_findings(
            [
                report(finding("问题 A", description="", suggestion="")),
                report(finding("问题 A", description="越界访问", suggestion="加边界判断")),
            ]
        )

        assert merged[0].description == "越界访问"
        assert merged[0].suggestion == "加边界判断"

    def test_different_lines_are_kept_separate(self):
        merged = ReporterAgent().merge_findings(
            [report(finding("问题 A", line=1), finding("问题 A", line=2))]
        )
        assert len(merged) == 2

    def test_same_line_is_merged_even_if_category_differs(self):
        """同一文件同一行**只有一个位置**，分类不同也合并。

        依据：离线复核 W8 的 54 次真实运行，同文件同行的 31 组残留重复里没有一组是
        两个真正不同的缺陷；按分类过滤挡不住错合，却会少合 10 组该合的。
        """
        merged = ReporterAgent().merge_findings(
            [
                report(
                    finding("空指针", category="bug", source="reviewer"),
                    finding(
                        "空指针",
                        category="performance",
                        source="security",
                        severity="high",
                    ),
                )
            ]
        )
        assert len(merged) == 1
        assert merged[0].source == "reviewer+security"
        assert merged[0].severity == "high"

    def test_reworded_same_issue_on_the_same_line_is_merged(self):
        """真实样本：blog_api ``app/api/posts.py:41``，Reviewer 与 Security 对同一处
        缺鉴权写了措辞完全不同的标题，但行号一字不差 → 直接合并。"""
        merged = ReporterAgent().merge_findings(
            [
                report(finding("删帖接口完全缺失鉴权与权限校验", line=41, source="reviewer")),
                report(
                    finding(
                        "删帖接口无任何鉴权，任何人可删任意帖子",
                        line=41,
                        source="security",
                        category="security",
                    ),
                    agent="security",
                ),
            ]
        )

        assert len(merged) == 1
        assert merged[0].source == "reviewer+security"

    def test_same_line_merge_does_not_apply_to_missing_line_numbers(self):
        """没有行号就没有"同一行"可言，不能借这条规则把无行号的问题混成一堆。"""
        merged = ReporterAgent().merge_findings(
            [report(finding("空指针", line=0), finding("越界访问", line=0))]
        )
        assert len(merged) == 2

    def test_reworded_same_issue_nearby_is_merged(self):
        """真实样本：Reviewer 与 Security 对同一处密钥硬编码各写了一条措辞不同的标题。"""
        merged = ReporterAgent().merge_findings(
            [
                report(finding("JWT 签名密钥硬编码在源码中", line=11, source="reviewer")),
                report(
                    finding(
                        "JWT 签名密钥硬编码，且令牌直接暴露密钥明文",
                        line=14,
                        source="security",
                        severity="high",
                    ),
                    agent="security",
                ),
            ]
        )

        assert len(merged) == 1
        assert merged[0].source == "reviewer+security"
        assert merged[0].severity == "high"

    def test_unrelated_issues_at_neighbouring_lines_are_kept(self):
        """相近行号不等于同一条问题：密钥硬编码与弱口令哈希是两码事，不能合并。"""
        merged = ReporterAgent().merge_findings(
            [
                report(finding("会话密钥 SECRET_KEY 硬编码在源码中", line=11)),
                report(finding("口令使用无盐 SHA-256 存储，可离线爆破与彩虹表还原", line=14)),
            ]
        )

        assert len(merged) == 2

    def test_reworded_duplicate_beyond_line_tolerance_is_kept(self):
        """行号差得远就不是"同一条被重复上报"，宁可多留一条也不错合。"""
        merged = ReporterAgent().merge_findings(
            [
                report(finding("登录接口缺少失败次数限制，可无限次暴力破解", line=9)),
                report(finding("登录失败无次数限制，可无限暴力破解口令", line=21)),
            ]
        )

        assert len(merged) == 2

    def test_output_is_sorted_by_severity(self):
        merged = ReporterAgent().merge_findings(
            [
                report(finding("低危", severity="low", line=3)),
                report(finding("高危", severity="high", line=9)),
                report(finding("中危", severity="medium", line=1)),
            ]
        )
        assert [item.title for item in merged] == ["高危", "中危", "低危"]

    def test_merge_does_not_mutate_input_findings(self):
        original = finding("问题 A", severity="low", source="reviewer")
        ReporterAgent().merge_findings([report(original), report(finding("问题 A", severity="high"))])

        assert original.severity == "low"
        assert original.source == "reviewer"


class TestCompose:
    def test_metadata_records_sources_and_counts(self):
        result = ReporterAgent().compose(
            "repo",
            [
                report(finding("问题 A", line=1), finding("问题 B", line=2)),
                report(finding("问题 C", line=3), agent="security"),
            ],
        )

        assert result.metadata["agent"] == "reporter"
        assert result.metadata["sources"] == {"reviewer": 2, "security": 1}
        assert result.metadata["degraded"] is False
        assert result.total == 3

    def test_summary_concatenates_role_summaries_without_rewriting(self):
        result = ReporterAgent().compose(
            "repo",
            [report(finding("A"), summary="Rev 结论"), report(finding("B"), summary="Sec 结论")],
        )

        assert "Rev 结论" in result.summary
        assert "Sec 结论" in result.summary
        assert "参与角色与问题条数" in result.summary

    def test_duplicate_summaries_are_listed_once(self):
        result = ReporterAgent().compose(
            "repo", [report(summary="同一段"), report(summary="同一段")]
        )
        assert result.summary.count("同一段") == 1

    def test_no_reports_still_produces_a_readable_summary(self):
        result = ReporterAgent().compose("repo", [])

        assert "（各角色均未给出摘要）" in result.summary
        assert result.total == 0

    def test_evidence_is_counted_in_metadata(self):
        result = ReporterAgent().compose(
            "repo",
            [report()],
            evidence=[Evidence(path="a.py", start_line=1, end_line=2), {"path": "b.py"}],
        )

        assert result.metadata["evidence_count"] == 2

    def test_plan_and_test_result_are_carried_through(self):
        plan = ReviewPlan(target="repo", tasks=[])
        plan.metadata["fallback"] = True
        result = ReporterAgent().compose(
            "repo", [report()], plan=plan, test_result={"available": True, "passed": False}
        )

        assert result.metadata["plan"]["metadata"]["fallback"] is True
        assert result.metadata["test_result"]["passed"] is False

    def test_parse_error_triggers_degraded(self):
        broken = report()
        broken.metadata["parse_error"] = "输出不是 JSON"
        result = ReporterAgent().compose("repo", [broken])

        assert result.metadata["degraded"] is True
        assert result.metadata["parse_errors"] == [{"agent": "reviewer", "error": "输出不是 JSON"}]
        assert DEGRADED_WARNING in result.summary

    def test_failed_stage_triggers_degraded(self):
        workflow = {
            "stages": [
                {"name": "plan", "status": "done"},
                {"name": "security", "status": "failed", "error": "超时"},
                {"name": "report", "status": "running"},
            ]
        }
        result = ReporterAgent().compose("repo", [report()], workflow=workflow)

        assert result.metadata["failed_stages"] == ["security"]
        assert result.metadata["degraded"] is True

    def test_fallback_plan_triggers_degraded(self):
        plan = ReviewPlan(target="repo", tasks=[])
        plan.metadata["fallback"] = True
        result = ReporterAgent().compose("repo", [report()], plan=plan)

        assert result.metadata["degraded"] is True
        assert DEGRADED_WARNING in result.summary

    def test_workflow_snapshot_is_recorded(self):
        from codeagentx.orchestrator.state import WorkflowState

        state = WorkflowState.create("repo")
        state.set_stage("plan", "done", detail="3 条子任务")
        result = ReporterAgent().compose("repo", [report()], workflow=state)

        assert result.metadata["workflow"]["target"] == "repo"
        assert result.metadata["workflow"]["stages"][0]["status"] == "done"


class TestRun:
    def test_clean_run_succeeds_and_renders_markdown(self):
        result = ReporterAgent().run("repo", reports=[report(finding("硬编码密钥", severity="high"))])

        assert result.success is True
        assert result.error is None
        assert result.metadata["agent"] == "reporter"
        assert "# 代码审查报告" in result.output
        assert "硬编码密钥" in result.metadata["markdown"]
        assert result.usage["calls"] == ZERO_USAGE["calls"]

    def test_degraded_run_keeps_report_but_flags_failure(self):
        broken = report()
        broken.metadata["parse_error"] = "输出不是 JSON"
        result = ReporterAgent().run("repo", reports=[broken])

        assert result.success is False
        assert result.error == DEGRADED_WARNING
        # 报告照样产出：部分结论 + 明确降级标记，比"什么都没有"更有用
        assert result.metadata["report"]["metadata"]["degraded"] is True
        assert "# 代码审查报告" in result.output

    def test_report_contains_no_timestamp(self):
        """报告必须可复现：同样输入两次运行结果完全一致。"""
        first = ReporterAgent().run("repo", reports=[report(finding("A"))]).output
        second = ReporterAgent().run("repo", reports=[report(finding("A"))]).output
        assert first == second

    def test_extra_metadata_is_merged(self):
        result = ReporterAgent().run(
            "repo",
            reports=[report()],
            extra_metadata={"usage": dict(ZERO_USAGE), "stages": "plan=done"},
        )

        assert result.metadata["report"]["metadata"]["stages"] == "plan=done"
        assert result.metadata["report"]["metadata"]["usage"]["calls"] == 0


class TestContract:
    def test_reporter_is_deterministic_and_llm_free(self):
        """Reporter 不接 LLM：构造参数里没有 llm，避免被误接上。"""
        agent = ReporterAgent()
        assert "llm" not in vars(agent)
        assert agent.name == "reporter"
