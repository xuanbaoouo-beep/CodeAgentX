"""工作流状态测试。

多 Agent 流水线最容易出的事故不是"某一步报错"，而是**某一步悄悄没跑**：
报告照常输出，读者以为覆盖了安全审查。因此状态机必须做到：
- 「失败」与「跳过」严格区分（前者是异常，后者是设计如此）；
- 阶段进度可查询、可序列化、可落盘、可恢复；
- 落盘失败是降级（只记 WARNING），不该把审查任务本身搞失败。
"""

from __future__ import annotations

import json

from codeagentx.orchestrator.state import (
    SETTLED_STATUSES,
    STAGE_STATUSES,
    STAGES,
    STATE_VERSION,
    StageState,
    WorkflowState,
)


class TestStageState:
    def test_defaults_to_pending(self):
        assert StageState(name="plan").status == "pending"

    def test_unknown_status_falls_back_to_pending(self):
        assert StageState(name="plan", status="weird").status == "pending"

    def test_status_is_case_insensitive(self):
        assert StageState(name="plan", status="DONE").status == "done"

    def test_duration_is_rounded(self):
        assert StageState(name="plan", duration=1.23456).duration == 1.235

    def test_settled_only_for_done_and_skipped(self):
        assert StageState(name="a", status="done").settled is True
        assert StageState(name="a", status="skipped").settled is True
        assert StageState(name="a", status="failed").settled is False
        assert StageState(name="a", status="running").settled is False

    def test_round_trip(self):
        item = StageState(name="review", status="done", detail="2 条问题", duration=0.5)
        assert StageState.from_dict(item.to_dict()) == item

    def test_non_dict_payload(self):
        assert StageState.from_dict("plan").name == "plan"


class TestWorkflowStateBasics:
    def test_create_covers_every_stage_in_order(self):
        state = WorkflowState.create("repo")

        assert [item.name for item in state.stages] == list(STAGES)
        assert state.pending_stages() == list(STAGES)
        assert state.is_complete() is False

    def test_stage_statuses_are_the_five_documented_ones(self):
        assert STAGE_STATUSES == ("pending", "running", "done", "failed", "skipped")
        assert SETTLED_STATUSES == ("done", "skipped")

    def test_set_stage_updates_status_and_detail(self):
        state = WorkflowState.create("repo")
        item = state.set_stage("plan", "done", detail="3 条子任务", duration=0.123)

        assert item.status == "done"
        assert item.detail == "3 条子任务"
        assert state.status_of("plan") == "done"
        assert state.is_settled("plan") is True

    def test_empty_detail_keeps_the_previous_value(self):
        state = WorkflowState.create("repo")
        state.set_stage("plan", "done", detail="3 条子任务")
        state.set_stage("plan", "failed", error="超时")

        assert state.stage("plan").detail == "3 条子任务"
        assert state.stage("plan").error == "超时"

    def test_failed_and_skipped_are_distinguishable(self):
        state = WorkflowState.create("repo")
        state.set_stage("security", "failed", error="超时")
        state.set_stage("test", "skipped", detail="未启用")

        assert state.failed_stages() == ["security"]
        # 跳过 = 已有结果（不再重跑）；失败 = 没有可用结果，仍算未完成
        assert "test" not in state.pending_stages()
        assert "security" in state.pending_stages()
        assert state.is_complete() is False

    def test_progress_counts(self):
        state = WorkflowState.create("repo")
        state.set_stage("plan", "done")
        state.set_stage("retrieve", "skipped")

        progress = state.progress()
        assert progress["done"] == 1
        assert progress["skipped"] == 1
        assert progress["pending"] == len(STAGES) - 2

    def test_describe_is_a_one_line_summary(self):
        state = WorkflowState.create("repo")
        state.set_stage("plan", "done")

        described = state.describe()
        assert described.startswith("plan=done | retrieve=pending")
        assert "\n" not in described

    def test_missing_stage_is_created_in_order(self):
        state = WorkflowState(target="repo", stages=[StageState(name="report")])
        state.stage("plan")

        assert [item.name for item in state.stages] == ["plan", "report"]


class TestSerialization:
    def test_round_trip_keeps_artifacts_and_metadata(self):
        state = WorkflowState.create("repo")
        state.set_stage("plan", "done", detail="3 条子任务")
        state.artifacts["plan"] = {"target": "repo", "tasks": []}
        state.metadata["duration"] = 1.5

        restored = WorkflowState.from_dict(json.loads(json.dumps(state.to_dict())))

        assert restored.target == "repo"
        assert restored.status_of("plan") == "done"
        assert restored.artifacts["plan"] == {"target": "repo", "tasks": []}
        assert restored.metadata["duration"] == 1.5

    def test_from_dict_fills_missing_stages(self):
        restored = WorkflowState.from_dict({"target": "repo", "stages": [{"name": "plan"}]})

        assert [item.name for item in restored.stages] == list(STAGES)
        assert restored.status_of("report") == "pending"

    def test_version_is_recorded(self):
        assert WorkflowState.create("repo").to_dict()["version"] == STATE_VERSION


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "state" / "run.json"
        state = WorkflowState.create("repo")
        state.set_stage("retrieve", "done", detail="5 条证据")

        assert state.save(path) is True
        loaded = WorkflowState.load(path)

        assert loaded is not None
        assert loaded.status_of("retrieve") == "done"
        assert loaded.stage("retrieve").detail == "5 条证据"

    def test_save_is_atomic_and_leaves_no_temp_file(self, tmp_path):
        path = tmp_path / "run.json"
        WorkflowState.create("repo").save(path)

        assert path.exists()
        assert not (tmp_path / "run.json.tmp").exists()

    def test_save_failure_is_only_a_warning(self, tmp_path):
        """落盘失败（这里让父路径是个文件）不能让审查任务本身失败。"""
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir", encoding="utf-8")

        assert WorkflowState.create("repo").save(blocker / "run.json") is False

    def test_load_missing_file_returns_none(self, tmp_path):
        assert WorkflowState.load(tmp_path / "nope.json") is None

    def test_load_corrupted_json_returns_none(self, tmp_path):
        path = tmp_path / "run.json"
        path.write_text("{不是 JSON", encoding="utf-8")

        assert WorkflowState.load(path) is None

    def test_load_list_payload_returns_none(self, tmp_path):
        path = tmp_path / "run.json"
        path.write_text("[1, 2]", encoding="utf-8")

        assert WorkflowState.load(path) is None

    def test_load_version_mismatch_returns_none(self, tmp_path):
        path = tmp_path / "run.json"
        payload = WorkflowState.create("repo").to_dict()
        payload["version"] = STATE_VERSION + 99
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        assert WorkflowState.load(path) is None
