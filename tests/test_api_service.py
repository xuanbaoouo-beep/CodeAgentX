"""作业队列测试：提交 → 排队 → 执行 → 取结果。

全部离线：工作流与 LLM 都是替身，目标用临时目录里的真实文件（这样
``prepare_target`` 走的是真实分支，单文件范围等规则也一并被验证）。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from codeagentx.api import service as service_module
from codeagentx.api.service import (
    DONE,
    FAILED,
    ReviewRequest,
    ReviewService,
)
from codeagentx.core.exceptions import ConfigError
from codeagentx.workspace import PreparedTarget


# ------------------------------------------------------------------ 替身
def make_result(*, success: bool = True, degraded: bool = False) -> SimpleNamespace:
    findings = [
        SimpleNamespace(to_dict=lambda: {"title": "弱哈希", "file": "app/main.py", "line": 3}),
        SimpleNamespace(to_dict=lambda: {"title": "日志泄露", "file": "app/main.py", "line": 9}),
    ]
    report = SimpleNamespace(
        total=len(findings),
        summary="整体尚可",
        findings=findings,
        metadata={"degraded": degraded},
        to_text=lambda: "报告正文",
    )
    state = SimpleNamespace(
        target="sample_repo",
        stages=[SimpleNamespace(name="plan", status="done", duration=1.5, detail="5 条子任务")],
        metadata={"usage": {"calls": 4, "total_tokens": 1234}},
    )
    return SimpleNamespace(
        success=success, report=report, state=state, to_markdown=lambda: "# 报告正文"
    )


class FakeWorkflow:
    """替身工作流：记下入参，按 ``result`` / ``error`` 决定行为。"""

    instances: list[FakeWorkflow] = []
    result: Any = None
    error: Exception | None = None
    #: 同时进入 ``run`` 的最大数量（用来证明队列真的把审查串起来了）
    concurrency = 0
    max_concurrency = 0
    gate: threading.Lock = threading.Lock()

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        self.llm = llm
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def run(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        with type(self).gate:
            type(self).concurrency += 1
            type(self).max_concurrency = max(type(self).max_concurrency, type(self).concurrency)
        try:
            time.sleep(0.01)  # 留出窗口：并发问题只在"两个真同时在跑"时才暴露
            if type(self).error is not None:
                raise type(self).error
            return type(self).result
        finally:
            with type(self).gate:
                type(self).concurrency -= 1


class FakeConfig:
    def __init__(self, configured: bool = True) -> None:
        self.is_llm_configured = configured
        self.llm_model_id = "test-model"
        self.github_token = ""


@pytest.fixture
def make_service() -> Any:
    """造一个用替身跑的服务，收尾时关线程池（不然线程会漏出测试之外）。"""
    services: list[ReviewService] = []

    def factory(
        *, target_loader: Any = None, max_workers: int = 1, configured: bool = True
    ) -> ReviewService:
        FakeWorkflow.instances = []
        FakeWorkflow.result = make_result()
        FakeWorkflow.error = None
        FakeWorkflow.concurrency = 0
        FakeWorkflow.max_concurrency = 0
        kwargs: dict[str, Any] = {
            "workflow_factory": FakeWorkflow,
            "llm_factory": lambda config: SimpleNamespace(model_id=config.llm_model_id),
            "config_getter": lambda: FakeConfig(configured),
            "max_workers": max_workers,
        }
        if target_loader is not None:
            kwargs["target_loader"] = target_loader
        service = ReviewService(**kwargs)
        services.append(service)
        return service

    yield factory

    for service in services:
        service.shutdown()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """一个最小的本地仓库：两个文件，方便验证"单文件只审该文件"。"""
    root = tmp_path / "sample_repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "app" / "other.py").write_text("print('other')\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------ 自检与受理
def test_health_reports_missing_llm_key(make_service: Any) -> None:
    service = make_service(configured=False)

    health = service.health()

    assert health["status"] == "ok"
    assert health["llm_configured"] is False
    assert health["jobs"] == 0


def test_submit_without_llm_key_raises_config_error(make_service: Any) -> None:
    """服务端没配密钥要在**同步路径**上就报出来，而不是让人排队半天再看到失败。"""
    service = make_service(configured=False)

    with pytest.raises(ConfigError) as excinfo:
        service.submit(ReviewRequest(target="acme/demo"))

    assert "LLM_API_KEY" in excinfo.value.message
    assert service.stats()["jobs"] == 0, "没受理成功就不该留下作业"
    assert FakeWorkflow.instances == [], "不该构造工作流"


# ------------------------------------------------------------------ 正常路径
def test_job_runs_and_exposes_report(make_service: Any, repo: Path) -> None:
    service = make_service()

    job = service.submit(ReviewRequest(target=str(repo)))
    done = service.wait(job.id, timeout=10)

    assert done.status == DONE
    assert done.finished
    assert done.duration >= 0
    payload = done.to_dict()
    assert payload["job_id"] == job.id
    assert payload["status"] == DONE
    result = payload["result"]
    assert result["success"] is True
    assert result["degraded"] is False
    assert result["total"] == 2
    assert result["markdown"] == "# 报告正文"
    assert [item["title"] for item in result["findings"]] == ["弱哈希", "日志泄露"]
    assert result["stages"][0]["name"] == "plan"
    assert result["usage"]["calls"] == 4

    assert FakeWorkflow.instances[0].kwargs["root"] == repo.resolve()
    assert FakeWorkflow.instances[0].kwargs["target"] == "sample_repo"
    assert FakeWorkflow.instances[0].kwargs["paths"] is None


def test_single_file_target_scopes_paths(make_service: Any, repo: Path) -> None:
    """给单文件时范围要一路传到工作流：这是"给文件只审那个文件"的接口侧入口。"""
    service = make_service()

    job = service.submit(ReviewRequest(target=str(repo / "app" / "main.py")))
    service.wait(job.id, timeout=10)

    kwargs = FakeWorkflow.instances[0].kwargs
    assert kwargs["paths"] == ("main.py",)
    assert kwargs["target"] == "app/main.py"
    assert kwargs["root"] == (repo / "app").resolve()


def test_explicit_paths_win_over_default_scope(make_service: Any, repo: Path) -> None:
    """调用方显式给了范围就以它为准。"""
    service = make_service()

    job = service.submit(
        ReviewRequest(target=str(repo), paths=("app/other.py", "app/main.py"))
    )
    service.wait(job.id, timeout=10)

    assert FakeWorkflow.instances[0].kwargs["paths"] == ("app/other.py", "app/main.py")


def test_switches_are_forwarded(make_service: Any, repo: Path) -> None:
    service = make_service()

    job = service.submit(ReviewRequest(target=str(repo), enable_refactor=True, reflect=True))
    service.wait(job.id, timeout=10)

    kwargs = FakeWorkflow.instances[0].kwargs
    assert kwargs["enable_refactor"] is True
    assert kwargs["reflect"] is True


# ------------------------------------------------------------------ 失败路径
def test_bad_target_fails_as_target_error(make_service: Any) -> None:
    """目标写错：作业失败但要标成"调用方的问题"，而不是 500 式的内部错误。"""
    service = make_service()

    job = service.submit(ReviewRequest(target="not-a-dir/nor-a-repo/extra"))
    failed = service.wait(job.id, timeout=10)

    assert failed.status == FAILED
    assert failed.error_kind == "target"
    assert "既不是已存在的路径" in failed.error
    assert "result" not in failed.to_dict()


def test_unexpected_error_does_not_leave_job_running(make_service: Any, repo: Path) -> None:
    """后台线程没人接异常：任何意外都必须落到作业状态上，不能永远停在 running。"""
    service = make_service()
    FakeWorkflow.error = RuntimeError("替身炸了")

    job = service.submit(ReviewRequest(target=str(repo)))
    failed = service.wait(job.id, timeout=10)

    assert failed.status == FAILED
    assert failed.error_kind == "internal"
    assert "替身炸了" in failed.error


def test_queue_keeps_running_after_a_failure(make_service: Any, repo: Path) -> None:
    """一个作业炸了不能把队列钉死——后面的作业还得能跑。"""
    service = make_service()
    FakeWorkflow.error = RuntimeError("替身炸了")
    broken = service.wait(service.submit(ReviewRequest(target=str(repo))).id, timeout=10)
    assert broken.status == FAILED

    FakeWorkflow.error = None
    healthy = service.wait(service.submit(ReviewRequest(target=str(repo))).id, timeout=10)

    assert healthy.status == DONE


def test_reviews_are_serialized(make_service: Any, repo: Path) -> None:
    """向量库是全局单集合（AD-82）：审查必须串行，不能两个仓库同时写。"""
    service = make_service()

    jobs = [service.submit(ReviewRequest(target=str(repo))) for _ in range(3)]
    for job in jobs:
        assert service.wait(job.id, timeout=30).status == DONE

    assert FakeWorkflow.max_concurrency == 1


def test_wait_raises_for_unknown_job(make_service: Any) -> None:
    service = make_service()

    assert service.get("nope") is None
    with pytest.raises(KeyError):
        service.wait("nope", timeout=0.01)


def test_remote_workdir_is_cleaned_after_job(make_service: Any, tmp_path: Path) -> None:
    """远端工作区用完必须删——接口天天被调，漏一个就是磁盘泄漏。"""
    workdir = tmp_path / "codeagentx-remote-fake"
    workdir.mkdir()
    (workdir / "app.py").write_text("print(1)\n", encoding="utf-8")

    def loader(target: str, **kwargs: Any) -> PreparedTarget:
        return PreparedTarget(
            root=workdir, display="acme/demo@main", remote=True, workdir=workdir, note="已下载"
        )

    service = make_service(target_loader=loader)
    job = service.submit(ReviewRequest(target="acme/demo"))
    done = service.wait(job.id, timeout=10)

    assert done.status == DONE
    assert done.note == "已下载"
    assert not workdir.exists()


def test_finished_jobs_are_pruned(
    make_service: Any, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """内存里不能无限堆积作业（接口长期在线时这是慢泄漏）。"""
    monkeypatch.setattr(service_module, "MAX_JOBS_KEPT", 3)
    service = make_service()

    for _ in range(6):
        service.wait(service.submit(ReviewRequest(target=str(repo))).id, timeout=30)

    assert service.stats()["jobs"] <= 3
