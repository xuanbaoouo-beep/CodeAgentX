"""HTTP 接口测试：端点形状、状态码、作业生命周期。

覆盖的是"外壳"该负责的东西——请求体校验、状态码、JSON 形状——流程本身由
`tests/test_api_service.py` 覆盖。用 FastAPI 的 TestClient 在内存里发请求，
不起服务器、不联网、不产生 LLM 调用。

需要 `.[serve]` 依赖（fastapi + httpx）；没装时这一组整体跳过，而不是让套件变红。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


# ------------------------------------------------------------------ 替身
class FakeWorkflow:
    instances: list[FakeWorkflow] = []
    error: Exception | None = None

    def __init__(self, llm: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)

    def run(self, **kwargs: Any) -> Any:
        if type(self).error is not None:
            raise type(self).error
        findings = [SimpleNamespace(to_dict=lambda: {"title": "弱哈希", "line": 3})]
        report = SimpleNamespace(
            total=1,
            summary="整体尚可",
            findings=findings,
            metadata={"degraded": False},
            to_text=lambda: "报告正文",
        )
        state = SimpleNamespace(
            target=str(self.kwargs.get("target", "")),
            stages=[SimpleNamespace(name="plan", status="done", duration=0.1, detail="1 条子任务")],
            metadata={"usage": {"calls": 1, "total_tokens": 10}},
        )
        return SimpleNamespace(
            success=True, report=report, state=state, to_markdown=lambda: "# 报告正文"
        )


@pytest.fixture
def make_client() -> Iterator[Callable[..., Any]]:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from codeagentx.api.main import create_app
    from codeagentx.api.service import ReviewService

    services: list[Any] = []

    def factory(*, configured: bool = True, error: Exception | None = None) -> TestClient:
        FakeWorkflow.instances = []
        FakeWorkflow.error = error
        service = ReviewService(
            workflow_factory=FakeWorkflow,
            llm_factory=lambda config: SimpleNamespace(model_id="test-model"),
            config_getter=lambda: SimpleNamespace(
                is_llm_configured=configured, llm_model_id="test-model", github_token=""
            ),
        )
        services.append(service)
        return TestClient(create_app(service))

    yield factory

    for service in services:
        service.shutdown()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "sample_repo"
    (root / "app").mkdir(parents=True)
    (root / "app" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return root


def poll(client: Any, job_id: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """等到作业结束；接口是"提交 + 轮询"，测试也照调用方的方式轮询。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/review/{job_id}").json()
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(0.02)
    raise AssertionError(f"作业 {job_id} 超时未结束")


# ------------------------------------------------------------------ 端点
def test_health_endpoint(make_client: Callable[..., Any]) -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True
    assert body["model"] == "test-model"
    assert body["jobs"] == 0


def test_submit_then_poll_returns_report(
    make_client: Callable[..., Any], repo: Path
) -> None:
    client = make_client()

    submitted = client.post("/review", json={"target": str(repo)})

    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    assert submitted.json()["poll"] == f"/review/{job_id}"

    payload = poll(client, job_id)
    assert payload["status"] == "done"
    assert payload["result"]["total"] == 1
    assert payload["result"]["markdown"] == "# 报告正文"
    assert payload["result"]["findings"][0]["title"] == "弱哈希"


def test_single_file_target_is_scoped(
    make_client: Callable[..., Any], repo: Path
) -> None:
    """HTTP 侧也要遵守"给文件只审那个文件"：范围必须传到工作流。"""
    client = make_client()

    job_id = client.post("/review", json={"target": str(repo / "app" / "main.py")}).json()[
        "job_id"
    ]
    assert poll(client, job_id)["status"] == "done"

    kwargs = FakeWorkflow.instances[0].kwargs
    assert kwargs["paths"] == ("main.py",)
    assert kwargs["target"] == "app/main.py"


def test_submit_without_llm_key_returns_503(make_client: Callable[..., Any], repo: Path) -> None:
    """服务端没配密钥：调用方没错，用 503 让它知道该找运维而不是改参数。"""
    response = make_client(configured=False).post("/review", json={"target": str(repo)})

    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]


def test_unknown_job_returns_404(make_client: Callable[..., Any]) -> None:
    response = make_client().get("/review/nope")

    assert response.status_code == 404
    assert "没有这个作业" in response.json()["detail"]


def test_bad_target_reports_target_error_kind(make_client: Callable[..., Any]) -> None:
    client = make_client()

    job_id = client.post("/review", json={"target": "not-a-dir/nor-a-repo/extra"}).json()["job_id"]
    payload = poll(client, job_id)

    assert payload["status"] == "failed"
    assert payload["error_kind"] == "target"
    assert "既不是已存在的路径" in payload["error"]


def test_unexpected_error_is_reported(make_client: Callable[..., Any], repo: Path) -> None:
    client = make_client(error=RuntimeError("替身炸了"))

    payload = poll(client, client.post("/review", json={"target": str(repo)}).json()["job_id"])

    assert payload["status"] == "failed"
    assert payload["error_kind"] == "internal"
    assert "替身炸了" in payload["error"]


def test_test_runner_switch_is_not_exposed(
    make_client: Callable[..., Any], repo: Path
) -> None:
    """接口不提供"执行目标仓库测试"的开关：传了就明确拒绝，不能悄悄忽略。"""
    response = make_client().post(
        "/review", json={"target": str(repo), "enable_test": True}
    )

    assert response.status_code == 422
    assert "enable_test" in str(response.json())


def test_missing_target_returns_422(make_client: Callable[..., Any]) -> None:
    response = make_client().post("/review", json={})

    assert response.status_code == 422


def test_empty_target_returns_422(make_client: Callable[..., Any]) -> None:
    response = make_client().post("/review", json={"target": ""})

    assert response.status_code == 422


def test_openapi_documents_all_endpoints(make_client: Callable[..., Any]) -> None:
    """文档能被自动生成（也意味着请求体 schema 是合法的）。"""
    schema = make_client().get("/openapi.json").json()

    assert set(schema["paths"]) == {"/health", "/review", "/review/{job_id}"}
    assert schema["info"]["title"] == "CodeAgentX 审查服务"
