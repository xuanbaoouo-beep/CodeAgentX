"""CLI 入口测试：目标解析、远端工作区生命周期、退出码。

为什么单独测这个文件：CLI 是**唯一一条用户真正会走的路径**，它自己做三件
容易被忽略的事——把 `owner/repo` 变成"已存在的本地目录"、把临时工作区
在成功/失败/异常三条路径上都清干净、用退出码区分"审查完成""有阶段失败"
"目标不合法"。这三件事错一件，用户看到的就是"命令没反应"或者"磁盘上
悄悄多了一堆仓库"。

全部离线：配置、LLM、工作流、GitHub 客户端都被替换成替身，
没有任何网络请求，也不产生 LLM 调用。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from codeagentx import cli, workspace
from codeagentx.core.exceptions import GitHubRateLimitError
from codeagentx.protocols.github_archive import ArchiveExtractResult


# ------------------------------------------------------------------ 替身
def make_result(*, success: bool = True) -> SimpleNamespace:
    """造一个够用的 WorkflowResult 替身（字段与 workflow.py 里的一致）。"""
    stage = SimpleNamespace(
        name="collect", status="ok", duration=0.12, detail="收集完成", error=""
    )
    state = SimpleNamespace(
        stages=[stage],
        metadata={"usage": {"calls": 2, "total_tokens": 40}, "duration": 0.5},
        progress=lambda: {"collect": 1},
        failed_stages=lambda: [] if success else ["verify"],
    )
    report = SimpleNamespace(
        metadata={"degraded": True} if not success else {},
        to_text=lambda: "报告正文",
        to_markdown=lambda: "# 报告",
    )
    return SimpleNamespace(
        state=state, report=report, success=success, to_markdown=lambda: "# 报告"
    )


class FakeWorkflow:
    """记录构造参数与 run 调用；返回预设结果。"""

    instances: list[FakeWorkflow] = []
    result: Any = None

    def __init__(self, llm: Any = None, **kwargs: Any) -> None:
        self.llm = llm
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        type(self).instances.append(self)

    def run(self, *, resume: bool = False, reset: bool = False) -> Any:
        self.calls.append({"resume": resume, "reset": reset})
        return type(self).result


class FakeGitHubClient:
    """替身：`error` 非空时模拟远端失败；否则在目标目录写一个文件并返回结果。"""

    error: Exception | None = None
    calls: list[tuple[Any, ...]] = []

    def __init__(self) -> None:
        type(self).calls = []

    @classmethod
    def from_config(cls, *args: Any, **kwargs: Any) -> FakeGitHubClient:
        return cls()

    def resolve_ref(self, repo: Any) -> str:
        type(self).calls.append(("resolve_ref", str(repo)))
        return "main"

    def download_archive(self, repo: Any, dest: Any, **kwargs: Any) -> ArchiveExtractResult:
        type(self).calls.append(("download_archive", str(repo), Path(dest), kwargs.get("ref")))
        root = Path(dest)
        root.mkdir(parents=True, exist_ok=True)
        (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
        if type(self).error is not None:
            raise type(self).error
        return ArchiveExtractResult(root=root, files=1, bytes=11)

    def close(self) -> None:
        type(self).calls.append(("close",))


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> type[FakeGitHubClient]:
    """把配置/LLM/工作流/远端客户端全部换成替身（含密钥，好走到真正的主流程）。"""
    monkeypatch.setattr(
        cli, "get_config", lambda: SimpleNamespace(is_llm_configured=True, github_token="")
    )
    monkeypatch.setattr(cli, "build_llm", lambda config: SimpleNamespace(model_id="test-model"))
    FakeWorkflow.instances = []
    FakeWorkflow.result = make_result()
    monkeypatch.setattr(cli, "CodeReviewWorkflow", FakeWorkflow)
    monkeypatch.setattr(workspace, "GitHubClient", FakeGitHubClient)
    FakeGitHubClient.error = None
    return FakeGitHubClient


# ------------------------------------------------------------------ 退出码与分支
def test_missing_llm_key_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """没密钥时必须在解析目标之前就退出 2，并且不去碰任何远端/本地资源。"""
    monkeypatch.setattr(cli, "get_config", lambda: SimpleNamespace(is_llm_configured=False))

    assert cli.main(["review", "acme/demo"]) == 2

    err = capsys.readouterr().err
    assert "未配置 LLM_API_KEY" in err
    assert "远端仓库准备失败" not in err


def test_unknown_target_exits_2(
    wired: type[FakeGitHubClient], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["review", "not-a-dir/nor-a-repo/extra"]) == 2

    assert "既不是已存在的路径" in capsys.readouterr().err
    assert FakeWorkflow.instances == [], "目标没解析成功就不该构造工作流"


def test_local_directory_runs_workflow(
    wired: type[FakeGitHubClient], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "sample_repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    assert cli.main(["review", str(repo), "--enable-refactor", "--reflect"]) == 0

    workflow = FakeWorkflow.instances[0]
    assert workflow.kwargs["root"] == repo.resolve()
    assert workflow.kwargs["target"] == "sample_repo"
    assert workflow.kwargs["enable_refactor"] is True
    assert workflow.kwargs["reflect"] is True
    assert workflow.kwargs["state_path"] is None
    assert workflow.calls == [{"resume": False, "reset": False}]

    out = capsys.readouterr().out
    assert "来源=本地" in out
    assert "报告正文" in out


def test_single_file_target_reviews_only_that_file(
    wired: type[FakeGitHubClient], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """沙箱根必须是目录：给单文件时根退到父目录，但审查范围只限这一个文件。"""
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    single = repo / "main.py"
    single.write_text("print('hi')\n", encoding="utf-8")
    (repo / "other.py").write_text("print('other')\n", encoding="utf-8")

    assert cli.main(["review", str(single)]) == 0

    kwargs = FakeWorkflow.instances[0].kwargs
    assert kwargs["root"] == repo.resolve()
    assert kwargs["paths"] == ("main.py",)
    assert kwargs["target"] == "sample_repo/main.py"
    assert "只审 sample_repo/main.py" in capsys.readouterr().out


def test_directory_target_has_no_file_scope(
    wired: type[FakeGitHubClient], tmp_path: Path
) -> None:
    """给目录时不给范围：整仓库审查是默认行为，不能被单文件逻辑串到。"""
    repo = tmp_path / "sample_repo"
    repo.mkdir()

    assert cli.main(["review", str(repo)]) == 0

    assert FakeWorkflow.instances[0].kwargs["paths"] is None


def test_state_and_out_are_wired_through(
    wired: type[FakeGitHubClient], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    state_file = tmp_path / "state.json"
    out_file = tmp_path / "report.md"

    assert cli.main(
        [
            "review",
            str(repo),
            "--state",
            str(state_file),
            "--reset",
            "--out",
            str(out_file),
        ]
    ) == 0

    assert FakeWorkflow.instances[0].kwargs["state_path"] == state_file
    assert FakeWorkflow.instances[0].calls == [{"resume": False, "reset": True}]
    assert out_file.read_text(encoding="utf-8") == "# 报告"
    assert "[输出] Markdown 报告已写入" in capsys.readouterr().out


def test_degraded_run_returns_1(
    wired: type[FakeGitHubClient], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """有阶段失败 → 退出码 1，并且必须提示"别据此判定没有问题"。"""
    FakeWorkflow.result = make_result(success=False)
    repo = tmp_path / "sample_repo"
    repo.mkdir()

    assert cli.main(["review", str(repo)]) == 1

    out = capsys.readouterr().out
    assert "⚠ 未完成的阶段：verify" in out
    assert "请勿据此判定「没有问题」" in out


# ------------------------------------------------------------------ 远端工作区
def test_remote_target_downloads_then_cleans_workdir(
    wired: type[FakeGitHubClient], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["review", "acme/demo"]) == 0

    _, repo, workdir, ref = FakeGitHubClient.calls[1]
    assert repo == "acme/demo"
    assert ref == "main"
    assert FakeWorkflow.instances[0].kwargs["target"] == "acme/demo@main"
    assert FakeWorkflow.instances[0].kwargs["root"] == workdir

    assert not workdir.exists(), "审查结束后临时工作区必须被删除"
    out = capsys.readouterr().out
    assert "[远端] 已下载 acme/demo@main" in out
    assert "来源=GitHub" in out


def test_remote_download_failure_cleans_workdir(
    wired: type[FakeGitHubClient], capsys: pytest.CaptureFixture[str]
) -> None:
    wired.error = GitHubRateLimitError("触发限流")

    assert cli.main(["review", "acme/demo"]) == 2

    workdir = FakeGitHubClient.calls[1][2]
    assert not workdir.exists(), "下载失败也要把临时工作区删掉"
    err = capsys.readouterr().err
    assert "远端仓库准备失败" in err
    assert "本意是本地路径" in err
    assert FakeWorkflow.instances == []


def test_keep_workdir_keeps_tree(
    wired: type[FakeGitHubClient], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["review", "acme/demo", "--keep-workdir"]) == 0

    workdir = FakeGitHubClient.calls[1][2]
    assert (workdir / "app.py").exists()
    assert f"已保留：{workdir}" in capsys.readouterr().out

    shutil.rmtree(workdir, ignore_errors=True)


def test_enable_test_on_remote_warns(
    wired: type[FakeGitHubClient], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["review", "acme/demo", "--enable-test"]) == 0

    out = capsys.readouterr().out
    assert "请确认目标可信" in out
    assert FakeWorkflow.instances[0].kwargs["enable_test"] is True


def test_local_target_does_not_warn_about_remote_tests(
    wired: type[FakeGitHubClient], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "sample_repo"
    repo.mkdir()

    assert cli.main(["review", str(repo), "--enable-test"]) == 0

    assert "请确认目标可信" not in capsys.readouterr().out
