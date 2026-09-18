"""审查目标的解析测试：本地目录 / 单文件 / ``owner/repo`` → 可审查的工作区。

为什么单独测这个模块：命令行与 HTTP 服务共用这一层，且这里有两条**必须一致**的规则
——单文件只审该文件、远端工作区用完即删。规则的实现只此一份，测试也只此一份。

全部离线：GitHub 客户端被替换成替身，没有网络请求，也不产生 LLM 调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from codeagentx import workspace
from codeagentx.core.exceptions import GitHubRateLimitError, TargetError
from codeagentx.protocols.github_archive import ArchiveExtractResult


# ------------------------------------------------------------------ 替身
class FakeGitHubClient:
    """替身 GitHub 客户端：记下调用，按需抛错，并真的在磁盘上造出"仓库"。"""

    calls: list[tuple[Any, ...]] = []
    error: Exception | None = None

    def __init__(self, **_: Any) -> None:
        self.closed = False

    @classmethod
    def from_config(cls) -> FakeGitHubClient:
        return cls()

    def resolve_ref(self, repo: str) -> str:
        type(self).calls.append(("resolve_ref", repo))
        return "main"

    def download_archive(
        self, repo: str, dest: Path, *, ref: str = "", **kwargs: Any
    ) -> ArchiveExtractResult:
        type(self).calls.append(("download_archive", repo, Path(dest), ref))
        if type(self).error is not None:
            raise type(self).error
        root = Path(dest)
        (root / "app").mkdir(parents=True, exist_ok=True)
        (root / "app" / "main.py").write_text("print('hi')\n", encoding="utf-8")
        return ArchiveExtractResult(root=root, files=1, bytes=11)

    def close(self) -> None:
        self.closed = True
        type(self).calls.append(("close",))


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> type[FakeGitHubClient]:
    FakeGitHubClient.calls = []
    FakeGitHubClient.error = None
    monkeypatch.setattr(workspace, "GitHubClient", FakeGitHubClient)
    return FakeGitHubClient


# ------------------------------------------------------------------ 远端判定
@pytest.mark.parametrize(
    "value",
    [
        "acme/demo",
        "acme/demo@v1.2.0",
        "https://github.com/acme/demo",
        "https://github.com/acme/demo/tree/main",
        "git@github.com:acme/demo.git",
    ],
)
def test_looks_like_repo_accepts_remote_forms(value: str) -> None:
    assert workspace.looks_like_repo(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "./sample",
        "../sample",
        "/tmp/sample",
        "~/sample",
        r"E:\code\sample",
        "acme/demo/extra",
    ],
)
def test_looks_like_repo_rejects_local_or_odd_forms(value: str) -> None:
    assert workspace.looks_like_repo(value) is False


def test_bare_relative_subdir_is_treated_as_remote() -> None:
    """``data/sample_repo`` 这种两段式写法与 ``owner/repo`` 无法区分。

    只要该路径**不存在**就当远端处理——但错误信息里必须补一句"也可能是路径拼错"，
    否则用户会拿着"仓库不存在"去查 GitHub。
    """
    assert workspace.looks_like_repo("data/sample_repo") is True


# ------------------------------------------------------------------ 本地目标
def test_directory_target_has_no_scope(tmp_path: Path) -> None:
    repo = tmp_path / "sample_repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "main.py").write_text("print('hi')\n", encoding="utf-8")

    prepared = workspace.prepare_target(str(repo))

    assert prepared.root == repo.resolve()
    assert prepared.display == "sample_repo"
    assert prepared.paths == ()
    assert prepared.remote is False
    assert prepared.workdir is None
    assert prepared.note == ""
    prepared.cleanup()  # 本地目标：空操作，不能误删用户的东西
    assert repo.exists()


def test_single_file_target_scopes_to_that_file(tmp_path: Path) -> None:
    """沙箱根必须退到父目录，但审查范围只留这一个文件。"""
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    single = repo / "main.py"
    single.write_text("print('hi')\n", encoding="utf-8")
    (repo / "other.py").write_text("print('other')\n", encoding="utf-8")

    prepared = workspace.prepare_target(str(single))

    assert prepared.root == repo.resolve()
    assert prepared.display == "sample_repo/main.py"
    assert prepared.paths == ("main.py",)
    assert "只审 sample_repo/main.py" in prepared.note


def test_unknown_target_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(TargetError) as excinfo:
        workspace.prepare_target("not-a-dir/nor-a-repo/extra")

    assert "既不是已存在的路径" in excinfo.value.message
    assert "拼写" in (excinfo.value.detail or "")


# ------------------------------------------------------------------ 远端目标
def test_remote_target_downloads_into_workdir(wired: type[FakeGitHubClient]) -> None:
    prepared = workspace.prepare_target("acme/demo")

    assert prepared.remote is True
    assert prepared.display == "acme/demo@main"
    assert prepared.workdir == prepared.root
    assert (prepared.root / "app" / "main.py").exists()
    assert prepared.paths == ()
    assert "已下载 acme/demo@main" in prepared.note
    assert ("close",) in wired.calls, "客户端必须被关闭"

    prepared.cleanup()
    assert not prepared.root.exists()


def test_explicit_ref_is_not_resolved_again(wired: type[FakeGitHubClient]) -> None:
    prepared = workspace.prepare_target("acme/demo", ref="v1.2.0")

    assert prepared.display == "acme/demo@v1.2.0"
    assert [call[0] for call in wired.calls] == ["download_archive", "close"]

    prepared.cleanup()


def test_cleanup_keep_leaves_workdir(wired: type[FakeGitHubClient]) -> None:
    prepared = workspace.prepare_target("acme/demo")

    prepared.cleanup(keep=True)
    assert prepared.root.exists()

    prepared.cleanup()


def test_failed_download_removes_half_written_workdir(
    wired: type[FakeGitHubClient],
) -> None:
    wired.error = GitHubRateLimitError("触发限流")

    with pytest.raises(GitHubRateLimitError):
        workspace.prepare_target("acme/demo")

    workdir = next(Path(call[2]) for call in wired.calls if call[0] == "download_archive")
    assert not workdir.exists(), "下载失败不能把半截工作区留在磁盘上"


def test_inline_ref_is_not_appended_twice(wired: type[FakeGitHubClient]) -> None:
    """目标里已经写了 ``@ref`` 时不要再拼一次，否则报告里出现 ``repo@main@main``。"""
    prepared = workspace.prepare_target("acme/demo@main")

    assert prepared.display == "acme/demo@main"

    prepared.cleanup()


def test_limits_are_forwarded_to_download(
    wired: type[FakeGitHubClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """体积上限必须一路传到下载层，否则接口侧的限制形同虚设。"""
    seen: dict[str, Any] = {}
    original = wired.download_archive

    def spy(self: Any, repo: str, dest: Path, **kwargs: Any) -> ArchiveExtractResult:
        seen.update(kwargs)
        return original(self, repo, dest, **kwargs)

    monkeypatch.setattr(wired, "download_archive", spy)
    prepared = workspace.prepare_target("acme/demo", max_files=7, max_bytes=99)

    assert seen == {"ref": "main", "max_files": 7, "max_bytes": 99}

    prepared.cleanup()
