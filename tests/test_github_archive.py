"""GitHub 归档解压测试：把"外部来的 zip"当成不可信输入来验。

覆盖四类事：

1. **正常路径**：剥掉 zipball 顶层目录，工作区根就是仓库根，内容一字不差；
2. **逃逸**：``..``、Windows 盘符写法、符号链接，一个都不许落到工作区里，
   更不许落到工作区之外；
3. **上限**：文件数/总字节超限要**显式失败**（不能默默写满磁盘），
   单文件超限则跳过并记账；
4. **下载**：``download_archive`` 打的确实是 zipball 接口，且解压结果可用。

全部离线：zip 在内存里拼，HTTP 用 ``httpx.MockTransport``。
"""

from __future__ import annotations

import stat
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import pytest

from codeagentx.core.exceptions import GitHubArchiveError
from codeagentx.protocols.github_archive import ArchiveExtractResult, extract_zipball
from codeagentx.protocols.github_client import GitHubClient


def zip_bytes(*entries: tuple[str, bytes], symlinks: tuple[str, ...] = ()) -> bytes:
    """拼一个内存 zip；``entries`` 是 (条目名, 内容)，``symlinks`` 只放链接名。"""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
        for name in symlinks:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "/etc/passwd")
    return buffer.getvalue()


def read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def test_extract_strips_top_level_directory(tmp_path: Path) -> None:
    data = zip_bytes(
        ("demo-abc123/README.md", b"# demo"),
        ("demo-abc123/src/app.py", b"print('hi')"),
    )

    result = extract_zipball(data, tmp_path / "work")

    assert isinstance(result, ArchiveExtractResult)
    assert result.files == 2
    assert result.skipped == []
    assert read(result.root, "README.md") == "# demo"
    assert read(result.root, "src/app.py") == "print('hi')"
    assert result.root == (tmp_path / "work").resolve()
    assert result.to_dict()["files"] == 2


def test_extract_keeps_layout_when_there_is_no_single_root(tmp_path: Path) -> None:
    """顶层不唯一就不是 zipball 的形态，此时不剥目录，避免把两层结构压成一层。"""
    data = zip_bytes(("a/one.py", b"1"), ("b/two.py", b"2"))

    result = extract_zipball(data, tmp_path / "work")

    assert result.files == 2
    assert read(result.root, "a/one.py") == "1"
    assert read(result.root, "b/two.py") == "2"


def test_extract_rejects_zip_slip_entries(tmp_path: Path) -> None:
    dest = tmp_path / "work"
    data = zip_bytes(
        ("demo-abc123/app.py", b"ok"),
        ("demo-abc123/../../evil.py", b"pwned"),
        ("demo-abc123/sub/../../../evil2.py", b"pwned"),
    )

    result = extract_zipball(data, dest)

    assert result.files == 1
    assert read(result.root, "app.py") == "ok"
    assert not (tmp_path / "evil.py").exists()
    assert not (tmp_path / "evil2.py").exists()
    assert len(result.skipped) == 2


def test_extract_rejects_windows_drive_style_names(tmp_path: Path) -> None:
    result = extract_zipball(
        zip_bytes(("demo-abc123/C:/Windows/system32/evil.py", b"pwned")),
        tmp_path / "work",
    )

    assert result.files == 0
    assert result.skipped == ["demo-abc123/C:/Windows/system32/evil.py"]
    assert [path for path in (tmp_path / "work").rglob("*") if path.is_file()] == []


def test_extract_skips_symlinks_and_vcs_directories(tmp_path: Path) -> None:
    data = zip_bytes(
        ("demo-abc123/app.py", b"ok"),
        ("demo-abc123/.git/config", b"[core]"),
        ("demo-abc123/data/.gitkeep", b""),
        symlinks=("demo-abc123/link.py",),
    )

    result = extract_zipball(data, tmp_path / "work")

    # 两个 .git 路径不落地；data/.gitkeep 不叫 .git，正常保留
    assert result.files == 2
    assert read(result.root, "app.py") == "ok"
    assert (result.root / "data" / ".gitkeep").exists()
    assert not (result.root / "link.py").exists()
    assert not (result.root / ".git").exists()
    assert any("符号链接" in item for item in result.skipped)


def test_extract_skips_oversized_single_file(tmp_path: Path) -> None:
    result = extract_zipball(
        zip_bytes(("demo-abc123/big.bin", b"x" * 64), ("demo-abc123/app.py", b"ok")),
        tmp_path / "work",
        max_file_bytes=16,
    )

    assert result.files == 1
    assert read(result.root, "app.py") == "ok"
    assert any("超过单文件上限" in item for item in result.skipped)


def test_extract_fails_loudly_when_file_count_exceeds_limit(tmp_path: Path) -> None:
    data = zip_bytes(*[(f"demo-abc123/file{index}.py", b"x") for index in range(5)])

    with pytest.raises(GitHubArchiveError) as error:
        extract_zipball(data, tmp_path / "work", max_files=3)

    assert "文件数超过上限" in error.value.message


def test_extract_fails_loudly_when_total_bytes_exceed_limit(tmp_path: Path) -> None:
    data = zip_bytes(
        ("demo-abc123/one.py", b"x" * 100),
        ("demo-abc123/two.py", b"x" * 100),
    )

    with pytest.raises(GitHubArchiveError) as error:
        extract_zipball(data, tmp_path / "work", max_bytes=150)

    assert "解压后超过上限" in error.value.message


def test_extract_reports_bad_zip(tmp_path: Path) -> None:
    with pytest.raises(GitHubArchiveError) as error:
        extract_zipball(b"<html>rate limited</html>", tmp_path / "work")

    assert "不是合法的 zip" in error.value.message
    assert error.value.detail and "字节" in error.value.detail


# ---------------------------------------------------------------- 下载


def make_site(archive: bytes, *, seen: list[str]) -> httpx.MockTransport:
    """最小离线站点：仓库元信息 + zipball。"""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        path = request.url.path
        if path == "/repos/acme/demo":
            return httpx.Response(
                200,
                json={
                    "name": "demo",
                    "full_name": "acme/demo",
                    "owner": {"login": "acme"},
                    "default_branch": "main",
                },
            )
        if path.startswith("/repos/acme/demo/zipball/"):
            return httpx.Response(200, content=archive, headers={"content-type": "application/zip"})
        return httpx.Response(404, json={"message": "Not Found"})

    return httpx.MockTransport(handler)


def test_download_archive_hits_zipball_endpoint_and_extracts(tmp_path: Path) -> None:
    archive = zip_bytes(("demo-abc123/app/auth.py", b"SECRET = 'x'"))
    seen: list[str] = []
    client = GitHubClient(client=httpx.Client(transport=make_site(archive, seen=seen)))

    result = client.download_archive("acme/demo", tmp_path / "work")

    assert any("/repos/acme/demo/zipball/main" in url for url in seen)
    assert result.files == 1
    assert read(result.root, "app/auth.py") == "SECRET = 'x'"


def test_download_archive_honours_explicit_ref_and_limits(tmp_path: Path) -> None:
    archive = zip_bytes(
        ("demo-abc123/one.py", b"1"),
        ("demo-abc123/two.py", b"2"),
    )
    seen: list[str] = []
    client = GitHubClient(client=httpx.Client(transport=make_site(archive, seen=seen)))

    with pytest.raises(GitHubArchiveError):
        client.download_archive("acme/demo@v1.2.0", tmp_path / "work", max_files=1)

    assert any("/zipball/v1.2.0" in url for url in seen)
    assert not any(url.endswith("/repos/acme/demo") for url in seen), "显式 ref 不该再查默认分支"


def test_download_archive_reports_html_body_as_archive_error(tmp_path: Path) -> None:
    """拿到 200 + HTML（限流页/代理页）时，报的必须是"不是合法 zip"，
    而不是静默写出半个仓库。"""
    client = GitHubClient(client=httpx.Client(transport=make_site(b"<html>nope</html>", seen=[])))

    with pytest.raises(GitHubArchiveError):
        client.download_archive("acme/demo", tmp_path / "work")
