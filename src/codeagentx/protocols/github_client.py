"""GitHub REST v3 只读客户端。

为什么自带一个而不装 SDK：与 AD-01 同款理由（依赖越少越稳、覆盖 3.10~3.13），
而且 GitHub 的"读仓库"需求本质就是几个 GET——自己实现可以**注入
``httpx.Client``**，于是离线也能用录制的响应跑通全链路（见 tests/fixtures/github）。

只做只读
--------
本模块只发 GET，不提供任何写操作（创建 PR、提交文件等一概没有）。
这一点是刻意的：Agent 可以"读世界的代码"，但不能替用户改远端仓库。

路径安全
--------
``path`` 一律走 :func:`normalize_repo_path`：拒绝绝对路径、``..`` 逃逸、
盘符与反斜杠，保证"模型给的路径"不可能跳到仓库之外。
"""

from __future__ import annotations

import base64
import binascii
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from codeagentx.core.exceptions import (
    GitHubAuthError,
    GitHubError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubResponseError,
    SecurityViolationError,
)
from codeagentx.core.logger import get_logger
from codeagentx.protocols.github_archive import (
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_FILE_BYTES,
    MAX_ARCHIVE_FILES,
    ArchiveExtractResult,
    extract_zipball,
)

logger = get_logger("protocols.github_client")

__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_FILE_CHARS",
    "DEFAULT_MAX_PR_FILES",
    "GitHubClient",
    "GitHubClientStats",
    "GitHubFile",
    "GitHubPullRequest",
    "GitHubPullRequestFile",
    "GitHubRepository",
    "GitHubRepoRef",
    "GitHubTree",
    "GitHubTreeEntry",
    "normalize_repo_path",
]

DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT = 20.0
DEFAULT_USER_AGENT = "CodeAgentX/0.1.0"
#: 固定 API 版本，避免 GitHub 侧行为漂移（文档推荐显式声明）
DEFAULT_API_VERSION = "2022-11-28"
DEFAULT_ACCEPT = "application/vnd.github+json"
DIFF_ACCEPT = "application/vnd.github.v3.diff"

#: 仓库树最多取多少条目（GitHub 自己的截断上限约 10 万，这里更保守）
DEFAULT_MAX_ENTRIES = 2000
DEFAULT_MAX_FILE_CHARS = 60_000
DEFAULT_MAX_PR_FILES = 100

#: Contents API 对 ``content`` 字段有 1MB 上限，超过时返回空字符串
CONTENTS_INLINE_LIMIT = 1_000_000


# ------------------------------------------------------------------ 仓库标识
@dataclass(frozen=True)
class GitHubRepoRef:
    """仓库标识：``owner/name``，可选 ``ref``（分支/标签/提交）。"""

    owner: str
    name: str
    ref: str = ""

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @classmethod
    def parse(cls, value: str | GitHubRepoRef) -> GitHubRepoRef:
        """解析多种常见写法。

        支持 ``owner/repo``、``owner/repo@tag``、``git@github.com:owner/repo.git``、
        ``https://github.com/owner/repo``、``https://github.com/owner/repo/tree/main``、
        ``https://api.github.com/repos/owner/repo``。
        """
        if isinstance(value, GitHubRepoRef):
            return value
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("仓库标识不能为空")

        ref = ""
        text = raw
        if text.startswith("git@"):
            _, _, text = text.partition(":")  # git@github.com:owner/repo.git
        elif "://" in text:
            segments = [part for part in urlparse(text).path.split("/") if part]
            if segments[:1] == ["repos"]:  # api.github.com/repos/owner/repo
                segments = segments[1:]
            if len(segments) >= 4 and segments[2] in {"tree", "blob"}:
                ref = "/".join(segments[3:])
            segments = segments[:2]
            text = "/".join(segments)

        if "@" in text:
            text, _, inline = text.partition("@")
            ref = ref or inline.strip()

        parts = [part for part in text.removesuffix(".git").strip("/").split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"无法解析仓库标识：{raw!r}（应形如 owner/repo）")
        return cls(owner=parts[0], name=parts[1], ref=ref)

    def to_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "name": self.name, "ref": self.ref, "slug": self.slug}


def normalize_repo_path(path: str | None) -> str:
    """把仓库内路径规范成 ``a/b/c``（相对、正斜杠），并拒绝越界。

    Raises:
        SecurityViolationError: 绝对路径、``..`` 逃逸或含盘符。
    """
    raw = str(path or "").strip().replace("\\", "/")
    if not raw or raw in {".", "./"}:
        return ""
    if raw.startswith("/") or raw.startswith("~"):
        raise SecurityViolationError(f"仓库内路径必须是相对路径：{path!r}")
    if len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":":
        raise SecurityViolationError(f"仓库内路径不允许盘符：{path!r}")

    parts: list[str] = []
    for part in raw.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise SecurityViolationError(f"仓库内路径不允许越界：{path!r}")
        parts.append(part)
    if not parts:
        raise SecurityViolationError(f"仓库内路径非法：{path!r}")
    return "/".join(parts)


# ------------------------------------------------------------------ 数据结构
@dataclass(frozen=True)
class GitHubTreeEntry:
    """仓库树里的一条记录。"""

    path: str
    type: str = "blob"
    sha: str = ""
    size: int = 0
    mode: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GitHubTreeEntry:
        path = str(payload.get("path") or "").strip()
        if not path:
            raise GitHubResponseError("仓库树条目缺少 path", detail=str(payload)[:200])
        try:
            size = int(payload.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        return cls(
            path=path,
            type=str(payload.get("type") or "blob"),
            sha=str(payload.get("sha") or ""),
            size=size,
            mode=str(payload.get("mode") or ""),
        )

    @property
    def is_file(self) -> bool:
        return self.type == "blob"

    @property
    def is_dir(self) -> bool:
        return self.type == "tree"

    @property
    def name(self) -> str:
        return PurePosixPath(self.path).name

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.path).suffix

    @property
    def depth(self) -> int:
        return len([part for part in self.path.split("/") if part])

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": self.type,
            "sha": self.sha,
            "size": self.size,
            "mode": self.mode,
        }


@dataclass
class GitHubTree:
    """一次仓库树读取的结果（含截断标记）。"""

    repo: str
    ref: str = ""
    sha: str = ""
    truncated: bool = False
    entries: list[GitHubTreeEntry] = field(default_factory=list)

    @property
    def files(self) -> list[GitHubTreeEntry]:
        return [entry for entry in self.entries if entry.is_file]

    @property
    def directories(self) -> list[GitHubTreeEntry]:
        return [entry for entry in self.entries if entry.is_dir]

    def find(self, path: str) -> GitHubTreeEntry | None:
        wanted = normalize_repo_path(path) if path else ""
        for entry in self.entries:
            if entry.path == wanted:
                return entry
        return None

    def to_text(self, *, max_lines: int = 400) -> str:
        """渲染成可读清单：``类型 大小 路径``，超出行数会标注省略。"""
        lines = [
            f"{'dir ' if entry.is_dir else 'file'} {entry.size:>8}  {entry.path}"
            for entry in self.entries[:max_lines]
        ]
        if len(self.entries) > max_lines:
            lines.append(f"...（共 {len(self.entries)} 条，仅显示前 {max_lines} 条）")
        note = "，GitHub 侧已截断" if self.truncated else ""
        header = f"# {self.repo}@{self.ref or self.sha} 共 {len(self.entries)} 条{note}"
        return "\n".join([header, *lines])

    def to_dict(self, *, with_entries: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "repo": self.repo,
            "ref": self.ref,
            "sha": self.sha,
            "truncated": self.truncated,
            "total": len(self.entries),
            "files": len(self.files),
            "directories": len(self.directories),
        }
        if with_entries:
            payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


@dataclass(frozen=True)
class GitHubRepository:
    """仓库元信息。"""

    owner: str
    name: str
    default_branch: str = "main"
    description: str = ""
    language: str = ""
    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    size_kb: int = 0
    private: bool = False
    archived: bool = False
    html_url: str = ""
    pushed_at: str = ""

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GitHubRepository:
        owner = payload.get("owner")
        owner_login = str(owner.get("login") or "") if isinstance(owner, Mapping) else ""
        name = str(payload.get("name") or "").strip()
        if not name:
            raise GitHubResponseError("仓库响应缺少 name", detail=str(payload)[:200])
        return cls(
            owner=str(payload.get("owner_login") or owner_login or ""),
            name=name,
            default_branch=str(payload.get("default_branch") or "main"),
            description=str(payload.get("description") or ""),
            language=str(payload.get("language") or ""),
            stars=_as_int(payload.get("stargazers_count")),
            forks=_as_int(payload.get("forks_count")),
            open_issues=_as_int(payload.get("open_issues_count")),
            size_kb=_as_int(payload.get("size")),
            private=bool(payload.get("private")),
            archived=bool(payload.get("archived")),
            html_url=str(payload.get("html_url") or ""),
            pushed_at=str(payload.get("pushed_at") or ""),
        )

    def to_text(self) -> str:
        return "\n".join(
            [
                f"# {self.slug}",
                f"默认分支：{self.default_branch}",
                f"语言：{self.language or '未知'} | 星标：{self.stars} | Fork：{self.forks}",
                f"开放 issue：{self.open_issues} | 体积：{self.size_kb} KB | "
                f"私有：{'是' if self.private else '否'} | 归档：{'是' if self.archived else '否'}",
                f"最近推送：{self.pushed_at or '未知'}",
                f"描述：{self.description or '（无）'}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "default_branch": self.default_branch,
            "description": self.description,
            "language": self.language,
            "stars": self.stars,
            "forks": self.forks,
            "open_issues": self.open_issues,
            "size_kb": self.size_kb,
            "private": self.private,
            "archived": self.archived,
            "html_url": self.html_url,
            "pushed_at": self.pushed_at,
        }


@dataclass(frozen=True)
class GitHubFile:
    """一个文件的内容与元信息。"""

    path: str
    text: str
    sha: str = ""
    size: int = 0
    truncated: bool = False
    html_url: str = ""
    ref: str = ""

    @property
    def line_count(self) -> int:
        return len(self.text.splitlines())

    def to_dict(self, *, with_text: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "sha": self.sha,
            "size": self.size,
            "lines": self.line_count,
            "truncated": self.truncated,
            "ref": self.ref,
            "html_url": self.html_url,
        }
        if with_text:
            payload["text"] = self.text
        return payload


@dataclass(frozen=True)
class GitHubPullRequest:
    """一个 PR 的摘要信息。"""

    number: int
    title: str = ""
    state: str = ""
    draft: bool = False
    author: str = ""
    base_ref: str = ""
    head_ref: str = ""
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    commits: int = 0
    body: str = ""
    created_at: str = ""
    updated_at: str = ""
    html_url: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GitHubPullRequest:
        try:
            number = int(payload.get("number") or 0)
        except (TypeError, ValueError) as exc:
            raise GitHubResponseError("PR 响应缺少合法的 number", detail=str(payload)[:200]) from exc
        if number <= 0:
            raise GitHubResponseError("PR 响应缺少 number", detail=str(payload)[:200])

        def _ref(key: str) -> str:
            branch = payload.get(key)
            return str(branch.get("ref") or "") if isinstance(branch, Mapping) else ""

        user = payload.get("user")
        return cls(
            number=number,
            title=str(payload.get("title") or ""),
            state=str(payload.get("state") or ""),
            draft=bool(payload.get("draft")),
            author=str(user.get("login") or "") if isinstance(user, Mapping) else "",
            base_ref=_ref("base"),
            head_ref=_ref("head"),
            additions=_as_int(payload.get("additions")),
            deletions=_as_int(payload.get("deletions")),
            changed_files=_as_int(payload.get("changed_files")),
            commits=_as_int(payload.get("commits")),
            body=str(payload.get("body") or ""),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            html_url=str(payload.get("html_url") or ""),
        )

    def to_text(self, *, body_chars: int = 1200) -> str:
        body = self.body.strip()
        if len(body) > body_chars:
            body = f"{body[:body_chars]}\n...（描述已截断，共 {len(self.body)} 字符）"
        return "\n".join(
            [
                f"# PR #{self.number} {self.title}",
                f"状态：{self.state}{'（草稿）' if self.draft else ''} | 作者：{self.author or '未知'}",
                f"分支：{self.head_ref} -> {self.base_ref}",
                f"变更：+{self.additions} / -{self.deletions}，{self.changed_files} 个文件，"
                f"{self.commits} 个提交",
                f"更新时间：{self.updated_at or '未知'}",
                f"链接：{self.html_url}",
                "",
                body or "（无描述）",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "state": self.state,
            "draft": self.draft,
            "author": self.author,
            "base_ref": self.base_ref,
            "head_ref": self.head_ref,
            "additions": self.additions,
            "deletions": self.deletions,
            "changed_files": self.changed_files,
            "commits": self.commits,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "html_url": self.html_url,
        }


@dataclass(frozen=True)
class GitHubPullRequestFile:
    """PR 里单个文件的变更（``patch`` 是 unified diff 片段）。"""

    filename: str
    status: str = ""
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    previous_filename: str = ""
    patch: str = ""
    blob_url: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GitHubPullRequestFile:
        filename = str(payload.get("filename") or "").strip()
        if not filename:
            raise GitHubResponseError("PR 文件缺少 filename", detail=str(payload)[:200])
        return cls(
            filename=filename,
            status=str(payload.get("status") or ""),
            additions=_as_int(payload.get("additions")),
            deletions=_as_int(payload.get("deletions")),
            changes=_as_int(payload.get("changes")),
            previous_filename=str(payload.get("previous_filename") or ""),
            patch=str(payload.get("patch") or ""),
            blob_url=str(payload.get("blob_url") or ""),
        )

    def to_text(self, *, patch_chars: int = 4000) -> str:
        header = f"### {self.filename}（{self.status}，+{self.additions} / -{self.deletions}）"
        if self.previous_filename:
            header += f"\n原名：{self.previous_filename}"
        patch = self.patch
        if not patch:
            return f"{header}\n（GitHub 未返回 patch，可能是二进制或超大文件）"
        if len(patch) > patch_chars:
            patch = f"{patch[:patch_chars]}\n...（patch 已截断，共 {len(self.patch)} 字符）"
        return f"{header}\n{patch}"

    def to_dict(self, *, with_patch: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "filename": self.filename,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "changes": self.changes,
        }
        if with_patch:
            payload["patch"] = self.patch
        return payload


@dataclass
class GitHubClientStats:
    """调用统计（可观测性 + 后续评估的输入）。"""

    requests: int = 0
    errors: int = 0
    duration: float = 0.0
    rate_limit_remaining: int | None = None
    truncated_responses: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "duration": round(self.duration, 3),
            "rate_limit_remaining": self.rate_limit_remaining,
            "truncated_responses": self.truncated_responses,
        }


# ------------------------------------------------------------------ 客户端
class GitHubClient:
    """GitHub 只读客户端。

    Args:
        token: 个人访问令牌；为空时不带 ``Authorization`` 头（公开仓库仍可读）。
        base_url: API 根地址，GitHub Enterprise 可换成自己的域名。
        client: 注入的 ``httpx.Client``（测试用 ``MockTransport`` 即可离线跑通）。
            注入的 client **不会被本类关闭**，由调用方负责。
    """

    def __init__(
        self,
        token: str = "",
        *,
        base_url: str = DEFAULT_API_BASE,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        api_version: str = DEFAULT_API_VERSION,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
        max_pr_files: int = DEFAULT_MAX_PR_FILES,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout 必须为正数")
        if max_entries <= 0 or max_file_chars <= 0 or max_pr_files <= 0:
            raise ValueError("容量上限必须为正数")

        self.token = (token or "").strip()
        self.base_url = (base_url or DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self.api_version = api_version
        self.max_entries = max_entries
        self.max_file_chars = max_file_chars
        self.max_pr_files = max_pr_files
        self.stats = GitHubClientStats()

        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._repo_cache: dict[str, GitHubRepository] = {}

    # -------------------------------------------------------- 构造与生命周期
    @classmethod
    def from_config(cls, config: Any = None, **kwargs: Any) -> GitHubClient:
        """从全局配置取令牌（``GITHUB_TOKEN``）。"""
        from codeagentx.config import get_config

        settings = config or get_config()
        kwargs.setdefault("token", getattr(settings, "github_token", "") or "")
        return cls(**kwargs)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.token)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def describe(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "authenticated": self.is_authenticated,
            "owns_client": self._owns_client,
            "stats": self.stats.as_dict(),
        }

    # -------------------------------------------------------- 仓库
    def get_repository(self, repo: str | GitHubRepoRef, *, use_cache: bool = True) -> GitHubRepository:
        """读取仓库元信息（同时用于解析默认分支）。"""
        ref = GitHubRepoRef.parse(repo)
        if use_cache and ref.slug in self._repo_cache:
            return self._repo_cache[ref.slug]
        payload = self._get_json(f"/repos/{ref.owner}/{ref.name}")
        repository = GitHubRepository.from_payload(self._as_object(payload, "仓库"))
        self._repo_cache[ref.slug] = repository
        return repository

    def resolve_ref(self, repo: str | GitHubRepoRef) -> str:
        """确定要用的 ref：显式给了就用显式的，否则取仓库默认分支。"""
        ref = GitHubRepoRef.parse(repo)
        if ref.ref:
            return ref.ref
        return self.get_repository(ref).default_branch

    # -------------------------------------------------------- 仓库树与文件
    def list_tree(
        self,
        repo: str | GitHubRepoRef,
        *,
        ref: str | None = None,
        recursive: bool = True,
        max_entries: int | None = None,
        path_prefix: str | None = None,
    ) -> GitHubTree:
        """读取仓库树（``git/trees`` 接口，``recursive`` 一次拿全）。

        Args:
            path_prefix: 只保留该目录下的条目；空表示全仓库。
        """
        identifier = GitHubRepoRef.parse(repo)
        if ref is None:
            ref = identifier.ref or self.resolve_ref(identifier)
        tree_sha = quote(ref, safe="")
        params = {"recursive": "1"} if recursive else None
        payload = self._as_object(
            self._get_json(f"/repos/{identifier.owner}/{identifier.name}/git/trees/{tree_sha}", params=params),
            "仓库树",
        )
        raw_entries = payload.get("tree") or []
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
            raise GitHubResponseError("仓库树的 tree 字段不是数组", detail=str(payload)[:200])

        entries = [
            GitHubTreeEntry.from_payload(item)
            for item in raw_entries
            if isinstance(item, Mapping)
        ]

        if path_prefix is not None:
            prefix = normalize_repo_path(path_prefix) if path_prefix else ""
            if prefix:
                entries = [e for e in entries if e.path == prefix or e.path.startswith(prefix + "/")]

        limit = self.max_entries if max_entries is None else max_entries
        locally_capped = len(entries) > limit
        if locally_capped:
            entries = entries[:limit]

        # 两种截断（GitHub 侧截断 / 本地按上限截断）都算"内容不完整"，
        # 都要计入统计并告知调用方——否则拿到的树是残缺的却看不出来。
        truncated = bool(payload.get("truncated")) or locally_capped
        if truncated:
            self.stats.truncated_responses += 1
            logger.warning(
                "仓库树被截断（GitHub 侧=%s，本地上限=%s）：%s@%s",
                bool(payload.get("truncated")),
                locally_capped,
                identifier.slug,
                ref,
            )

        return GitHubTree(
            repo=identifier.slug,
            ref=ref,
            sha=str(payload.get("sha") or ""),
            truncated=truncated,
            entries=entries,
        )

    def download_archive(
        self,
        repo: str | GitHubRepoRef,
        dest: str | Path,
        *,
        ref: str | None = None,
        max_files: int = MAX_ARCHIVE_FILES,
        max_bytes: int = MAX_ARCHIVE_BYTES,
        max_file_bytes: int = MAX_ARCHIVE_FILE_BYTES,
    ) -> ArchiveExtractResult:
        """下载整仓库归档（zipball）并解压到 ``dest``，返回仓库根目录。

        与 :meth:`read_file` 的分工：那个用于"按需读几个文件"，本方法用于
        "要把整个仓库当作审查目标"（工具沙箱需要一个真实的本地根目录）。
        下载与解压都不执行远端代码，解压的安全约束见
        :func:`~codeagentx.protocols.github_archive.extract_zipball`。
        """
        identifier = GitHubRepoRef.parse(repo)
        target_ref = ref or identifier.ref or self.resolve_ref(identifier)
        path = f"/repos/{identifier.owner}/{identifier.name}/zipball/{quote(target_ref, safe='')}"
        # 归档接口会 302 到 codeload，客户端已开启 follow_redirects
        response = self._send(path, params=None, accept=DEFAULT_ACCEPT)
        return extract_zipball(
            response.content,
            dest,
            max_files=max_files,
            max_bytes=max_bytes,
            max_file_bytes=max_file_bytes,
        )

    def read_file(
        self,
        repo: str | GitHubRepoRef,
        path: str,
        *,
        ref: str | None = None,
        max_chars: int | None = None,
    ) -> GitHubFile:
        """读取仓库内一个文本文件（``contents`` 接口 + base64 解码）。

        Raises:
            GitHubNotFoundError: 路径不存在（或指向目录）。
            GitHubResponseError: 文件超过 Contents API 的 1MB 内联上限。
        """
        identifier = GitHubRepoRef.parse(repo)
        safe_path = normalize_repo_path(path)
        target_ref = ref or identifier.ref or ""
        params = {"ref": target_ref} if target_ref else None
        payload = self._get_json(
            f"/repos/{identifier.owner}/{identifier.name}/contents/{quote(safe_path, safe='/')}",
            params=params,
        )
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            raise GitHubNotFoundError(
                f"{safe_path} 是目录而不是文件", detail=f"repo={identifier.slug} ref={target_ref}"
            )
        data = self._as_object(payload, "文件")

        size = _as_int(data.get("size"))
        raw_content = str(data.get("content") or "")
        if not raw_content and size > CONTENTS_INLINE_LIMIT:
            raise GitHubResponseError(
                f"文件过大（{size} 字节），超出 Contents API 的 1MB 内联上限",
                detail=f"path={safe_path}，可改用 list_tree + 分批读取",
            )
        text = _decode_content(raw_content, str(data.get("encoding") or ""), safe_path)

        limit = self.max_file_chars if max_chars is None else max_chars
        truncated = False
        if len(text) > limit:
            total_chars = len(text)  # 先记下原始长度，截断后就取不到了
            head = text[:limit]
            cut = head.rfind("\n")
            text = f"{head if cut <= 0 else head[:cut]}\n...（已截断，仅返回前 {limit} 字符 / 共 {total_chars} 字符）"
            truncated = True
            self.stats.truncated_responses += 1

        return GitHubFile(
            path=safe_path,
            text=text,
            sha=str(data.get("sha") or ""),
            size=size,
            truncated=truncated,
            html_url=str(data.get("html_url") or ""),
            ref=target_ref,
        )

    def list_directory(
        self,
        repo: str | GitHubRepoRef,
        path: str = "",
        *,
        ref: str | None = None,
    ) -> list[GitHubTreeEntry]:
        """列出仓库内某个目录（``contents`` 接口返回数组时即目录）。"""
        identifier = GitHubRepoRef.parse(repo)
        safe_path = normalize_repo_path(path) if path else ""
        target_ref = ref or identifier.ref or ""
        params = {"ref": target_ref} if target_ref else None
        suffix = f"/{quote(safe_path, safe='/')}" if safe_path else ""
        payload = self._get_json(f"/repos/{identifier.owner}/{identifier.name}/contents{suffix}", params=params)
        if isinstance(payload, Mapping):
            raise GitHubNotFoundError(
                f"{safe_path or '/'} 是文件而不是目录", detail=f"repo={identifier.slug}"
            )
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise GitHubResponseError("目录列表响应不是数组", detail=str(payload)[:200])

        parent = safe_path
        entries: list[GitHubTreeEntry] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            entries.append(
                GitHubTreeEntry(
                    path=f"{parent}/{name}" if parent else name,
                    type="tree" if str(item.get("type") or "") == "dir" else "blob",
                    sha=str(item.get("sha") or ""),
                    size=_as_int(item.get("size")),
                )
            )
        return entries

    # -------------------------------------------------------- PR
    def list_pull_requests(
        self,
        repo: str | GitHubRepoRef,
        *,
        state: str = "open",
        limit: int = 20,
        sort: str = "updated",
        direction: str = "desc",
    ) -> list[GitHubPullRequest]:
        """列出 PR（``state`` 取 ``open`` / ``closed`` / ``all``）。"""
        identifier = GitHubRepoRef.parse(repo)
        if state not in {"open", "closed", "all"}:
            raise ValueError(f"state 非法：{state!r}（应为 open/closed/all）")
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        payload = self._get_json(
            f"/repos/{identifier.owner}/{identifier.name}/pulls",
            params={"state": state, "sort": sort, "direction": direction, "per_page": min(limit, 100)},
        )
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise GitHubResponseError("PR 列表响应不是数组", detail=str(payload)[:200])
        pulls = [
            GitHubPullRequest.from_payload(item) for item in payload if isinstance(item, Mapping)
        ]
        return pulls[:limit]

    def get_pull_request(self, repo: str | GitHubRepoRef, number: int) -> GitHubPullRequest:
        """读取单个 PR 的详情。"""
        identifier = GitHubRepoRef.parse(repo)
        payload = self._get_json(f"/repos/{identifier.owner}/{identifier.name}/pulls/{int(number)}")
        return GitHubPullRequest.from_payload(self._as_object(payload, "PR"))

    def get_pull_request_files(
        self,
        repo: str | GitHubRepoRef,
        number: int,
        *,
        limit: int | None = None,
    ) -> list[GitHubPullRequestFile]:
        """读取 PR 的文件变更列表（含 unified diff 片段 ``patch``）。"""
        identifier = GitHubRepoRef.parse(repo)
        cap = self.max_pr_files if limit is None else limit
        if cap <= 0:
            raise ValueError("limit 必须为正数")
        payload = self._get_json(
            f"/repos/{identifier.owner}/{identifier.name}/pulls/{int(number)}/files",
            params={"per_page": min(cap, 100)},
        )
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise GitHubResponseError("PR 文件响应不是数组", detail=str(payload)[:200])
        files = [
            GitHubPullRequestFile.from_payload(item) for item in payload if isinstance(item, Mapping)
        ]
        return files[:cap]

    def get_pull_request_diff(
        self, repo: str | GitHubRepoRef, number: int, *, max_chars: int | None = None
    ) -> str:
        """读取 PR 的完整 unified diff（``Accept: ...v3.diff``，返回纯文本）。"""
        identifier = GitHubRepoRef.parse(repo)
        text = self._get_text(
            f"/repos/{identifier.owner}/{identifier.name}/pulls/{int(number)}",
            accept=DIFF_ACCEPT,
        )
        limit = self.max_file_chars if max_chars is None else max_chars
        if len(text) > limit:
            self.stats.truncated_responses += 1
            return f"{text[:limit]}\n...（diff 已截断，共 {len(text)} 字符）"
        return text

    # -------------------------------------------------------- HTTP
    def _headers(self, *, accept: str = DEFAULT_ACCEPT) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": self.user_agent,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _send(self, path: str, *, params: Mapping[str, Any] | None, accept: str) -> httpx.Response:
        url = f"{self.base_url}{path}"
        clean = {key: value for key, value in (params or {}).items() if value is not None}
        started = time.perf_counter()
        self.stats.requests += 1
        try:
            response = self._client.request(
                "GET", url, params=clean or None, headers=self._headers(accept=accept)
            )
        except httpx.HTTPError as exc:
            self.stats.errors += 1
            raise GitHubError(f"GitHub 请求失败：{path}", detail=f"{type(exc).__name__}: {exc}") from exc
        finally:
            self.stats.duration += time.perf_counter() - started

        remaining = response.headers.get("x-ratelimit-remaining")
        if remaining is not None:
            self.stats.rate_limit_remaining = _as_int(remaining)
        if response.status_code >= 400:
            self.stats.errors += 1
            self._raise_for_status(response, path)
        return response

    def _get_json(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        response = self._send(path, params=params, accept=DEFAULT_ACCEPT)
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubResponseError(
                "GitHub 响应不是合法 JSON", detail=f"{path} -> {response.text[:200]}"
            ) from exc

    def _get_text(self, path: str, *, params: Mapping[str, Any] | None = None, accept: str) -> str:
        response = self._send(path, params=params, accept=accept)
        return response.text

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        code = response.status_code
        message = _error_message(response)
        detail = f"GET {path} -> HTTP {code}：{message[:300]}"

        if code == 401:
            raise GitHubAuthError("GitHub 令牌无效或缺失", detail=detail)
        if code == 429:
            raise GitHubRateLimitError("GitHub 触发速率限制（429）", detail=detail)
        if code == 403:
            if self.stats.rate_limit_remaining == 0 or "rate limit" in message.lower():
                raise GitHubRateLimitError("GitHub 触发速率限制", detail=detail)
            raise GitHubAuthError("GitHub 拒绝访问（403）：令牌权限不足", detail=detail)
        if code == 404:
            raise GitHubNotFoundError("GitHub 资源不存在（404）", detail=detail)
        raise GitHubResponseError("GitHub 返回异常状态", detail=detail)

    @staticmethod
    def _as_object(payload: Any, what: str) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise GitHubResponseError(f"{what}响应不是 JSON 对象", detail=str(payload)[:200])
        return payload


# ------------------------------------------------------------------ 辅助
def _decode_content(raw: str, encoding: str, path: str) -> str:
    """解码 Contents API 的 ``content`` 字段（GitHub 默认 base64 + 换行分段）。"""
    if not raw:
        return ""
    if encoding and encoding != "base64":  # pragma: no cover - 目前只有 base64
        raise GitHubResponseError(f"不支持的 content 编码：{encoding}", detail=f"path={path}")
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise GitHubResponseError("文件内容不是合法 base64", detail=f"path={path}：{exc}") from exc
    return data.decode("utf-8", errors="replace")


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip() or "（无错误详情）"
    if isinstance(payload, Mapping):
        return str(payload.get("message") or payload)
    return str(payload)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
