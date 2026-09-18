"""GitHub 只读客户端单元测试（离线 fixture 回放）。

本机无法直连 ``api.github.com``（SSL 证书校验与吊销检查均失败），因此这里用
``httpx.MockTransport`` 把 ``tests/fixtures/github`` 下**按官方文档字段构造**的
响应回放给客户端——走的仍是真实的 ``httpx`` 请求/响应对象与真实 URL 拼装逻辑，
所以"离线"只影响数据来源，不影响被测代码路径。

覆盖：
1. **仓库标识与路径安全**：``GitHubRepoRef.parse`` 的各种写法、``normalize_repo_path`` 的越界拒绝；
2. **读取能力**：仓库元信息、仓库树（前缀过滤/截断/限制）、文件（base64 解码/截断）、目录、PR 列表/详情/diff；
3. **错误映射**：401/403（限流与权限两种）/404/429/5xx/非 JSON/连接异常，以及统计计数；
4. **请求契约**：只发 GET、令牌进 Authorization 头（无令牌则不带）、Accept 与 API 版本头。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from codeagentx.core.exceptions import (
    GitHubAuthError,
    GitHubError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubResponseError,
    SecurityViolationError,
)
from codeagentx.protocols.github_client import (
    DEFAULT_ACCEPT,
    DEFAULT_API_BASE,
    DIFF_ACCEPT,
    GitHubClient,
    GitHubRepoRef,
    GitHubTree,
    GitHubTreeEntry,
    normalize_repo_path,
)
from codeagentx.protocols.mcp_github import (
    GitHubMCPServer,
    github_server_command,
)
from codeagentx.protocols.mcp_protocol import JSONRPC_VERSION, MCPErrorCode

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "github"
REPO = "acme/payments-api"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> Any:
    return json.loads(fixture(name))


# ------------------------------------------------------------------ 离线站点
class FakeGitHub:
    """按路径路由的离线 GitHub：记录每个请求，回放注册好的响应。

    未注册的路径一律按官方 404 响应体返回——这样"客户端请求了不该请求的地址"
    会以 GitHubNotFoundError 暴露出来，而不是被悄悄放过。
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, tuple[int, str, dict[str, str]]] = {}

    def add(
        self,
        path: str,
        body: str,
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
    ) -> FakeGitHub:
        self.routes[path] = (status, body, dict(headers or {}))
        return self

    def add_fixture(self, path: str, name: str, **kwargs: Any) -> FakeGitHub:
        return self.add(path, fixture(name), **kwargs)

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        accept = request.headers.get("accept") or ""
        if "v3.diff" in accept:
            return httpx.Response(
                200, text=fixture("pull_101.diff"), headers={"content-type": "text/plain"}
            )
        entry = self.routes.get(request.url.path)
        if entry is None:
            return httpx.Response(
                404, text=fixture("not_found.json"), headers={"content-type": "application/json"}
            )
        status, body, headers = entry
        return httpx.Response(
            status, text=body, headers={"content-type": "application/json", **headers}
        )


def standard_site() -> FakeGitHub:
    """把 fixture 目录里的正常响应全部挂上。"""
    site = FakeGitHub()
    site.add_fixture(f"/repos/{REPO}", "repo.json")
    site.add_fixture(f"/repos/{REPO}/git/trees/main", "tree.json")
    site.add_fixture(f"/repos/{REPO}/contents/README.md", "contents_readme.json")
    site.add_fixture(f"/repos/{REPO}/contents/src/payments/auth.py", "contents_auth_py.json")
    site.add_fixture(f"/repos/{REPO}/contents/src/payments", "contents_dir_src_payments.json")
    site.add_fixture(f"/repos/{REPO}/pulls", "pulls.json")
    site.add_fixture(f"/repos/{REPO}/pulls/101", "pull_101.json")
    site.add_fixture(f"/repos/{REPO}/pulls/101/files", "pull_101_files.json")
    return site


def make_client(site: FakeGitHub, *, token: str = "", **kwargs: Any) -> GitHubClient:
    return GitHubClient(
        token=token, client=httpx.Client(transport=httpx.MockTransport(site)), **kwargs
    )


def site_with(site: FakeGitHub, **kwargs: Any) -> GitHubClient:
    return make_client(site, **kwargs)


# ================================================================== 1. 仓库标识
def test_repo_ref_parses_common_forms() -> None:
    ref = GitHubRepoRef.parse("acme/payments-api")
    assert (ref.owner, ref.name, ref.ref, ref.slug) == ("acme", "payments-api", "", REPO)

    assert GitHubRepoRef.parse("acme/payments-api@v1.2").ref == "v1.2"
    assert GitHubRepoRef.parse("git@github.com:acme/payments-api.git").slug == REPO
    assert GitHubRepoRef.parse("https://github.com/acme/payments-api").slug == REPO
    assert GitHubRepoRef.parse(f"https://api.github.com/repos/{REPO}").slug == REPO

    # 带 tree/blob 的网页链接要把后面的路径当 ref
    assert GitHubRepoRef.parse(f"https://github.com/{REPO}/tree/feature/x").ref == "feature/x"

    # 已经是对象时原样返回（不重复解析）
    original = GitHubRepoRef(owner="a", name="b")
    assert GitHubRepoRef.parse(original) is original

    assert GitHubRepoRef.parse(REPO).to_dict() == {
        "owner": "acme",
        "name": "payments-api",
        "ref": "",
        "slug": REPO,
    }


def test_repo_ref_rejects_unparsable_values() -> None:
    for bad in ("", "   ", "just-a-name", "/"):
        with pytest.raises(ValueError):
            GitHubRepoRef.parse(bad)


def test_normalize_repo_path_flattens_and_rejects_escapes() -> None:
    assert normalize_repo_path("") == ""
    assert normalize_repo_path(".") == ""
    assert normalize_repo_path(None) == ""
    assert normalize_repo_path("src\\payments\\auth.py") == "src/payments/auth.py"
    assert normalize_repo_path("a/./b") == "a/b"

    for bad in ("/etc/passwd", "~/.ssh/id_rsa", "../secret", "src/../../secret", "C:/Windows"):
        with pytest.raises(SecurityViolationError):
            normalize_repo_path(bad)


# ================================================================== 2. 读取
def test_get_repository_reads_metadata_and_caches() -> None:
    site = standard_site()
    client = make_client(site)

    repo = client.get_repository(REPO)
    assert repo.slug == REPO
    assert repo.default_branch == "main"
    assert repo.language == "Python"
    assert repo.stars == 128
    assert repo.forks == 17
    assert repo.open_issues == 6
    assert repo.private is False
    assert repo.archived is False
    assert "evaluation fixture" in repo.description
    assert repo.pushed_at.startswith("2024-11-02")
    assert REPO in repo.to_text() and "默认分支：main" in repo.to_text()

    # 第二次走缓存：不再产生请求
    client.get_repository(REPO)
    assert client.stats.requests == 1
    assert site.paths() == [f"/repos/{REPO}"]


def test_resolve_ref_prefers_explicit_ref_over_default_branch() -> None:
    site = standard_site()
    client = make_client(site)

    assert client.resolve_ref(REPO) == "main"
    assert client.stats.requests == 1

    # 显式给了 ref 就该直接用，不必再问默认分支
    assert client.resolve_ref(f"{REPO}@release/2.x") == "release/2.x"
    assert client.stats.requests == 1


def test_list_tree_reports_counts_and_lookup() -> None:
    site = standard_site()
    client = make_client(site)

    tree = client.list_tree(REPO)
    assert tree.repo == REPO
    assert tree.ref == "main"
    assert tree.sha == "3f1e2c9b7a4d5e6f8a0b1c2d3e4f5a6b7c8d9e0f"
    assert tree.truncated is False
    assert len(tree.entries) == 12
    assert len(tree.files) == 8
    assert len(tree.directories) == 4

    entry = tree.find("src/payments/auth.py")
    assert entry is not None
    assert entry.is_file and not entry.is_dir
    assert entry.name == "auth.py"
    assert entry.suffix == ".py"
    assert entry.depth == 3
    assert entry.size == 202
    assert tree.find("src") is not None and tree.find("src").is_dir
    assert tree.find("不存在.py") is None

    text = tree.to_text()
    assert "共 12 条" in text
    assert "file" in text and "dir " in text
    assert tree.to_dict(with_entries=False)["files"] == 8

    assert site.paths() == [f"/repos/{REPO}", f"/repos/{REPO}/git/trees/main"]


def test_list_tree_filters_by_path_prefix_and_marks_truncation() -> None:
    site = standard_site()
    client = make_client(site)

    scoped = client.list_tree(REPO, path_prefix="src/payments")
    # 前缀语义是"含该目录自身"，所以 1 个目录 + 4 个文件
    assert [entry.path for entry in scoped.entries] == [
        "src/payments",
        "src/payments/__init__.py",
        "src/payments/api.py",
        "src/payments/auth.py",
        "src/payments/db.py",
    ]
    assert scoped.truncated is False

    capped = client.list_tree(REPO, max_entries=3)
    assert len(capped.entries) == 3
    assert capped.truncated is True  # 本地截断也要如实标注
    assert client.stats.truncated_responses == 1
    assert "已截断" in capped.to_text()

    with pytest.raises(SecurityViolationError):
        client.list_tree(REPO, path_prefix="../secrets")


def test_list_tree_reports_github_side_truncation() -> None:
    site = standard_site()
    payload = fixture_json("tree.json")
    payload["truncated"] = True
    site.add(f"/repos/{REPO}/git/trees/main", json.dumps(payload))
    client = make_client(site)

    tree = client.list_tree(REPO)
    assert tree.truncated is True
    assert "GitHub 侧已截断" in tree.to_text()
    assert client.stats.truncated_responses == 1


def test_read_file_decodes_base64_and_truncates() -> None:
    site = standard_site()
    client = make_client(site)

    readme = client.read_file(REPO, "README.md")
    assert readme.path == "README.md"
    assert "payments-api" in readme.text
    assert readme.sha == "8e4b1c0d9f2a3b4c5d6e7f8091a2b3c4d5e6f708"
    assert readme.size == 199
    assert readme.truncated is False
    assert readme.line_count > 1
    assert "Layout" in readme.text

    auth = client.read_file(REPO, "src\\payments\\auth.py")  # 反斜杠会被规范化
    assert auth.path == "src/payments/auth.py"
    assert "compare_digest" in auth.text
    assert client.stats.requests == 2

    clipped = client.read_file(REPO, "README.md", max_chars=20)
    assert clipped.truncated is True
    assert "已截断" in clipped.text
    assert clipped.line_count <= 3
    assert client.stats.truncated_responses == 1

    # 截断的是文本，但 size 仍是 GitHub 报的原始字节数
    assert clipped.size == readme.size


def test_read_file_rejects_escape_and_directory() -> None:
    site = standard_site()
    client = make_client(site)

    with pytest.raises(SecurityViolationError):
        client.read_file(REPO, "../.env")
    assert client.stats.requests == 0  # 越界在发请求前就被拦下

    # 该路径在 GitHub 上是目录（contents 接口返回数组）
    with pytest.raises(GitHubNotFoundError, match="目录而不是文件"):
        client.read_file(REPO, "src/payments")


def test_read_file_reports_uninlinable_and_undecodable_content() -> None:
    site = standard_site()
    # Contents API 对超过 1MB 的文件返回空 content（size 仍在）
    site.add(
        f"/repos/{REPO}/contents/big.bin",
        json.dumps({"name": "big.bin", "size": 2_000_000, "encoding": "base64", "content": ""}),
    )
    site.add(
        f"/repos/{REPO}/contents/broken.bin",
        json.dumps({"name": "broken.bin", "size": 12, "encoding": "base64", "content": "AAAAA"}),
    )
    site.add(
        f"/repos/{REPO}/contents/weird.txt",
        json.dumps({"name": "weird.txt", "size": 3, "encoding": "utf-8", "content": "abc"}),
    )
    client = make_client(site)

    # Contents API 对超过 1MB 的文件返回空 content，此时必须明确报错而不是返回空字符串
    with pytest.raises(GitHubResponseError, match="过大"):
        client.read_file(REPO, "big.bin")

    with pytest.raises(GitHubResponseError, match="base64"):
        client.read_file(REPO, "broken.bin")

    with pytest.raises(GitHubResponseError, match="不支持的 content 编码"):
        client.read_file(REPO, "weird.txt")


def test_list_directory_builds_paths_and_rejects_files() -> None:
    site = standard_site()
    client = make_client(site)

    entries = client.list_directory(REPO, "src/payments")
    assert [entry.path for entry in entries] == [
        "src/payments/__init__.py",
        "src/payments/api.py",
        "src/payments/auth.py",
        "src/payments/db.py",
    ]
    assert all(entry.is_file for entry in entries)
    assert entries[2].size == 202

    with pytest.raises(GitHubNotFoundError, match="文件而不是目录"):
        client.list_directory(REPO, "README.md")


# ================================================================== 3. PR
def test_list_and_get_pull_requests() -> None:
    site = standard_site()
    client = make_client(site)

    pulls = client.list_pull_requests(REPO)
    assert [pr.number for pr in pulls] == [101, 102]

    first = pulls[0]
    assert first.title.startswith("fix(auth)")
    assert first.state == "open"
    assert first.draft is False
    assert first.author == "alice"
    assert (first.base_ref, first.head_ref) == ("main", "fix/auth-timing")
    assert (first.additions, first.deletions, first.changed_files, first.commits) == (41, 12, 2, 3)
    assert first.updated_at.startswith("2024-11-01")
    assert "PR #101" in first.to_text()
    assert pulls[1].draft is True

    assert client.list_pull_requests(REPO, limit=1) == [first]

    detail = client.get_pull_request(REPO, 101)
    assert detail.number == 101 and detail.author == "alice"

    assert site.paths()[0] == f"/repos/{REPO}/pulls"
    assert site.requests[0].url.params["state"] == "open"
    assert site.requests[0].url.params["per_page"] == "20"


def test_pull_request_argument_validation() -> None:
    client = make_client(standard_site())
    with pytest.raises(ValueError, match="state"):
        client.list_pull_requests(REPO, state="merged")
    with pytest.raises(ValueError, match="limit"):
        client.list_pull_requests(REPO, limit=0)
    with pytest.raises(ValueError, match="limit"):
        client.get_pull_request_files(REPO, 101, limit=0)


def test_pull_request_files_include_patches() -> None:
    site = standard_site()
    client = make_client(site)

    files = client.get_pull_request_files(REPO, 101)
    assert [item.filename for item in files] == ["src/payments/auth.py", "tests/test_auth.py"]
    assert files[0].status == "modified"
    assert files[0].changes == 7
    assert "compare_digest" in files[0].patch
    assert files[1].status == "added"
    assert "### tests/test_auth.py（added" in files[1].to_text()

    assert len(client.get_pull_request_files(REPO, 101, limit=1)) == 1


def test_pull_request_patch_may_be_absent() -> None:
    site = standard_site()
    site.add(
        f"/repos/{REPO}/pulls/102/files",
        json.dumps([{"filename": "assets/logo.png", "status": "modified", "additions": 0}]),
    )
    client = make_client(site)

    files = client.get_pull_request_files(REPO, 102)
    assert files[0].patch == ""
    assert "GitHub 未返回 patch" in files[0].to_text()
    assert files[0].to_dict(with_patch=False) == {
        "filename": "assets/logo.png",
        "status": "modified",
        "additions": 0,
        "deletions": 0,
        "changes": 0,
    }


def test_pull_request_diff_uses_diff_accept_header() -> None:
    site = standard_site()
    client = make_client(site)

    diff = client.get_pull_request_diff(REPO, 101)
    assert diff.startswith("diff --git a/src/payments/auth.py")
    assert "compare_digest" in diff
    assert site.requests[0].headers["accept"] == DIFF_ACCEPT

    clipped = client.get_pull_request_diff(REPO, 101, max_chars=40)
    assert "diff 已截断" in clipped
    assert client.stats.truncated_responses == 1


# ================================================================== 4. 错误映射
def _error_client(
    body: Any, *, status: int, headers: Mapping[str, str] | None = None
) -> GitHubClient:
    """只挂一个"报错"路由的客户端（用于验证状态码 → 异常类型的映射）。"""
    text = body if isinstance(body, str) else json.dumps(body)
    return make_client(FakeGitHub().add(f"/repos/{REPO}", text, status=status, headers=headers))


def test_error_status_mapping() -> None:
    not_found = _error_client(fixture_json("not_found.json"), status=404)
    with pytest.raises(GitHubNotFoundError, match="404") as excinfo:
        not_found.get_repository(REPO)
    assert "Not Found" in (excinfo.value.detail or "")
    assert not_found.stats.errors == 1

    unauthorized = _error_client(fixture_json("unauthorized.json"), status=401)
    with pytest.raises(GitHubAuthError, match="令牌无效"):
        unauthorized.get_repository(REPO)

    limited = _error_client(
        fixture_json("rate_limited.json"), status=403, headers={"x-ratelimit-remaining": "0"}
    )
    with pytest.raises(GitHubRateLimitError):
        limited.get_repository(REPO)
    assert limited.stats.rate_limit_remaining == 0

    # 403 但没有限流标记，且报错文本提到 rate limit：仍按限流处理
    limited_by_text = _error_client(fixture_json("rate_limited.json"), status=403)
    with pytest.raises(GitHubRateLimitError):
        limited_by_text.get_repository(REPO)

    # 403 且是权限问题：按鉴权失败处理（与限流区分开）
    forbidden = _error_client(
        {"message": "Resource not accessible by personal access token"},
        status=403,
        headers={"x-ratelimit-remaining": "42"},
    )
    with pytest.raises(GitHubAuthError, match="权限不足"):
        forbidden.get_repository(REPO)
    assert forbidden.stats.rate_limit_remaining == 42

    too_many = _error_client({"message": "slow down"}, status=429)
    with pytest.raises(GitHubRateLimitError, match="429"):
        too_many.get_repository(REPO)

    server_error = _error_client({"message": "boom"}, status=500)
    with pytest.raises(GitHubResponseError, match="异常状态"):
        server_error.get_repository(REPO)


def test_non_json_and_transport_failures_are_reported() -> None:
    not_json = FakeGitHub().add(f"/repos/{REPO}", "<html>502 Bad Gateway</html>")
    with pytest.raises(GitHubResponseError, match="不是合法 JSON"):
        make_client(not_json).get_repository(REPO)

    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("网络不可达", request=request)

    broken = GitHubClient(client=httpx.Client(transport=httpx.MockTransport(explode)))
    with pytest.raises(GitHubError, match="请求失败") as excinfo:
        broken.get_repository(REPO)
    assert "ConnectError" in (excinfo.value.detail or "")
    assert broken.stats.errors == 1
    assert broken.stats.requests == 1


def test_unknown_endpoint_hits_offline_404() -> None:
    client = make_client(standard_site())
    with pytest.raises(GitHubNotFoundError):
        client.get_repository("acme/ghost")


def test_response_shape_violations_are_reported() -> None:
    missing_name = FakeGitHub().add(f"/repos/{REPO}", json.dumps({"full_name": REPO}))
    with pytest.raises(GitHubResponseError, match="缺少 name"):
        make_client(missing_name).get_repository(REPO)

    site = standard_site()
    site.add(f"/repos/{REPO}/git/trees/main", json.dumps({"tree": "not-an-array", "sha": "x"}))
    with pytest.raises(GitHubResponseError, match="不是数组"):
        make_client(site).list_tree(REPO)

    with pytest.raises(GitHubResponseError):
        GitHubTreeEntry.from_payload({"type": "blob"})  # 缺 path


def test_tree_to_text_marks_truncated_entries() -> None:
    tree = GitHubTree(
        repo=REPO,
        ref="main",
        truncated=True,
        entries=[GitHubTreeEntry(path="a.py", size=3), GitHubTreeEntry(path="src", type="tree")],
    )
    text = tree.to_text()
    assert "GitHub 侧已截断" in text
    assert "file " in text and "dir " in text


# ================================================================== 5. 请求契约
def test_requests_are_get_only_and_carry_expected_headers() -> None:
    site = standard_site()
    anonymous = make_client(site)
    anonymous.read_file(REPO, "README.md")

    request = site.requests[0]
    assert request.method == "GET"
    assert str(request.url).startswith(f"{DEFAULT_API_BASE}/repos/{REPO}/contents/README.md")
    assert request.headers["accept"] == DEFAULT_ACCEPT
    assert request.headers["x-github-api-version"] == "2022-11-28"
    assert request.headers["user-agent"].startswith("CodeAgentX/")
    assert "authorization" not in request.headers  # 无令牌就不该带鉴权头
    assert anonymous.is_authenticated is False

    authed_site = standard_site()
    authed = make_client(authed_site, token="ghp_secret")
    authed.read_file(REPO, "README.md")
    assert authed_site.requests[0].headers["authorization"] == "Bearer ghp_secret"
    assert authed.is_authenticated is True
    assert authed.describe()["authenticated"] is True


def test_ref_is_passed_as_query_parameter() -> None:
    site = standard_site()
    client = make_client(site)
    client.read_file(f"{REPO}@release/2.x", "README.md")
    assert site.requests[0].url.params["ref"] == "release/2.x"


def test_constructor_validation_and_owned_client() -> None:
    with pytest.raises(ValueError, match="timeout"):
        GitHubClient(timeout=0)
    with pytest.raises(ValueError, match="容量上限"):
        GitHubClient(max_entries=0)

    http = httpx.Client(transport=httpx.MockTransport(FakeGitHub()))
    client = GitHubClient(client=http)
    assert client.describe()["owns_client"] is False
    client.close()
    assert http.is_closed is False  # 注入的 client 由调用方负责关闭


def test_client_closes_its_own_http_client() -> None:
    client = GitHubClient(token="")  # 未注入 -> 自己创建一个
    inner = client._client
    assert client.describe()["owns_client"] is True
    client.close()
    assert inner.is_closed is True


def test_stats_snapshot() -> None:
    site = standard_site()
    client = make_client(site)
    client.read_file(REPO, "README.md")
    client.read_file(REPO, "src/payments/auth.py")

    payload = client.stats.as_dict()
    assert payload["requests"] == 2
    assert payload["errors"] == 0
    assert payload["truncated_responses"] == 0
    assert payload["rate_limit_remaining"] is None
    assert isinstance(payload["duration"], float)


# ================================================================== 6. MCP 服务端
# W7 交付物「MCP GitHub：读仓库/PR」：把上面的只读客户端包成一个 MCP 服务端。
# 这里全部走 MCPServerBase.handle（即真实报文分发路径），而不是直接调方法——
# 要验证的正是"协议层能不能把它正确暴露出去"。
def mcp_server(site: FakeGitHub, **kwargs: Any) -> GitHubMCPServer:
    return GitHubMCPServer(client=make_client(site), **kwargs)


def rpc(server: GitHubMCPServer, method: str, params: Any = None, *, request_id: int = 1):
    payload: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    response = server.handle(payload)
    assert response is not None
    return response


def call_tool(server: GitHubMCPServer, name: str, arguments: dict[str, Any]):
    """调用一个工具，返回（文本，结构化内容，是否 isError）。

    协议级错误（工具名/arguments 结构非法）会以 ``error`` 返回——这里直接断言它不该发生。
    """
    response = rpc(server, "tools/call", {"name": name, "arguments": arguments})
    assert "error" not in response, response["error"]
    result = response["result"]
    text = "\n".join(block.get("text", "") for block in result["content"])
    return text, result.get("structuredContent") or {}, bool(result.get("isError"))


def tool_failure(server: GitHubMCPServer, name: str, arguments: dict[str, Any]) -> str:
    """调用一个应当被拒的工具，返回失败说明。

    口径与 filesystem 服务端一致（见 tests/test_mcp.py）：
    **包内**的参数校验失败、安全策略拒绝都属于"工具这次没跑成"（``isError``），
    只有**报文层**的问题（未知工具名、arguments 不是对象）才是协议级 ``error``。
    """
    response = rpc(server, "tools/call", {"name": name, "arguments": arguments})
    assert "error" not in response, "工具级失败不该以协议 error 返回"
    assert response["result"]["isError"] is True, response["result"]
    return response["result"]["content"][0]["text"]


def test_mcp_github_server_info_and_tool_specs() -> None:
    server = mcp_server(standard_site())
    info = server.server_info_payload()

    assert info["serverInfo"] == {"name": "codeagentx-github", "version": "0.1.0"}
    assert "只读" in info["instructions"]

    tools = {tool.name: tool for tool in server.list_tools()}
    assert set(tools) == {
        "get_repository",
        "list_tree",
        "read_file",
        "list_pull_requests",
        "get_pull_request",
        "get_pull_request_files",
        "get_pull_request_diff",
    }
    assert all(tool.description for tool in tools.values())
    assert all(tool.input_schema.get("required") for tool in tools.values())


def test_mcp_github_reads_repository_and_tree() -> None:
    server = mcp_server(standard_site())

    text, structured, is_error = call_tool(server, "get_repository", {"repo": REPO})
    assert is_error is False
    assert "默认分支" in text and "acme/payments-api" in text
    assert structured["slug"] == REPO and structured["default_branch"] == "main"

    text, structured, _ = call_tool(server, "list_tree", {"repo": REPO, "path_prefix": "src/payments"})
    assert "src/payments/auth.py" in text
    assert "README.md" not in text  # 前缀之外的文件不该出现
    assert structured["files"] == 4 and structured["directories"] == 1
    assert structured["truncated"] is False


def test_mcp_github_reads_file_and_reports_missing_one_as_tool_error() -> None:
    server = mcp_server(standard_site())

    text, structured, is_error = call_tool(
        server, "read_file", {"repo": REPO, "path": "src/payments/auth.py"}
    )
    assert is_error is False and "hmac.compare_digest" in text
    assert structured["path"] == "src/payments/auth.py"
    assert structured["lines"] > 0
    assert structured["ref"] == ""  # 没显式给 ref 时不假装知道用了哪个分支

    # 未注册的路径 → 404 → 属于"工具这次没跑成"（isError），而不是协议错误
    text, _, is_error = call_tool(server, "read_file", {"repo": REPO, "path": "src/payments/db.py"})
    assert is_error is True and "GitHubNotFoundError" in text


def test_mcp_github_reads_pull_requests() -> None:
    server = mcp_server(standard_site())

    text, structured, is_error = call_tool(server, "list_pull_requests", {"repo": REPO})
    assert is_error is False and "101" in text and "compare_digest" in text
    assert structured["count"] == 2
    assert [item["number"] for item in structured["pull_requests"]] == [101, 102]

    text, structured, _ = call_tool(server, "get_pull_request", {"repo": REPO, "number": 101})
    assert "fix(auth)" in text and structured["number"] == 101

    text, structured, _ = call_tool(
        server, "get_pull_request_files", {"repo": REPO, "number": 101}
    )
    assert structured["count"] == 2
    assert "tests/test_auth.py" in text and "@@" in text

    diff, _, _ = call_tool(server, "get_pull_request_diff", {"repo": REPO, "number": 101})
    assert diff.startswith("diff --git") and "hmac.compare_digest" in diff


def test_mcp_github_truncates_file_content_by_max_chars() -> None:
    server = mcp_server(standard_site())

    text, structured, _ = call_tool(
        server, "read_file", {"repo": REPO, "path": "src/payments/auth.py", "max_chars": 20}
    )
    assert structured["truncated"] is True
    assert "已截断" in text and len(text) < 202


def test_mcp_github_enforces_repo_allowlist() -> None:
    blocked = mcp_server(standard_site(), allowed_repos=("other/repo",))
    text = tool_failure(blocked, "get_repository", {"repo": REPO})
    assert "不在允许列表" in text

    allowed = mcp_server(standard_site(), allowed_repos=(REPO,))
    _, structured, is_error = call_tool(allowed, "get_repository", {"repo": REPO})
    assert is_error is False and structured["slug"] == REPO


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("read_file", {"repo": REPO}, "缺少必填参数：path"),
        ("get_repository", {}, "缺少必填参数：repo"),
        ("get_repository", {"repo": "just-a-name"}, "仓库标识非法"),
        ("get_pull_request", {"repo": REPO}, "缺少必填参数：number"),
        ("list_tree", {"repo": REPO, "recursive": "maybe"}, "布尔值"),
        ("read_file", {"repo": REPO, "path": "a.py", "max_chars": "abc"}, "整数"),
    ],
)
def test_mcp_github_rejects_bad_arguments(name: str, arguments: dict, expected: str) -> None:
    server = mcp_server(standard_site())

    assert expected in tool_failure(server, name, arguments)


def test_mcp_github_rejects_unknown_tool() -> None:
    server = mcp_server(standard_site())
    response = rpc(server, "tools/call", {"name": "delete_repo", "arguments": {}})

    assert response["error"]["code"] == int(MCPErrorCode.INVALID_PARAMS)
    assert "未知工具" in response["error"]["message"]


def test_github_server_command_keeps_token_off_the_command_line() -> None:
    executable, args = github_server_command(
        repos=[REPO, "other/repo"], python="py", base_url="https://ghe.example.com"
    )

    assert executable == "py"
    assert args == [
        "-m",
        "codeagentx.protocols.mcp_github",
        "--repo",
        REPO,
        "--repo",
        "other/repo",
        "--base-url",
        "https://ghe.example.com",
    ]
    # 令牌必须走环境变量：命令行会出现在进程列表里
    assert not any("token" in item.lower() for item in args)
