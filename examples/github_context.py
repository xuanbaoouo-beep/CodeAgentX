"""W7 端到端示例：自动读取 GitHub 仓库 → 生成上下文（GSSC 流水线）。

用法::

    # 离线演示站点：真实的 GitHubClient 协议代码全跑，只把 HTTP 层换成内存响应
    python examples/github_context.py
    # 只读某个目录，并演示「最多取几个文件」
    python examples/github_context.py --prefix src/payments --max-files 4
    # 明确点名要读的文件（给了 --paths 就不再拉整棵仓库树）
    python examples/github_context.py --paths README.md src/payments/auth.py
    # 收紧预算，观察分级压缩怎么把上下文压回预算内
    python examples/github_context.py --budget 600
    # 连真实 GitHub（只读；建议配 GITHUB_TOKEN，否则很快限流）
    python examples/github_context.py --live acme/payments-api --ref main
    # 把生成好的上下文落盘 / 直接打印
    python examples/github_context.py --out .context.md --show-text

验收标准（脚本据此决定退出码）

1. **证据非空**：从仓库里至少取回一个文件，并作为"证据"节进入上下文；
2. **不超预算**：``BuiltContext.within_budget`` 为真，且渲染后的 token ≤ ``--budget``；
3. **账目可查**：打印 GSSC 四个阶段与压缩各分级的前后 token 变化。

诚实的边界
----------
离线模式的 HTTP 响应是**按官方文档字段手工构造**的（不是抓包录制），
文件内容也只是为演示准备的样例代码。协议行为本身——URL 拼装、请求头、
base64 解码、超长截断、错误状态映射——都由
``src/codeagentx/protocols/github_client.py`` 真实执行；同款断言在
``tests/test_github.py`` 里用 fixture 跑过。
要真读互联网上的仓库，去掉 ``--offline`` 之外的 ``--live`` 即可（只读，不改远端）。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from codeagentx.context import (  # noqa: E402
    SECTION_EVIDENCE,
    SECTION_TASK,
    ContextBuilder,
    GitHubSource,
)
from codeagentx.context.builder import DEFAULT_MAX_GITHUB_FILES, in_text_suffixes  # noqa: E402
from codeagentx.core.exceptions import CodeAgentXError  # noqa: E402
from codeagentx.core.logger import setup_logging  # noqa: E402
from codeagentx.protocols.github_client import (  # noqa: E402
    DEFAULT_API_BASE,
    GitHubClient,
    GitHubRepoRef,
)

DEMO_REPO = "acme/payments-api"
DEMO_BRANCH = "main"
DEFAULT_TASK = "审查这个支付服务的鉴权与数据访问实现，找出逻辑漏洞与安全风险"
DEFAULT_BUDGET = 1000

OFFLINE_BANNER = (
    "[离线] HTTP 层被替换为内存响应（按官方文档字段构造）：GitHubClient 的协议代码、"
    "路径校验、base64 解码、GSSC 流水线都是真实执行的。\n"
    "       要读真实仓库，加 --live owner/repo（只读，需要网络与 GITHUB_TOKEN）。"
)

# ---------------------------------------------------------------- 演示仓库
#: 离线站点提供的文件（path → 正文）。内容刻意留了几个典型问题，
#: 便于观察"证据是怎么进上下文"的；这不是真实项目代码。
MINI_REPO: dict[str, str] = {
    "README.md": """\
# payments-api

支付服务（W7 离线演示仓库，内容为样例，非真实项目）。

## 约定
- 所有对外入口必须校验入参，金额一律用整数分表示，禁止浮点；
- 生产环境禁止开启 DEBUG 路由；
- 生产凭据只能来自环境变量或密钥服务。
""",
    "pyproject.toml": """\
[project]
name = "payments-api"
version = "0.3.1"
requires-python = ">=3.10"
dependencies = ["fastapi>=0.110", "pydantic>=2"]

[tool.ruff]
line-length = 100
""",
    "docs/architecture.md": """\
# 架构说明

```
routes（HTTP 层）
  → service（业务逻辑）
    → repository（数据访问）
      → PostgreSQL
```

## 分层约定
1. 路由层只做参数解析与错误包装，不写业务判断；
2. 数据访问层必须使用参数化查询，禁止拼接 SQL；
3. 鉴权失败一律返回 401，且不得回显内部异常信息。
""",
    "src/payments/__init__.py": '''\
"""支付核心包。"""

from payments.api import create_router

__all__ = ["create_router"]
''',
    "src/payments/api.py": '''\
"""HTTP 层：路由与错误包装。"""

import logging

from payments.auth import require_token
from payments.db import find_user

logger = logging.getLogger(__name__)


def create_router(app):
    """注册路由。"""

    @app.post("/login")
    def handle_login(payload: dict):
        username = payload.get("username", "")
        password = payload.get("password", "")
        user = find_user(username)
        if user is None or not require_token(user, password):
            return {"ok": False, "error": "用户名或密码错误"}
        return {"ok": True, "token": user["token"]}

    @app.get("/debug/login")
    def handle_debug_login(username: str):
        # DEBUG 路由没有鉴权，且把内部异常原样回显
        try:
            return {"ok": True, "user": find_user(username)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    return app
''',
    "src/payments/auth.py": '''\
"""鉴权与口令处理。"""

import hashlib

# 硬编码密钥（生产环境应从密钥服务读取）
SECRET_KEY = "payments-demo-secret-key"


def require_token(user: dict, password: str) -> bool:
    """校验口令是否匹配。"""
    return hash_password(password) == user.get("password_hash")


def hash_password(password: str) -> str:
    """口令散列。"""
    return hashlib.md5((password + SECRET_KEY).encode()).hexdigest()


def logout(session_token: str) -> bool:
    """登出。"""
    return True
''',
    "src/payments/db.py": '''\
"""数据访问层。"""

import sqlite3

CONNECTION = sqlite3.connect("payments.db")


def find_user(username: str) -> dict | None:
    """按用户名查询用户。"""
    cursor = CONNECTION.cursor()
    # 字符串拼接构造 SQL：username 直接来自请求参数
    cursor.execute("SELECT * FROM users WHERE username = '" + username + "'")
    row = cursor.fetchone()
    if row is None:
        return None
    return {"username": row[0], "password_hash": row[1], "token": row[2]}


def record_transaction(amount_cents: int, note: str) -> None:
    """记录一笔交易。"""
    CONNECTION.execute(
        "INSERT INTO transactions (amount_cents, note) VALUES (?, ?)",
        (amount_cents, note),
    )
    CONNECTION.commit()
''',
    "tests/test_auth.py": '''\
"""鉴权相关测试。"""

from payments.auth import hash_password, logout, require_token


def test_hash_password_is_stable() -> None:
    assert hash_password("s3cret") == hash_password("s3cret")


def test_require_token_rejects_wrong_password() -> None:
    user = {"password_hash": hash_password("right")}
    assert require_token(user, "wrong") is False


def test_logout_is_implemented() -> None:
    # 这个断言是假的：logout 直接 return True，并没有让 token 失效
    assert logout("any-token") is True
''',
}
#: 只出现在仓库树里、不提供正文的二进制文件（验证"后缀过滤"确实生效）
BINARY_FILES: dict[str, int] = {"assets/logo.png": 2048}


def _sha(name: str) -> str:
    """给离线站点造一个稳定（非随机）的假 sha，便于对照输出。"""
    return hashlib.sha1(name.encode("utf-8")).hexdigest()


def _tree_payload() -> list[dict[str, Any]]:
    """按 ``git/trees`` 的结构造仓库树：目录条目 + 文件条目。"""
    entries: dict[str, dict[str, Any]] = {}
    for path in (*MINI_REPO, *BINARY_FILES):
        parts = path.split("/")
        for depth in range(1, len(parts)):
            directory = "/".join(parts[:depth])
            entries.setdefault(
                directory,
                {"path": directory, "mode": "040000", "type": "tree", "sha": _sha(directory)},
            )
        size = BINARY_FILES.get(path) or len(MINI_REPO[path].encode("utf-8"))
        entries[path] = {
            "path": path,
            "mode": "100644",
            "type": "blob",
            "sha": _sha(path),
            "size": size,
        }
    return list(entries.values())


def build_offline_site(repo: str) -> httpx.MockTransport:
    """内存版 GitHub：只实现本示例会用到的三个只读端点。"""
    slug = GitHubRepoRef.parse(repo).slug
    prefix = f"/repos/{slug}"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == prefix:
            return httpx.Response(
                200,
                json={
                    "name": slug.split("/")[-1],
                    "owner": {"login": slug.split("/")[0]},
                    "default_branch": DEMO_BRANCH,
                    "description": "W7 离线演示仓库（虚构）",
                    "language": "Python",
                    "size": 128,
                    "private": False,
                    "html_url": f"https://github.com/{slug}",
                    "pushed_at": "2026-09-01T00:00:00Z",
                },
            )
        if path == f"{prefix}/git/trees/{DEMO_BRANCH}":
            return httpx.Response(
                200, json={"sha": _sha("tree"), "truncated": False, "tree": _tree_payload()}
            )
        if path.startswith(f"{prefix}/contents/"):
            target = unquote(path[len(f"{prefix}/contents/") :])
            if target in MINI_REPO:
                body = MINI_REPO[target].encode("utf-8")
                return httpx.Response(
                    200,
                    json={
                        "name": target.split("/")[-1],
                        "path": target,
                        "sha": _sha(target),
                        "size": len(body),
                        "encoding": "base64",
                        "content": base64.b64encode(body).decode("ascii"),
                        "html_url": f"https://github.com/{slug}/blob/{DEMO_BRANCH}/{target}",
                    },
                )
            return httpx.Response(404, json={"message": "Not Found", "status": "404"})
        return httpx.Response(404, json={"message": "Not Found", "status": "404"})

    return httpx.MockTransport(handler)


@contextmanager
def open_client(*, repo: str, live: bool) -> Iterator[GitHubClient]:
    """打开客户端：离线用注入的内存站点，在线用配置里的 GITHUB_TOKEN。"""
    if live:
        client = GitHubClient.from_config()
        try:
            yield client
        finally:
            client.close()
        return
    with httpx.Client(transport=build_offline_site(repo), base_url=DEFAULT_API_BASE) as http:
        yield GitHubClient(token="offline-demo", client=http)


# ---------------------------------------------------------------- 跑一次构建
def explain_empty_evidence(
    client: GitHubClient, repo: str, args: argparse.Namespace
) -> list[str]:
    """证据为 0 时给出**可核对**的解释，而不是让人怀疑"是不是网络/链路坏了"。

    只在"证据为空"这条分支里被调用，所以正常路径不会多花这一次请求。
    实测教训：`octocat/Hello-World` 的唯一文件是无后缀的 `README`，
    被文本白名单挡掉后示例只报"证据非空：不通过"，看起来就像 live 链路没打通。
    """
    if args.paths:
        return [f"[诊断] --paths 指定的 {len(args.paths)} 个路径都没取到内容（原因见上方的跳过日志）"]

    try:
        tree = client.list_tree(repo, ref=args.ref or None, path_prefix=args.prefix or None)
    except CodeAgentXError as exc:
        return [f"[诊断] 为解释原因重新拉取仓库树也失败了：{exc}"]

    files = list(tree.files)
    kept = [entry for entry in files if in_text_suffixes(entry)]
    skipped = [entry for entry in files if not in_text_suffixes(entry)]
    scope = f"{repo} 的 {args.prefix} 目录" if args.prefix else repo
    lines = [
        f"[诊断] {scope} 共 {len(files)} 个文件：符合文本白名单 {len(kept)} 个，被过滤 {len(skipped)} 个"
    ]
    if skipped:
        sample = "、".join(str(entry.path) for entry in skipped[:5])
        lines.append(f"        被过滤：{sample}{'…' if len(skipped) > 5 else ''}")

    if not files:
        lines.append("        这个范围里一个文件都没有（路径前缀写错了？或该分支下确实没有文件）。")
    elif not kept:
        lines.append(
            "        证据为空的直接原因是**文本白名单**：仓库里没有后缀在白名单内、"
            "或名字属于 README / LICENSE / Makefile / Dockerfile 这类约定文本的文件。"
        )
        lines.append("        可用 --paths <具体文件> 强制读取指定文件。")
    else:
        lines.append("        白名单内有文件却没取到内容，请看上方的跳过日志（404 / 超大 / 编码异常）。")
    return lines


def run(*, client: GitHubClient, repo: str, args: argparse.Namespace) -> int:
    repository = client.get_repository(repo)
    print(repository.to_text())
    print()

    source = GitHubSource(
        client=client,
        repo=repo,
        paths=tuple(args.paths),
        path_prefix=args.prefix or "",
        ref=args.ref or None,
        max_files=args.max_files,
    )
    how = f"指定 {len(args.paths)} 个路径" if args.paths else f"按仓库树挑选（上限 {args.max_files}）"
    print(f"[取回] {how}{f'，限定目录 {args.prefix}' if args.prefix else ''}")

    built = ContextBuilder(budget_tokens=args.budget).build(
        args.task,
        github=source,
        notes=["项目约定：所有对外入口必须校验入参；金额用整数分表示"],
    )

    evidence = next(
        (section for section in built.sections if section.title == SECTION_EVIDENCE), None
    )
    github_documents = [d for d in built.documents if d.source == "github"]
    for document in github_documents:
        print(f"  - {document.location or document.path}  {document.metadata.get('size', 0)} 字节")
    print()

    print("[GSSC 阶段]")
    print(f"  {'阶段':<10}{'进':>4}{'出':>5}{'token':>8}{'耗时':>8}  说明")
    for stage in built.stats.stages:
        print(
            f"  {stage.name:<10}{stage.documents_in:>4}{stage.documents_out:>5}"
            f"{stage.tokens:>8}{stage.duration * 1000:>7.1f}ms  {stage.detail}"
        )
    print()

    compression = built.stats.compression
    print(
        f"[压缩] {compression['before_tokens']} → {compression['after_tokens']} token"
        f"（降幅 {compression['reduction'] * 100:.1f}%），"
        f"丢弃 {compression['dropped_documents']} 条，截断 {compression['truncated_documents']} 条，"
        f"硬收敛 {compression['hard_trimmed']} 条"
    )
    print(f"[请求] {client.stats.as_dict()}")
    print()

    print(
        f"[上下文] {len(built.sections)} 节 / {len(built.documents)} 条文档 / "
        f"{built.tokens} token（预算 {built.budget}，余量 {built.budget - built.tokens}）"
    )
    for index, section in enumerate(built.sections, start=1):
        # 任务节的 note 就是任务本身，已经在上面打印过，这里不再重复
        note = section.note if section.note and section.title != SECTION_TASK else ""
        suffix = f"（{note[:40]}…）" if len(note) > 40 else (f"（{note}）" if note else "")
        print(f"  {index}. {section.title}：{len(section.documents)} 条{suffix}")
    print()

    if args.out:
        target = Path(args.out)
        target.write_text(built.text, encoding="utf-8")
        print(f"[落盘] {target}（{len(built.text)} 字符）")
    if args.show_text:
        print("-------------------------------- 上下文正文 --------------------------------")
        print(built.text)
        print("---------------------------------------------------------------------------")

    evidence_count = len(evidence.documents) if evidence else 0
    checks = [
        ("仓库证据非空", evidence_count > 0, f"证据节 {evidence_count} 条"),
        ("渲染后不超预算", built.within_budget, f"{built.tokens} / {built.budget} token"),
    ]
    for name, passed, detail in checks:
        print(f"[{'通过' if passed else '不通过'}] {name}：{detail}")
    if evidence_count == 0:
        # 只报"不通过"会让人误以为 live 链路坏了，这里补一条可核对的解释
        for line in explain_empty_evidence(client, repo, args):
            print(line)
    return 0 if all(passed for _, passed, _ in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeAgentX W7 示例：读 GitHub 仓库生成上下文")
    parser.add_argument("--task", default=DEFAULT_TASK, help="审查任务（决定检索与排序的相关性）")
    parser.add_argument("--prefix", default="", help="只读仓库里的这个目录")
    parser.add_argument("--paths", nargs="*", default=[], help="明确要读的文件，给了它就不再拉树")
    parser.add_argument("--ref", default="", help="分支/标签/提交，缺省用仓库默认分支")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_GITHUB_FILES, help="最多取回几个文件")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="上下文 token 预算")
    parser.add_argument("--live", metavar="OWNER/REPO", default="", help="连真实 GitHub（只读）")
    parser.add_argument("--out", default="", help="把生成的上下文写到文件")
    parser.add_argument("--show-text", action="store_true", help="打印生成的上下文正文")
    args = parser.parse_args(argv)

    # 示例的输出重点是自己打印的阶段账目，库内 INFO 日志会把版面打散。
    # 导入期已经有模块取过 logger（默认 INFO），所以必须 force 重建 handler 才能改级别。
    setup_logging("WARNING", force=True)

    if args.budget <= 0 or args.max_files <= 0:
        print("[参数错误] --budget 与 --max-files 必须为正整数", file=sys.stderr)
        return 2

    repo = args.live or DEMO_REPO
    if not args.live:
        print(OFFLINE_BANNER)
        print()

    try:
        with open_client(repo=repo, live=bool(args.live)) as client:
            return run(client=client, repo=repo, args=args)
    except CodeAgentXError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
