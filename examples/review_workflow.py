"""W6 端到端示例：七个 Agent 协作完成一次代码审查。

流水线：``plan → retrieve → review → security →（test）→（refactor）→ report``

用法::

    # 离线演示（无需 API Key）
    python examples/review_workflow.py --mock
    # 七阶段全开：测试生成 + 重构规划
    python examples/review_workflow.py --mock --enable-test --enable-refactor
    # 真实审查（需在 .env 配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID）
    python examples/review_workflow.py data/sample_repo
    # 落盘状态 → 中断恢复（第二次跑会跳过已完成的阶段，不重复烧 Token）
    python examples/review_workflow.py --state .workflow_state.json data/sample_repo
    python examples/review_workflow.py --state .workflow_state.json --resume data/sample_repo
    # 把 Markdown 报告写到文件
    python examples/review_workflow.py --mock --out review_workflow.md

验收标准：一次运行必须给出三样东西

1. **七阶段状态表**：``done / failed / skipped`` 三者可区分，跳过与失败不能混为一谈；
2. **合并后的报告**：多角色结论去重合并（同一条问题保留更严重等级与合并来源）；
3. **降级标记**：任何一步没跑成时，报告摘要里必须有警告，且退出码非 0。

诚实的边界
----------
``--mock`` 只把 **LLM 的输出**脚本化；检索（建索引 + 混合检索）与汇总（确定性合并）
仍然是真实执行的，所以"证据来自哪个文件"是真的。但"发现了哪些问题"来自写死的脚本，
**不是真实结论**——真实结论必须去掉 ``--mock``，或依赖 ``static_analyzer`` 这类确定性工具。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from codeagentx.agents.schemas import ReviewPlan  # noqa: E402
from codeagentx.config import get_config  # noqa: E402
from codeagentx.core.exceptions import CodeAgentXError  # noqa: E402
from codeagentx.core.llm import MockLLM, build_llm  # noqa: E402
from codeagentx.orchestrator import CodeReviewWorkflow, WorkflowResult  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "data" / "sample_repo"

MOCK_BANNER = (
    "[!] 离线演示模式（--mock）：检索与汇总真实执行，但「发现了哪些问题」来自脚本化输出，\n"
    "    不是真实审查结论。要得到真实结论，请在 .env 配置 LLM_API_KEY 后去掉 --mock。"
)

# ---------------------------------------------------------------- 演示脚本
_DEMO_PLAN = """\
{
  "summary": "按「认证实现 → 数据存取 → 入口异常 → 会话失效」四条线索分工，先定位再下结论。",
  "tasks": [
    {
      "description": "检查 login / hash_password / SECRET_KEY：口令校验与密钥管理是否安全",
      "focus": "security",
      "files": ["app/auth/service.py"],
      "reason": "登录是唯一入口，密钥与口令处理是最容易被利用的两点",
      "assignee": "security"
    },
    {
      "description": "检查 build_user_query / find_user：SELECT 语句构造与 username 入参校验",
      "focus": "security",
      "files": ["app/db/repository.py"],
      "reason": "username 来自请求参数，字符串拼接即注入风险",
      "assignee": "security"
    },
    {
      "description": "检查 handle_login / handle_debug_login：异常处理与错误信息回显",
      "focus": "bug",
      "files": ["app/api/routes.py"],
      "reason": "调试入口未鉴权且回显内部异常，登录失败也没有次数限制",
      "assignee": "reviewer"
    },
    {
      "description": "检查 logout / current_user：登出后 token 是否真的失效",
      "focus": "bug",
      "files": ["app/auth/service.py"],
      "reason": "登出是空实现，属于会话管理缺陷",
      "assignee": "reviewer"
    }
  ]
}
"""

_DEMO_REVIEW = """\
{
  "summary": "认证服务的会话管理有缺陷：logout 未使 token 失效；另有调试入口回显内部异常。",
  "findings": [
    {
      "title": "logout 未真正注销会话 token",
      "category": "bug",
      "severity": "medium",
      "file": "app/auth/service.py",
      "line": 34,
      "description": "logout 直接 return True，没有让 token 失效，登出后旧 token 仍可继续访问。",
      "suggestion": "引入服务端会话表或签发时间戳校验，logout 时显式作废该 token，并补充「登出后再访问应失败」的回归用例。",
      "evidence": "def logout(session_token):\\n    return True",
      "confidence": 0.8
    },
    {
      "title": "调试入口把内部异常原样返回给调用方",
      "category": "security",
      "severity": "medium",
      "file": "app/api/routes.py",
      "line": 27,
      "description": "handle_debug_login 把 repr(exc) 放进响应体，可能泄露内部路径与实现细节，且该入口没有任何鉴权。",
      "suggestion": "对外只返回统一错误码与提示语，详细异常写入日志；调试入口加鉴权，或在生产环境不注册该路由。",
      "evidence": "return {\\"status\\": 500, \\"error\\": repr(exc)}",
      "confidence": 0.7
    }
  ]
}
"""

_DEMO_SECURITY = """\
{
  "summary": "存在两处高危与两处中危：硬编码密钥、无盐 sha256、SQL 字符串拼接、异常信息泄露。",
  "findings": [
    {
      "title": "SECRET_KEY 硬编码在源码中",
      "category": "security",
      "severity": "high",
      "file": "app/auth/service.py",
      "line": 11,
      "description": "会话签名密钥以常量形式写在源码里，任何拿到仓库的人都能伪造 token。",
      "suggestion": "改为从环境变量读取并在启动时校验其存在，同时轮换已泄露的密钥。",
      "evidence": "SECRET_KEY = \\"hardcoded-secret-key-for-demo\\"",
      "confidence": 0.95
    },
    {
      "title": "口令使用无盐 sha256 存储",
      "category": "security",
      "severity": "high",
      "file": "app/auth/service.py",
      "line": 19,
      "description": "hash_password 直接对明文做 sha256，无盐且无密钥派生，口令表泄露后可被彩虹表与 GPU 高速破解。",
      "suggestion": "改用 bcrypt / argon2 等带盐的密钥派生算法，并对历史口令做一次重置或迁移。",
      "evidence": "return hashlib.sha256(password.encode()).hexdigest()",
      "confidence": 0.9
    },
    {
      "title": "SQL 语句由字符串拼接而成",
      "category": "security",
      "severity": "high",
      "file": "app/db/repository.py",
      "line": 24,
      "description": "build_user_query 直接把 username 拼进 SELECT，攻击者可用 ' OR '1'='1 绕过条件判断。",
      "suggestion": "改用参数化查询：cursor.execute(\\"SELECT name, password_hash FROM users WHERE name = ?\\", (username,))。",
      "evidence": "return f\\"SELECT name, password_hash FROM users WHERE name = '{username}'\\"",
      "confidence": 0.9
    },
    {
      "title": "调试入口把内部异常原样返回给调用方",
      "category": "security",
      "severity": "high",
      "file": "app/api/routes.py",
      "line": 27,
      "description": "异常信息回显给未鉴权的调用方，可用于探测内部实现与路径（与 Reviewer 结论一致，等级按高处置）。",
      "suggestion": "对外返回统一错误码，详细异常只写日志；调试入口加鉴权或仅在开发环境启用。",
      "evidence": "return {\\"status\\": 500, \\"error\\": repr(exc)}",
      "confidence": 0.85
    }
  ]
}
"""

_DEMO_TEST_CODE = """\
```python
\"\"\"复现「logout 后 token 仍可用」这一缺陷。\"\"\"

from app.auth.models import User
from app.auth.service import current_user, hash_password, login, logout
from app.db.repository import save_user


def make_user(name="alice", password="p@ssw0rd"):
    save_user(User(name=name, password_hash=hash_password(password)))
    return name, password


def test_token_is_rejected_after_logout():
    name, password = make_user()
    token = login(name, password)
    assert token is not None, "登录应返回 token"

    logout(token)

    # 缺陷：logout 是空实现，旧 token 依然能还原出用户
    assert current_user(token) is None
```

该测试断言「登出后旧 token 不再能还原用户」；在当前示例仓库上会失败，
因为 logout 并未真正作废 token（可先复制到仓库根目录，再用 pytest 运行）。
"""

_DEMO_REFACTOR_PLAN = """\
{
  "goal": "在保持对外接口不变的前提下，修复认证链路的密钥管理、口令存储与会话失效问题",
  "steps": [
    {
      "description": "把 SECRET_KEY 改为从环境变量读取，并在启动时校验其存在",
      "files": ["app/auth/service.py"],
      "rationale": "密钥必须与代码分离；缺失时快速失败优于静默使用默认值"
    },
    {
      "description": "把 hash_password 换成带盐的密钥派生算法（bcrypt/argon2），保留函数签名",
      "files": ["app/auth/service.py"],
      "rationale": "对外仍是 password_hash 字符串，调用方无需改动"
    },
    {
      "description": "把 build_user_query 改为参数化查询",
      "files": ["app/db/repository.py"],
      "rationale": "参数化是消除注入的标准做法，且不影响返回结构"
    },
    {
      "description": "为 logout 增加服务端 token 作废机制，并补回归用例",
      "files": ["app/auth/service.py", "tests/test_auth_service.py"],
      "rationale": "登出必须真正生效；用例用于锁定该行为不被回退"
    }
  ],
  "risks": [
    "本地未设置 SECRET_KEY 时服务将启动失败，需同步更新 .env.example",
    "切换口令算法后，历史口令摘要无法直接比对，需要一次性重置或迁移"
  ],
  "verification": [
    "python -m pytest tests/",
    "手工调用 /login 验证成功、口令错误、登出后复用旧 token 三条路径"
  ]
}
"""

#: 路由表：按提示词里必然出现的标记选择脚本响应（顺序即优先级）
_ROUTES: tuple[tuple[str, str], ...] = (
    ("请输出不超过", _DEMO_PLAN),
    ("本轮重点关注：逻辑正确性", _DEMO_REVIEW),
    ("本轮重点关注：注入", _DEMO_SECURITY),
    ("请给出可直接运行的 pytest 测试文件", _DEMO_TEST_CODE),
    ("请输出重构计划 JSON", _DEMO_REFACTOR_PLAN),
)


class ScriptedLLM(MockLLM):
    """按提示词路由的脚本化 LLM（离线演示用）。

    :class:`~codeagentx.core.llm.MockLLM` 是按调用顺序发牌的，而本流水线有可选的
    test / refactor 阶段：少开一个阶段，后续响应就会整体前移、脚本全部错位。
    这里改成"看当前提示词命中哪条标记"来选响应，可选阶段怎么增减都不会串味。
    """

    def __init__(
        self,
        routes: Sequence[tuple[str, str]] = _ROUTES,
        *,
        default_response: str = _DEMO_REVIEW,
    ) -> None:
        super().__init__([], default_response=default_response)
        self._routes = list(routes)

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        # 复用父类的记账逻辑：把选中的脚本当作"唯一一张牌"发出去
        self._responses = [self._pick(messages)]
        self._cursor = 0
        return super().chat(messages, **kwargs)

    def _pick(self, messages: Any) -> str:
        prompt = _last_user_text(self._to_openai_messages(messages))
        for marker, reply in self._routes:
            if marker in prompt:
                return reply
        return str(self.default_response)


# ---------------------------------------------------------------- 辅助
def _last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    """取最后一条 user 消息（各角色的任务描述都在这里）。"""
    for item in reversed(list(messages)):
        if str(item.get("role")) == "user":
            return str(item.get("content") or "")
    return ""


def _display(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_root(raw: str) -> Path:
    """把命令行参数解析成"允许审查的仓库根目录"。"""
    root = Path(raw)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"目标不存在：{root}")
    return root if root.is_dir() else root.parent


def _print_stages(result: WorkflowResult) -> None:
    """打印阶段状态表：跳过与失败必须一眼可分辨。"""
    state = result.state
    counts = state.progress()
    print("[阶段状态] " + " | ".join(f"{name} {count}" for name, count in counts.items()))
    for item in state.stages:
        line = f"  {item.name:<9}{item.status:<8}{item.duration:>6.2f}s  {item.detail}"
        if item.error:
            line += f"  ⟵ {item.error}"
        print(line)
    failed = state.failed_stages()
    if failed:
        print(f"  ⚠ 未完成的阶段：{', '.join(failed)}（报告已按降级标注）")


def _print_pipeline_detail(result: WorkflowResult) -> None:
    """展示各阶段留下的产物（计划 / 证据 / 重构计划），证明阶段之间真的传了东西。"""
    artifacts = result.state.artifacts
    payload = artifacts.get("plan")
    if isinstance(payload, dict):
        plan = ReviewPlan.from_dict(payload)
        print(f"[计划] {plan.total} 条子任务（检索阶段会逐条当作查询词）")
        for index, task in enumerate(plan.tasks, start=1):
            owner = f" @{task.assignee}" if task.assignee else ""
            print(f"  {index}. [{task.focus}]{owner} {task.description}")
    evidence = artifacts.get("evidence") or []
    print(f"[证据] 检索命中 {len(evidence)} 条片段（来自真实索引，可回溯到文件与行号）")
    for item in evidence[:3]:
        location = f"{item.get('path')}:{item.get('start_line')}-{item.get('end_line')}"
        print(f"  - {location}  score={item.get('score')}")
    if len(evidence) > 3:
        print(f"  ...（其余 {len(evidence) - 3} 条见报告 metadata）")
    payload = artifacts.get("refactor")
    if isinstance(payload, dict):
        steps = payload.get("steps") or []
        print(f"[重构] {len(steps)} 步（仅规划，未改动任何代码）")
    test_payload = artifacts.get("test")
    if isinstance(test_payload, dict):
        test_result = test_payload.get("test_result")
        if isinstance(test_result, dict):
            if test_result.get("available"):
                # 有没有收集到用例、退出码是多少，都要如实说：exit_code=5 表示"没收集到用例"，
                # 不能因为"跑过了"就写成通过
                detail = test_result.get("summary_line") or f"exit_code={test_result.get('exit_code')}"
                verdict = "通过" if test_result.get("passed") else "未通过"
                print(f"[测试] 现有用例：{verdict}（{detail}）")
            else:
                print(f"[测试] 现有用例：未运行（{test_result.get('error')}）")


def _print_summary(result: WorkflowResult, llm: Any) -> None:
    state = result.state
    report = result.report
    counts = report.severity_counts()
    print(
        f"[结论] 问题 {report.total} 条"
        f"（high {counts['high']} / medium {counts['medium']} / low {counts['low']}）"
        f" | 降级={bool(report.metadata.get('degraded'))} | 整体成功={result.success}"
    )
    sources = report.metadata.get("sources") or {}
    if sources:
        print("[来源] " + "、".join(f"{name} {count} 条" for name, count in sources.items()))
    usage = state.metadata.get("usage") or {}
    print(
        f"[用量] 角色累计：LLM 调用 {usage.get('calls', 0)} 次"
        f" | token {usage.get('total_tokens', 0)}"
        f" | 耗时 {state.metadata.get('duration', 0.0)}s"
    )
    print(
        f"[用量] 注入的 LLM 实例统计：调用 {llm.stats.calls} 次"
        f"（含未经 AgentResult 的直连调用，例如重构规划）"
    )
    if report.metadata.get("degraded"):
        print("[提示] 本次存在降级环节，结论不完整，请勿据此判定「没有问题」。")


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeAgentX 多 Agent 审查流水线示例（W6）")
    parser.add_argument("target", nargs="?", default=str(DEFAULT_TARGET), help="要审查的仓库目录")
    parser.add_argument("--mock", action="store_true", help="离线演示（脚本化结论，检索真实执行）")
    parser.add_argument("--enable-test", action="store_true", help="启用测试阶段（生成复现测试）")
    parser.add_argument("--enable-refactor", action="store_true", help="启用重构规划阶段（只出计划）")
    parser.add_argument("--reflect", action="store_true", help="主审查角色改用 Reflection 范式")
    parser.add_argument("--state", default="", help="状态文件路径；给了才具备中断恢复能力")
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument("--resume", action="store_true", help="从状态文件恢复，跳过已完成阶段")
    switch.add_argument("--reset", action="store_true", help="先丢弃旧状态再从头执行")
    parser.add_argument("--out", default="", help="把 Markdown 报告写入该文件")
    args = parser.parse_args(argv)

    config = get_config()
    if not args.mock and not config.is_llm_configured:
        print(
            "[错误] 未配置 LLM_API_KEY，无法进行真实审查。\n"
            "       请复制 .env.example 为 .env 并填写密钥，或加 --mock 查看演示输出。",
            file=sys.stderr,
        )
        return 2
    try:
        root = _resolve_root(args.target)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    llm = ScriptedLLM() if args.mock else build_llm(config)
    print(
        f"[配置] 模型={llm.model_id} | 目标={_display(root)}"
        f" | 测试阶段={'开' if args.enable_test else '关'}"
        f" | 重构规划={'开' if args.enable_refactor else '关'}"
        f" | 审查范式={'reflection' if args.reflect else 'react'}"
    )
    if args.mock:
        print(MOCK_BANNER)
    print()

    workflow = CodeReviewWorkflow(
        llm,
        root=root,
        target=_display(root),
        config=config,
        state_path=Path(args.state) if args.state else None,
        enable_test=args.enable_test,
        enable_refactor=args.enable_refactor,
        reflect=args.reflect,
    )
    try:
        result = workflow.run(resume=args.resume, reset=args.reset)
    except CodeAgentXError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1

    print(result.report.to_text())
    print()
    _print_pipeline_detail(result)
    print()
    _print_stages(result)
    print()
    _print_summary(result, llm)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.write_text(result.to_markdown(), encoding="utf-8")
        print(f"[输出] Markdown 报告已写入 {out_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
