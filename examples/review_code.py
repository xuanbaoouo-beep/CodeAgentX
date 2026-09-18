"""W5 端到端示例：对目标代码做一次审查，输出"问题列表 + 修复建议"。

用法::

    # 离线演示（无需 API Key，输出格式与真实模式一致，但结论是脚本化示例）
    python examples/review_code.py --mock data/sample_repo/app/auth/service.py

    # 真实审查（需在 .env 配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID）
    python examples/review_code.py data/sample_repo/app/auth/service.py
    python examples/review_code.py data/sample_repo                          # 审查整个目录

    # 换范式
    python examples/review_code.py --mock --mode reflection data/sample_repo
    python examples/review_code.py --mock --mode plan-refactor data/sample_repo

    # 把 Markdown 报告写到文件
    python examples/review_code.py --mock --out review_report.md data/sample_repo

验收标准：输入 Python 文件，输出结构化问题列表（含文件/行号/严重度/修复建议）。

诚实的边界
----------
``--mock`` 只是把链路跑通给不配密钥的场景看格式，**它不会真的发现问题**；
真实结论必须来自真实 LLM（去掉 ``--mock``），或来自 ``static_analyzer`` 等确定性工具。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from codeagentx.agents.plan_solve_refactor import PlanSolveRefactor  # noqa: E402
from codeagentx.agents.react_reviewer import ReActReviewer  # noqa: E402
from codeagentx.agents.reflection import ReflectionReviewer  # noqa: E402
from codeagentx.agents.schemas import ReviewReport  # noqa: E402
from codeagentx.agents.toolkit import build_review_toolkit  # noqa: E402
from codeagentx.config import get_config  # noqa: E402
from codeagentx.core.exceptions import CodeAgentXError  # noqa: E402
from codeagentx.core.llm import MockLLM, build_llm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = PROJECT_ROOT / "data" / "sample_repo"
DEFAULT_TASK = "找出目标代码中的缺陷、安全风险与可维护性问题，并给出可落地的修复建议。"

MOCK_BANNER = (
    "[!] 离线演示模式（--mock）：下面的结论由脚本化 MockLLM 生成，只用于演示输出格式，\n"
    "    不是真实审查结果。要得到真实结论，请在 .env 配置 LLM_API_KEY 后去掉 --mock。"
)

# ---------------------------------------------------------------- 演示数据
_DEMO_REPORT = """\
{
  "summary": "该登录模块存在两个高危问题：密钥硬编码与 SQL 语句字符串拼接；另有异常处理泄露内部信息。",
  "findings": [
    {
      "title": "SECRET_KEY 硬编码在源码中",
      "category": "security",
      "severity": "high",
      "file": "app/auth/service.py",
      "line": 8,
      "description": "全局常量 SECRET_KEY 直接写在源码里，任何拿到仓库的人都能伪造会话 token。",
      "suggestion": "改为从环境变量读取：SECRET_KEY = os.environ[\\"SECRET_KEY\\"]，并在启动时校验其存在；同时轮换已泄露的密钥。",
      "evidence": "SECRET_KEY = \\"dev-secret-key-please-change\\"",
      "confidence": 0.95
    },
    {
      "title": "SQL 语句由字符串拼接而成",
      "category": "security",
      "severity": "high",
      "file": "app/db/repository.py",
      "line": 24,
      "description": "build_user_query 直接把 username 拼进 SQL，攻击者可用 ' OR '1'='1 绕过条件判断。",
      "suggestion": "改用参数化查询：cursor.execute(\\"SELECT * FROM users WHERE username = ?\\", (username,))。",
      "evidence": "return f\\"SELECT * FROM users WHERE username = '{username}'\\"",
      "confidence": 0.9
    },
    {
      "title": "异常信息直接返回给调用方",
      "category": "bug",
      "severity": "medium",
      "file": "app/api/routes.py",
      "line": 18,
      "description": "handle_debug_login 把 repr(exc) 原样返回，可能泄露文件路径与内部实现细节。",
      "suggestion": "对外返回统一错误码与提示语，把详细异常写入日志而不是响应体。",
      "evidence": "return {\\"error\\": repr(exc)}",
      "confidence": 0.75
    }
  ]
}
"""

_DEMO_PLAN = """\
{
  "goal": "消除登录模块中的硬编码密钥与 SQL 注入风险，且不改变对外行为",
  "steps": [
    {
      "description": "把 SECRET_KEY 改为从环境变量读取并补充启动校验",
      "files": ["app/auth/service.py"],
      "rationale": "密钥必须与代码分离；缺失时快速失败优于静默使用默认值"
    },
    {
      "description": "把 build_user_query 改为参数化查询",
      "files": ["app/db/repository.py"],
      "rationale": "参数化查询是消除 SQL 注入的标准做法"
    }
  ],
  "risks": ["本地未设置 SECRET_KEY 时服务将启动失败，需要同步更新 .env.example"],
  "verification": ["python -m pytest tests/", "手工调用 /login 确认正常登录与错误密码两种路径"]
}
"""

_DEMO_STEP_RESULT = (
    "已按该步骤完成等价重构：新增环境变量读取与缺失校验，对外函数签名与返回结构保持不变；\n"
    "验证方式：运行 python -m pytest tests/ 全部通过，并手工验证登录成功与失败两条路径。"
)

_DEMO_ACCEPT = '{"score": 0.9, "issues": [], "missing": [], "verdict": "accept"}'


def _build_demo_llm(mode: str) -> MockLLM:
    """构造离线演示用的脚本化 LLM（响应与所选范式匹配，避免演示时解析失败）。"""
    if mode == "plan-refactor":
        return MockLLM(
            [_DEMO_PLAN, _DEMO_STEP_RESULT, _DEMO_STEP_RESULT], default_response=_DEMO_STEP_RESULT
        )
    if mode == "reflection":
        return MockLLM([_DEMO_REPORT, _DEMO_ACCEPT], default_response=_DEMO_REPORT)
    return MockLLM([_DEMO_REPORT], default_response=_DEMO_REPORT)


# ---------------------------------------------------------------- 辅助
def _display(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _resolve_target(raw: str) -> tuple[Path, Path]:
    """返回 (目标绝对路径, 沙箱根目录)。目标是文件时，把其父目录作为允许根。"""
    target = Path(raw)
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if not target.exists():
        raise FileNotFoundError(f"目标不存在：{target}")
    return target, (target if target.is_dir() else target.parent)


def _build_tools(config, root: Path, *, no_tools: bool):
    if no_tools:
        return None
    return build_review_toolkit(
        config, root=root, tools=("code_search", "static_analyzer", "terminal")
    )


def _make_agent(mode: str, llm, tools, args):
    if mode == "plan-refactor":
        return PlanSolveRefactor(llm, tools=tools)
    if mode == "reflection":
        return ReflectionReviewer(llm, tools=tools, reflection_rounds=args.reflection_rounds)
    return ReActReviewer(llm, tools=tools)


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeAgentX 代码审查示例（W5）")
    parser.add_argument("target", nargs="?", default=str(DEFAULT_TARGET), help="要审查的文件或目录")
    parser.add_argument(
        "--mode",
        choices=("react", "reflection", "plan-refactor"),
        default="react",
        help="Agent 范式：react（默认）/ reflection / plan-refactor",
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="审查任务描述")
    parser.add_argument("--mock", action="store_true", help="离线演示模式（脚本化输出，非真实结论）")
    parser.add_argument("--no-tools", action="store_true", help="不挂载任何工具（纯 LLM 直出）")
    parser.add_argument("--reflection-rounds", type=int, default=1, help="Reflection 最大轮次")
    parser.add_argument("--out", default="", help="把 Markdown 报告写入该文件")
    args = parser.parse_args(argv)

    config = get_config()
    try:
        target, root = _resolve_target(args.target)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    if not args.mock and not config.is_llm_configured:
        print(
            "[错误] 未配置 LLM_API_KEY，无法进行真实审查。\n"
            "       请复制 .env.example 为 .env 并填写密钥，或加 --mock 查看演示输出。",
            file=sys.stderr,
        )
        return 2

    llm = _build_demo_llm(args.mode) if args.mock else build_llm(config)
    tools = _build_tools(config, root, no_tools=args.no_tools)
    tool_names = ", ".join(tools.names()) if tools is not None else "（未挂载）"
    print(f"[配置] 模式={args.mode} | 模型={llm.model_id} | 工具={tool_names}")
    print(f"[审查] 目标={_display(target)}")
    if args.mock:
        print(MOCK_BANNER)
    print()

    agent = _make_agent(args.mode, llm, tools, args)
    try:
        if args.mode == "plan-refactor":
            result = agent.run(args.task, scope=f"只处理 {_display(root)}", execute=True)
            markdown = result.output
            body = result.output
        else:
            result = agent.run(args.task, target=_display(target))
            report = ReviewReport.from_dict(result.metadata.get("report") or {})
            markdown = report.to_markdown()
            body = report.to_text()
    except CodeAgentXError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 1

    print(body)
    usage = result.usage or {}
    print()
    print(
        f"[统计] LLM 调用 {usage.get('calls', 0)} 次 | token {usage.get('total_tokens', 0)} "
        f"| 耗时 {usage.get('latency', 0.0)}s | 工具调用 {len(result.tool_calls)} 次"
        f" | 收敛={result.success}"
    )
    if result.error:
        print(f"[提示] {result.error}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        out_path.write_text(markdown, encoding="utf-8")
        print(f"[输出] Markdown 报告已写入 {out_path}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
