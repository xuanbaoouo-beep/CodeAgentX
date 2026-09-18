"""CodeAgentX 命令行入口：对本地目录或 GitHub 仓库做一次多 Agent 代码审查。

用法::

    codeagentx review data/sample_repo
    codeagentx review acme/demo --out review.md
    codeagentx review owner/repo@v1.2.0 --ref main
    codeagentx review acme/demo --enable-refactor --state .workflow_state.json

三条边界（和 README 的说法保持一致）
-----------------------------------
1. **远端仓库先落到本地再审查**：``owner/repo`` 会被下载成一个临时工作区
   （zipball，不执行任何远端代码），审查完即删除，除非显式 ``--keep-workdir``。
2. **``--enable-test`` 会执行目标仓库自带的测试**——那是在跑别人的代码。
   默认关闭；对远端仓库会额外打印一条警告，请只对信得过的目标开启。
3. **不产出补丁**：本项目沙箱没有写文件工具，重构只到"计划"（``--enable-refactor``
   只给步骤与验证方式，不改任何文件）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codeagentx.config import get_config
from codeagentx.core.exceptions import CodeAgentXError, TargetError
from codeagentx.core.llm import build_llm
from codeagentx.orchestrator import CodeReviewWorkflow, WorkflowResult
from codeagentx.protocols.github_archive import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_FILES
from codeagentx.workspace import prepare_target

__all__ = ["build_parser", "main"]


def _print_stages(result: WorkflowResult) -> None:
    state = result.state
    print("[阶段] " + " | ".join(f"{name} {count}" for name, count in state.progress().items()))
    for item in state.stages:
        line = f"  {item.name:<9}{item.status:<8}{item.duration:>6.2f}s  {item.detail}"
        if item.error:
            line += f"  ⟵ {item.error}"
        print(line)
    failed = state.failed_stages()
    if failed:
        print(f"  ⚠ 未完成的阶段：{', '.join(failed)}（报告已按降级标注）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeagentx",
        description="CodeAgentX：多智能体代码审查与重构助手（本地目录 / GitHub 仓库）",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review", help="审查一个本地目录或 GitHub 仓库")
    review.add_argument(
        "target",
        help="本地目录、单个 .py 文件（只审这一个文件），或 owner/repo（可写 owner/repo@ref 或仓库 URL）",
    )
    review.add_argument("--ref", default="", help="远端仓库的分支/标签/提交（默认用仓库默认分支）")
    review.add_argument("--out", default="", help="把 Markdown 报告写入该文件")
    review.add_argument("--state", default="", help="状态文件路径；给了才具备中断恢复能力")
    switch = review.add_mutually_exclusive_group()
    switch.add_argument("--resume", action="store_true", help="从状态文件恢复，跳过已完成阶段")
    switch.add_argument("--reset", action="store_true", help="先丢弃旧状态再从头执行")
    review.add_argument("--enable-test", action="store_true", help="启用测试阶段（跑目标仓库自带用例）")
    review.add_argument("--enable-refactor", action="store_true", help="启用重构规划阶段（只出计划）")
    review.add_argument("--reflect", action="store_true", help="主审查角色改用 Reflection 范式")
    review.add_argument("--keep-workdir", action="store_true", help="保留远端下载的临时工作区")
    review.add_argument(
        "--max-files", type=int, default=MAX_ARCHIVE_FILES, help="远端归档最多解压多少文件"
    )
    review.add_argument(
        "--max-mb",
        type=float,
        default=MAX_ARCHIVE_BYTES / 1024 / 1024,
        help="远端归档解压后的总字节上限（MB）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "review":  # pragma: no cover - argparse 已限定
        print(f"[错误] 未知子命令：{args.command}", file=sys.stderr)
        return 2

    config = get_config()
    if not config.is_llm_configured:
        print(
            "[错误] 未配置 LLM_API_KEY，无法进行真实审查。\n"
            "       请复制 .env.example 为 .env 并填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID。",
            file=sys.stderr,
        )
        return 2

    prepared = None
    try:
        prepared = prepare_target(
            args.target,
            ref=args.ref,
            max_files=args.max_files,
            max_bytes=int(args.max_mb * 1024 * 1024),
        )
    except TargetError as exc:
        print(f"[错误] {exc.message}", file=sys.stderr)
        if exc.detail:
            print(f"       {exc.detail}", file=sys.stderr)
        return 2
    except CodeAgentXError as exc:
        print(f"[错误] 远端仓库准备失败：{exc}", file=sys.stderr)
        # 两层含义都能写成 owner/repo：本地相对子目录不存在时会被当成远端标识，
        # 这里补一句，避免用户对着"仓库不存在"去查 GitHub 而其实是路径写错
        print(
            f"       若 {args.target} 本意是本地路径，请检查拼写"
            "（当前目录下没有这个路径，所以按远端仓库处理了）。",
            file=sys.stderr,
        )
        return 2

    root, target = prepared.root, prepared.display
    if prepared.remote:
        print(f"[远端] {prepared.note}")
    elif prepared.note:
        print(f"[提示] {prepared.note}")

    llm = build_llm(config)
    print(
        f"[配置] 模型={llm.model_id} | 目标={target} | 来源={'GitHub' if prepared.remote else '本地'}"
        f" | 测试阶段={'开' if args.enable_test else '关'}"
        f" | 重构规划={'开' if args.enable_refactor else '关'}"
        f" | 审查范式={'reflection' if args.reflect else 'react'}"
    )
    if prepared.remote and args.enable_test:
        print("[警告] --enable-test 会执行刚下载的远端仓库自带测试，请确认目标可信。")

    workflow = CodeReviewWorkflow(
        llm,
        root=root,
        target=target,
        paths=prepared.paths or None,
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
    finally:
        prepared.cleanup(keep=args.keep_workdir)

    print()
    print(result.report.to_text())
    print()
    _print_stages(result)
    state = result.state
    usage = state.metadata.get("usage") or {}
    print(
        f"[用量] LLM 调用 {usage.get('calls', 0)} 次"
        f" | token {usage.get('total_tokens', 0)}"
        f" | 耗时 {state.metadata.get('duration', 0.0)}s"
        f" | 整体成功={result.success}"
    )
    if result.report.metadata.get("degraded"):
        print("[提示] 本次存在降级环节，结论不完整，请勿据此判定「没有问题」。")

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path = out_path if out_path.is_absolute() else (Path.cwd() / out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_markdown(), encoding="utf-8")
        print(f"[输出] Markdown 报告已写入 {out_path}")
    if prepared.workdir is not None and args.keep_workdir:
        print(f"[工作区] 已保留：{prepared.workdir}")

    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
