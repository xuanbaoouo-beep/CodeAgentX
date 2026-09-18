"""SimpleAgent 示例：最小可运行的对话 Agent（W2 验收项）。

用法::

    python examples/simple_agent.py                      # 交互式多轮对话（真实 LLM）
    python examples/simple_agent.py "用一句话介绍你自己"   # 单轮提问
    python examples/simple_agent.py --mock               # 离线 Mock 演示，无需 API Key
    python examples/simple_agent.py --stream "讲个短笑话"  # 流式输出（单轮）

验收标准：运行后能正常回复。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from codeagentx.config import get_config  # noqa: E402
from codeagentx.core.agent import Agent, AgentResult  # noqa: E402
from codeagentx.core.exceptions import CodeAgentXError  # noqa: E402
from codeagentx.core.llm import BaseLLM, MockLLM, build_llm  # noqa: E402
from codeagentx.core.logger import get_logger  # noqa: E402

logger = get_logger("examples.simple_agent")

SYSTEM_PROMPT = (
    "你是一个简洁、严谨的中文助手。"
    "回答控制在三句话以内，不要使用 Markdown 标题，不要编造事实。"
)


class SimpleAgent(Agent):
    """最小 Agent：无工具，纯对话。"""

    name = "simple"

    def __init__(self, llm: BaseLLM, *, system_prompt: str | None = SYSTEM_PROMPT, **kwargs: object) -> None:
        super().__init__(llm, system_prompt=system_prompt, max_iterations=1, **kwargs)  # type: ignore[arg-type]

    def run(self, task: str, **kwargs: object) -> AgentResult:
        return self.tool_loop(self._seed_messages(task))


def _build_agent(use_mock: bool) -> SimpleAgent:
    config = get_config()
    llm = MockLLM() if use_mock else build_llm(config)
    print(f"[配置] 模型={llm.model_id} | 模式={'Mock（离线）' if use_mock else '真实调用'}")
    return SimpleAgent(llm)


def _run_once(agent: SimpleAgent, question: str, *, stream: bool) -> None:
    if stream:
        messages = agent._seed_messages(question)
        print("助手: ", end="", flush=True)
        for piece in agent.llm.stream_chat(messages):
            print(piece, end="", flush=True)
        print()
        return

    result = agent.run(question)
    if result.success:
        print(f"助手: {result.output}")
    else:
        print(f"助手: [执行未成功] {result.error}\n最后输出: {result.output}")
    print(
        f"      [本次调用 {result.usage['calls']} 次 | "
        f"token {result.usage['total_tokens']} | "
        f"耗时 {result.usage['latency']}s]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CodeAgentX SimpleAgent 示例")
    parser.add_argument("question", nargs="*", help="单轮提问；不传则进入交互模式")
    parser.add_argument("--mock", action="store_true", help="离线 Mock 模式，不需要 API Key")
    parser.add_argument("--stream", action="store_true", help="流式输出（仅单轮提问时有效）")
    args = parser.parse_args(argv)

    try:
        agent = _build_agent(args.mock)
    except CodeAgentXError as exc:
        print(f"[启动失败] {exc}", file=sys.stderr)
        return 2

    if args.question:
        _run_once(agent, " ".join(args.question), stream=args.stream)
        return 0

    print("进入交互模式，输入 exit / quit 退出。\n")
    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break
        try:
            _run_once(agent, question, stream=False)
        except CodeAgentXError as exc:
            logger.error("调用失败：%s", exc)
            print(f"[调用失败] {exc}", file=sys.stderr)
        print()

    print("已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
