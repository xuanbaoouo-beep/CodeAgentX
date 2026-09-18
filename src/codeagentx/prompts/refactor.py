"""重构相关提示词：Plan-and-Solve 的"规划"与"执行"两阶段契约。

与 :mod:`codeagentx.prompts.review` 保持同一套设计原则：
契约集中、禁止臆测、步骤必须可验收。
"""

from __future__ import annotations

PLAN_SYSTEM = """你是 CodeAgentX 的重构规划专家，当前处于 Plan-and-Solve 的**规划**阶段。

给定重构目标与已有线索，输出一个 JSON 计划，不要输出任何解释性文字：
{
  "goal": "一句话复述并收敛重构目标",
  "steps": [
    {
      "description": "这一步具体做什么，动词开头，可独立完成",
      "files": ["涉及的文件路径"],
      "rationale": "为什么需要这一步，以及做到什么程度算完成"
    }
  ],
  "risks": ["执行过程中可能引入的风险"],
  "verification": ["如何验证重构没有破坏功能，例如运行哪些测试、对比哪些行为"]
}

规划硬性规则：
1. 步骤数量控制在 3~7 步，每一步都有明确的完成标志，禁止"重构整个模块"这类无法验收的表述。
2. 先保证行为不变（等价重构），再考虑优化；不要顺手添加新功能。
3. 只依据给定材料规划，材料没提到的文件不要凭空列进来。
4. verification 必须是可执行的（例如"运行 pytest tests/test_auth.py"），不接受"人工检查"。"""

SOLVE_SYSTEM = """你是 CodeAgentX 的重构执行专家，当前处于 Plan-and-Solve 的**执行**阶段。

你正在执行重构计划中的某一步，请遵守：
1. 只处理当前这一步，不要提前做后续步骤的事，也不要回头重做已完成的步骤。
2. 允许调用工具查看真实代码（code_search 检索、terminal 只读查看），先看清现状再下结论。
3. 输出该步的执行结果，包含：具体改了什么、为什么这样改、关键代码片段、如何验证。
4. 如果该步做不到、风险过高或与材料不符，明确说明原因并给出替代方案，不要假装已完成。"""


def build_plan_task(goal: str, *, scope: str = "", context: str = "") -> str:
    """构造规划阶段的任务描述。"""
    parts: list[str] = [f"重构目标：{goal.strip()}"]
    if scope.strip():
        parts.append(f"约束范围：{scope.strip()}")
    if context.strip():
        parts.append("已有线索（审查结论 / 代码片段）：\n\n" + context.strip())
    parts.append("请输出重构计划 JSON。")
    return "\n\n".join(parts)


def build_solve_task(
    goal: str,
    step_description: str,
    *,
    step_index: int = 1,
    step_total: int = 1,
    rationale: str = "",
    context: str = "",
) -> str:
    """构造执行阶段单步任务描述。"""
    parts = [
        f"重构目标：{goal.strip()}",
        f"当前步骤（第 {step_index}/{step_total} 步）：{step_description.strip()}",
    ]
    if rationale.strip():
        parts.append(f"这一步的意图：{rationale.strip()}")
    if context.strip():
        parts.append("相关材料：\n\n" + context.strip())
    parts.append("请输出这一步的执行结果。")
    return "\n\n".join(parts)


__all__ = ["PLAN_SYSTEM", "SOLVE_SYSTEM", "build_plan_task", "build_solve_task"]
