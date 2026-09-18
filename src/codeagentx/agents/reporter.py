"""Reporter 角色：把多个角色的结论合并成一份可交付报告。

为什么 Reporter 刻意不用 LLM
---------------------------
报告是最终交付物，必须"同样的输入给出同样的输出"。合并这一步如果用 LLM 复述：

- 同一条问题会被换个说法重复写进报告（去重失效）；
- 摘要会被"润色"成更顺耳的说法，而润色就是编造的入口；
- 每次运行结果不同，W8 的评估数字会飘。

所以合并规则全部写成确定性代码：

1. **去重**：三档判据，严格按「先精确、后粗略」的顺序找目标（见
   :meth:`ReporterAgent.merge_findings`）；同一条问题被多个角色报出时
   合并来源、取更严重的等级、置信度取高者、空缺字段互相补齐。
2. **排序**：复用 :meth:`~codeagentx.agents.schemas.ReviewReport.sorted_findings`
   （严重度 → 文件 → 行号 → 标题）。
3. **摘要**：各角色摘要按顺序拼接，不重写。改写摘要等于替模型编结论。

降级必须显式（关键）
--------------------
任一角色结论解析失败、或流水线某一环失败时，``metadata["degraded"] = True``，
摘要开头直接打印警告。绝不能让读者以为"报告里没写的问题就是没有问题"。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from codeagentx.agents.schemas import (
    SEVERITY_ORDER,
    Evidence,
    Finding,
    ReviewPlan,
    ReviewReport,
    normalize_str_list,
)
from codeagentx.core.agent import ZERO_USAGE, AgentResult
from codeagentx.core.logger import get_logger, log_event

logger = get_logger("agents.reporter")

#: 降级时写在摘要最前面的警告（不含时间戳，保证报告可复现）
DEGRADED_WARNING = (
    "⚠️ 本次审查存在降级环节（见 metadata.workflow 中失败的阶段），"
    "结论可能不完整，请勿据此判定「没有问题」。"
)

#: 「同一缺陷被重复上报」的行号容差：真实运行中观测到的重复行号偏差最大为 6 行
NEAR_LINE_TOLERANCE = 6
#: 标题相似度阈值（重叠系数）。刻意保守：漏合只是多一条冗余，
#: 错合会把另一个真实缺陷吞掉，直接掉召回，代价更大。
TITLE_SIMILARITY = 0.5
#: 参与近似去重的标题最少要有多少个特征。标题太短时相似度没有判别力
#: （"问题 A" 与 "问题 B" 只差一个字，会被算成完全相似），这类一律不合并。
MIN_TITLE_TOKENS = 4


def _title_tokens(title: str) -> set[str]:
    """把标题拆成可比特征：ASCII 词整体保留 + 中文二元组。

    中文没有词边界，用字符二元组近似；ASCII 词不切碎，
    否则 ``login`` 与 ``logout`` 会因为共享 ``lo``/``og`` 而被判成相似。
    """
    text = (title or "").lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", text))
    han = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    grams = {han[index : index + 2] for index in range(len(han) - 1)}
    return words | grams


def _title_similarity(left: str, right: str) -> float:
    """重叠系数（交集 / 较小集合）：同一缺陷被写长写短都能对上。"""
    left_tokens, right_tokens = _title_tokens(left), _title_tokens(right)
    shortest = min(len(left_tokens), len(right_tokens))
    if shortest < MIN_TITLE_TOKENS:
        return 0.0
    return len(left_tokens & right_tokens) / shortest


class ReporterAgent:
    """汇总角色：确定性合并各角色结论并渲染最终报告。"""

    name = "reporter"

    # ------------------------------------------------------------ 合并
    def _same_line_key(
        self, merged: Mapping[tuple[str, int, str, str], Finding], finding: Finding
    ) -> tuple[str, int, str, str] | None:
        """找「同一文件、**行号一字不差**」的已合并项（没有则返回 ``None``）。

        这是介于「精确」与「邻近」之间的**第二道**去重。离线复核 W8 的 54 次真实运行
        后确定：同文件同行的残留重复里**没有一组是两个不同的缺陷**——25 组经标注确认
        是同一缺陷，其余是"同一件事被另一个角色换了措辞，但标注落在另一条文案上"。
        按分类过滤挡不住错合，却会少合 10 组该合的，所以同一行直接合并，
        不再要求标题相似。
        """
        if not finding.line:
            return None
        for key, existing in merged.items():
            if existing.line and existing.file == finding.file and existing.line == finding.line:
                return key
        return None

    def _near_duplicate_key(
        self, merged: Mapping[tuple[str, int, str, str], Finding], finding: Finding
    ) -> tuple[str, int, str, str] | None:
        """找「同一文件、行号邻近、**换了措辞**」的已合并项（没有则返回 ``None``）。

        这是**第三道**去重：只按 ``(文件, 行号, 分类, 标题)`` 精确比对时，
        同一条问题被两个角色换种措辞、行号差几行报出来就会各留一条——
        报告里看着像两个问题，评估里第二次上报还要按误报计。

        只处理"换了措辞"的情形（标题逐字相同者一律放过）：标题一模一样但行号不同，
        更可能是同一个问题出现在**两处不同的位置**，两条都该留在报告里。
        """
        if not finding.line:
            return None
        title = finding.title.strip().lower()
        if not title:
            return None
        for key, existing in merged.items():
            if existing.file != finding.file or not existing.line:
                continue
            if abs(existing.line - finding.line) > NEAR_LINE_TOLERANCE:
                continue
            if existing.title.strip().lower() == title:
                continue
            if _title_similarity(finding.title, existing.title) >= TITLE_SIMILARITY:
                return key
        return None

    def merge_findings(self, reports: Sequence[ReviewReport]) -> list[Finding]:
        """合并多个报告的问题列表（去重 + 合并来源 + 取更严重等级）。

        去重分三档，严格按由严到宽的固定顺序找目标，找不到才新增：

        1. 精确：``(文件, 行号, 分类, 标题)`` 全同；
        2. 同行：文件 + 行号一字不差（见 :meth:`_same_line_key`）；
        3. 邻近：文件 + 行号相差 ≤ ``NEAR_LINE_TOLERANCE`` + 标题换了措辞。

        顺序不能变：先精确再放宽，避免把一条新问题错并入邻近的旧问题。
        """
        merged: dict[tuple[str, int, str, str], Finding] = {}
        for report in reports:
            for finding in report.findings:
                key = (
                    finding.file,
                    finding.line,
                    finding.category,
                    finding.title.strip().lower(),
                )
                candidate = Finding.from_dict(finding.to_dict())
                if key in merged:
                    target = key
                else:
                    target = self._same_line_key(merged, candidate) or self._near_duplicate_key(
                        merged, candidate
                    )
                current = merged.pop(target) if target is not None else None
                if current is None:
                    merged[key] = candidate
                    continue
                # SEVERITY_ORDER 数值越小越严重：保留下标更小的那一档
                if SEVERITY_ORDER.get(candidate.severity, 99) > SEVERITY_ORDER.get(
                    current.severity, 99
                ):
                    candidate.severity = current.severity
                candidate.confidence = max(current.confidence, candidate.confidence)
                candidate.source = "+".join(_merge_tokens(current.source, candidate.source))
                candidate.suggestion = candidate.suggestion or current.suggestion
                candidate.description = candidate.description or current.description
                candidate.evidence = candidate.evidence or current.evidence
                merged[key] = candidate
        return ReviewReport(findings=list(merged.values())).sorted_findings()

    def compose(
        self,
        target: str,
        reports: Sequence[ReviewReport],
        *,
        plan: ReviewPlan | None = None,
        evidence: Sequence[Evidence] | None = None,
        test_result: dict[str, Any] | None = None,
        workflow: Any = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ReviewReport:
        """组装最终报告（不调用 LLM，纯确定性合并）。"""
        evidence_items = [_as_evidence(item) for item in (evidence or [])]
        parse_errors = _parse_errors(reports)
        failed_stages = _failed_stages(workflow)
        degraded = (
            bool(parse_errors)
            or bool(failed_stages)
            or bool(plan is not None and plan.metadata.get("fallback"))
        )

        metadata: dict[str, Any] = {
            "agent": self.name,
            "sources": _source_counts(reports),
            "parse_errors": parse_errors,
            "failed_stages": failed_stages,
            "degraded": degraded,
            "evidence_count": len(evidence_items),
            "plan": plan.to_dict() if plan is not None else None,
            "test_result": test_result,
            "workflow": _as_dict(workflow),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return ReviewReport(
            target=target,
            summary=self._compose_summary(reports, degraded=degraded),
            findings=self.merge_findings(reports),
            metadata=metadata,
        )

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        target: str = "",
        *,
        reports: Sequence[ReviewReport] = (),
        plan: ReviewPlan | None = None,
        evidence: Sequence[Evidence] | None = None,
        test_result: dict[str, Any] | None = None,
        workflow: Any = None,
        extra_metadata: dict[str, Any] | None = None,
        **_: Any,
    ) -> AgentResult:
        """汇总并渲染最终报告。

        ``success=False`` 表示本次审查存在降级环节（解析失败 / 阶段失败），
        调用方据此决定是重试还是提示人工复核；报告本身仍然产出，
        因为"部分结论 + 明确的降级标记"比"什么都没有"更有用。
        """
        report = self.compose(
            target,
            reports,
            plan=plan,
            evidence=evidence,
            test_result=test_result,
            workflow=workflow,
            extra_metadata=extra_metadata,
        )
        degraded = bool(report.metadata.get("degraded"))
        markdown = report.to_markdown()
        log_event(
            logger,
            "reporter_finished",
            agent=self.name,
            findings=report.total,
            degraded=degraded,
        )
        return AgentResult(
            output=markdown,
            success=not degraded,
            error=DEGRADED_WARNING if degraded else None,
            usage=dict(ZERO_USAGE),
            metadata={
                "agent": self.name,
                "report": report.to_dict(),
                "markdown": markdown,
            },
        )

    # ------------------------------------------------------------ 内部
    def _compose_summary(self, reports: Sequence[ReviewReport], *, degraded: bool) -> str:
        parts: list[str] = []
        for report in reports:
            text = report.summary.strip()
            if text and text not in parts:
                parts.append(text)
        body = "\n\n".join(parts) if parts else "（各角色均未给出摘要）"
        counts = _source_counts(reports)
        coverage = "参与角色与问题条数：" + ("、".join(f"{name} {count}" for name, count in counts.items()) or "无")
        blocks = [body, coverage]
        if degraded:
            blocks.insert(0, DEGRADED_WARNING)
        return "\n\n".join(blocks)


# ---------------------------------------------------------------- 工具函数
def _as_evidence(item: Any) -> Evidence:
    if isinstance(item, Evidence):
        return item
    return Evidence.from_dict(item)


def _as_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _merge_tokens(*texts: str) -> list[str]:
    """合并来源标记（``"reviewer+security"`` 这类组合会被拆开去重）。"""
    tokens: list[str] = []
    for text in texts:
        for part in normalize_str_list(text):
            for token in part.split("+"):
                token = token.strip()
                if token and token not in tokens:
                    tokens.append(token)
    return tokens


def _source_counts(reports: Sequence[ReviewReport]) -> dict[str, int]:
    """各角色贡献的问题条数（按报告顺序保序）。"""
    counts: dict[str, int] = {}
    for report in reports:
        name = str(report.metadata.get("agent") or "unknown")
        counts[name] = counts.get(name, 0) + report.total
    return counts


def _parse_errors(reports: Sequence[ReviewReport]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for report in reports:
        error = report.metadata.get("parse_error")
        if error:
            errors.append(
                {"agent": str(report.metadata.get("agent") or "unknown"), "error": str(error)}
            )
    return errors


def _failed_stages(workflow: Any) -> list[str]:
    payload = _as_dict(workflow)
    if not isinstance(payload, dict):
        return []
    failed: list[str] = []
    for stage in payload.get("stages") or []:
        if isinstance(stage, dict) and str(stage.get("status")) == "failed":
            failed.append(str(stage.get("name") or "unknown"))
    return failed


__all__ = ["DEGRADED_WARNING", "ReporterAgent"]
