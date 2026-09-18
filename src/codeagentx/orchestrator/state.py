"""工作流状态：让一次多 Agent 审查可观测、可持久化、可中断恢复。

为什么需要它
------------
多 Agent 流水线的典型事故不是"某一步报错"，而是**某一步悄悄没跑**：
报告照常输出，读者以为覆盖了安全审查，其实那一步因为超时被跳过了。
所以每个阶段都必须留下状态：``pending / running / done / failed / skipped``，
并且"失败"与"跳过"要能区分（前者是异常，后者是设计如此）。

持久化策略
----------
- 每完成一个阶段就落盘一次（``save()``），进程被杀也只丢当前阶段；
- 落盘用"写临时文件 + 原子替换"，避免半截 JSON 让恢复直接失效；
- **落盘失败只记 WARNING 不抛异常**：状态丢失是降级，不该让审查任务失败
  （与记忆层的处理一致）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagentx.core.logger import get_logger, log_event

logger = get_logger("orchestrator.state")

#: 流水线阶段（顺序即执行顺序）
STAGES: tuple[str, ...] = ("plan", "retrieve", "review", "security", "test", "refactor", "report")
#: 合法阶段状态
STAGE_STATUSES: tuple[str, ...] = ("pending", "running", "done", "failed", "skipped")
#: 视为"这一步已有结果、恢复时可跳过"的状态
SETTLED_STATUSES: tuple[str, ...] = ("done", "skipped")
#: 状态文件格式版本（不匹配时拒绝恢复，避免读到结构不一致的旧文件）
STATE_VERSION = 1


@dataclass
class StageState:
    """单个阶段的状态。"""

    name: str
    status: str = "pending"
    detail: str = ""
    error: str = ""
    duration: float = 0.0

    def __post_init__(self) -> None:
        status = str(self.status or "pending").strip().lower()
        self.status = status if status in STAGE_STATUSES else "pending"
        self.detail = str(self.detail or "")
        self.error = str(self.error or "")
        self.duration = round(float(self.duration or 0.0), 3)

    @property
    def settled(self) -> bool:
        """是否已有最终结果（成功或有意跳过）。"""
        return self.status in SETTLED_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "error": self.error,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Any) -> StageState:
        if not isinstance(data, dict):
            return cls(name=str(data))
        return cls(
            name=str(data.get("name") or ""),
            status=data.get("status") or "pending",
            detail=data.get("detail") or "",
            error=data.get("error") or "",
            duration=data.get("duration") or 0.0,
        )


@dataclass
class WorkflowState:
    """一次审查流水线的完整状态（含各阶段产物，供中断恢复）。"""

    target: str = ""
    stages: list[StageState] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = STATE_VERSION

    # ------------------------------------------------------------ 构造
    @classmethod
    def create(cls, target: str = "", *, stages: Sequence[str] = STAGES) -> WorkflowState:
        return cls(target=target, stages=[StageState(name=name) for name in stages])

    # ------------------------------------------------------------ 阶段读写
    def stage(self, name: str) -> StageState:
        """取阶段状态；缺失时按 :data:`STAGES` 顺序补一个 pending。"""
        for item in self.stages:
            if item.name == name:
                return item
        created = StageState(name=name)
        self.stages.append(created)
        self._sort_stages()
        return created

    def set_stage(
        self,
        name: str,
        status: str,
        *,
        detail: str = "",
        error: str = "",
        duration: float | None = None,
    ) -> StageState:
        """更新阶段状态（``detail`` 为空时保留原值，避免覆盖有用信息）。"""
        item = self.stage(name)
        item.status = status
        if detail:
            item.detail = detail
        item.error = error
        if duration is not None:
            item.duration = round(float(duration), 3)
        log_event(
            logger,
            "stage_finished",
            level=logging.WARNING if status == "failed" else logging.INFO,
            target=self.target,
            stage=name,
            status=item.status,
            detail=item.detail,
            error=item.error,
        )
        return item

    def status_of(self, name: str) -> str:
        return self.stage(name).status

    def is_settled(self, name: str) -> bool:
        """该阶段是否已有最终结果（用于判断能否跳过）。"""
        return self.stage(name).settled

    def failed_stages(self) -> list[str]:
        return [item.name for item in self.stages if item.status == "failed"]

    def pending_stages(self) -> list[str]:
        return [item.name for item in self.stages if not item.settled]

    def is_complete(self) -> bool:
        return not self.pending_stages()

    def progress(self) -> dict[str, int]:
        counts = dict.fromkeys(STAGE_STATUSES, 0)
        for item in self.stages:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def describe(self) -> str:
        """一行摘要，便于日志与报告页脚。"""
        parts = [f"{item.name}={item.status}" for item in self.stages]
        return " | ".join(parts)

    # ------------------------------------------------------------ 序列化
    def snapshot(self) -> dict[str, Any]:
        """给报告用的轻量快照：只带阶段状态与运行元信息，**不带各阶段产物**。

        两个理由：
        1. 产物可能很大（证据里带代码片段），最终报告没必要内嵌一份；
        2. 产物里会存回报告本身，若快照里带着 ``artifacts`` 就会形成循环引用，
           导致 ``save()`` 直接失败（json 无法序列化）。
        """
        return {
            "version": self.version,
            "target": self.target,
            "stages": [item.to_dict() for item in self.stages],
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": self.target,
            "stages": [item.to_dict() for item in self.stages],
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        state = cls(
            target=str(data.get("target") or ""),
            stages=[StageState.from_dict(item) for item in (data.get("stages") or [])],
            artifacts=dict(data.get("artifacts") or {}),
            metadata=dict(data.get("metadata") or {}),
            version=int(data.get("version") or STATE_VERSION),
        )
        for name in STAGES:
            state.stage(name)
        return state

    def save(self, path: str | Path) -> bool:
        """原子落盘；失败只记 WARNING（状态丢失是降级，不该让任务失败）。"""
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(target)
            return True
        except (OSError, TypeError, ValueError) as exc:
            log_event(
                logger,
                "state_save_failed",
                level=logging.WARNING,
                path=str(target),
                error=f"{type(exc).__name__}: {exc}",
            )
            return False

    @classmethod
    def load(cls, path: str | Path) -> WorkflowState | None:
        """读取状态文件；文件缺失、损坏或版本不匹配时返回 ``None``（不抛异常）。"""
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            payload = json.loads(text)
        except ValueError as exc:
            log_event(
                logger,
                "state_load_failed",
                level=logging.WARNING,
                path=str(source),
                error=f"JSON 解析失败：{exc}",
            )
            return None
        if not isinstance(payload, dict):
            return None
        version = int(payload.get("version") or 0)
        if version != STATE_VERSION:
            log_event(
                logger,
                "state_version_mismatch",
                level=logging.WARNING,
                path=str(source),
                version=version,
                expected=STATE_VERSION,
            )
            return None
        return cls.from_dict(payload)

    # ------------------------------------------------------------ 内部
    def _sort_stages(self) -> None:
        order = {name: index for index, name in enumerate(STAGES)}
        self.stages.sort(key=lambda item: order.get(item.name, len(order)))


__all__ = [
    "SETTLED_STATUSES",
    "STAGES",
    "STAGE_STATUSES",
    "STATE_VERSION",
    "StageState",
    "WorkflowState",
]
