"""A2A（Agent-to-Agent）最小实现：消息信封 + 本地路由。

为什么"最小"：A2A 规范的完整形态是"Agent Card 发现 + Task 生命周期 + HTTP/JSON
传输"，需要独立服务端进程。本项目 W7 只需要把"**Agent 之间怎么把活递过去、
结果怎么回来**"这件事讲清楚、且可离线复现，于是只实现三样东西：

1. **信封**（:class:`A2APart` / :class:`A2AMessage`）——字段与 A2A 的 ``Part`` /
   ``Message`` 对齐，可无损转 JSON；将来换成 HTTP 传输时线路格式不用改。
2. **名片**（:class:`AgentCard` / :class:`AgentSkill`）——回答"有谁能接什么活"，
   等价于 A2A 的 Agent Card，用于**按能力路由**（:meth:`A2ANetwork.find`）。
3. **路由**（:class:`A2ANetwork`）——注册／按名字派发／按技能派发／广播／留痕
   （:class:`A2ATask` 记录 ``submitted → completed|failed`` 全过程的请求与回复）。

失败处理沿用 MCP 那套口径（见 :mod:`codeagentx.protocols.mcp_server`）：

- **协议级**（API 用错：消息既没文本也没数据块、目标名不是字符串）→ 抛
  :class:`~codeagentx.core.exceptions.A2AError`；
- **运行时**（目标 Agent 没注册、处理器自己抛异常、处理器返回了不认识的类型）→
  返回 ``ok=False`` 的响应并如实带上 ``error_code``；一次派发失败不会让整张网炸掉，
  调用方能看到失败继续走。

本实现是**进程内、同步**的：不做线程池、不做超时、不联网。这两个特性属于
"传输层"，与信封/路由正交，等真有跨进程需求时再换 ``send`` 的实现即可。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from codeagentx.core.agent import AgentResult
from codeagentx.core.exceptions import A2AError
from codeagentx.core.logger import get_logger, log_event

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    from codeagentx.core.agent import Agent

logger = get_logger("protocols.a2a")

__all__ = [
    "A2A_PROTOCOL_VERSION",
    "A2A_ROLE_AGENT",
    "A2A_ROLE_USER",
    "A2AErrorCode",
    "A2AMessage",
    "A2ANetwork",
    "A2ANetworkStats",
    "A2APart",
    "A2AResponse",
    "A2ATask",
    "A2ATaskState",
    "AgentCard",
    "AgentSkill",
    "Handler",
    "agent_handler",
    "card_from_agent",
]

#: 对齐的 A2A 规范版本（本项目只实现其中的信封/名片/任务字段，不含 HTTP 传输）
A2A_PROTOCOL_VERSION = "0.2.0"

#: 消息角色：A2A 里只有"调用方"与"被调用的 Agent"两方
A2A_ROLE_USER = "user"
A2A_ROLE_AGENT = "agent"

_KIND_TEXT = "text"
_KIND_DATA = "data"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class A2AErrorCode(str, Enum):
    """派发失败的**运行时**原因（出现在 :attr:`A2AResponse.error_code`）。

    刻意不复用 MCP 的 JSON-RPC 数字码：A2A 的失败是业务级的（找不到 Agent、
    处理器抛异常），不涉及报文解析，混用数字码只会让两边都难读。
    """

    #: 目标 Agent 未注册，或没有 Agent 声明该技能
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    #: 处理器自己抛了异常（含返回了不认识的类型）
    HANDLER_FAILED = "HANDLER_FAILED"
    #: 处理器返回的类型不认识
    INVALID_RESPONSE = "INVALID_RESPONSE"


# ------------------------------------------------------------------ 信封
@dataclass(frozen=True)
class A2APart:
    """消息的一部分：文本或结构化数据（对齐 A2A 的 ``Part``）。"""

    kind: str = _KIND_TEXT
    text: str = ""
    data: Mapping[str, Any] | None = None

    @classmethod
    def text_part(cls, text: str) -> A2APart:
        return cls(kind=_KIND_TEXT, text=str(text))

    @classmethod
    def data_part(cls, data: Mapping[str, Any]) -> A2APart:
        return cls(kind=_KIND_DATA, data=dict(data))

    @property
    def is_text(self) -> bool:
        return self.kind == _KIND_TEXT

    def as_text(self) -> str:
        """统一转字符串：文本块直出，数据块转 JSON。"""
        if self.is_text:
            return self.text
        return json.dumps(dict(self.data or {}), ensure_ascii=False, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.is_text:
            payload["text"] = self.text
        else:
            payload["data"] = dict(self.data or {})
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> A2APart:
        if not isinstance(payload, Mapping):
            raise A2AError("A2A Part 必须是对象", detail=str(payload)[:200])
        kind = str(payload.get("kind") or (_KIND_DATA if "data" in payload else _KIND_TEXT))
        if kind == _KIND_DATA:
            data = payload.get("data")
            return cls(kind=_KIND_DATA, data=dict(data) if isinstance(data, Mapping) else {})
        return cls(kind=_KIND_TEXT, text=str(payload.get("text") or ""))


@dataclass
class A2AMessage:
    """一次派发的请求/回复信封。"""

    role: str = A2A_ROLE_USER
    parts: list[A2APart] = field(default_factory=list)
    message_id: str = ""
    task_id: str = ""
    context_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message_id:
            self.message_id = _new_id("msg")
        self.parts = [
            item if isinstance(item, A2APart) else A2APart.text_part(str(item))
            for item in self.parts
        ]

    # ------------------------------------------------------------ 快捷构造
    @classmethod
    def user(
        cls,
        text: str = "",
        *,
        parts: Sequence[A2APart] | None = None,
        context_id: str = "",
        task_id: str = "",
        **metadata: Any,
    ) -> A2AMessage:
        return cls(
            role=A2A_ROLE_USER,
            parts=list(parts) if parts is not None else [A2APart.text_part(text)],
            context_id=context_id,
            task_id=task_id,
            metadata=dict(metadata),
        )

    @classmethod
    def agent(
        cls,
        text: str = "",
        *,
        parts: Sequence[A2APart] | None = None,
        context_id: str = "",
        task_id: str = "",
        **metadata: Any,
    ) -> A2AMessage:
        return cls(
            role=A2A_ROLE_AGENT,
            parts=list(parts) if parts is not None else [A2APart.text_part(text)],
            context_id=context_id,
            task_id=task_id,
            metadata=dict(metadata),
        )

    # ------------------------------------------------------------ 读取
    @property
    def text(self) -> str:
        """所有文本块的拼接（数据块不参与，避免把 JSON 塞进给模型的提示里）。"""
        return "\n".join(part.text for part in self.parts if part.is_text and part.text)

    @property
    def is_empty(self) -> bool:
        """既没有非空文本、也没有数据块。"""
        return not self.text.strip() and not any(not part.is_text for part in self.parts)

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts],
            "messageId": self.message_id,
        }
        if self.task_id:
            payload["taskId"] = self.task_id
        if self.context_id:
            payload["contextId"] = self.context_id
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> A2AMessage:
        if not isinstance(payload, Mapping):
            raise A2AError("A2A 消息必须是对象", detail=str(payload)[:200])
        raw_parts = payload.get("parts") or []
        if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, (str, bytes)):
            raise A2AError("A2A 消息的 parts 必须是数组", detail=str(payload)[:200])
        metadata = payload.get("metadata")
        return cls(
            role=str(payload.get("role") or A2A_ROLE_USER),
            parts=[A2APart.from_dict(item) for item in raw_parts if isinstance(item, Mapping)],
            message_id=str(payload.get("messageId") or ""),
            task_id=str(payload.get("taskId") or ""),
            context_id=str(payload.get("contextId") or ""),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


# ------------------------------------------------------------------ 名片
@dataclass(frozen=True)
class AgentSkill:
    """Agent 声明的一项能力（用于按能力路由）。"""

    id: str
    name: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"id": self.id, "name": self.name or self.id}
        if self.description:
            payload["description"] = self.description
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AgentSkill:
        skill_id = str(payload.get("id") or "").strip()
        if not skill_id:
            raise A2AError("Agent 技能缺少 id", detail=str(payload)[:200])
        tags = payload.get("tags") or ()
        return cls(
            id=skill_id,
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            tags=tuple(str(item) for item in tags) if isinstance(tags, Sequence) else (),
        )


@dataclass(frozen=True)
class AgentCard:
    """Agent 名片：我是谁、我能接什么活（对齐 A2A 的 Agent Card）。"""

    name: str
    description: str = ""
    version: str = "1.0.0"
    url: str = ""
    skills: tuple[AgentSkill, ...] = ()
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise A2AError("Agent 名片必须有名字")

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(skill.id for skill in self.skills)

    def handles(self, skill_id: str) -> bool:
        """是否声明了某项技能。"""
        return str(skill_id) in self.skill_ids

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "skills": [skill.to_dict() for skill in self.skills],
        }
        if self.url:
            payload["url"] = self.url
        if self.capabilities:
            payload["capabilities"] = dict(self.capabilities)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AgentCard:
        if not isinstance(payload, Mapping):
            raise A2AError("Agent 名片必须是对象", detail=str(payload)[:200])
        raw_skills = payload.get("skills") or []
        if not isinstance(raw_skills, Sequence) or isinstance(raw_skills, (str, bytes)):
            raise A2AError("Agent 名片的 skills 必须是数组", detail=str(payload)[:200])
        capabilities = payload.get("capabilities")
        metadata = payload.get("metadata")
        return cls(
            name=str(payload.get("name") or "").strip(),
            description=str(payload.get("description") or ""),
            version=str(payload.get("version") or "1.0.0"),
            url=str(payload.get("url") or ""),
            skills=tuple(
                AgentSkill.from_dict(item) for item in raw_skills if isinstance(item, Mapping)
            ),
            capabilities=dict(capabilities) if isinstance(capabilities, Mapping) else {},
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


# ------------------------------------------------------------------ 任务与结果
class A2ATaskState(str, Enum):
    """任务状态。

    只保留三个状态：本项目是同步进程内派发，"排队中/处理中"这类**长任务**
    才需要的中间态在这里没有对应实现，硬加进来只会是永远走不到的死代码。
    """

    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class A2ATask:
    """一次派发的完整记录：请求消息 + 回复消息 + 结果状态。

    ``A2ANetwork.history`` 存的就是它——"谁把什么活派给了谁、结果如何"。
    """

    target: str = ""
    task_id: str = ""
    context_id: str = ""
    state: A2ATaskState = A2ATaskState.SUBMITTED
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = _new_id("task")

    @property
    def ok(self) -> bool:
        return self.state is A2ATaskState.COMPLETED

    @property
    def text(self) -> str:
        """回复正文：取最后一条 agent 消息（没有则空串）。"""
        for message in reversed(self.messages):
            if message.role == A2A_ROLE_AGENT:
                return message.text
        return ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "taskId": self.task_id,
            "target": self.target,
            "state": self.state.value,
            "messages": [message.to_dict() for message in self.messages],
            "duration": round(self.duration, 6),
        }
        if self.context_id:
            payload["contextId"] = self.context_id
        if self.artifacts:
            payload["artifacts"] = [dict(item) for item in self.artifacts]
        if self.error:
            payload["error"] = self.error
        if self.error_code:
            payload["errorCode"] = self.error_code
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> A2ATask:
        if not isinstance(payload, Mapping):
            raise A2AError("A2A 任务必须是对象", detail=str(payload)[:200])
        try:
            state = A2ATaskState(str(payload.get("state") or A2ATaskState.SUBMITTED.value))
        except ValueError:
            state = A2ATaskState.SUBMITTED
        raw_messages = payload.get("messages") or []
        artifacts = payload.get("artifacts") or []
        metadata = payload.get("metadata")
        return cls(
            target=str(payload.get("target") or ""),
            task_id=str(payload.get("taskId") or ""),
            context_id=str(payload.get("contextId") or ""),
            state=state,
            messages=[
                A2AMessage.from_dict(item) for item in raw_messages if isinstance(item, Mapping)
            ],
            artifacts=[dict(item) for item in artifacts if isinstance(item, Mapping)],
            error=str(payload.get("error") or "") or None,
            error_code=str(payload.get("errorCode") or "") or None,
            duration=float(payload.get("duration") or 0.0),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


@dataclass(frozen=True)
class A2AResponse:
    """一次派发给调用方看的结果视图（由 :class:`A2ATask` 派生）。"""

    ok: bool = False
    text: str = ""
    target: str = ""
    task_id: str = ""
    state: A2ATaskState = A2ATaskState.FAILED
    artifacts: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None
    error_code: str | None = None
    duration: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: A2ATask) -> A2AResponse:
        return cls(
            ok=task.ok,
            text=task.text,
            target=task.target,
            task_id=task.task_id,
            state=task.state,
            artifacts=tuple(task.artifacts),
            error=task.error,
            error_code=task.error_code,
            duration=task.duration,
            metadata=dict(task.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "target": self.target,
            "taskId": self.task_id,
            "state": self.state.value,
            "duration": round(self.duration, 6),
        }
        if self.text:
            payload["text"] = self.text
        if self.artifacts:
            payload["artifacts"] = [dict(item) for item in self.artifacts]
        if self.error:
            payload["error"] = self.error
        if self.error_code:
            payload["errorCode"] = self.error_code
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class A2ANetworkStats:
    """路由统计（"派了多少次、失败几次、各目标几次"）。"""

    sent: int = 0
    failed: int = 0
    by_target: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "failed": self.failed,
            "by_target": dict(sorted(self.by_target.items())),
        }


# ------------------------------------------------------------------ 路由
#: 处理器：收到请求消息，返回 ``str`` / :class:`A2AResponse` / ``AgentResult``
Handler = Callable[[A2AMessage], Any]


@dataclass
class _Registration:
    card: AgentCard
    handler: Handler


def _require_message(message: Any) -> None:
    """派发前的入参校验：类型不对或空消息属于**协议级**用错 API。"""
    if not isinstance(message, A2AMessage):
        raise A2AError("派发内容必须是 A2AMessage", detail=type(message).__name__)
    if message.is_empty:
        raise A2AError("消息既没有文本也没有数据块", detail=message.to_dict())


class A2ANetwork:
    """进程内的 Agent 路由表：注册名片 → 按名字/技能派发 → 留痕。

    用法::

        network = A2ANetwork()
        network.register(card_from_agent(agent), agent_handler(agent))
        result = network.send("reviewer", A2AMessage.user("看看 auth.py"))
        result.ok, result.text
    """

    def __init__(self, *, name: str = "local", max_history: int = 100) -> None:
        if max_history < 0:
            raise A2AError("max_history 不能为负")
        self.name = name
        #: 最多保留多少条任务记录（0 表示不留痕）
        self.max_history = max_history
        self._agents: dict[str, _Registration] = {}
        self._history: list[A2ATask] = []
        self.stats = A2ANetworkStats()

    # ------------------------------------------------------------ 注册
    def register(self, card: AgentCard, handler: Handler, *, replace: bool = False) -> None:
        """登记一个 Agent。名字重复且 ``replace=False`` 时报错，避免悄悄顶掉别人。"""
        if not isinstance(card, AgentCard):
            raise A2AError("register 需要 AgentCard", detail=type(card).__name__)
        if not callable(handler):
            raise A2AError("register 的 handler 必须可调用", detail=str(type(handler)))
        if card.name in self._agents and not replace:
            raise A2AError(f"Agent 已注册：{card.name}", detail="如需覆盖请传 replace=True")
        self._agents[card.name] = _Registration(card=card, handler=handler)

    def unregister(self, name: str) -> bool:
        """摘掉一个 Agent，返回是否真的摘掉了。"""
        return self._agents.pop(str(name), None) is not None

    def card(self, name: str) -> AgentCard | None:
        entry = self._agents.get(str(name))
        return entry.card if entry else None

    def cards(self) -> list[AgentCard]:
        return [entry.card for entry in self._agents.values()]

    def find(self, skill_id: str) -> list[AgentCard]:
        """按技能找 Agent（"谁能干这个活"）。"""
        return [entry.card for entry in self._agents.values() if entry.card.handles(skill_id)]

    # ------------------------------------------------------------ 派发
    def send(self, target: str, message: A2AMessage) -> A2AResponse:
        """把消息派给指定 Agent。

        返回值永远不抛"运行时"异常：目标不存在或处理器失败都会拿到 ``ok=False``
        的响应（见模块 docstring）；只有调用方用错 API 才抛 :class:`A2AError`。
        """
        if not isinstance(target, str) or not target.strip():
            raise A2AError("派发目标必须是非空字符串", detail=repr(target))
        _require_message(message)

        target = target.strip()
        task = A2ATask(
            target=target,
            task_id=message.task_id or "",
            context_id=message.context_id,
            messages=[message],
        )
        started = time.perf_counter()

        entry = self._agents.get(target)
        if entry is None:
            task.state = A2ATaskState.FAILED
            task.error_code = A2AErrorCode.AGENT_NOT_FOUND.value
            task.error = f"未注册的 Agent：{target}"
            return self._record(task, started)

        try:
            self._apply_outcome(task, entry.handler(message), message)
        except Exception as exc:  # noqa: BLE001 - 一个 Agent 崩掉不该带垮整张网（同 AD-45）
            logger.warning("Agent %s 处理失败：%s", target, exc)
            task.state = A2ATaskState.FAILED
            task.error_code = (
                A2AErrorCode.INVALID_RESPONSE.value
                if isinstance(exc, TypeError)
                else A2AErrorCode.HANDLER_FAILED.value
            )
            task.error = f"{type(exc).__name__}: {exc}"
        return self._record(task, started)

    def send_to_skill(self, skill_id: str, message: A2AMessage) -> A2AResponse:
        """按技能派发：取第一个声明该技能的 Agent；没人声明则如实失败。"""
        _require_message(message)
        candidates = self.find(skill_id)
        if not candidates:
            task = A2ATask(
                target=f"skill:{skill_id}",
                task_id=message.task_id,
                state=A2ATaskState.FAILED,
                error_code=A2AErrorCode.AGENT_NOT_FOUND.value,
                error=f"没有 Agent 声明技能：{skill_id}",
            )
            return self._record(task, time.perf_counter())
        return self.send(candidates[0].name, message)

    def broadcast(
        self,
        message: A2AMessage,
        *,
        targets: Iterable[str] | None = None,
        skill: str | None = None,
    ) -> dict[str, A2AResponse]:
        """一次发给多个 Agent；某个失败不影响其余（返回表里逐项带状态）。

        Args:
            targets: 指定目标名；与 ``skill`` 二选一，都不给则发给全部已注册 Agent。
            skill: 按技能筛选目标。
        """
        if targets is not None:
            names = [str(item) for item in targets]
        elif skill is not None:
            names = [card.name for card in self.find(skill)]
        else:
            names = self.targets()
        return {name: self.send(name, message) for name in names}

    def targets(self) -> list[str]:
        return list(self._agents)

    @property
    def history(self) -> list[A2ATask]:
        return list(self._history)

    # ------------------------------------------------------------ 内部
    def _apply_outcome(self, task: A2ATask, outcome: Any, message: A2AMessage) -> None:
        """把处理器的返回值归一化到任务记录上。"""
        artifacts: Sequence[Mapping[str, Any]] = ()
        metadata: dict[str, Any] = {}
        if isinstance(outcome, A2AResponse):
            task.state = A2ATaskState.COMPLETED if outcome.ok else A2ATaskState.FAILED
            task.error = outcome.error
            task.error_code = outcome.error_code
            artifacts = outcome.artifacts
            metadata = dict(outcome.metadata)
            text = outcome.text
        elif isinstance(outcome, AgentResult):
            task.state = A2ATaskState.COMPLETED if outcome.success else A2ATaskState.FAILED
            task.error = outcome.error
            task.error_code = None if outcome.success else A2AErrorCode.HANDLER_FAILED.value
            metadata = {"iterations": outcome.iterations, "usage": dict(outcome.usage)}
            artifacts = outcome.metadata.get("artifacts") or ()
            text = outcome.output
        elif isinstance(outcome, str):
            task.state = A2ATaskState.COMPLETED
            text = outcome
        else:
            raise TypeError(f"处理器返回类型不支持：{type(outcome).__name__}")

        task.artifacts = [dict(item) for item in artifacts if isinstance(item, Mapping)]
        task.metadata.update(metadata)
        task.messages.append(
            A2AMessage.agent(
                text,
                task_id=task.task_id,
                context_id=message.context_id,
                source=task.target,
            )
        )

    def _record(self, task: A2ATask, started: float) -> A2AResponse:
        task.duration = time.perf_counter() - started
        if self.max_history:
            self._history.append(task)
            if len(self._history) > self.max_history:
                del self._history[: len(self._history) - self.max_history]
        self.stats.sent += 1
        self.stats.by_target[task.target] = self.stats.by_target.get(task.target, 0) + 1
        if not task.ok:
            self.stats.failed += 1
        log_event(
            logger,
            "a2a.send",
            target=task.target,
            ok=task.ok,
            state=task.state.value,
            duration=round(task.duration, 4),
            error=task.error,
        )
        return A2AResponse.from_task(task)

    def describe(self) -> str:
        return (
            f"A2ANetwork({self.name}) agents={len(self._agents)} "
            f"sent={self.stats.sent} failed={self.stats.failed}"
        )


# ------------------------------------------------------------------ 与 core.Agent 的桥
def agent_handler(agent: Agent) -> Handler:
    """把 :class:`~codeagentx.core.agent.Agent` 包成 A2A 处理器。

    只做"信封文本 → ``agent.run(task)``"这一件事：Agent 的运行细节
    （工具循环、迭代上限）由 Agent 自己管，路由层不该插手。
    """

    def handler(message: A2AMessage) -> AgentResult:
        return agent.run(message.text)

    return handler


def card_from_agent(
    agent: Agent,
    *,
    description: str = "",
    skills: Sequence[AgentSkill] = (),
    url: str = "",
    capabilities: Mapping[str, Any] | None = None,
) -> AgentCard:
    """用 :class:`~codeagentx.core.agent.Agent` 的名字生成名片。

    技能必须显式给：从 Agent 类里"猜"能力只会猜错，名片写错了比没有更糟。
    """
    return AgentCard(
        name=str(getattr(agent, "name", "") or ""),
        description=description,
        url=url,
        skills=tuple(skills),
        capabilities=dict(capabilities or {}),
    )
