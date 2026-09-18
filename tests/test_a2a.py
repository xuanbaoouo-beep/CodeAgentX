"""A2A（Agent 间通信）单元测试。

覆盖三块：
1. **信封**：Part / Message / Task 的构造、序列化往返、字段推断与非法输入；
2. **名片**：AgentCard / AgentSkill 的往返与 `handles` 路由依据；
3. **路由**：注册冲突、按名字/技能/广播派发、留痕与统计，
   以及"**协议级失败抛异常、运行时失败返回 ok=False**"这条分界（同 MCP 口径）。
"""

from __future__ import annotations

import pytest

from codeagentx.core.agent import AgentResult
from codeagentx.core.exceptions import A2AError
from codeagentx.protocols.a2a import (
    A2A_PROTOCOL_VERSION,
    A2A_ROLE_AGENT,
    A2A_ROLE_USER,
    A2AErrorCode,
    A2AMessage,
    A2ANetwork,
    A2ANetworkStats,
    A2APart,
    A2AResponse,
    A2ATask,
    A2ATaskState,
    AgentCard,
    AgentSkill,
    agent_handler,
    card_from_agent,
)


# ------------------------------------------------------------------ 替身
class FakeAgent:
    """鸭子类型的 Agent 替身：只需要 ``name`` 与 ``run(task)``。"""

    def __init__(self, name: str, output: str = "ok", *, success: bool = True) -> None:
        self.name = name
        self.output = output
        self.success = success
        self.tasks: list[str] = []

    def run(self, task: str, **kwargs) -> AgentResult:
        self.tasks.append(task)
        return AgentResult(
            output=self.output,
            success=self.success,
            error=None if self.success else "模型没收敛",
            iterations=2,
            usage={"total_tokens": 33},
        )


def card(name: str, *skill_ids: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=f"{name} 的名片",
        skills=tuple(AgentSkill(id=item, name=item) for item in skill_ids),
    )


# ------------------------------------------------------------------ 1. 信封
def test_protocol_version_exposed() -> None:
    assert A2A_PROTOCOL_VERSION.count(".") == 2


def test_text_part_to_dict_and_back() -> None:
    part = A2APart.text_part("你好")
    assert part.to_dict() == {"kind": "text", "text": "你好"}
    assert A2APart.from_dict(part.to_dict()) == part
    assert part.as_text() == "你好"
    assert part.is_text


def test_data_part_serializes_to_sorted_json() -> None:
    part = A2APart.data_part({"b": 1, "a": [2]})
    assert part.to_dict() == {"kind": "data", "data": {"b": 1, "a": [2]}}
    assert part.as_text() == '{"a": [2], "b": 1}'
    assert not part.is_text


def test_part_from_dict_infers_kind() -> None:
    assert A2APart.from_dict({"data": {"x": 1}}).kind == "data"
    assert A2APart.from_dict({}).kind == "text"


def test_part_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(A2AError):
        A2APart.from_dict("not-a-part")  # type: ignore[arg-type]


def test_message_defaults_and_ids() -> None:
    message = A2AMessage.user("看看 auth.py")
    assert message.role == A2A_ROLE_USER
    assert message.message_id.startswith("msg-")
    assert message.text == "看看 auth.py"
    assert not message.is_empty


def test_message_ids_are_unique() -> None:
    assert A2AMessage.user("a").message_id != A2AMessage.user("a").message_id


def test_message_text_skips_data_parts() -> None:
    message = A2AMessage.user(
        parts=[A2APart.text_part("标题"), A2APart.data_part({"n": 1})],
    )
    assert message.text == "标题"


def test_message_with_only_data_part_is_not_empty() -> None:
    message = A2AMessage.user(parts=[A2APart.data_part({"n": 1})])
    assert message.text == ""
    assert not message.is_empty


def test_message_with_blank_text_is_empty() -> None:
    assert A2AMessage.user("   \n").is_empty


def test_message_without_parts_is_empty() -> None:
    assert A2AMessage(parts=[]).is_empty


def test_message_round_trip() -> None:
    message = A2AMessage.agent(
        "改好了",
        context_id="ctx-1",
        task_id="task-1",
        source="reviewer",
    )
    payload = message.to_dict()
    assert payload["role"] == A2A_ROLE_AGENT
    assert payload["taskId"] == "task-1"
    assert payload["contextId"] == "ctx-1"
    assert payload["metadata"] == {"source": "reviewer"}

    restored = A2AMessage.from_dict(payload)
    assert restored == message


def test_message_round_trip_omits_empty_optional_fields() -> None:
    payload = A2AMessage.user("hi").to_dict()
    assert "taskId" not in payload
    assert "contextId" not in payload
    assert "metadata" not in payload


def test_message_from_dict_rejects_bad_parts() -> None:
    with pytest.raises(A2AError):
        A2AMessage.from_dict({"parts": "not-a-list"})
    with pytest.raises(A2AError):
        A2AMessage.from_dict("not-a-message")  # type: ignore[arg-type]


def test_message_from_dict_uses_generated_id_when_missing() -> None:
    restored = A2AMessage.from_dict({"parts": [{"kind": "text", "text": "hi"}]})
    assert restored.message_id.startswith("msg-")


# ------------------------------------------------------------------ 2. 名片
def test_skill_round_trip() -> None:
    skill = AgentSkill(id="review", name="代码审查", description="看代码", tags=("py",))
    payload = skill.to_dict()
    assert payload["tags"] == ["py"]
    assert AgentSkill.from_dict(payload) == skill


def test_skill_requires_id() -> None:
    with pytest.raises(A2AError):
        AgentSkill.from_dict({"name": "没有 id"})


def test_card_round_trip_and_handles() -> None:
    original = card("reviewer", "review", "security")
    restored = AgentCard.from_dict(original.to_dict())
    assert restored == original
    assert restored.handles("review") and not restored.handles("deploy")
    assert restored.skill_ids == ("review", "security")


def test_card_requires_name() -> None:
    with pytest.raises(A2AError):
        AgentCard(name="   ")
    with pytest.raises(A2AError):
        AgentCard.from_dict({"skills": []})


def test_card_from_dict_rejects_bad_skills() -> None:
    with pytest.raises(A2AError):
        AgentCard.from_dict({"name": "x", "skills": "review"})


def test_card_omits_empty_optional_fields() -> None:
    payload = AgentCard(name="x").to_dict()
    assert set(payload) == {"name", "description", "version", "skills"}


def test_card_from_agent_uses_agent_name() -> None:
    agent = FakeAgent("refactor")
    built = card_from_agent(
        agent,  # type: ignore[arg-type]
        description="重构",
        skills=[AgentSkill(id="refactor")],
    )
    assert built.name == "refactor"
    assert built.handles("refactor")


# ------------------------------------------------------------------ 3. 任务与结果
def test_task_generates_id_and_ok_state() -> None:
    task = A2ATask(target="reviewer")
    assert task.task_id.startswith("task-")
    assert not task.ok
    task.state = A2ATaskState.COMPLETED
    assert task.ok


def test_task_text_comes_from_last_agent_message() -> None:
    task = A2ATask(target="a", messages=[A2AMessage.user("问"), A2AMessage.agent("答")])
    assert task.text == "答"
    assert A2ATask(target="a").text == ""


def test_task_round_trip_keeps_failure_fields() -> None:
    task = A2ATask(
        target="reviewer",
        state=A2ATaskState.FAILED,
        messages=[A2AMessage.user("问")],
        artifacts=[{"kind": "diff"}],
        error="坏了",
        error_code=A2AErrorCode.HANDLER_FAILED.value,
        duration=0.5,
        metadata={"k": "v"},
    )
    restored = A2ATask.from_dict(task.to_dict())
    assert restored.state is A2ATaskState.FAILED
    assert restored.error_code == A2AErrorCode.HANDLER_FAILED.value
    assert restored.artifacts == [{"kind": "diff"}]
    assert restored.metadata == {"k": "v"}
    assert restored.text == ""


def test_task_from_dict_falls_back_on_unknown_state() -> None:
    assert A2ATask.from_dict({"state": "weird"}).state is A2ATaskState.SUBMITTED
    with pytest.raises(A2AError):
        A2ATask.from_dict(42)  # type: ignore[arg-type]


def test_response_from_task_projects_fields() -> None:
    task = A2ATask(
        target="reviewer",
        state=A2ATaskState.COMPLETED,
        messages=[A2AMessage.agent("没问题")],
        artifacts=[{"kind": "report"}],
        duration=0.25,
    )
    response = A2AResponse.from_task(task)
    assert response.ok and response.text == "没问题"
    assert response.state is A2ATaskState.COMPLETED
    assert response.artifacts == ({"kind": "report"},)
    payload = response.to_dict()
    assert payload["ok"] is True and payload["taskId"] == task.task_id


def test_stats_as_dict_is_sorted() -> None:
    stats = A2ANetworkStats(sent=3, failed=1, by_target={"b": 2, "a": 1})
    assert stats.as_dict() == {"sent": 3, "failed": 1, "by_target": {"a": 1, "b": 2}}


# ------------------------------------------------------------------ 4. 注册与派发
def test_register_and_lookup() -> None:
    network = A2ANetwork()
    network.register(card("reviewer", "review"), lambda message: "ok")
    assert network.targets() == ["reviewer"]
    assert network.card("reviewer") is not None
    assert network.card("nobody") is None
    assert [item.name for item in network.cards()] == ["reviewer"]


def test_register_rejects_duplicate_unless_replaced() -> None:
    network = A2ANetwork()
    network.register(card("reviewer"), lambda message: "v1")
    with pytest.raises(A2AError):
        network.register(card("reviewer"), lambda message: "v2")

    network.register(card("reviewer"), lambda message: "v2", replace=True)
    assert network.send("reviewer", A2AMessage.user("hi")).text == "v2"


def test_register_validates_inputs() -> None:
    network = A2ANetwork()
    with pytest.raises(A2AError):
        network.register("not-a-card", lambda message: "x")  # type: ignore[arg-type]
    with pytest.raises(A2AError):
        network.register(card("x"), "not-callable")  # type: ignore[arg-type]


def test_unregister() -> None:
    network = A2ANetwork()
    network.register(card("reviewer"), lambda message: "ok")
    assert network.unregister("reviewer") is True
    assert network.unregister("reviewer") is False


def test_find_by_skill() -> None:
    network = A2ANetwork()
    network.register(card("a", "review"), lambda message: "a")
    network.register(card("b", "review", "test"), lambda message: "b")
    assert [item.name for item in network.find("review")] == ["a", "b"]
    assert [item.name for item in network.find("test")] == ["b"]
    assert network.find("deploy") == []


def test_send_returns_text_and_records_history() -> None:
    network = A2ANetwork()
    network.register(card("reviewer"), lambda message: f"收到：{message.text}")
    message = A2AMessage.user("看看 auth.py", context_id="ctx-1")

    response = network.send("reviewer", message)

    assert response.ok and response.text == "收到：看看 auth.py"
    assert response.target == "reviewer"
    assert response.state is A2ATaskState.COMPLETED
    assert len(network.history) == 1
    task = network.history[0]
    assert task.context_id == "ctx-1"
    assert [item.role for item in task.messages] == [A2A_ROLE_USER, A2A_ROLE_AGENT]
    assert task.duration >= 0


def test_send_keeps_caller_supplied_task_id() -> None:
    network = A2ANetwork()
    network.register(card("reviewer"), lambda message: "ok")
    response = network.send("reviewer", A2AMessage.user("hi", task_id="task-fixed"))
    assert response.task_id == "task-fixed"


def test_send_accepts_agent_result() -> None:
    network = A2ANetwork()
    agent = FakeAgent("reviewer", output="审完了", success=False)
    network.register(card("reviewer"), agent_handler(agent))  # type: ignore[arg-type]

    response = network.send("reviewer", A2AMessage.user("审一下"))

    assert not response.ok
    assert response.text == "审完了"  # 失败也要把输出带回来，不能让调用方看不到
    assert response.error == "模型没收敛"
    assert response.error_code == A2AErrorCode.HANDLER_FAILED.value
    assert response.metadata["iterations"] == 2
    assert response.metadata["usage"] == {"total_tokens": 33}
    assert agent.tasks == ["审一下"]


def test_send_accepts_response_object_with_artifacts() -> None:
    network = A2ANetwork()
    network.register(
        card("reporter"),
        lambda message: A2AResponse(
            ok=True,
            text="报告已生成",
            artifacts=({"kind": "report", "path": "out.md"},),
            metadata={"format": "markdown"},
        ),
    )

    response = network.send("reporter", A2AMessage.user("出报告"))

    assert response.ok and response.text == "报告已生成"
    assert response.artifacts == ({"kind": "report", "path": "out.md"},)
    assert response.metadata == {"format": "markdown"}


def test_send_reports_failing_handler_and_keeps_network_alive() -> None:
    network = A2ANetwork()

    def boom(message: A2AMessage) -> str:
        raise RuntimeError("处理器炸了")

    network.register(card("broken"), boom)
    network.register(card("good"), lambda message: "正常")

    failed = network.send("broken", A2AMessage.user("hi"))
    assert not failed.ok
    assert failed.error_code == A2AErrorCode.HANDLER_FAILED.value
    assert "RuntimeError" in (failed.error or "")

    assert network.send("good", A2AMessage.user("hi")).ok
    assert network.stats.failed == 1
    assert network.stats.sent == 2


def test_send_marks_unsupported_handler_return_type() -> None:
    network = A2ANetwork()
    network.register(card("weird"), lambda message: 42)

    response = network.send("weird", A2AMessage.user("hi"))

    assert not response.ok
    assert response.error_code == A2AErrorCode.INVALID_RESPONSE.value


def test_send_to_unknown_agent_is_runtime_failure() -> None:
    network = A2ANetwork()
    response = network.send("nobody", A2AMessage.user("hi"))
    assert not response.ok
    assert response.error_code == A2AErrorCode.AGENT_NOT_FOUND.value
    assert response.state is A2ATaskState.FAILED
    assert len(network.history) == 1


def test_send_validates_arguments() -> None:
    network = A2ANetwork()
    network.register(card("reviewer"), lambda message: "ok")
    with pytest.raises(A2AError):
        network.send("", A2AMessage.user("hi"))
    with pytest.raises(A2AError):
        network.send("reviewer", "纯字符串不是消息")  # type: ignore[arg-type]
    with pytest.raises(A2AError):
        network.send("reviewer", A2AMessage(parts=[]))


def test_send_to_skill_routes_to_first_match() -> None:
    network = A2ANetwork()
    network.register(card("a", "review"), lambda message: "a 来干")
    network.register(card("b", "review"), lambda message: "b 来干")
    response = network.send_to_skill("review", A2AMessage.user("审一下"))
    assert response.ok and response.target == "a" and response.text == "a 来干"


def test_send_to_skill_without_candidate_fails_softly() -> None:
    network = A2ANetwork()
    response = network.send_to_skill("deploy", A2AMessage.user("上线"))
    assert not response.ok
    assert response.target == "skill:deploy"
    assert response.error_code == A2AErrorCode.AGENT_NOT_FOUND.value
    with pytest.raises(A2AError):
        network.send_to_skill("deploy", "不是消息")  # type: ignore[arg-type]


def test_broadcast_to_all_and_by_skill() -> None:
    network = A2ANetwork()
    network.register(card("a", "review"), lambda message: "a")
    network.register(card("b", "test"), lambda message: "b")

    everyone = network.broadcast(A2AMessage.user("开工"))
    assert sorted(everyone) == ["a", "b"]
    assert all(item.ok for item in everyone.values())

    only_test = network.broadcast(A2AMessage.user("跑测试"), skill="test")
    assert list(only_test) == ["b"]


def test_broadcast_isolates_failures() -> None:
    network = A2ANetwork()
    network.register(card("broken"), lambda message: 1 / 0)
    network.register(card("good"), lambda message: "ok")

    results = network.broadcast(A2AMessage.user("hi"), targets=["broken", "good", "ghost"])

    assert not results["broken"].ok
    assert results["good"].ok
    assert results["ghost"].error_code == A2AErrorCode.AGENT_NOT_FOUND.value


def test_history_is_capped() -> None:
    network = A2ANetwork(max_history=2)
    network.register(card("reviewer"), lambda message: "ok")
    for index in range(4):
        network.send("reviewer", A2AMessage.user(f"第 {index} 次"))

    assert len(network.history) == 2
    assert network.history[-1].text == "ok"
    assert network.stats.sent == 4


def test_history_can_be_disabled() -> None:
    network = A2ANetwork(max_history=0)
    network.register(card("reviewer"), lambda message: "ok")
    network.send("reviewer", A2AMessage.user("hi"))
    assert network.history == []


def test_negative_max_history_rejected() -> None:
    with pytest.raises(A2AError):
        A2ANetwork(max_history=-1)


def test_stats_and_describe() -> None:
    network = A2ANetwork(name="demo")
    network.register(card("reviewer"), lambda message: "ok")
    network.send("reviewer", A2AMessage.user("hi"))
    network.send("ghost", A2AMessage.user("hi"))

    assert network.stats.as_dict() == {
        "sent": 2,
        "failed": 1,
        "by_target": {"ghost": 1, "reviewer": 1},
    }
    text = network.describe()
    assert "demo" in text and "sent=2" in text and "failed=1" in text


def test_history_is_a_copy() -> None:
    network = A2ANetwork()
    network.register(card("reviewer"), lambda message: "ok")
    network.send("reviewer", A2AMessage.user("hi"))
    network.history.clear()
    assert len(network.history) == 1
