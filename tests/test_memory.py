"""记忆层测试：WorkingMemory（Token 预算与安全裁剪）、EpisodicMemory、NoteTool。"""

from __future__ import annotations

import json

import pytest

from codeagentx.core.message import Message, Role, ToolCall, estimate_messages_tokens
from codeagentx.memory import (
    Episode,
    EpisodicMemory,
    Note,
    NoteBook,
    NoteTool,
    WorkingMemory,
    append_jsonl,
    read_jsonl,
)

# ---------------------------------------------------------------- 辅助


def assert_pairs_intact(memory: WorkingMemory) -> None:
    """断言消息序列合法：``assistant(tool_calls)`` 与 ``tool`` 结果必须成对。"""
    pending: set[str] = set()
    for message in memory.messages:
        if message.role is Role.ASSISTANT:
            assert not pending, f"上一条 assistant 的 tool_calls 缺少结果：{sorted(pending)}"
            pending = {call.id for call in message.tool_calls}
        elif message.role is Role.TOOL:
            assert message.tool_call_id in pending, f"孤立的 tool 消息：{message.tool_call_id}"
            pending.discard(message.tool_call_id)
    assert not pending, f"assistant 的 tool_calls 缺少结果：{sorted(pending)}"


# ---------------------------------------------------------------- JSONL 落盘


class TestJsonlPersistence:
    def test_read_missing_file_returns_empty(self, tmp_path):
        assert read_jsonl(tmp_path / "not-created.jsonl") == []

    def test_append_creates_parent_dirs_and_roundtrips(self, tmp_path):
        path = tmp_path / "nested" / "memory.jsonl"
        assert append_jsonl(path, {"a": 1}) is True
        assert append_jsonl(path, {"b": "中文"}) is True
        assert read_jsonl(path) == [{"a": 1}, {"b": "中文"}]

    def test_broken_line_is_skipped(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            "{这不是 json\n" + json.dumps({"ok": True}, ensure_ascii=False) + "\n\n",
            encoding="utf-8",
        )
        assert read_jsonl(path) == [{"ok": True}]


# ---------------------------------------------------------------- WorkingMemory


class TestWorkingMemory:
    def test_rejects_invalid_budget(self):
        with pytest.raises(ValueError):
            WorkingMemory(max_tokens=0)
        with pytest.raises(ValueError):
            WorkingMemory(max_tokens=100, reserve_tokens=100)
        with pytest.raises(ValueError):
            WorkingMemory(max_tokens=100, reserve_tokens=-1)

    def test_budget_and_usage(self):
        memory = WorkingMemory(max_tokens=1000, reserve_tokens=200)
        memory.add_user("你好")
        assert memory.budget == 800
        assert memory.tokens > 0
        assert memory.describe()["roles"] == {"user": 1}

    def test_default_reserve_scales_with_window(self):
        # 小窗口不应因为固定的默认预留量而无法构造
        assert WorkingMemory(max_tokens=1000).budget == 800
        assert WorkingMemory(max_tokens=10_000).budget == 9_000

    def test_system_prompt_is_replaced_not_duplicated(self):
        memory = WorkingMemory(system_prompt="第一版提示", max_tokens=1000)
        memory.set_system_prompt("第二版提示")
        assert memory.system_prompt == "第二版提示"
        assert len(memory) == 1

    def test_message_helpers_and_openai_payload(self):
        memory = WorkingMemory(max_tokens=1000)
        memory.add_user("问题")
        memory.add_assistant("调用工具", tool_calls=[ToolCall(id="c1", name="note", arguments={})])
        memory.add_tool("已记录", "c1", name="note")
        payload = memory.to_openai()
        assert [item["role"] for item in payload] == ["user", "assistant", "tool"]
        assert payload[1]["tool_calls"][0]["function"]["name"] == "note"
        assert payload[2]["tool_call_id"] == "c1"
        assert_pairs_intact(memory)

    def test_trim_drops_oldest_and_keeps_latest(self):
        memory = WorkingMemory(max_tokens=120, reserve_tokens=20)
        for index in range(20):
            memory.add_user(f"消息{index}" + "内容" * 30)
        assert memory.tokens <= memory.budget
        assert memory.dropped_count > 0
        assert len(memory) < 20
        assert memory.messages[-1].text.startswith("消息19")

    def test_latest_message_is_never_trimmed(self):
        memory = WorkingMemory(max_tokens=60, reserve_tokens=0)
        for index in range(5):
            memory.add_user(f"第{index}条" + "x" * 400)
        assert len(memory) == 1
        assert memory.messages[-1].text.startswith("第4条")
        assert memory.tokens > memory.budget  # 无法再裁剪时只告警，不删最新消息

    def test_pinned_messages_survive_trimming(self):
        memory = WorkingMemory(system_prompt="你是代码审查员", max_tokens=200, reserve_tokens=50)
        memory.add_user("请审查登录模块", pinned=True)
        for index in range(30):
            memory.add_user(f"填充{index}" + "x" * 100)
        assert memory.system_prompt == "你是代码审查员"
        assert memory.messages[0].metadata["_pinned"] is True
        assert memory.messages[1].text == "请审查登录模块"

    def test_tool_message_cannot_be_pinned(self):
        memory = WorkingMemory(max_tokens=1000)
        with pytest.raises(ValueError):
            memory.add(Message.tool("结果", "c1", "note"), pinned=True)

    def test_pair_is_dropped_as_a_whole(self):
        """assistant 被裁掉时，其 tool 结果必须一起裁掉，否则形成非法消息序列。"""
        call = ToolCall(id="c1", name="code_search", arguments={})
        assistant = Message.assistant("调用工具", [call])
        tool = Message.tool("y" * 200, "c1", name="code_search")
        filler = Message.user("z" * 168)

        # 预算恰好等于"tool + filler"：只裁 assistant 也能满足预算，
        # 那样会把 tool 结果留成孤儿——正是本用例要拦住的错误行为。
        budget = estimate_messages_tokens([tool]) + estimate_messages_tokens([filler])
        memory = WorkingMemory(max_tokens=budget, reserve_tokens=0, auto_trim=False)
        memory.add(assistant)
        memory.add(tool)
        memory.add(filler)

        assert memory.tokens > memory.budget
        assert memory.trim() == 2
        assert [message.text for message in memory.messages] == [filler.text]
        assert_pairs_intact(memory)

    def test_pinned_assistant_keeps_its_tool_results(self):
        call = ToolCall(id="c1", name="code_search", arguments={})
        memory = WorkingMemory(max_tokens=60, reserve_tokens=0, auto_trim=False)
        memory.add(Message.assistant("调用工具", [call]), pinned=True)
        memory.add(Message.tool("检索结果" * 5, "c1", "code_search"))
        memory.add(Message.user("x" * 400))
        memory.add(Message.user("y" * 40))

        assert memory.trim() == 1  # 只裁掉较旧的那条填充消息
        assert [message.role for message in memory.messages] == [
            Role.ASSISTANT,
            Role.TOOL,
            Role.USER,
        ]
        assert_pairs_intact(memory)

    def test_auto_trim_can_be_disabled(self):
        memory = WorkingMemory(max_tokens=50, reserve_tokens=0, auto_trim=False)
        memory.add_user("x" * 400)
        assert memory.tokens > memory.budget
        assert memory.trim() == 0  # 唯一的一条就是最新消息，不裁剪

    def test_snapshot_restore_roundtrip(self):
        memory = WorkingMemory(system_prompt="系统提示", max_tokens=1000)
        memory.add_user("问题", pinned=True)
        memory.add_assistant("回答")
        snapshot = memory.snapshot()

        restored = WorkingMemory(max_tokens=1000)
        restored.restore(snapshot)
        assert [message.to_dict() for message in restored.messages] == snapshot
        assert restored.system_prompt == "系统提示"
        assert restored.messages[1].metadata.get("_pinned") is True

    def test_clear_keeps_system_prompt(self):
        memory = WorkingMemory(system_prompt="系统提示", max_tokens=1000)
        memory.add_user("问题")
        memory.clear()
        assert memory.system_prompt == "系统提示"
        assert len(memory) == 1
        memory.clear(keep_system=False)
        assert len(memory) == 0


# ---------------------------------------------------------------- EpisodicMemory


class TestEpisodicMemory:
    def test_record_validates_input(self):
        memory = EpisodicMemory()
        with pytest.raises(ValueError):
            memory.record("   ")
        with pytest.raises(ValueError):
            memory.record("审查登录模块", outcome="maybe")
        with pytest.raises(ValueError):
            EpisodicMemory(limit=0)

    def test_record_and_read_back(self):
        memory = EpisodicMemory()
        episode = memory.record(
            "审查用户登录模块",
            summary="发现 3 处问题",
            outcome="partial",
            findings=["缺少参数校验"],
            files=["pkg/service.py"],
            tags=["security"],
        )
        assert episode.episode_id.startswith("ep-")
        assert memory.get(episode.episode_id) is episode
        assert memory.get("不存在") is None
        assert len(memory) == 1
        assert Episode.from_dict(episode.to_dict()).to_dict() == episode.to_dict()

    def test_episode_text_covers_all_fields(self):
        episode = Episode(
            task="审查登录模块",
            summary="缺少校验",
            findings=["password 可为空"],
            files=["pkg/service.py"],
            tags=["security"],
        )
        text = episode.text
        for fragment in ("审查登录模块", "缺少校验", "password 可为空", "pkg/service.py", "security"):
            assert fragment in text
        rendered = episode.to_text()
        assert "结论：缺少校验" in rendered
        assert "涉及文件：pkg/service.py" in rendered

    def test_recall_prefers_related_episode(self):
        memory = EpisodicMemory()
        memory.record(
            "审查用户登录模块",
            summary="发现 login 接口缺少参数校验",
            outcome="partial",
            tags=["security"],
        )
        memory.record("审查日志落盘策略", summary="日志文件未按天轮转", tags=["maintainability"])

        hits = memory.recall("登录接口缺少校验")
        assert [episode.task for episode in hits] == ["审查用户登录模块"]

    def test_recall_filters_by_outcome_and_tags(self):
        memory = EpisodicMemory()
        first = memory.record("修复登录校验", summary="已补上 password 校验", outcome="success", tags=["security"])
        second = memory.record("修复登录校验的边界情况", summary="发现漏网分支", outcome="partial", tags=["bug"])

        assert [episode.episode_id for episode in memory.recall("登录校验", outcome="success")] == [
            first.episode_id
        ]
        assert [episode.episode_id for episode in memory.recall("登录校验", tags=["bug"])] == [
            second.episode_id
        ]
        assert memory.recall("登录校验", tags=["不存在的标签"]) == []

    def test_empty_query_returns_most_recent(self):
        memory = EpisodicMemory()
        for index in range(3):
            memory.record(f"任务{index}")
        assert [episode.task for episode in memory.recall(limit=2)] == ["任务2", "任务1"]

    def test_recall_validates_limit(self):
        with pytest.raises(ValueError):
            EpisodicMemory().recall("任意查询", limit=0)

    def test_limit_drops_oldest_episodes(self):
        memory = EpisodicMemory(limit=2)
        first = memory.record("任务一")
        memory.record("任务二")
        memory.record("任务三")
        assert len(memory) == 2
        assert memory.get(first.episode_id) is None
        assert memory.describe()["dropped"] == 1

    def test_index_follows_evicted_episodes(self):
        memory = EpisodicMemory(limit=1)
        memory.record("审查登录校验")
        memory.record("审查缓存策略")
        # 索引随条目变化自动重建，不会召回已被淘汰的内容
        assert memory.recall("登录校验") == []

    def test_describe_counts_outcomes(self):
        memory = EpisodicMemory()
        memory.record("任务一", outcome="success")
        memory.record("任务二", outcome="failure")
        memory.record("任务三", outcome="failure")
        described = memory.describe()
        assert described["episodes"] == 3
        assert described["outcomes"] == {"success": 1, "failure": 2}
        assert described["path"] is None

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "episodes.jsonl"
        memory = EpisodicMemory(path=path)
        memory.record("审查用户登录模块", summary="缺少校验", outcome="partial", tags=["security"])
        assert path.exists()

        reloaded = EpisodicMemory(path=path)
        assert len(reloaded) == 1
        assert reloaded.recall("登录模块缺少校验")[0].summary == "缺少校验"

    def test_load_skips_broken_lines(self, tmp_path):
        path = tmp_path / "episodes.jsonl"
        path.write_text(
            "不是 json\n"
            + json.dumps({"task": "审查登录模块"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        memory = EpisodicMemory(path=path)
        assert [episode.task for episode in memory] == ["审查登录模块"]

    def test_load_can_be_disabled(self, tmp_path):
        path = tmp_path / "episodes.jsonl"
        append_jsonl(path, {"task": "历史任务"})
        assert len(EpisodicMemory(path=path, load_existing=False)) == 0


# ---------------------------------------------------------------- NoteBook


class TestNoteBook:
    def test_add_validates_content(self):
        with pytest.raises(ValueError):
            NoteBook().add("   ")

    def test_list_is_newest_first(self):
        book = NoteBook()
        first = book.add("第一条", tags=["security"])
        second = book.add("第二条")
        assert [note.note_id for note in book.list()] == [second.note_id, first.note_id]
        assert [note.note_id for note in book.list(limit=1)] == [second.note_id]
        assert [note.note_id for note in book.list(tags=["security"])] == [first.note_id]

    def test_search_matches_identifiers(self):
        book = NoteBook()
        book.add("login 接口缺少 password 校验", tags=["security"])
        book.add("数据库连接池大小需要调整", tags=["performance"])
        hits = book.search("password")
        assert len(hits) == 1
        assert "password" in hits[0].content

    def test_search_validates_query_and_limit(self):
        book = NoteBook()
        book.add("任意内容")
        with pytest.raises(ValueError):
            book.search("  ")
        with pytest.raises(ValueError):
            book.search("内容", limit=0)

    def test_search_without_candidates_returns_empty(self):
        book = NoteBook()
        book.add("login 接口缺少校验", tags=["security"])
        assert book.search("login", tags=["不存在的标签"]) == []

    def test_get_and_limit(self):
        book = NoteBook(limit=2)
        first = book.add("第一条")
        book.add("第二条")
        book.add("第三条")
        assert len(book) == 2
        assert book.get(first.note_id) is None
        assert book.describe()["dropped"] == 1

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "notes.jsonl"
        book = NoteBook(path=path)
        book.add("login 缺少 password 校验", tags=["security"], source="reviewer")
        assert path.exists()

        reloaded = NoteBook(path=path)
        assert len(reloaded) == 1
        note = reloaded.search("password")[0]
        assert note.tags == ["security"]
        assert note.source == "reviewer"

    def test_note_dict_roundtrip(self):
        note = Note(
            content="内容",
            note_id="note-1",
            tags=["a"],
            source="reviewer",
            created_at="2026-01-01T00:00:00+08:00",
            metadata={"k": "v"},
        )
        assert Note.from_dict(note.to_dict()).to_dict() == note.to_dict()
        # 检索文本要覆盖正文、标签与来源，但不含 id（id 只用于定位）
        assert note.text == "内容\na\nreviewer"


# ---------------------------------------------------------------- NoteTool


class TestNoteTool:
    def test_tool_identity_and_schema(self):
        tool = NoteTool()
        assert tool.name == "note"
        assert tool.dangerous is False
        schema = tool.to_openai_schema()["function"]
        assert schema["name"] == "note"
        assert set(schema["parameters"]["properties"]) == {
            "action",
            "content",
            "query",
            "tags",
            "source",
            "limit",
        }
        assert "required" not in schema["parameters"]
        assert schema["parameters"]["properties"]["action"]["default"] == "add"

    def test_add_then_list(self):
        tool = NoteTool()
        result = tool.run(
            action="add", content="login 未校验 password", tags=["security"], source="reviewer"
        )
        assert result.success
        assert result.metadata["count"] == 1
        assert result.metadata["note"]["tags"] == ["security"]
        assert result.metadata["note"]["note_id"].startswith("note-")

        listed = tool.run(action="list")
        assert listed.success
        assert listed.metadata["count"] == 1
        assert "login 未校验 password" in listed.output
        assert "reviewer" in listed.output

    def test_default_action_is_add(self):
        tool = NoteTool()
        result = tool.run(content="默认当作 add")
        assert result.success
        assert result.metadata["count"] == 1

    def test_add_requires_content(self):
        result = NoteTool().run(action="add", content="   ")
        assert not result.success
        assert result.error_type == "ToolValidationError"

    def test_search_requires_query(self):
        result = NoteTool().run(action="search")
        assert not result.success
        assert result.error_type == "ToolValidationError"

    def test_unknown_action_is_rejected(self):
        result = NoteTool().run(action="boom")
        assert not result.success
        assert "未知的 action" in result.error

    def test_unknown_argument_is_rejected(self):
        result = NoteTool().run(action="add", content="内容", unexpected="x")
        assert not result.success
        assert result.error_type == "ToolValidationError"

    def test_search_hits_recorded_note(self):
        tool = NoteTool()
        tool.run(action="add", content="login 接口缺少 password 校验", tags=["security"])
        tool.run(action="add", content="日志未按天轮转")
        result = tool.run(action="search", query="password")
        assert result.success
        assert result.metadata["count"] == 1
        assert "password" in result.output

    def test_search_without_match_returns_friendly_text(self):
        result = NoteTool().run(action="search", query="zzz 不存在的关键词")
        assert result.success
        assert result.metadata["count"] == 0
        assert "未找到" in result.output

    def test_list_without_notes(self):
        result = NoteTool().run(action="list")
        assert result.success
        assert result.output == "暂无笔记。"

    def test_tags_accept_comma_separated_string(self):
        result = NoteTool().run(action="add", content="待确认", tags="security, p0")
        assert result.success
        assert result.metadata["note"]["tags"] == ["security", "p0"]

    def test_invalid_tags_type_is_rejected(self):
        result = NoteTool().run(action="add", content="内容", tags=123)
        assert not result.success
        assert result.error_type == "ToolValidationError"

    def test_limit_validation(self):
        tool = NoteTool()
        tool.run(action="add", content="任意内容")
        assert not tool.run(action="list", limit=0).success
        assert not tool.run(action="list", limit="x").success

    def test_limit_is_clamped(self):
        tool = NoteTool(max_limit=2)
        for index in range(4):
            tool.run(action="add", content=f"笔记{index}")
        result = tool.run(action="list", limit=100)
        assert result.metadata["count"] == 2

    def test_long_content_truncated_in_text_but_kept_in_metadata(self):
        tool = NoteTool(max_content_chars=20)
        content = "x" * 300
        tool.run(action="add", content=content)
        result = tool.run(action="list")
        assert "已截断" in result.output
        assert result.metadata["notes"][0]["content"] == content

    def test_uses_provided_notebook(self):
        book = NoteBook()
        book.add("已有笔记")
        tool = NoteTool(notebook=book)
        assert tool.run(action="list").metadata["count"] == 1
        assert tool.run(action="add", content="新笔记").metadata["count"] == 2
        assert len(book) == 2
