"""上下文工程（W7）测试：schemas / compressor / builder。

三块分别对应"数据结构对不对""压得够不够、压完还在不在预算内""GSSC 流水线是否稳"。
两条硬约束是重点，且都必须**按实际渲染结果**验证，不能只看文档估算：
- 压缩后 token ≤ 预算（``compress_documents``）；
- 构建后 ``BuiltContext.text`` 的 token ≤ 预算（``build``），压不动时如实标 ``False``。

测试刻意不依赖真实索引与检索：检索器用鸭子类型替身，文件用 ``tmp_path``，
保证确定性、可离线复现；真实 RAG 联调放在端到端示例里做。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from codeagentx.context.builder import (
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_MAX_GITHUB_FILES,
    STAGES,
    UNTRUSTED_NOTICE,
    ContextBuilder,
    GitHubSource,
    build_context_builder,
    document_from_retrieved,
    documents_from_github,
    github_file_to_document,
    in_text_suffixes,
)
from codeagentx.context.compressor import (
    ELISION_MARKER,
    LEVEL_BUDGET,
    LEVEL_DEDUPE,
    LEVEL_MERGE,
    LEVEL_TRUNCATE,
    CompressionReport,
    ContextCompressor,
    LevelReport,
    hard_fit,
)
from codeagentx.context.schemas import (
    SECTION_CONSTRAINTS,
    SECTION_EVIDENCE,
    SECTION_TASK,
    BuiltContext,
    ContextDocument,
    ContextSection,
    ContextStats,
    StageStat,
    document_identity,
    estimate_documents_tokens,
    estimate_sections_tokens,
    render_sections,
)
from codeagentx.core.exceptions import (
    GitHubAuthError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    SecurityViolationError,
    ToolValidationError,
)
from codeagentx.protocols.github_client import GitHubFile, GitHubTree, GitHubTreeEntry
from codeagentx.tools.sandbox import SandboxPolicy

TASK = "审查登录与鉴权实现，找出逻辑漏洞与安全风险"
#: 12 行纯 ASCII 正文：token 估算 ≈ 字符数 / 4，便于精确构造预算
BODY = "\n".join(f"line_{index} = compute({index})" for index in range(12))


def make_document(
    index: int = 0,
    *,
    path: str = "app/auth/service.py",
    name: str = "login",
    kind: str = "function",
    score: float = 0.5,
    content: str = BODY,
    start_line: int | None = None,
    end_line: int | None = None,
    **kwargs,
) -> ContextDocument:
    start = 10 + index * 20 if start_line is None else start_line
    return ContextDocument(
        doc_id=kwargs.pop("doc_id", f"{path}:{start}"),
        content=content,
        path=path,
        start_line=start,
        end_line=start + content.count("\n") if end_line is None else end_line,
        kind=kind,
        name=name,
        score=score,
        **kwargs,
    )


class FakeRetriever:
    """检索器替身：只要具备 ``retrieve`` 或 ``search`` 就应被接受。"""

    def __init__(self, chunks, *, method: str = "retrieve") -> None:
        self.chunks = list(chunks)
        self.calls: list[tuple[str, int]] = []
        setattr(self, method, self._search)

    def _search(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return list(self.chunks)


# ================================================================== 1. 数据结构
def test_document_location_and_header() -> None:
    document = make_document(0, start_line=10, end_line=21, name="login")
    assert document.location == "app/auth/service.py:10-21"
    assert document.header == "[app/auth/service.py:10-21] function login"

    without_lines = ContextDocument(doc_id="x", content="c", path="a.py")
    assert without_lines.location == "a.py"
    assert without_lines.header == "[a.py]"

    anonymous = ContextDocument(doc_id="x", content="c")
    assert anonymous.location == ""
    assert anonymous.header == "[无位置信息]"


def test_document_header_marks_non_retrieval_source() -> None:
    document = make_document(0, source="mcp")
    assert "来源=mcp" in document.header
    assert "来源" not in make_document(1, source="retrieval").header


def test_document_to_text_and_token_estimate() -> None:
    document = make_document(0, start_line=1, end_line=12, path="a.py", name="f")
    assert document.to_text(with_header=False) == BODY
    assert document.to_text().startswith("[a.py:1-12] function f\n")
    assert document.token_estimate() > document.token_estimate(with_header=False)


def test_document_derived_copies_do_not_mutate_original() -> None:
    document = make_document(0, metadata={"query": "登录"})
    replaced = document.with_content("新正文", elided_lines=3)
    assert replaced.content == "新正文"
    assert replaced.metadata == {"query": "登录", "elided_lines": 3}
    assert document.content == BODY and "elided_lines" not in document.metadata

    assert document.with_score(0.9).score == 0.9
    assert document.with_section(SECTION_CONSTRAINTS).section == SECTION_CONSTRAINTS
    assert document.score == 0.5


def test_document_to_dict_can_drop_content() -> None:
    document = make_document(0, start_line=1, end_line=12)
    full = document.to_dict()
    assert full["location"] == "app/auth/service.py:1-12"
    assert full["tokens"] > 0 and full["content"] == BODY
    assert "content" not in document.to_dict(with_content=False)


def test_document_identity_prefers_location() -> None:
    first = make_document(0, start_line=10, end_line=21, doc_id="retrieval::1")
    second = make_document(0, start_line=10, end_line=21, doc_id="mcp::9")
    assert document_identity(first) == document_identity(second)

    id_only = ContextDocument(doc_id="note::1", content="c")
    assert document_identity(id_only) == ("id", "note::1")
    assert document_identity(ContextDocument(doc_id="", content="c")) == ("content", "c")


def test_section_rendering_includes_headings_and_note() -> None:
    section = ContextSection(
        title=SECTION_EVIDENCE,
        documents=(make_document(0, start_line=1, end_line=12),),
        note="说明",
    )
    text = section.to_text()
    assert text.startswith(f"## {SECTION_EVIDENCE}\n\n说明")
    assert section.to_text(headings=False).startswith("说明")
    assert section.tokens() == estimate_sections_tokens([section])


def test_section_tokens_count_headings_and_note() -> None:
    empty = ContextSection(title=SECTION_TASK, note="审查任务：登录")
    with_documents = ContextSection(title=SECTION_TASK, note="审查任务：登录", documents=())
    assert empty.tokens() == with_documents.tokens()
    assert empty.tokens() > 0

    filled = empty.with_documents([make_document(0, start_line=1, end_line=12)])
    assert filled.tokens() > empty.tokens()
    assert filled.to_dict()["documents"] == 1


def test_render_sections_joins_blocks_with_blank_line() -> None:
    sections = [
        ContextSection(title=SECTION_TASK, note="任务说明"),
        ContextSection(title="补充材料", note="更多"),
    ]
    assert render_sections(sections) == (
        f"## {SECTION_TASK}\n\n任务说明\n\n## 补充材料\n\n更多"
    )
    assert render_sections([]) == ""


def test_stage_stat_and_context_stats() -> None:
    stat = StageStat(name="gather", documents_out=3, tokens=42, duration=0.123456, detail="x")
    payload = stat.as_dict()
    assert payload["duration"] == 0.123
    assert payload["documents_out"] == 3

    stats = ContextStats(budget=100, tokens=80, documents=2, stages=[stat])
    assert stats.stage("gather") is stat
    assert stats.stage("nope") is None
    assert stats.as_dict()["stages"][0]["name"] == "gather"


def test_built_context_renders_from_sections() -> None:
    sections = [
        ContextSection(title=SECTION_TASK, note="任务"),
        ContextSection(title=SECTION_EVIDENCE, documents=(make_document(0),)),
    ]
    built = BuiltContext(task="t", sections=sections, stats=ContextStats(budget=999, tokens=1))
    assert built.text == render_sections(sections)
    assert built.tokens == 1 and built.budget == 999 and built.within_budget
    assert len(built.documents) == 1

    payload = built.to_dict(with_text=False)
    assert "text" not in payload
    assert payload["sections"][1]["documents"] == 1
    assert built.to_dict(with_content=True)["sections"][1]["items"][0]["content"] == BODY


# ================================================================== 2. 压缩器
def test_compressor_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        ContextCompressor(max_lines_per_document=0)
    with pytest.raises(ValueError):
        ContextCompressor(head_lines=-1)
    with pytest.raises(ValueError):
        ContextCompressor(max_lines_per_document=10, head_lines=8, tail_lines=4)


def test_compressor_describe_and_budget_validation() -> None:
    compressor = ContextCompressor(max_lines_per_document=50, head_lines=30, tail_lines=10)
    assert compressor.describe() == {
        "max_lines_per_document": 50,
        "head_lines": 30,
        "tail_lines": 10,
    }
    with pytest.raises(ValueError):
        compressor.compress_documents([make_document(0)], budget_tokens=0)


def test_dedupe_keeps_highest_score_and_drops_blank() -> None:
    kept, report = ContextCompressor().compress_documents(
        [
            make_document(0, score=0.1),
            make_document(0, doc_id="dup", score=0.9),
            make_document(1, path="blank.py", content="   \n"),
        ],
        budget_tokens=10_000,
    )
    assert [document.score for document in kept] == [0.9]
    assert report.level(LEVEL_DEDUPE).applied is True


def test_merge_adjacent_fragments_of_same_symbol() -> None:
    first = make_document(0, start_line=10)
    second = make_document(
        1, start_line=first.end_line + 1, end_line=first.end_line + 1 + BODY.count("\n")
    )
    merged, _ = ContextCompressor().compress_documents([first, second], budget_tokens=10_000)

    assert len(merged) == 1
    assert merged[0].start_line == first.start_line
    assert merged[0].end_line == second.end_line
    assert merged[0].metadata["merged_count"] == 2
    assert merged[0].doc_id == first.doc_id


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "logout"},  # 不同符号
        {"path": "app/auth/other.py"},  # 不同文件
        {"source": "mcp"},  # 不同来源
        {"kind": "class"},  # 不同粒度
    ],
)
def test_merge_never_crosses_symbol_boundaries(overrides: dict) -> None:
    first = make_document(0, start_line=10)
    second = make_document(1, start_line=first.end_line + 1, **overrides)
    merged, _ = ContextCompressor().compress_documents([first, second], budget_tokens=10_000)
    assert len(merged) == 2


def test_merge_requires_contiguous_lines() -> None:
    first = make_document(0, start_line=10)
    second = make_document(1, start_line=first.end_line + 5)
    merged, _ = ContextCompressor().compress_documents([first, second], budget_tokens=10_000)
    assert len(merged) == 2


def test_truncate_keeps_head_and_tail_with_marker() -> None:
    compressor = ContextCompressor(max_lines_per_document=10, head_lines=6, tail_lines=2)
    long_text = "\n".join(f"print({index})" for index in range(20))
    kept, report = compressor.compress_documents(
        [make_document(0, content=long_text, start_line=1)], budget_tokens=10_000
    )

    assert report.truncated_documents == 1
    assert report.level(LEVEL_TRUNCATE).applied is True
    content = kept[0].content
    assert content.startswith("print(0)")
    assert content.endswith("print(19)")
    assert ELISION_MARKER.format(elided=12) in content
    assert kept[0].metadata["elided_lines"] == 12


def test_truncate_leaves_short_document_untouched() -> None:
    compressor = ContextCompressor(max_lines_per_document=12, head_lines=6, tail_lines=2)
    kept, report = compressor.compress_documents([make_document(0)], budget_tokens=10_000)
    assert report.truncated_documents == 0
    assert kept[0].content == BODY


def test_budget_keeps_high_score_and_reports_drops() -> None:
    documents = [
        make_document(0, path="a.py", score=0.9),
        make_document(1, path="b.py", score=0.5),
        make_document(2, path="c.py", score=0.1),
    ]
    cost = estimate_documents_tokens([documents[0]])
    kept, report = ContextCompressor().compress_documents(
        documents, budget_tokens=cost + 1
    )

    assert [document.path for document in kept] == ["a.py"]
    assert report.dropped_documents == 2
    assert report.after_tokens <= cost + 1
    assert report.within_budget


def test_budget_preserves_original_order() -> None:
    documents = [make_document(index, path=f"file_{index}.py", score=0.9) for index in range(4)]
    kept, _ = ContextCompressor().compress_documents(documents, budget_tokens=10_000)
    assert [document.path for document in kept] == [f"file_{index}.py" for index in range(4)]


def test_tiny_budget_hard_trims_instead_of_returning_nothing() -> None:
    documents = [make_document(index, path=f"file_{index}.py") for index in range(6)]
    kept, report = ContextCompressor().compress_documents(documents, budget_tokens=60)

    assert len(kept) == 1
    assert kept[0].metadata["hard_trimmed"] is True
    assert report.hard_trimmed == 1
    assert report.after_tokens <= 60
    assert report.within_budget


def test_hard_fit_gives_up_when_budget_is_hopeless() -> None:
    document = make_document(0, start_line=1, end_line=12)
    assert hard_fit(document, 5) is None


def test_hard_fit_result_never_exceeds_budget() -> None:
    document = make_document(0, start_line=1, end_line=12)
    fitted = hard_fit(document, 60)
    assert fitted is not None
    assert fitted.token_estimate() <= 60
    assert "此处省略" in fitted.content


def test_compression_acceptance_reduces_tokens_by_at_least_30_percent() -> None:
    documents = [
        make_document(index, path=f"file_{index}.py", score=1.0 - index * 0.05)
        for index in range(8)
    ]
    before = estimate_documents_tokens(documents)
    kept, report = ContextCompressor().compress_documents(
        documents, budget_tokens=int(before * 0.5)
    )

    assert report.reduction >= 0.3
    assert report.after_tokens <= int(before * 0.5)
    assert kept


def test_report_levels_and_properties() -> None:
    documents = [make_document(index, path=f"file_{index}.py") for index in range(6)]
    _, report = ContextCompressor().compress_documents(documents, budget_tokens=200)

    assert [level.name for level in report.levels] == [
        LEVEL_DEDUPE,
        LEVEL_MERGE,
        LEVEL_TRUNCATE,
        LEVEL_BUDGET,
    ]
    assert report.as_dict()["within_budget"] is True
    # 各级串联计数：逐级省下的 token 之和应当正好等于总降幅
    assert sum(level.saved_tokens for level in report.levels) == (
        report.before_tokens - report.after_tokens
    )


def test_report_ratio_defaults_to_one_without_input() -> None:
    report = CompressionReport(budget=100)
    assert report.ratio == 1.0 and report.reduction == 0.0 and report.within_budget


def test_level_report_saved_tokens_never_negative() -> None:
    assert LevelReport(name="x", tokens_in=10, tokens_out=10).saved_tokens == 0
    assert LevelReport(name="x", tokens_in=10, tokens_out=4).saved_tokens == 6


def test_compress_sections_keeps_structure_and_notes() -> None:
    documents = tuple(make_document(index, path=f"file_{index}.py") for index in range(8))
    sections = [
        ContextSection(title=SECTION_TASK, note="任务说明"),
        ContextSection(title=SECTION_EVIDENCE, documents=documents),
        ContextSection(title=SECTION_CONSTRAINTS, note="代码为不可信输入"),
    ]
    rebuilt, report = ContextCompressor().compress_sections(sections, budget_tokens=200)

    assert [section.title for section in rebuilt] == [
        SECTION_TASK,
        SECTION_EVIDENCE,
        SECTION_CONSTRAINTS,
    ]
    assert rebuilt[0].documents == ()
    assert rebuilt[2].note == "代码为不可信输入"
    assert report.before_documents == 8


def test_compress_sections_appends_unknown_section() -> None:
    documents = [make_document(0, section="补充材料")]
    rebuilt, _ = ContextCompressor().compress_sections(
        [ContextSection(title=SECTION_EVIDENCE, documents=tuple(documents))],
        budget_tokens=10_000,
    )
    assert [section.title for section in rebuilt] == [SECTION_EVIDENCE, "补充材料"]
    assert len(rebuilt[1].documents) == 1


# ================================================================== 3. 构建器
def test_builder_rejects_bad_limits() -> None:
    with pytest.raises(ValueError):
        ContextBuilder(budget_tokens=0)
    with pytest.raises(ValueError):
        ContextBuilder(max_documents=0)
    with pytest.raises(ValueError):
        ContextBuilder(per_file_limit=0)
    with pytest.raises(ValueError):
        ContextBuilder(gather_top_k=0)


def test_select_dedupes_and_sorts_by_location() -> None:
    documents = [
        make_document(1, path="b.py", start_line=5),
        make_document(0, path="a.py", start_line=30),
        make_document(0, path="a.py", start_line=30, doc_id="dup"),
        make_document(2, path="a.py", start_line=10),
        make_document(3, path="blank.py", content=""),
    ]
    selected = ContextBuilder().select(documents)

    assert [(item.path, item.start_line) for item in selected] == [
        ("a.py", 10),
        ("a.py", 30),
        ("b.py", 5),
    ]


def test_select_applies_score_threshold_but_keeps_unscored() -> None:
    scored_low = make_document(0, path="low.py", score=0.1)
    scored_high = make_document(1, path="high.py", score=0.9)
    unscored = make_document(2, path="note.py", score=0.0)

    selected = ContextBuilder(min_score=0.5).select([scored_low, scored_high, unscored])
    assert [item.path for item in selected] == ["high.py", "note.py"]


def test_select_limits_per_file_and_total() -> None:
    builder = ContextBuilder(max_documents=3, per_file_limit=2)
    documents = [make_document(index, path="same.py", score=0.5) for index in range(3)]
    documents += [make_document(index, path="other.py", score=0.4) for index in range(3)]

    selected = builder.select(documents)
    assert len(selected) == 3
    assert sum(1 for item in selected if item.path == "same.py") == 2
    assert sum(1 for item in selected if item.path == "other.py") == 1


def test_select_rejects_zero_limits() -> None:
    with pytest.raises(ValueError):
        ContextBuilder().select([make_document(0)], max_documents=0)
    with pytest.raises(ValueError):
        ContextBuilder().select([make_document(0)], per_file_limit=0)


def test_gather_from_notes_skips_blank_and_marks_section() -> None:
    documents = ContextBuilder().gather_from_notes(["  项目约定：必须校验入参 ", ""])
    assert len(documents) == 1
    assert documents[0].section == SECTION_CONSTRAINTS
    assert documents[0].source == "note"
    assert documents[0].content == "项目约定：必须校验入参"


def test_gather_from_retriever_uses_each_query() -> None:
    retriever = FakeRetriever(
        [{"chunk_id": "c1", "content": BODY, "path": "a.py", "start_line": 1, "end_line": 12}]
    )
    documents = ContextBuilder().gather_from_retriever(
        retriever, ["登录校验", "登录校验", "token 校验"], top_k=3
    )

    assert retriever.calls == [("登录校验", 3), ("token 校验", 3)]
    assert len(documents) == 2
    assert documents[0].metadata["query"] == "登录校验"


def test_gather_from_retriever_accepts_search_only_object() -> None:
    retriever = FakeRetriever([], method="search")
    assert ContextBuilder().gather_from_retriever(retriever, "查询") == []


def test_gather_from_retriever_requires_queries_and_method() -> None:
    retriever = FakeRetriever([])
    assert ContextBuilder().gather_from_retriever(retriever, None) == []
    with pytest.raises(ToolValidationError):
        ContextBuilder().gather_from_retriever(object(), "查询")


def test_gather_from_files_requires_policy() -> None:
    with pytest.raises(ToolValidationError):
        ContextBuilder().gather_from_files(["a.py"])


def test_gather_from_files_reads_and_skips_missing(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "service.py").write_text(BODY, encoding="utf-8")

    builder = ContextBuilder(policy=SandboxPolicy.for_roots([tmp_path]))
    documents = builder.gather_from_files(["app/service.py", "缺失.py", "app"])

    assert len(documents) == 1
    assert documents[0].path == "app/service.py"
    assert documents[0].source == "file"
    assert documents[0].metadata["truncated"] is False


def test_gather_from_files_truncates_large_file(tmp_path: Path) -> None:
    target = tmp_path / "big.py"
    target.write_text("\n".join(f"line_{index} = {index}" for index in range(50)), encoding="utf-8")

    builder = ContextBuilder(
        policy=SandboxPolicy.for_roots([tmp_path]), max_file_bytes=40
    )
    document = builder.gather_from_files(["big.py"])[0]

    assert document.metadata["truncated"] is True
    assert "文件过大" in document.content
    assert len(document.content) < len(target.read_text(encoding="utf-8"))


def test_gather_from_files_raises_on_escape(tmp_path: Path) -> None:
    builder = ContextBuilder(policy=SandboxPolicy.for_roots([tmp_path]))
    with pytest.raises(SecurityViolationError):
        builder.gather_from_files(["../.env"])


def test_structure_always_adds_untrusted_notice() -> None:
    builder = ContextBuilder()
    sections = builder.structure([], task=TASK)

    assert [section.title for section in sections] == [
        SECTION_TASK,
        SECTION_EVIDENCE,
        SECTION_CONSTRAINTS,
    ]
    assert UNTRUSTED_NOTICE in sections[2].note
    assert TASK in sections[0].note


def test_structure_appends_instructions_and_keeps_custom_section_order() -> None:
    builder = ContextBuilder()
    documents = [make_document(0, section="补充材料")]
    sections = builder.structure(documents, task=TASK, instructions="只报高危问题")

    assert [section.title for section in sections] == [
        SECTION_TASK,
        SECTION_EVIDENCE,
        "补充材料",
        SECTION_CONSTRAINTS,
    ]
    assert sections[-1].note.endswith("只报高危问题")


def test_build_runs_all_stages_within_budget() -> None:
    documents = [
        make_document(index, path=f"file_{index}.py", score=1.0 - index * 0.05)
        for index in range(10)
    ]
    builder = ContextBuilder(budget_tokens=400, max_documents=10)
    built = builder.build(TASK, documents=documents, notes=["项目约定：校验入参"])

    assert [stage.name for stage in built.stats.stages] == list(STAGES)
    assert built.within_budget
    assert built.tokens <= 400
    assert built.stats.tokens == estimate_sections_tokens(built.sections)
    assert built.stats.documents == len(built.documents)
    assert built.stats.compression["attempts"] >= 1
    assert built.stats.dropped_documents >= 0


def test_build_reports_stage_details() -> None:
    built = ContextBuilder(budget_tokens=500).build(
        TASK,
        documents=[make_document(0, source="mcp")],
        notes=["约定"],
    )

    gather = built.stats.stage("gather")
    assert gather.detail == "mcp 1 / note 1"
    assert "丢弃" in built.stats.stage("select").detail
    assert "降幅" in built.stats.stage("compress").detail
    assert built.stats.stage("structure").detail.startswith("3 节")


def test_build_is_honest_when_budget_is_impossible() -> None:
    documents = [make_document(index, path=f"file_{index}.py") for index in range(5)]
    built = ContextBuilder(budget_tokens=40).build(TASK, documents=documents)

    assert not built.within_budget
    assert built.tokens > 40
    assert built.stats.within_budget is False


def test_build_without_candidates_still_returns_skeleton() -> None:
    built = ContextBuilder(budget_tokens=300).build(TASK)

    assert built.stats.documents == 0
    assert built.within_budget
    assert [section.title for section in built.sections] == [
        SECTION_TASK,
        SECTION_EVIDENCE,
        SECTION_CONSTRAINTS,
    ]
    assert UNTRUSTED_NOTICE in built.text


def test_build_requires_task() -> None:
    with pytest.raises(ValueError):
        ContextBuilder().build("   ")


def test_build_rejects_non_positive_budget_override() -> None:
    with pytest.raises(ValueError):
        ContextBuilder().build(TASK, budget_tokens=-1)


def test_build_puts_notes_into_constraints_section() -> None:
    built = ContextBuilder(budget_tokens=400).build(TASK, notes=["项目约定：必须校验入参"])
    constraints = built.sections[-1]
    assert constraints.title == SECTION_CONSTRAINTS
    assert [document.content for document in constraints.documents] == ["项目约定：必须校验入参"]


def test_build_reads_files_from_sandbox(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(BODY, encoding="utf-8")
    builder = ContextBuilder(
        budget_tokens=800, policy=SandboxPolicy.for_roots([tmp_path])
    )
    built = builder.build(TASK, files=["service.py"])

    assert [document.source for document in built.documents] == ["file"]
    assert built.within_budget


def test_build_uses_retriever_queries() -> None:
    retriever = FakeRetriever(
        [{"chunk_id": "c1", "content": BODY, "path": "a.py", "start_line": 1, "end_line": 12}]
    )
    built = ContextBuilder(budget_tokens=600).build(TASK, queries=["登录校验"], retriever=retriever)

    assert retriever.calls == [("登录校验", 8)]
    assert built.stats.documents == 1


def test_build_context_builder_reads_config() -> None:
    config = SimpleNamespace(context_budget_tokens=123, context_max_documents=3)
    builder = build_context_builder(config, policy=None)
    assert builder.budget_tokens == 123
    assert builder.max_documents == 3
    assert builder.describe()["sandbox"] is None


def test_build_context_builder_overrides_and_describe(tmp_path: Path) -> None:
    config = SimpleNamespace(context_budget_tokens=123, context_max_documents=3)
    policy = SandboxPolicy.for_roots([tmp_path])
    builder = build_context_builder(config, policy=policy, budget_tokens=999, max_documents=7)

    assert builder.budget_tokens == 999 and builder.max_documents == 7
    described = builder.describe()
    assert described["sandbox"] == [str(tmp_path.resolve())]
    assert described["compressor"]["max_lines_per_document"] > 0

    fallback = build_context_builder()
    assert fallback.budget_tokens == DEFAULT_BUDGET_TOKENS


# ------------------------------------------------------------------ 检索命中 → 文档
def test_document_from_retrieved_mapping() -> None:
    document = document_from_retrieved(
        {
            "chunk_id": "c1",
            "content": BODY,
            "path": "app/auth/service.py",
            "start_line": 10,
            "end_line": 21,
            "kind": "function",
            "name": "login",
            "parent": "AuthService",
            "score": "0.75",
            "sources": ("semantic", "bm25"),
            "language": "python",
        },
        query="登录校验",
    )

    assert document.name == "AuthService.login"
    assert document.score == 0.75
    assert document.section == SECTION_EVIDENCE
    assert document.metadata["query"] == "登录校验"
    assert document.metadata["sources"] == ["semantic", "bm25"]
    assert document.metadata["language"] == "python"


def test_document_from_retrieved_object_uses_attributes() -> None:
    chunk = SimpleNamespace(
        chunk_id="",
        content=BODY,
        path="a.py",
        start_line=1,
        end_line=12,
        kind="function",
        name="f",
        parent="",
        language="",
        score=None,
        sources=None,
        metadata={"index": 3},
    )
    document = document_from_retrieved(chunk)

    assert document.doc_id == "a.py:1-12"
    assert document.score == 0.0
    assert document.metadata["index"] == 3
    assert document.metadata["sources"] == []


def test_document_from_retrieved_tolerates_bad_types() -> None:
    document = document_from_retrieved(
        {"path": "a.py", "start_line": "x", "end_line": None, "score": "not-a-number"}
    )
    assert document.start_line == 0 and document.end_line == 0
    assert document.score == 0.0


# ================================================================== 4. GitHub 仓库 → 上下文
# 这一节兑现 W7 的验收："能自动读取 GitHub 仓库并生成上下文"。
# 刻意用鸭子类型替身（只提供 list_tree / read_file），既保证离线可复现，
# 也把"上下文层只认能力、不认具体实现"这条约束固化成测试。
def make_tree(*entries: tuple[str, str]) -> GitHubTree:
    """按 ``(type, path)`` 造一棵仓库树（size 无关紧要，置 0）。"""
    return GitHubTree(
        repo="acme/demo",
        ref="main",
        sha="tree-sha",
        entries=[GitHubTreeEntry(path=path, type=kind) for kind, path in entries],
    )


def make_github_file(
    path: str,
    *,
    text: str | None = None,
    sha: str = "sha-1",
    size: int = 0,
    truncated: bool = False,
    ref: str = "main",
) -> GitHubFile:
    """造一个真实 ``GitHubFile``：内容默认按行号生成，便于断言行数。"""
    body = text if text is not None else "\n".join(f"# {path} 第 {i} 行" for i in range(1, 6))
    return GitHubFile(
        path=path,
        text=body,
        sha=sha,
        size=size or len(body),
        truncated=truncated,
        ref=ref,
    )


class FakeGitHubClient:
    """GitHub 客户端替身：只实现 ``documents_from_github`` 依赖的两个方法。

    刻意不继承真实客户端——上下文层应当只经由鸭子类型使用它；
    能换一个等价实现接进来，说明这层没有偷偷耦合 ``GitHubClient`` 的类型。
    ``path_prefix`` / ``max_entries`` 等过滤由真实客户端完成（见 tests/test_github.py），
    这里只记录收到的参数，不重复实现一遍过滤逻辑。
    """

    def __init__(
        self,
        *,
        tree: GitHubTree | None = None,
        files: dict[str, Any] | None = None,
        unreadable: Iterable[str] = (),
        tree_error: Exception | None = None,
        file_error: Exception | None = None,
    ) -> None:
        self.tree = tree
        self.files = dict(files or {})
        self.unreadable = set(unreadable)
        self.tree_error = tree_error
        self.file_error = file_error
        self.tree_calls: list[dict[str, Any]] = []
        self.file_calls: list[tuple[str, str, str | None]] = []

    def list_tree(self, repo: str, *, ref: str | None = None, path_prefix: str | None = None, **_: Any):
        self.tree_calls.append({"repo": repo, "ref": ref, "path_prefix": path_prefix})
        if self.tree_error is not None:
            raise self.tree_error
        return self.tree

    def read_file(self, repo: str, path: str, *, ref: str | None = None) -> Any:
        self.file_calls.append((repo, path, ref))
        if self.file_error is not None:
            raise self.file_error
        if path in self.unreadable:
            raise GitHubNotFoundError(f"仓库里没有 {path}")
        return self.files[path]


# ------------------------------------------------------------------ 单文件 → 文档
def test_github_file_to_document_maps_real_file() -> None:
    file = make_github_file("src/payments/charge.py", text="a = 1\nb = 2\n", sha="abc123", size=42)
    document = github_file_to_document(file, repo="acme/demo")

    assert document.doc_id == "github::acme/demo/src/payments/charge.py"
    assert document.source == "github"
    assert document.kind == "file"
    assert document.name == "charge.py"
    assert document.section == SECTION_EVIDENCE
    assert (document.start_line, document.end_line) == (1, 2)
    assert document.metadata == {
        "repo": "acme/demo",
        "ref": "main",
        "sha": "abc123",
        "size": 42,
        "truncated": False,
    }


def test_github_file_to_document_accepts_duck_typed_object() -> None:
    fake = SimpleNamespace(
        path="README.md", text="# 标题", line_count=1, ref="", sha="", size=0, truncated=True
    )
    document = github_file_to_document(fake)

    assert document.doc_id == "github::README.md"
    assert document.name == "README.md"
    assert document.metadata["repo"] == ""
    assert document.metadata["truncated"] is True


# ------------------------------------------------------------------ 仓库 → 文档
def test_documents_from_github_prefers_explicit_paths() -> None:
    client = FakeGitHubClient(
        tree=make_tree(("blob", "src/other.py")),
        files={path: make_github_file(path) for path in ("src/a.py", "src/b.py")},
    )
    documents = documents_from_github(client, "acme/demo", paths=("src/a.py", "src/b.py"), ref="dev")

    assert [document.path for document in documents] == ["src/a.py", "src/b.py"]
    assert client.tree_calls == []  # 给了明确路径就不该再去拉整棵树
    assert client.file_calls == [("acme/demo", "src/a.py", "dev"), ("acme/demo", "src/b.py", "dev")]


def test_documents_from_github_blank_paths_fall_back_to_tree() -> None:
    client = FakeGitHubClient(
        tree=make_tree(("blob", "only.py")),
        files={"only.py": make_github_file("only.py")},
    )
    documents = documents_from_github(client, "acme/demo", paths=("  ", ""))

    assert [document.path for document in documents] == ["only.py"]
    assert client.tree_calls == [{"repo": "acme/demo", "ref": None, "path_prefix": None}]


def test_documents_from_github_selects_text_files_shallow_first() -> None:
    client = FakeGitHubClient(
        tree=make_tree(
            ("tree", "src"),
            ("blob", "README.md"),
            ("blob", "src/logo.png"),
            ("blob", "src/app.py"),
            ("blob", "src/deep/module.py"),
            ("blob", "src/run.sh"),
        ),
        files={
            path: make_github_file(path)
            for path in ("README.md", "src/app.py", "src/deep/module.py", "src/run.sh")
        },
    )
    documents = documents_from_github(client, "acme/demo")

    # 目录与二进制（.png）不取；其余按"层数浅优先、同层按路径"排序，避免深层文件挤掉入口文件
    assert [document.path for document in documents] == [
        "README.md",
        "src/app.py",
        "src/run.sh",
        "src/deep/module.py",
    ]
    assert all(document.source == "github" for document in documents)


def test_documents_from_github_accepts_extensionless_text_files() -> None:
    """无后缀但业内约定为文本的入口文件要取到。

    实测来源：`octocat/Hello-World` 唯一的文件就是无后缀的 `README`，
    只按后缀判断会让整条 live 链路的证据变成 0 条。
    """
    client = FakeGitHubClient(
        tree=make_tree(
            ("blob", "README"),
            ("blob", "LICENSE"),
            ("blob", "Makefile"),
            ("blob", "Dockerfile"),
            ("blob", "a.out"),
            ("blob", "data"),
            ("tree", "src"),
        ),
        files={
            path: make_github_file(path) for path in ("README", "LICENSE", "Makefile", "Dockerfile")
        },
    )
    documents = documents_from_github(client, "acme/demo")

    # 同层按路径排序；a.out / data / 目录条目都不该发请求
    assert [document.path for document in documents] == [
        "Dockerfile",
        "LICENSE",
        "Makefile",
        "README",
    ]
    assert [path for _, path, _ in client.file_calls] == [
        "Dockerfile",
        "LICENSE",
        "Makefile",
        "README",
    ]


def test_in_text_suffixes_rejects_dirs_and_unknown_extensionless() -> None:
    """白名单放宽到"无后缀的约定文件名"，但不能顺带把看不懂的文件也放进来。"""
    assert in_text_suffixes(GitHubTreeEntry(path="src", type="tree")) is False
    assert in_text_suffixes(GitHubTreeEntry(path="src/logo.png")) is False
    assert in_text_suffixes(GitHubTreeEntry(path="bin/a.out")) is False
    assert in_text_suffixes(GitHubTreeEntry(path="data")) is False
    assert in_text_suffixes(GitHubTreeEntry(path="src/app.py")) is True
    assert in_text_suffixes(GitHubTreeEntry(path="Makefile")) is True
    assert in_text_suffixes(GitHubTreeEntry(path="docs/Readme")) is True  # 大小写不敏感


def test_documents_from_github_passes_prefix_and_caps_files() -> None:
    paths = [f"src/payments/pay_{index}.py" for index in range(5)]
    client = FakeGitHubClient(
        files={path: make_github_file(path) for path in paths},
    )
    documents = documents_from_github(
        client, "acme/demo", paths=tuple(paths), path_prefix="src/payments", max_files=2
    )

    assert [document.path for document in documents] == paths[:2]
    assert len(client.file_calls) == 2  # 上限之外的文件根本不该发请求


def test_documents_from_github_uses_tree_when_selection_is_empty() -> None:
    client = FakeGitHubClient(
        tree=make_tree(("blob", "src/logo.png")),
        files={},
    )
    assert documents_from_github(client, "acme/demo") == []
    assert client.file_calls == []


def test_documents_from_github_skips_unreadable_file() -> None:
    client = FakeGitHubClient(
        files={
            "src/ok.py": make_github_file("src/ok.py"),
            "src/tail.py": make_github_file("src/tail.py"),
        },
        unreadable={"src/broken.py"},
    )
    documents = documents_from_github(
        client, "acme/demo", paths=("src/ok.py", "src/broken.py", "src/tail.py")
    )

    # 单个文件取不到只是少一条证据，不能让整次上下文构建崩掉
    assert [document.path for document in documents] == ["src/ok.py", "src/tail.py"]
    assert len(client.file_calls) == 3


@pytest.mark.parametrize("error", [GitHubAuthError("令牌无效"), GitHubRateLimitError("已限流")])
def test_documents_from_github_propagates_link_level_failure(error: Exception) -> None:
    client = FakeGitHubClient(file_error=error)

    # 认证/限流意味着"整条链路都不通"，证据必然残缺，不能静默当成"没有证据"
    with pytest.raises(type(error)):
        documents_from_github(client, "acme/demo", paths=("src/a.py",))


def test_documents_from_github_propagates_tree_failure() -> None:
    client = FakeGitHubClient(tree_error=GitHubAuthError("令牌无效"))

    with pytest.raises(GitHubAuthError):
        documents_from_github(client, "acme/demo")


def test_documents_from_github_rejects_bad_max_files() -> None:
    client = FakeGitHubClient()

    with pytest.raises(ValueError):
        documents_from_github(client, "acme/demo", max_files=0)


# ------------------------------------------------------------------ 接入 GSSC 流水线
def test_gather_includes_github_documents() -> None:
    client = FakeGitHubClient(files={"src/a.py": make_github_file("src/a.py")})
    builder = ContextBuilder(budget_tokens=800)
    gathered = builder.gather(github=GitHubSource(client=client, repo="acme/demo", paths=("src/a.py",)))

    assert [document.source for document in gathered] == ["github"]
    assert gathered[0].section == SECTION_EVIDENCE


def test_build_with_github_source_reads_repo_into_evidence() -> None:
    client = FakeGitHubClient(
        tree=make_tree(("blob", "src/a.py"), ("blob", "src/b.py")),
        files={path: make_github_file(path) for path in ("src/a.py", "src/b.py")},
    )
    built = ContextBuilder(budget_tokens=1200).build(
        TASK,
        github=GitHubSource(client=client, repo="acme/demo", path_prefix="src"),
    )

    evidence = next(section for section in built.sections if section.title == SECTION_EVIDENCE)
    assert [document.path for document in evidence.documents] == ["src/a.py", "src/b.py"]
    assert "src/a.py" in built.text  # 仓库内容真的进了最终上下文
    assert client.tree_calls[0]["path_prefix"] == "src"
    # 预算硬约束同样适用于 GitHub 来的证据
    assert built.within_budget
    assert built.tokens <= 1200
    assert built.stats.stage("gather").detail == "github 2"


def test_build_with_github_source_defaults_to_twelve_files() -> None:
    paths = [f"pkg/mod_{index}.py" for index in range(20)]
    client = FakeGitHubClient(
        tree=make_tree(*[("blob", path) for path in paths]),
        files={path: make_github_file(path) for path in paths},
    )
    built = ContextBuilder(budget_tokens=4000, max_documents=24).build(
        TASK, github=GitHubSource(client=client, repo="acme/demo")
    )

    assert DEFAULT_MAX_GITHUB_FILES == 12
    assert len(client.file_calls) == 12  # 不做上限控制就会把整个仓库拉下来烧 Token
    assert built.stats.stage("gather").detail == "github 12"
