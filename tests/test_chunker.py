"""AST 分块测试：边界保留、装饰器、超长切分、非 Python 兜底与目录遍历。"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagentx.rag.chunker import (
    DEFAULT_EXCLUDE_DIRS,
    CodeChunk,
    chunk_file,
    chunk_python_source,
    chunk_repository,
    chunk_text_source,
)

SAMPLE = '''"""模块文档。"""

import os
from pathlib import Path

MAX = 10


class LoginService:
    """处理用户登录。"""

    def __init__(self, store: Path) -> None:
        self.store = store

    @property
    def ready(self) -> bool:
        return self.store.exists()

    def authenticate(self, user: str, password: str) -> bool:
        if not user:
            raise ValueError("empty user")
        return password == "secret"


def login(user: str, password: str) -> bool:
    """模块级登录函数。"""
    return LoginService(Path(".")).authenticate(user, password)


if __name__ == "__main__":
    print(login("a", "secret"))
'''


def _by_name(chunks: list[CodeChunk], name: str) -> list[CodeChunk]:
    return [chunk for chunk in chunks if chunk.name == name]


class TestPythonBoundaries:
    def test_module_header_holds_docstring_and_imports(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        header = [chunk for chunk in chunks if chunk.kind == "module_header"]
        assert len(header) == 1
        assert "模块文档" in header[0].content
        assert "import os" in header[0].content
        assert "from pathlib import Path" in header[0].content
        assert header[0].start_line == 1
        assert header[0].end_line == 4

    def test_module_level_statements_grouped(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        module_chunks = [chunk for chunk in chunks if chunk.kind == "module"]
        # MAX = 10 与 if __name__ 块分成两个 module 块
        assert len(module_chunks) == 2
        assert any("MAX = 10" in chunk.content for chunk in module_chunks)
        assert any('__main__' in chunk.content for chunk in module_chunks)

    def test_class_header_is_separate_from_methods(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        class_chunks = _by_name(chunks, "LoginService")
        assert len(class_chunks) == 1
        header = class_chunks[0]
        assert header.kind == "class"
        assert "class LoginService" in header.content
        assert "处理用户登录" in header.content
        # 类头块不应包含方法体
        assert "def authenticate" not in header.content

    def test_each_method_becomes_own_chunk(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        names = [chunk.name for chunk in chunks if chunk.kind == "method"]
        assert names == ["__init__", "ready", "authenticate"]
        assert all(
            chunk.parent == "LoginService" for chunk in chunks if chunk.kind == "method"
        )

    def test_method_lines_match_source(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        authenticate = _by_name(chunks, "authenticate")[0]
        lines = SAMPLE.splitlines()
        assert lines[authenticate.start_line - 1].strip().startswith("def authenticate")
        assert lines[authenticate.end_line - 1].strip() == 'return password == "secret"'

    def test_decorator_is_included_in_method_chunk(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        ready = _by_name(chunks, "ready")[0]
        assert ready.content.lstrip().startswith("@property")

    def test_top_level_function_chunk(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        function = _by_name(chunks, "login")[0]
        assert function.kind == "function"
        assert function.parent == ""
        assert function.content.startswith("def login")

    def test_no_content_loss_on_simple_module(self) -> None:
        # 所有非空行都应至少出现在一个块里
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        covered: set[int] = set()
        for chunk in chunks:
            covered.update(range(chunk.start_line, chunk.end_line + 1))
        non_empty = {
            number
            for number, line in enumerate(SAMPLE.splitlines(), start=1)
            if line.strip()
        }
        assert non_empty <= covered

    def test_nested_class_parent_is_recorded(self) -> None:
        source = "class Outer:\n    class Inner:\n        def m(self):\n            pass\n"
        chunks = chunk_python_source(source, "nested.py")
        inner = _by_name(chunks, "Inner")[0]
        assert inner.parent == "Outer"
        method = _by_name(chunks, "m")[0]
        assert method.parent == "Inner"


class TestChunkIdentityAndText:
    def test_chunk_id_is_stable(self) -> None:
        first = chunk_python_source(SAMPLE, "pkg/service.py")
        second = chunk_python_source(SAMPLE, "pkg/service.py")
        assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]

    def test_chunk_ids_are_unique(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    def test_to_text_contains_location_and_symbol(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        text = _by_name(chunks, "authenticate")[0].to_text()
        assert text.startswith("# pkg/service.py:")
        assert "method authenticate (in class LoginService)" in text

    def test_to_metadata_excludes_content(self) -> None:
        chunks = chunk_python_source(SAMPLE, "pkg/service.py")
        metadata = chunks[0].to_metadata()
        assert "content" not in metadata
        assert metadata["path"] == "pkg/service.py"


class TestLongChunkSplitting:
    def test_long_function_is_split_into_parts(self) -> None:
        body = "\n".join(f"    value_{index} = {index}" for index in range(60))
        source = f"def big():\n{body}\n"
        chunks = chunk_python_source(source, "big.py", max_chunk_lines=20)
        parts = _by_name(chunks, "big")
        assert len(parts) > 1
        assert [chunk.part for chunk in parts] == list(range(len(parts)))

    def test_split_parts_do_not_overlap_or_lose_lines(self) -> None:
        body = "\n".join(f"    value_{index} = {index}" for index in range(60))
        source = f"def big():\n{body}\n"
        chunks = _by_name(chunk_python_source(source, "big.py", max_chunk_lines=20), "big")
        for previous, current in zip(chunks, chunks[1:], strict=False):
            assert current.start_line == previous.end_line + 1
        assert chunks[0].start_line == 1
        assert chunks[-1].end_line == len(source.splitlines())

    def test_char_limit_splits_long_lines(self) -> None:
        source = "def wide():\n" + "".join(f"    x{index} = '{'y' * 200}'\n" for index in range(10))
        chunks = _by_name(chunk_python_source(source, "wide.py", max_chunk_chars=300), "wide")
        assert len(chunks) > 1
        assert all(len(chunk.content) <= 400 for chunk in chunks)


class TestFallbacks:
    def test_syntax_error_falls_back_to_text_window(self) -> None:
        source = "def broken(:\n    pass\n" + "\n".join(f"line {i}" for i in range(80))
        chunks = chunk_python_source(source, "broken.py")
        assert chunks
        assert all(chunk.kind == "block" for chunk in chunks)

    def test_text_window_overlaps(self) -> None:
        source = "\n".join(f"行 {index}" for index in range(1, 21))
        chunks = chunk_text_source(source, "notes.md", window_lines=10, overlap_lines=2)
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10
        assert chunks[1].start_line == 9  # 重叠 2 行
        assert all(chunk.kind == "block" for chunk in chunks)

    def test_text_window_covers_last_line(self) -> None:
        source = "\n".join(f"行 {index}" for index in range(1, 21))
        chunks = chunk_text_source(source, "notes.md", window_lines=10, overlap_lines=2)
        assert chunks[-1].end_line == 20

    def test_invalid_window_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap_lines"):
            chunk_text_source("a\nb", "x.md", window_lines=5, overlap_lines=5)
        with pytest.raises(ValueError, match="window_lines"):
            chunk_text_source("a\nb", "x.md", window_lines=0)

    def test_markdown_language_is_detected(self) -> None:
        chunks = chunk_text_source("# 标题\n正文", "docs/readme.md")
        assert chunks[0].language == "markdown"

    def test_plain_text_language_is_detected(self) -> None:
        chunks = chunk_text_source("hello", "notes.txt")
        assert chunks[0].language == "text"


class TestRepositoryChunking:
    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text(SAMPLE, encoding="utf-8")
        (tmp_path / "README.md").write_text("# 项目\n说明\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("随手记\n", encoding="utf-8")
        # 应被排除的内容
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "junk.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / ".venv" / "lib").mkdir(parents=True)
        (tmp_path / ".venv" / "lib" / "site.py").write_text("y = 2\n", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        return tmp_path

    def test_only_supported_suffixes_are_indexed(self, repo: Path) -> None:
        paths = {chunk.path for chunk in chunk_repository(repo)}
        assert "src/app.py" in paths
        assert "README.md" in paths
        assert "notes.txt" in paths
        assert not any(path.endswith(".png") for path in paths)

    def test_excluded_directories_are_skipped(self, repo: Path) -> None:
        paths = {chunk.path for chunk in chunk_repository(repo)}
        assert not any(path.startswith("__pycache__/") for path in paths)
        assert not any(path.startswith(".venv/") for path in paths)

    def test_paths_are_relative_and_use_posix_separator(self, repo: Path) -> None:
        paths = {chunk.path for chunk in chunk_repository(repo)}
        assert "src/app.py" in paths
        assert all("\\" not in path for path in paths)

    def test_result_is_deterministic(self, repo: Path) -> None:
        first = [chunk.chunk_id for chunk in chunk_repository(repo)]
        second = [chunk.chunk_id for chunk in chunk_repository(repo)]
        assert first == second

    def test_max_files_limits_output(self, repo: Path) -> None:
        chunks = chunk_repository(repo, max_files=1)
        assert {chunk.path for chunk in chunks} == {"README.md"}

    def test_oversized_file_is_skipped(self, repo: Path) -> None:
        chunks = chunk_repository(repo, max_file_bytes=20)
        assert "src/app.py" not in {chunk.path for chunk in chunks}

    def test_missing_root_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="不存在"):
            chunk_repository(tmp_path / "nope")

    def test_default_excludes_cover_dependency_dirs(self) -> None:
        assert {".venv", "__pycache__", ".git", "node_modules"} <= DEFAULT_EXCLUDE_DIRS


class TestChunkFile:
    def test_reads_python_file(self, tmp_path: Path) -> None:
        target = tmp_path / "src" / "app.py"
        target.parent.mkdir()
        target.write_text(SAMPLE, encoding="utf-8")
        chunks = chunk_file(target, root=tmp_path)
        assert chunks
        assert chunks[0].path == "src/app.py"

    def test_path_outside_root_keeps_absolute_form(self, tmp_path: Path) -> None:
        target = tmp_path / "a.py"
        target.write_text("x = 1\n", encoding="utf-8")
        chunks = chunk_file(target, root=tmp_path / "other")
        assert chunks[0].path == target.as_posix()

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert chunk_file(tmp_path / "nope.py", root=tmp_path) == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.py"
        target.write_text("", encoding="utf-8")
        assert chunk_file(target, root=tmp_path) == []
