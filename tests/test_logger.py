"""日志系统单元测试。"""

from __future__ import annotations

import json
import logging

from codeagentx.core.logger import get_logger, log_event, setup_logging


def test_setup_logging_creates_log_files(tmp_path) -> None:
    log_dir = setup_logging(level="DEBUG", log_dir=tmp_path / "logs", force=True)

    assert log_dir is not None
    logger = get_logger("test")
    logger.debug("一条调试日志")
    for handler in logging.getLogger("codeagentx").handlers:
        handler.flush()

    assert (tmp_path / "logs" / "codeagentx.log").exists()
    assert (tmp_path / "logs" / "codeagentx.jsonl").exists()

    setup_logging(log_dir=tmp_path / "logs", force=True)


def test_log_event_writes_structured_fields(tmp_path) -> None:
    log_dir = setup_logging(level="INFO", log_dir=tmp_path / "logs", force=True)
    assert log_dir is not None

    log_event(get_logger("test"), "llm_call", model="m1", total_tokens=42)
    for handler in logging.getLogger("codeagentx").handlers:
        handler.flush()

    lines = (
        (tmp_path / "logs" / "codeagentx.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    payload = json.loads(lines[-1])

    assert payload["event"] == "llm_call"
    assert payload["model"] == "m1"
    assert payload["total_tokens"] == 42
    assert payload["level"] == "INFO"


def test_get_logger_is_namespaced() -> None:
    logger = get_logger("core.llm")
    assert logger.name == "codeagentx.core.llm"


def test_setup_logging_is_idempotent(tmp_path) -> None:
    first = setup_logging(level="INFO", log_dir=tmp_path / "a", force=True)
    second = setup_logging(level="INFO", log_dir=tmp_path / "b")
    assert first == second  # 未 force 时直接返回上一次结果


def test_logger_does_not_propagate_to_root() -> None:
    setup_logging(force=True)
    assert logging.getLogger("codeagentx").propagate is False
