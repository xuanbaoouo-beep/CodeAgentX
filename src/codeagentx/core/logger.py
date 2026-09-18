"""日志系统：控制台（rich 美化） + 文件（可读文本 + JSON Lines 结构化）双通道。

设计要点
--------
1. 所有日志挂载在 ``codeagentx`` 这个 logger 下，``propagate=False``，不污染 root logger。
2. ``get_logger`` 首次调用会自动完成一次默认初始化，业务代码无需关心初始化顺序。
3. ``log_event`` 用于记录结构化事件（LLM 调用、工具调用），字段会写入 JSONL，便于后续统计。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT_NAME = "codeagentx"
_CONSOLE_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False
_LOG_DIR: Path | None = None


class JsonLineFormatter(logging.Formatter):
    """把日志记录序列化为单行 JSON，便于离线分析 Token 与耗时。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_console_handler() -> logging.Handler:
    """优先使用 rich 美化输出，rich 不可用时退回标准 StreamHandler。"""
    try:
        from rich.logging import RichHandler

        return RichHandler(
            show_path=False,
            rich_tracebacks=True,
            markup=False,
            log_time_format="%H:%M:%S",
        )
    except Exception:  # pragma: no cover - rich 缺失时的兜底
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_CONSOLE_FMT, _DATE_FMT))
        return handler


def _ensure_dir(path: Path) -> Path | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def setup_logging(
    level: str | None = None,
    log_dir: str | Path | None = None,
    *,
    json_log: bool = True,
    force: bool = False,
) -> Path | None:
    """初始化日志。

    Args:
        level: 日志级别，缺省读环境变量 ``LOG_LEVEL``，再缺省 ``INFO``。
        log_dir: 日志目录，缺省读环境变量 ``LOG_DIR``，再缺省 ``logs``。
        json_log: 是否额外输出 ``.jsonl`` 结构化日志。
        force: 是否强制重建 handler（重复调用时默认直接返回）。

    Returns:
        实际启用的日志目录；目录创建失败（如只读文件系统）时返回 ``None``。
    """
    global _CONFIGURED, _LOG_DIR

    root = logging.getLogger(_ROOT_NAME)
    if _CONFIGURED and not force:
        return _LOG_DIR

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    level_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    root.propagate = False

    root.addHandler(_build_console_handler())

    _LOG_DIR = None
    target_dir = _ensure_dir(Path(log_dir or os.environ.get("LOG_DIR", "logs")))
    if target_dir is not None:
        text_handler = logging.FileHandler(target_dir / "codeagentx.log", encoding="utf-8")
        text_handler.setFormatter(logging.Formatter(_FILE_FMT, _DATE_FMT))
        root.addHandler(text_handler)

        if json_log:
            json_handler = logging.FileHandler(target_dir / "codeagentx.jsonl", encoding="utf-8")
            json_handler.setFormatter(JsonLineFormatter())
            root.addHandler(json_handler)

        _LOG_DIR = target_dir

    _CONFIGURED = True
    return _LOG_DIR


def get_logger(name: str = "") -> logging.Logger:
    """获取 ``codeagentx`` 命名空间下的 logger。"""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"{_ROOT_NAME}.{name}" if name else _ROOT_NAME)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """记录一条结构化事件日志。

    字段会同时进入文本日志的 message 与 JSONL 的顶层字段。
    """
    extra = {"extra_fields": {"event": event, **fields}}
    if logger.isEnabledFor(level):
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.log(level, f"{event} {rendered}".strip(), extra=extra)
