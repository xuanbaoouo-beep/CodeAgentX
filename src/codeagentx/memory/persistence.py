"""JSONL 持久化：情景记忆与任务笔记的跨会话落盘。

为什么用 JSONL（每行一条 JSON）而不是单个 JSON 数组：
- 追加写不必重写整个文件，进程中途退出最多丢最后一行；
- 记忆与笔记天然是"事件流"，一行一条便于事后用脚本或 grep 检索。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeagentx.core.logger import get_logger

logger = get_logger("memory.persistence")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> bool:
    """追加一条记录，返回是否写入成功。

    落盘失败（只读目录、磁盘满等）**不应让"记笔记"这类动作失败**——
    记忆丢失只是降级，任务本身还得继续，因此这里只记警告不抛异常。
    """
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning("记忆落盘失败：%s（%s）", path, exc)
        return False
    return True


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取全部记录。

    文件不存在返回空列表；单行损坏时跳过该行而不是丢掉整份记忆。
    """
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.debug("跳过损坏的记忆记录：%s", stripped[:120])
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
    except OSError as exc:
        logger.warning("记忆读取失败：%s（%s）", path, exc)
        return []
    return records
