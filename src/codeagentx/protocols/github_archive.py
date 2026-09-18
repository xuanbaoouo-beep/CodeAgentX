"""把 GitHub 仓库归档（zipball）解压成本地工作区。

为什么用归档而不是逐文件 API
----------------------------
一次 zipball 请求就能拿到整棵仓库（含测试套件与 README），而逐文件 ``contents``
接口按文件计次——未认证时每小时只有 60 次，真实仓库根本读不完。
下载归档**不执行任何远端代码**，只是把字节写进本地临时目录；真正会执行远端代码的
只有"跑仓库自带测试"这一步，那一步默认关闭。

解压必须当成不可信输入
----------------------
归档来自外部，因此这里做四件事：

1. 剥掉 zipball 的顶层目录（``<repo>-<sha>/``），让工作区根就是仓库根；
2. 拒绝符号链接（链接可能指向工作区之外），拒绝 ``..`` / 盘符 / 反斜杠等逃逸写法，
   并在落盘前再校验一次目标路径仍在工作区内（zip-slip）；
3. 文件数、总字节、单文件字节都有上限，**超限直接失败**而不是默默写满磁盘；
4. 单文件超限的条目跳过并记入 ``skipped``，不静默丢弃。
"""

from __future__ import annotations

import io
import shutil
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagentx.core.exceptions import GitHubArchiveError
from codeagentx.core.logger import get_logger

logger = get_logger("protocols.github_archive")

#: 解压文件数上限（纯源码仓库远小于此；超出说明拿错了目标或对方仓库异常）
MAX_ARCHIVE_FILES = 2000
#: 解压总字节上限（200MB）
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
#: 单个文件字节上限（超过的多为误提交的二进制/数据文件，跳过并记录）
MAX_ARCHIVE_FILE_BYTES = 8 * 1024 * 1024

#: 归档里的版本控制目录，一律不落地
SKIP_DIRS = frozenset({".git", ".hg", ".svn"})


@dataclass
class ArchiveExtractResult:
    """一次解压的结果：落在哪、写了多少、跳过了什么。"""

    root: Path
    files: int = 0
    bytes: int = 0
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "files": self.files,
            "bytes": self.bytes,
            "skipped": list(self.skipped),
        }


def _archive_root(names: list[str]) -> str:
    """算出要剥掉的顶层目录名；顶层不唯一（或没有）时返回空串表示不剥。"""
    tops = {name.replace("\\", "/").strip("/").split("/")[0] for name in names if name.strip("/")}
    tops.discard("")
    return tops.pop() if len(tops) == 1 else ""


def _safe_relative(name: str, root: str) -> str | None:
    """把归档条目名规整成相对路径；不该落盘时返回 ``None``。"""
    parts = [part for part in name.replace("\\", "/").strip("/").split("/") if part not in ("", ".")]
    if root and parts and parts[0] == root:
        parts = parts[1:]
    if not parts:
        return None
    if any(part == ".." for part in parts):
        return None
    # 冒号在 Windows 上是盘符/数据流语法，归档来自外部，一律拒绝
    if any(":" in part for part in parts):
        return None
    if parts[0] in SKIP_DIRS:
        return None
    return "/".join(parts)


def extract_zipball(
    data: bytes,
    dest: str | Path,
    *,
    max_files: int = MAX_ARCHIVE_FILES,
    max_bytes: int = MAX_ARCHIVE_BYTES,
    max_file_bytes: int = MAX_ARCHIVE_FILE_BYTES,
) -> ArchiveExtractResult:
    """把 zipball 字节解压到 ``dest``，返回结果（``root`` 即仓库根目录）。

    Raises:
        GitHubArchiveError: 不是合法 zip、或超出文件数/总字节上限。
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise GitHubArchiveError(
            "仓库归档不是合法的 zip",
            detail=f"收到 {len(data)} 字节，可能是限流页或错误响应",
        ) from exc

    destination = Path(dest).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    infos = [info for info in archive.infolist() if not info.is_dir()]
    root = _archive_root([info.filename for info in infos])
    result = ArchiveExtractResult(root=destination)

    for info in infos:
        relative = _safe_relative(info.filename, root)
        if relative is None:
            result.skipped.append(info.filename)
            continue
        if stat.S_ISLNK(info.external_attr >> 16):
            result.skipped.append(f"{info.filename}（符号链接）")
            continue
        if info.file_size > max_file_bytes:
            result.skipped.append(f"{info.filename}（{info.file_size} 字节，超过单文件上限）")
            continue
        if result.files + 1 > max_files:
            raise GitHubArchiveError(
                f"仓库归档文件数超过上限（{max_files}）",
                detail="请改用更小的目标仓库，或调大 download_archive 的 max_files",
            )
        if result.bytes + info.file_size > max_bytes:
            raise GitHubArchiveError(
                f"仓库归档解压后超过上限（{max_bytes} 字节）",
                detail="请改用更小的目标仓库，或调大 download_archive 的 max_bytes",
            )

        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            result.skipped.append(f"{info.filename}（解压路径逃出工作区）")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)

        result.files += 1
        result.bytes += info.file_size

    if result.skipped:
        logger.warning("归档解压跳过 %d 个条目：%s", len(result.skipped), result.skipped[:5])
    return result


__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_ARCHIVE_FILES",
    "MAX_ARCHIVE_FILE_BYTES",
    "ArchiveExtractResult",
    "extract_zipball",
]
