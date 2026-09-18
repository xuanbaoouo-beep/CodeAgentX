"""审查目标 → 可审查的本地工作区。

为什么单独一层：命令行（`codeagentx review`）与 HTTP 服务（W10）都要把"用户给的目标"
变成"工具沙箱能用的根目录"，而这里有两条规则**必须两边完全一致**：

1. 工具沙箱的根**必须是目录**（RAG 索引与路径守卫都要求目录），所以给单文件时根退到
   父目录，但审查**范围**只留这一个文件；
2. 远端仓库落到临时工作区，审查完即删。

放在同一个函数里实现，才不会有"命令行只审单文件、接口却审了整个目录"这种走偏。
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from codeagentx.core.exceptions import CodeAgentXError, TargetError
from codeagentx.protocols import GitHubClient
from codeagentx.protocols.github_archive import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_FILES

__all__ = [
    "WORKDIR_PREFIX",
    "PreparedTarget",
    "looks_like_repo",
    "prepare_target",
]

#: 远端工作区目录前缀（方便 `--keep-workdir` 时一眼认出）
WORKDIR_PREFIX = "codeagentx-remote-"


def looks_like_repo(value: str) -> bool:
    """判断目标像不像 ``owner/repo`` 形式的远端标识。

    只在"本地路径不存在"时才会被问到，所以这里不需要也不会去访问网络。
    ``data/sample_repo`` 这种**写错的本地相对路径**同样符合两段式写法，
    这里一律按远端处理，失败时由调用方额外提示"也可能是路径拼错"。
    """
    text = value.strip()
    if not text or text.startswith((".", "/", "~", "\\")):
        return False
    if "://" in text or text.startswith("git@"):
        return True
    head, _, _ = text.partition("@")
    parts = head.strip("/").split("/")
    return len(parts) == 2 and all(parts)


@dataclass
class PreparedTarget:
    """一次审查的落地结果。

    Attributes:
        root: 工具沙箱允许根，**一定是目录**。
        display: 报告里显示的审查目标（远端会带上解析后的 ref）。
        remote: 是否来自远端仓库（决定要不要清理工作区、要不要打印可信性警告）。
        paths: 审查范围（相对 ``root`` 的文件路径）；空表示整个 ``root``。
        workdir: 远端临时工作区；本地目标为 ``None``。
        note: 给用户看的一句话（单文件提示或下载结果），空表示没什么可说的。
    """

    root: Path
    display: str
    remote: bool = False
    paths: tuple[str, ...] = field(default_factory=tuple)
    workdir: Path | None = None
    note: str = ""

    def cleanup(self, *, keep: bool = False) -> None:
        """删除远端临时工作区；``keep=True`` 时保留，本地目标是空操作。"""
        if self.workdir is not None and not keep:
            shutil.rmtree(self.workdir, ignore_errors=True)


def prepare_target(
    target: str,
    *,
    ref: str = "",
    max_files: int = MAX_ARCHIVE_FILES,
    max_bytes: int = MAX_ARCHIVE_BYTES,
) -> PreparedTarget:
    """把用户给的目标解析成可审查的工作区。

    Args:
        target: 本地目录、单个 ``.py`` 文件，或 ``owner/repo``（也接受仓库 URL / ``@ref``）。
        ref: 远端仓库的分支/标签/提交；留空则查仓库默认分支。
        max_files: 远端归档最多解压多少文件。
        max_bytes: 远端归档解压后的总字节上限。

    Raises:
        TargetError: 目标既不是已存在的路径，也不像 ``owner/repo``。
        GitHubError: 远端仓库准备失败（限流、404、归档不合法等）。
    """
    local = Path(target).expanduser()
    if local.exists():
        resolved = local.resolve()
        if resolved.is_dir():
            return PreparedTarget(root=resolved, display=resolved.name)
        # 沙箱根必须是目录：给单文件时退到父目录，同时把审查范围限定在这一个文件上
        parent = resolved.parent
        display = f"{parent.name}/{resolved.name}"
        return PreparedTarget(
            root=parent,
            display=display,
            paths=(resolved.name,),
            note=f"目标是文件：审查根为 {parent.name}/（工具沙箱需要一个目录），只审 {display}",
        )

    if not looks_like_repo(target):
        raise TargetError(
            f"目标既不是已存在的路径，也不像 owner/repo：{target}",
            detail="若本意是本地路径，请检查拼写（当前目录下没有这个路径，所以按远端仓库处理了）",
        )
    return _prepare_remote(target, ref=ref, max_files=max_files, max_bytes=max_bytes)


def _prepare_remote(
    target: str, *, ref: str, max_files: int, max_bytes: int
) -> PreparedTarget:
    """下载远端仓库归档到临时工作区；任何失败都要把半截工作区删掉再抛。"""
    workdir = Path(tempfile.mkdtemp(prefix=WORKDIR_PREFIX))
    client = GitHubClient.from_config()
    try:
        # 显式解析 ref：报告里的目标名要写清是哪个分支/标签，不能只说"某仓库"
        resolved = ref or client.resolve_ref(target)
        result = client.download_archive(
            target, workdir, ref=resolved, max_files=max_files, max_bytes=max_bytes
        )
    except CodeAgentXError:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    finally:
        client.close()

    display = target if f"@{resolved}" in target else f"{target}@{resolved}"
    note = (
        f"已下载 {display} → {result.root}"
        f"（{result.files} 个文件 / {result.bytes / 1024:.0f} KB"
        + (f"，跳过 {len(result.skipped)} 个条目" if result.skipped else "")
        + "）"
    )
    return PreparedTarget(
        root=result.root, display=display, remote=True, workdir=workdir, note=note
    )
