"""审查作业的执行与状态管理（不依赖任何 Web 框架）。

为什么要有这一层：FastAPI 路由与 Streamlit 界面都只是"外壳"，真正的流程是
「准备目标 → 跑七阶段 → 存报告 → 清理工作区」。放在这里两边共用，也让这一层
能在**不起服务**的情况下被离线测试。

两条刻意的约束
--------------
1. **同时只跑一个审查**：向量库是全局单集合（AD-82），两个仓库同时写同一个集合
   会互相污染检索结果，所以这里是单工作线程的队列，超出并发数的作业排队等待。
2. **作业状态只存在内存里**：重启服务即清空。W10 的目标是"能被远程调用"，
   不是"任务调度平台"；持久化留给后续需要时再加。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeagentx.config import get_config
from codeagentx.core.exceptions import CodeAgentXError, ConfigError, TargetError
from codeagentx.core.llm import build_llm
from codeagentx.core.logger import get_logger
from codeagentx.orchestrator import CodeReviewWorkflow, WorkflowResult
from codeagentx.protocols.github_archive import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_FILES
from codeagentx.workspace import PreparedTarget, prepare_target

logger = get_logger("api.service")

__all__ = [
    "MAX_CONCURRENT_REVIEWS",
    "MAX_JOBS_KEPT",
    "ReviewJob",
    "ReviewRequest",
    "ReviewService",
    "describe_error",
    "local_path_exists",
]

#: 同时执行的审查数（见模块 docstring：单集合向量库决定这里只能是 1）
MAX_CONCURRENT_REVIEWS = 1

#: 内存里最多保留多少个作业（超出后先淘汰最老的已完成作业）
MAX_JOBS_KEPT = 50

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

#: 失败原因分类：接口据此决定是"用户改入参"还是"服务端自己出问题了"
ERROR_TARGET = "target"
ERROR_CONFIG = "config"
ERROR_INTERNAL = "internal"


@dataclass(frozen=True)
class ReviewRequest:
    """一次审查的入参（HTTP 请求体与界面表单共用同一份定义）。

    刻意**没有** ``enable_test``：接口不提供"执行目标仓库自带测试"的能力，
    因为沙箱还没有容器隔离（AD-15），把"跑别人的代码"暴露成远程开关风险太大。
    """

    target: str
    ref: str = ""
    paths: tuple[str, ...] = ()
    enable_refactor: bool = False
    reflect: bool = False
    max_files: int = MAX_ARCHIVE_FILES
    max_bytes: int = MAX_ARCHIVE_BYTES


@dataclass
class ReviewJob:
    """一次审查作业的状态与结果。"""

    id: str
    request: ReviewRequest
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: WorkflowResult | None = None
    error: str = ""
    error_kind: str = ""
    note: str = ""

    @property
    def duration(self) -> float:
        """已耗时（秒）。"""
        end = self.finished_at if self.finished_at is not None else time.time()
        return round(end - (self.started_at or self.created_at), 2)

    @property
    def finished(self) -> bool:
        return self.status in (DONE, FAILED)

    def result_payload(self) -> dict[str, Any]:
        """审查成功时的结果体：报告正文 + 结构化问题 + 阶段与用量。"""
        if self.result is None:  # pragma: no cover - 只有 done 状态才会走到
            return {}
        report = self.result.report
        state = self.result.state
        return {
            "success": self.result.success,
            "target": state.target,
            "degraded": bool(report.metadata.get("degraded")),
            "total": report.total,
            "summary": report.summary,
            "markdown": self.result.to_markdown(),
            "findings": [item.to_dict() for item in report.findings],
            "stages": [
                {
                    "name": item.name,
                    "status": item.status,
                    "duration": item.duration,
                    "detail": item.detail,
                }
                for item in state.stages
            ],
            "usage": state.metadata.get("usage") or {},
        }

    def to_dict(self) -> dict[str, Any]:
        """给调用方的 JSON。进行中的作业只回状态，省得让人以为"没结果就是失败"。"""
        payload: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "target": self.request.target,
            "paths": list(self.request.paths),
            "duration": self.duration,
            "note": self.note,
        }
        if self.status == DONE:
            payload["result"] = self.result_payload()
        elif self.status == FAILED:
            payload["error"] = self.error
            payload["error_kind"] = self.error_kind
        return payload


class ReviewService:
    """作业队列：提交 → 排队 → 执行 → 取结果。

    线程模型：``submit`` 只是把作业丢进单工作线程池，立刻返回；调用方拿
    ``job_id`` 轮询 ``get``。这样 HTTP 请求不会被一两个小时的审查吊住。
    """

    def __init__(
        self,
        *,
        workflow_factory: Callable[..., CodeReviewWorkflow] = CodeReviewWorkflow,
        llm_factory: Callable[[Any], Any] = build_llm,
        config_getter: Callable[[], Any] = get_config,
        target_loader: Callable[..., PreparedTarget] = prepare_target,
        max_workers: int = MAX_CONCURRENT_REVIEWS,
    ) -> None:
        self._jobs: dict[str, ReviewJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="codeagentx-review"
        )
        self._workflow_factory = workflow_factory
        self._llm_factory = llm_factory
        self._config_getter = config_getter
        self._target_loader = target_loader

    # ---------------------------------------------------------------- 提交
    def submit(self, request: ReviewRequest) -> ReviewJob:
        """登记作业并排队；服务端没配好 LLM 时直接抛 ``ConfigError``。

        配置问题在**同步路径**上就报出来，比让人排队半天再拿到一个失败作业好。
        """
        config = self._config_getter()
        if not getattr(config, "is_llm_configured", False):
            raise ConfigError(
                "服务端未配置 LLM_API_KEY",
                detail="请在服务端 .env 里配置模型密钥；接口不接受调用方传入密钥",
            )

        job = ReviewJob(id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._jobs[job.id] = job
            self._prune_locked()
        self._executor.submit(self._run, job, config)
        logger.info("已受理审查作业 %s：target=%s", job.id, request.target)
        return job

    # ---------------------------------------------------------------- 查询
    def get(self, job_id: str) -> ReviewJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def wait(self, job_id: str, timeout: float = 3600.0, interval: float = 0.05) -> ReviewJob:
        """等到作业结束（或超时）再返回——给界面与测试用，省得自己写轮询。"""
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.finished or time.monotonic() >= deadline:
                return job
            time.sleep(interval)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            return {"jobs": len(self._jobs), "by_status": counts}

    def health(self) -> dict[str, Any]:
        """服务自检：没配密钥时接口会受理但作业必然失败，所以提前告诉调用方。"""
        config = self._config_getter()
        return {
            "status": "ok",
            "llm_configured": bool(getattr(config, "is_llm_configured", False)),
            "model": getattr(config, "llm_model_id", ""),
            "github_token": bool(getattr(config, "github_token", "")),
            **self.stats(),
        }

    def shutdown(self, wait: bool = False) -> None:
        """停止接受新作业并释放线程池（测试收尾与优雅退出用）。"""
        self._executor.shutdown(wait=wait, cancel_futures=True)

    # ---------------------------------------------------------------- 执行
    def _run(self, job: ReviewJob, config: Any) -> None:
        """工作线程里跑完一次审查；任何异常都必须落到作业状态上，不能让它永远 queued。"""
        job.status = RUNNING
        job.started_at = time.time()
        prepared: PreparedTarget | None = None
        try:
            prepared = self._target_loader(
                job.request.target,
                ref=job.request.ref,
                max_files=job.request.max_files,
                max_bytes=job.request.max_bytes,
            )
            job.note = prepared.note
            # 调用方显式给了范围就以它为准（含 .py 后缀等更精确的写法），
            # 否则用目标准备阶段得出的范围（单文件审查就是这条）
            paths = job.request.paths or prepared.paths or None
            workflow = self._workflow_factory(
                self._llm_factory(config),
                root=prepared.root,
                target=prepared.display,
                paths=paths,
                config=config,
                enable_refactor=job.request.enable_refactor,
                reflect=job.request.reflect,
            )
            job.result = workflow.run()
            job.status = DONE
        except (TargetError, ValueError) as exc:
            # 目标写错或范围越界：属于调用方的问题
            job.status = FAILED
            job.error_kind = ERROR_TARGET
            job.error = describe_error(exc)
            logger.warning("作业 %s 目标不合法：%s", job.id, job.error)
        except CodeAgentXError as exc:
            job.status = FAILED
            job.error_kind = ERROR_INTERNAL
            job.error = describe_error(exc)
            logger.warning("作业 %s 执行失败：%s", job.id, job.error)
        except Exception as exc:  # noqa: BLE001 - 后台线程没人接异常，必须兜住
            job.status = FAILED
            job.error_kind = ERROR_INTERNAL
            job.error = f"{type(exc).__name__}: {exc}"
            logger.exception("作业 %s 出现未预期错误", job.id)
        finally:
            if prepared is not None:
                prepared.cleanup()
            job.finished_at = time.time()
            logger.info("作业 %s 结束：status=%s 耗时 %.2fs", job.id, job.status, job.duration)

    def _prune_locked(self) -> None:
        """淘汰最老的已完成作业，避免内存里无限堆积（调用方需持有 ``_lock``）。"""
        if len(self._jobs) <= MAX_JOBS_KEPT:
            return
        finished = sorted(
            (job for job in self._jobs.values() if job.finished), key=lambda job: job.created_at
        )
        for job in finished[: len(self._jobs) - MAX_JOBS_KEPT]:
            self._jobs.pop(job.id, None)


def describe_error(exc: Exception) -> str:
    """把异常压成一行给调用方看的文字（``detail`` 里有补充说明时一并带上）。"""
    detail = getattr(exc, "detail", None)
    message = getattr(exc, "message", None) or str(exc)
    return f"{message}（{detail}）" if detail else str(message)


def local_path_exists(target: str) -> bool:
    """目标在当前机器上是否已是存在的路径（界面用它决定要不要提示"远端下载"）。"""
    return bool(target.strip()) and Path(target).expanduser().exists()
