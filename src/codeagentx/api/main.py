"""CodeAgentX 的 HTTP 接口（W10）。

三个端点::

    GET  /health            存活与配置自检
    POST /review            提交一次审查，立刻返回 job_id（202）
    GET  /review/{job_id}   查询状态；完成后带报告正文与结构化问题

为什么是"提交 + 轮询"而不是一个请求等到底：一次完整审查要几分钟，让 HTTP 连接
吊在那儿既容易超时也占着连接。所以提交只排队，结果靠 ``job_id`` 取。

本文件只做"HTTP 外壳"：参数校验、状态码、JSON 形状。真正的流程在
:mod:`codeagentx.api.service`，界面与测试都走同一份实现。

三条边界
--------
1. **不暴露"执行目标仓库自带测试"**：沙箱还没有容器隔离（AD-15），
   所以请求体里根本没有这个开关。
2. **默认无鉴权**：请绑 ``127.0.0.1`` 或放在内网。公网部署前必须自己加鉴权，
   否则等于把你的 LLM 额度开放给所有人。
3. **审查只读**：不会修改被测仓库，也不产出补丁；远端仓库下载到临时工作区，
   审查完即删。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from codeagentx.api.service import (
    ReviewRequest,
    ReviewService,
    describe_error,
)
from codeagentx.core.exceptions import ConfigError
from codeagentx.protocols.github_archive import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_FILES

__all__ = ["ReviewRequestModel", "app", "create_app"]

DESCRIPTION = """多智能体代码审查服务。

- `POST /review` 提交审查：目标是**服务端可见的**本地路径，或 `owner/repo`
  （服务端会去下载它）。
- 结果用 `GET /review/{job_id}` 轮询；单工作线程排队执行，因为向量库是全局单集合。
- 不提供"执行目标仓库测试"的开关；不产出补丁。
"""


class ReviewRequestModel(BaseModel):
    """``POST /review`` 的请求体。

    ``extra="forbid"``：字段拼错时直接 422，而不是悄悄按默认值跑一遍——
    比如把 ``enable_test`` 写进来，会明确告诉你这个开关不存在。
    """

    model_config = ConfigDict(extra="forbid")

    target: str = Field(
        ...,
        min_length=1,
        description="服务端可见的本地目录 / 单个 .py 文件，或 owner/repo（也接受仓库 URL、@ref）",
        examples=["data/sample_repo", "pypa/sampleproject"],
    )
    ref: str = Field("", description="远端仓库的分支/标签/提交；留空用默认分支")
    paths: list[str] = Field(
        default_factory=list,
        description="只审这些文件（相对仓库根）；留空表示整个目标。给单文件目标时默认只审该文件",
    )
    enable_refactor: bool = Field(False, description="额外跑一次重构规划（只出计划，不改文件）")
    reflect: bool = Field(False, description="用 reflection 范式代替默认的 react")
    max_files: int = Field(MAX_ARCHIVE_FILES, ge=1, description="远端归档最多解压多少文件")
    max_bytes: int = Field(MAX_ARCHIVE_BYTES, ge=1, description="远端归档解压后的总字节上限")

    def to_request(self) -> ReviewRequest:
        return ReviewRequest(
            target=self.target,
            ref=self.ref,
            paths=tuple(self.paths),
            enable_refactor=self.enable_refactor,
            reflect=self.reflect,
            max_files=self.max_files,
            max_bytes=self.max_bytes,
        )


def create_app(service: ReviewService | None = None) -> FastAPI:
    """建应用。传入 ``service`` 即可注入替身（测试与界面都用得到）。"""
    review_service = service or ReviewService()
    application = FastAPI(title="CodeAgentX 审查服务", description=DESCRIPTION, version="0.1.0")
    application.state.service = review_service

    @application.get("/health", summary="存活与配置自检")
    def health() -> dict[str, Any]:
        return review_service.health()

    @application.post(
        "/review",
        status_code=status.HTTP_202_ACCEPTED,
        summary="提交一次审查",
    )
    def submit_review(payload: ReviewRequestModel, request: Request) -> dict[str, Any]:
        service_: ReviewService = request.app.state.service
        try:
            job = service_.submit(payload.to_request())
        except ConfigError as exc:
            # 服务端自己没配好密钥：不是调用方的错，用 503 而不是 4xx
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=describe_error(exc)
            ) from exc
        return {
            "job_id": job.id,
            "status": job.status,
            "target": job.request.target,
            "poll": f"/review/{job.id}",
        }

    @application.get("/review/{job_id}", summary="查询审查状态与结果")
    def get_review(job_id: str, request: Request) -> dict[str, Any]:
        service_: ReviewService = request.app.state.service
        job = service_.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"没有这个作业：{job_id}（作业只存在内存里，服务重启即清空）",
            )
        return job.to_dict()

    return application


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("codeagentx.api.main:app", host="127.0.0.1", port=8000)
