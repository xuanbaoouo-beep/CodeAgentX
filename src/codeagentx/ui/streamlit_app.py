"""网页界面（W10）：填个路径或仓库名，点一下，看报告。

启动::

    streamlit run src/codeagentx/ui/streamlit_app.py

界面只是**外壳**：提交与执行都走 :mod:`codeagentx.api.service`（与 HTTP 接口同一份
实现），所以"给单文件只审该文件""远端审查完即删""不产出补丁"这些规则在这里也自动成立。

三条边界
--------
1. **只读审查**：不改被测代码，也不产出补丁。
2. **不提供"执行目标仓库测试"的开关**：沙箱还没有容器隔离（AD-15）。
3. **审查是串行的**：向量库是全局单集合（AD-82），所以同一时刻只跑一个作业，
   多个请求会排队。
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st

from codeagentx.api.service import DONE, FAILED, ReviewJob, ReviewRequest, ReviewService
from codeagentx.core.exceptions import CodeAgentXError, ConfigError
from codeagentx.protocols.github_archive import MAX_ARCHIVE_BYTES, extract_zipball

__all__ = ["main"]

DEFAULT_MAX_MB = MAX_ARCHIVE_BYTES // 1024 // 1024
POLL_SECONDS = 0.5
SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")


@st.cache_resource
def get_service() -> ReviewService:
    """整个进程共用一个作业队列（审查串行，见模块 docstring）。"""
    return ReviewService()


def finding_rows(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    """把结构化问题转成表格行：界面要给的是"哪一行、多严重、怎么办"。"""
    rows: list[dict[str, object]] = []
    for item in findings:
        file = str(item.get("file") or "")
        line = item.get("line") or 0
        rows.append(
            {
                "严重度": item.get("severity", ""),
                "分类": item.get("category", ""),
                "标题": item.get("title", ""),
                "位置": f"{file}:{line}" if file else "（未给位置）",
                "来源": item.get("source", ""),
                "建议": item.get("suggestion", ""),
            }
        )
    return rows


def render_sidebar() -> tuple[ReviewRequest | None, Path | None]:
    """收集入参；返回 ``(请求, 上传解压出的临时目录)``。"""
    with st.sidebar:
        st.subheader("审查入参")
        target = st.text_input(
            "审查目标",
            placeholder="data/sample_repo 或 pypa/sampleproject",
            help="服务端可见的本地目录、单个 .py 文件，或 owner/repo（服务端会去下载它）",
        )
        ref = st.text_input("ref（分支/标签，仅远端用）", placeholder="留空用默认分支")
        uploaded = st.file_uploader("或上传仓库压缩包（.zip）", type=["zip"])

        with st.expander("高级选项"):
            enable_refactor = st.checkbox("附加重构规划（只出计划，不改文件）", value=False)
            reflect = st.checkbox("用 reflection 范式（默认 react）", value=False)
            max_mb = st.number_input(
                "远端解压上限（MB）", min_value=1, max_value=2048, value=DEFAULT_MAX_MB
            )

        submitted = st.button("开始审查", type="primary", use_container_width=True)

    if not submitted:
        return None, None

    workdir: Path | None = None
    if uploaded is not None:
        workdir = Path(tempfile.mkdtemp(prefix="codeagentx-upload-"))
        try:
            with st.spinner("正在解压上传的压缩包……"):
                extracted = extract_zipball(uploaded.getvalue(), workdir)
        except CodeAgentXError as exc:
            st.sidebar.error(f"压缩包无法使用：{exc}")
            return None, None
        st.sidebar.success(f"已解压 {extracted.files} 个文件到临时目录")
        target = str(extracted.root)

    if not target.strip():
        st.sidebar.error("请填写审查目标，或上传一个压缩包。")
        return None, None

    return (
        ReviewRequest(
            target=target,
            ref=ref.strip(),
            enable_refactor=enable_refactor,
            reflect=reflect,
            max_bytes=int(max_mb) * 1024 * 1024,
        ),
        workdir,
    )


def render_job(job: ReviewJob) -> None:
    """边跑边显示状态，结束后渲染报告。"""
    if not job.finished:
        with st.status(f"正在审查 {job.request.target}", expanded=True) as box:
            if job.note:
                st.write(job.note)
            while not job.finished:
                time.sleep(POLL_SECONDS)
                box.update(label=f"状态：{job.status}｜已用 {job.duration:.0f}s")
            box.update(
                label=f"审查结束：{job.status}（{job.duration:.0f}s）",
                state="complete" if job.status == DONE else "error",
            )

    payload = job.to_dict()
    if job.status == FAILED:
        st.error(f"审查失败（{payload.get('error_kind')}）：{payload.get('error')}")
        return

    result = payload["result"]
    columns = st.columns(4)
    columns[0].metric("问题总数", result["total"])
    for column, severity in zip(columns[1:], SEVERITY_ORDER, strict=False):
        count = sum(1 for item in result["findings"] if item.get("severity") == severity)
        column.metric(severity, count)

    if result["degraded"]:
        st.warning("本次存在降级环节，结论不完整，请勿据此判定「没有问题」。")

    st.subheader("问题清单")
    if result["findings"]:
        st.dataframe(finding_rows(result["findings"]), use_container_width=True)
    else:
        st.info("没有发现待上报的问题。注意这**不等于**代码没有问题，只说明本次没报出来。")

    with st.expander("阶段与用量", expanded=False):
        st.dataframe(result["stages"], use_container_width=True)
        usage = result["usage"]
        st.write(
            f"LLM 调用 {usage.get('calls', 0)} 次｜token {usage.get('total_tokens', 0)}"
            f"｜总耗时 {payload['duration']}s"
        )

    st.subheader("完整报告")
    st.markdown(result["markdown"])
    st.download_button(
        "下载 Markdown 报告",
        data=result["markdown"].encode("utf-8"),
        file_name=f"codeagentx-{job.id}.md",
        mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(page_title="CodeAgentX 代码审查", page_icon="🔎", layout="wide")
    st.title("CodeAgentX 多智能体代码审查")
    st.caption(
        "输入本地目录、单个 .py 文件或 GitHub 仓库（`owner/repo`），输出结构化审查报告、"
        "风险定位与重构建议。**只读**：不改你的代码，也不产出补丁。"
    )

    service = get_service()
    health = service.health()
    if not health["llm_configured"]:
        st.error(
            "服务端未配置 `LLM_API_KEY`，现在提交也会失败。请先在 `.env` 里配置模型密钥。"
        )

    request, upload_workdir = render_sidebar()
    if request is None:
        st.info(
            "左侧填写目标后点 **开始审查**。第一次跑建议用仓库里的示例："
            "`data/sample_repo`（约 1 分钟，含 7 处故意留的缺陷）。"
        )
        return

    current: ReviewJob | None = None
    try:
        current = service.submit(request)
        st.session_state["job_id"] = current.id
    except ConfigError as exc:
        st.error(f"提交失败：{exc.message}")
    finally:
        if upload_workdir is not None and current is None:
            # 没提交成功就别把用户上传的东西留在磁盘上
            _cleanup(upload_workdir)

    if current is None:
        return

    try:
        render_job(current)
    finally:
        if upload_workdir is not None:
            _cleanup(upload_workdir)


def _cleanup(path: Path) -> None:
    """删掉界面自己造的临时目录（只删这里造的，不碰用户的东西）。"""
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    main()
