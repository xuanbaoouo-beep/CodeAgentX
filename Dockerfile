# CodeAgentX 服务镜像（W10）：把 HTTP 接口与网页界面做成"一条命令能起来"。
#
# 只装 `serve` 依赖（fastapi/uvicorn/streamlit）；核心能力本身不依赖 Web 框架。
# 用法见 README「4. 当作服务跑」一节。
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先复制打包所需的最小集合，让依赖层能命中缓存
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[serve]"

# 示例仓库随镜像带上：容器里也能直接试 `data/sample_repo`
COPY data/sample_repo ./data/sample_repo

# 非 root 运行；审查会在 /tmp 下建临时工作区，容器里那个目录本来就属于该用户
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

# 密钥用环境变量传入，变量名与 .env 完全一致（见 .env.example）：
#   docker run --rm -p 8000:8000 --env-file .env codeagentx
# 想审宿主机上的代码就挂一个卷，目标填容器里的那个路径：
#   docker run --rm -p 8000:8000 --env-file .env -v /path/to/code:/work codeagentx
EXPOSE 8000 8501

# 默认起接口；界面用 docker compose 的另一个 service 或改 CMD 来起
CMD ["uvicorn", "codeagentx.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
