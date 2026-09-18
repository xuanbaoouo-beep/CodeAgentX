"""全局配置：从 ``.env`` / 环境变量加载并校验。

字段名与环境变量名严格一一对应（字段名大写即为环境变量名），
例如 ``llm_api_key`` <-> ``LLM_API_KEY``，新增字段无需改动加载逻辑。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from codeagentx.core.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def _manual_load_env_file(path: Path) -> None:
    """不依赖 python-dotenv 的极简 ``.env`` 解析（兜底用）。"""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_file(env_file: str | Path | None = None, *, required: bool = False) -> Path:
    """把 ``.env`` 载入进程环境变量（已存在的环境变量优先，不被覆盖）。"""
    path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    if not path.exists():
        if required:
            raise ConfigError(f"未找到环境变量文件：{path}")
        return path
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:  # pragma: no cover - python-dotenv 缺失时的兜底
        _manual_load_env_file(path)
    return path


class Config(BaseModel):
    """CodeAgentX 运行时配置。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    # ---------- LLM（OpenAI 兼容接口） ----------
    llm_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    llm_api_key: str = ""
    llm_base_url: str = "https://api-inference.modelscope.cn/v1/"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=1)
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    # ---------- 向量库 ----------
    #: 向量库后端：memory（默认，零依赖）/ qdrant（需装 qdrant-client 并起服务）/ auto
    vector_backend: Literal["memory", "qdrant", "auto"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "codeagentx_code"

    # ---------- Embedding ----------
    embedding_model_id: str = "text-embedding-v3"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_dim: int = Field(default=1024, ge=1)

    # ---------- GitHub ----------
    github_token: str = ""

    # ---------- 上下文工程（GSSC） ----------
    #: 单次构建的上下文 token 预算（含任务说明、证据头等全部渲染开销）
    context_budget_tokens: int = Field(default=4000, ge=256)
    #: 单次构建最多保留的证据条数
    context_max_documents: int = Field(default=24, ge=1)

    # ---------- 运行时 ----------
    log_level: str = "INFO"
    log_dir: str = "logs"
    workspace_dir: str = "./data/outputs"

    # ------------------------------------------------------------ 构造
    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = None,
        *,
        load_dotenv_file: bool = True,
        overrides: dict[str, Any] | None = None,
    ) -> Config:
        """从环境变量构造配置。

        Args:
            env_file: 指定 ``.env`` 路径；``None`` 表示使用项目根目录下的 ``.env``。
            load_dotenv_file: 是否读取 ``.env``（测试中可关闭以隔离环境）。
            overrides: 显式覆盖项，优先级最高。
        """
        if load_dotenv_file:
            load_env_file(env_file)

        raw: dict[str, Any] = {}
        for field_name in cls.model_fields:
            env_name = field_name.upper()
            if env_name in os.environ:
                raw[field_name] = os.environ[env_name]
        if overrides:
            raw.update(overrides)

        try:
            return cls(**raw)
        except ValidationError as exc:
            raise ConfigError("环境变量解析失败", detail=str(exc)) from exc

    # ------------------------------------------------------------ 派生属性
    @property
    def is_llm_configured(self) -> bool:
        """是否已提供 LLM 密钥。"""
        return bool(self.llm_api_key.strip())

    @property
    def log_dir_path(self) -> Path:
        """日志目录绝对路径（相对路径按项目根目录解析）。"""
        return self._resolve(self.log_dir)

    @property
    def workspace_path(self) -> Path:
        """产出目录绝对路径（相对路径按项目根目录解析）。"""
        return self._resolve(self.workspace_dir)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    # ------------------------------------------------------------ 校验与展示
    def require_llm_credentials(self) -> None:
        """确保 LLM 密钥已配置，否则抛出 :class:`ConfigError`。"""
        if not self.is_llm_configured:
            raise ConfigError(
                "未配置 LLM_API_KEY",
                detail="请复制 .env.example 为 .env 并填写 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID",
            )

    def masked_summary(self) -> dict[str, Any]:
        """用于日志输出的脱敏摘要，绝不包含完整密钥。"""
        return {
            "llm_model_id": self.llm_model_id,
            "llm_base_url": self.llm_base_url,
            "llm_api_key": _mask(self.llm_api_key),
            "qdrant_url": self.qdrant_url,
            "github_token": _mask(self.github_token),
            "log_level": self.log_level,
            "workspace_dir": str(self.workspace_path),
        }


def _mask(secret: str) -> str:
    """密钥脱敏：仅保留首尾各 3 位。"""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:3]}***{secret[-3:]}"


_config: Config | None = None


def get_config(reload: bool = False, env_file: str | Path | None = None) -> Config:
    """获取全局配置单例。"""
    global _config
    if _config is None or reload:
        _config = Config.from_env(env_file)
    return _config


def set_config(config: Config | None) -> None:
    """显式注入配置（测试用），传 ``None`` 表示清空缓存。"""
    global _config
    _config = config
