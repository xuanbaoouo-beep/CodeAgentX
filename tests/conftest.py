"""pytest 全局夹具：隔离环境变量、提供离线工具与配置。"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

# 必须在导入 codeagentx 之前设置，避免测试向真实 logs/ 目录写日志。
# 进程退出时把这个临时目录删掉：否则每跑一次 pytest 就在系统 TEMP 里留一个，
# 实测三天积了 173 个（约 7MB）——测试写的日志只在本次运行内有意义。
_TEST_LOG_DIR = tempfile.mkdtemp(prefix="codeagentx-tests-")
atexit.register(shutil.rmtree, _TEST_LOG_DIR, ignore_errors=True)
os.environ.setdefault("LOG_DIR", _TEST_LOG_DIR)

import pytest  # noqa: E402

from codeagentx.config import Config, set_config  # noqa: E402
from codeagentx.core.exceptions import SecurityViolationError  # noqa: E402
from codeagentx.tools.base import BaseTool, ToolParameter, ToolResult  # noqa: E402
from codeagentx.tools.registry import ToolRegistry  # noqa: E402

#: 会被隔离（删除）的环境变量前缀
ENV_PREFIXES = ("LLM_", "QDRANT_", "EMBEDDING_", "GITHUB_", "LOG_", "WORKSPACE_")


class EchoTool(BaseTool):
    """原样返回输入文本，用于验证工具调用链路。"""

    name = "echo"
    description = "回显输入文本，用于测试。"
    parameters = [
        ToolParameter(name="text", type="string", description="要回显的文本"),
        ToolParameter(
            name="repeat",
            type="integer",
            description="重复次数",
            required=False,
            default=1,
        ),
    ]

    def _run(self, text: str, repeat: int = 1) -> str:
        return str(text) * int(repeat)


class FailingTool(BaseTool):
    """总是抛异常，用于验证异常不会击穿编排层。"""

    name = "boom"
    description = "总是失败，用于测试异常归一化。"
    parameters = []

    def _run(self) -> ToolResult:
        raise RuntimeError("预期内的失败")


class GuardedTool(BaseTool):
    """抛出安全违规异常，用于验证沙箱拦截路径。"""

    name = "guarded"
    description = "触发安全策略，用于测试拦截。"
    parameters = []

    def _run(self) -> ToolResult:
        raise SecurityViolationError("拒绝执行危险操作")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """删除所有项目相关环境变量，并把日志/产出目录指向临时目录。"""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "outputs"))
    set_config(None)
    yield
    set_config(None)


@pytest.fixture
def config() -> Config:
    """不含任何密钥的默认配置。"""
    return Config.from_env(load_dotenv_file=False)


@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


@pytest.fixture
def registry() -> ToolRegistry:
    """挂载 echo / boom / guarded 三个工具，并挂到全局配置上的注册表。"""
    tool_registry = ToolRegistry(name="test")
    tool_registry.register_many(EchoTool(), FailingTool(), GuardedTool())
    return tool_registry
