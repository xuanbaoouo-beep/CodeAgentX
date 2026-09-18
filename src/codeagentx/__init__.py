"""CodeAgentX - 多智能体代码审查与重构助手。

分层依赖方向（只允许向左依赖）：

    core <- tools <- rag <- memory <- context <- agents <- orchestrator <- api/ui
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
