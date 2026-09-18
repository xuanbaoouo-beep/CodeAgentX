"""HTTP 服务层（W10）：把七阶段审查包装成可远程调用的接口。

本包**刻意不在导入时**加载 Web 框架：`codeagentx` 的核心能力不依赖 FastAPI，
只有装了 `.[serve]` 才有这些依赖。要用接口就显式导入子模块：

    from codeagentx.api.main import app, create_app   # HTTP 端点
    from codeagentx.api.service import ReviewService  # 作业队列（不依赖 Web 框架）
"""
