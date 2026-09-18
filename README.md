# CodeAgentX

> 多智能体代码审查与重构助手：输入本地目录或 GitHub 仓库，输出一份带文件、行号与修复建议的 Markdown 审查报告。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-%E5%8F%AF%E7%94%A8-brightgreen)

## 📌 简介

CodeAgentX 面向 **Python 项目**的代码审查：把静态分析、语义检索和多个专职 Agent 串成一条流水线，
对每个问题给出**位置、等级、证据与修复建议**，最后输出一份可以直接贴进 Issue 或评审记录的 Markdown 报告。

传统静态工具只能看语法、普通 LLM 审查缺少上下文和证据——CodeAgentX 用「代码检索 + 多角色分工 + 结果去重」
补上这两块，并且用一套可复现的评测集把效果量化出来（见 [评测](#-评测)）。

## ✨ 特性

- **仓库级理解**：AST 智能分块（保留函数 / 类边界）+ 语义 / 词法混合检索
- **多智能体协作**：Planner / Retriever / Reviewer / Security / Tester / Refactor / Reporter 七个角色分工，结论自动合并去重
- **三种审查范式**：ReAct / Plan-and-Solve / Reflection，可切换对比
- **上下文工程**：GSSC（gather → select → structure → compress）+ 四级压缩，token 预算为硬约束
- **外部接入**：MCP 只读工具（文件系统 / GitHub）、GitHub REST、A2A 信封
- **可观测、可恢复**：每个阶段留痕（`done` / `failed` / `skipped` 三者可区分），支持 `--resume` 续跑
- **四种运行方式**：CLI / HTTP API / Web UI / Docker
- **有评测**：6 仓库 / 47 文件 / 36 条标注，每个组合重复 3 次，数字可复现

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/xuanbaoouo-beep/CodeAgentX.git && cd CodeAgentX

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

只跑 HTTP API / Web UI：`pip install -e ".[serve]"`；静态分析工具链在 `.[tools]` 里。

### 2. 配置

```bash
# Windows
copy .env.example .env
# Linux / macOS
cp .env.example .env
```

`.env` 里**至少填 LLM 三项**（任意 OpenAI 兼容接口）：

```env
LLM_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
```

以下三项是可选增强，都可以不配，不配时按离线方式降级运行（日志里会写明）：

| 变量 | 作用 | 不配时 |
| --- | --- | --- |
| `EMBEDDING_*` | 让检索具备语义能力 | 用离线 `HashEmbedder` + BM25 词法，仍能靠标识符命中 |
| `VECTOR_BACKEND` / `QDRANT_*` | 索引落盘、跨进程复用 | `memory`：进程内向量库，零依赖 |
| `GITHUB_TOKEN` | 读私有仓库、更高 API 额度 | 公开仓库匿名可读（60 次/小时） |

Embedding 换服务只改 `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL_ID` / `EMBEDDING_DIM` 三项即可
（SiliconFlow 的 `BAAI/bge-m3`、本地 Ollama 的 `nomic-embed-text` 都验证过）；
**维度变了要清空旧集合并重建索引**。完整参数说明见 [.env.example](.env.example)。

### 3. 运行

```powershell
# CLI：本地目录 / 单个文件 / GitHub 仓库（自动下载到临时工作区，审完删除）
codeagentx review ./your-project
codeagentx review ./your-project/app/service.py
codeagentx review owner/repo

# HTTP API（默认只绑 127.0.0.1）
python -m uvicorn codeagentx.api.main:app --port 8000

# Web UI（本地路径 / owner/repo / 上传 zip）
streamlit run src/codeagentx/ui/streamlit_app.py

# Docker：一键起 API(8000) + UI(8501)
docker compose up --build
```

CLI 常用参数：`--out report.md` 落盘报告、`--enable-test` 跑目标仓库自带用例、
`--enable-refactor` 加重构规划、`--reflect` 换 Reflection 范式、`--state s.json --resume` 断点续跑。
退出码：`0` 全部阶段成功 / `1` 有阶段失败或报告降级 / `2` 目标或配置有误（此时不会调用模型）。

### 4. 更多示例脚本

仓库自带可离线跑通的示例（`examples/`）：

```powershell
python examples/simple_agent.py --mock "你好"                  # 最小对话 Agent
python examples/rag_search.py "用户登录逻辑在哪"                # 代码检索
python examples/review_code.py --mock data/sample_repo         # 三种单 Agent 范式
python examples/review_workflow.py --mock data/sample_repo     # 多 Agent 流水线
python examples/github_context.py                              # 读 GitHub 仓库 → 生成上下文
python examples/evaluate_sample_repo.py --judge --repeat 3     # 在标注集上评估（需 API Key）
```

## 📄 输出示例

报告为 Markdown，每条问题都带文件、行号、等级、说明与修复建议（下例为离线演示，格式与真实运行一致）：

```text
审查目标：data/sample_repo/app/auth/service.py
问题总数：3（high 2 / medium 1 / low 0）
摘要：该登录模块存在两个高危问题：密钥硬编码与 SQL 语句字符串拼接；另有异常处理泄露内部信息。

[1] [high][security] SECRET_KEY 硬编码在源码中 @ app/auth/service.py:8
    说明：全局常量 SECRET_KEY 直接写在源码里，任何拿到仓库的人都能伪造会话 token。
    建议：改为从环境变量读取：SECRET_KEY = os.environ["SECRET_KEY"]，并在启动时校验其存在；同时轮换已泄露的密钥。
```

七阶段流水线执行后同时给出各阶段状态与耗时（`failed` 与 `skipped` 分开统计）：

```text
[阶段状态] pending 0 | running 0 | done 7 | failed 0 | skipped 0
  plan     done      0.02s  4 条子任务
  retrieve done      0.05s  13 条证据 / 4 次查询
  review   done      0.00s  2 条问题
  security done      0.00s  4 条问题
  test     done      0.95s  已生成复现测试，并运行现有用例
  refactor done      0.00s  4 步（仅规划）
  report   done      0.00s  5 条问题
[结论] 问题 5 条（high 4 / medium 1 / low 0） | 降级=False | 整体成功=True
```

也可以在代码里直接调用：

```python
from codeagentx.orchestrator import CodeReviewWorkflow

workflow = CodeReviewWorkflow(root="./data/sample_repo", enable_test=False, enable_refactor=True)
result = workflow.run(resume=False, reset=False)

print(result.report.to_markdown())   # 可交付报告（Markdown，不含时间戳，可复现）
print(result.success)                # 有阶段失败或报告降级时为 False
```

## 📊 评测

在自建标注集（6 仓库 / 47 文件 / 36 条真实缺陷）上测量，每个「数据集 × 方案」重复 3 次，
共 54 次运行全部有效：

| 方案 | Precision | Recall | F1 | 误报率 | 判官均分 | Token |
| --- | --- | --- | --- | --- | --- | --- |
| react（ReAct + 三工具） | **0.7815** | 0.9722 | **0.8590** | **0.2185** | **4.322** | **450,373** |
| reflection（Reflection） | 0.7062 | 0.9630 | 0.8075 | 0.2938 | 4.289 | 849,488 |
| workflow（七阶段流水线） | 0.5024 | **1.0000** | 0.6653 | 0.4976 | 3.911 | 1,458,608 |

复现：`python examples/evaluate_sample_repo.py --judge --repeat 3`

- 三种方案的单次波动很大（单组极差平均 0.132），**小于 0.13 的差距不构成结论**，表中排序按跨仓库宏平均给出。
- 单 Agent 的 react 综合最优且最省；多 Agent 查全率满分（36 条标注全部命中）但误报偏高，**合并去重仍需加强**。

## 🏗️ 架构与技术栈

```text
输入（本地目录 / GitHub 仓库 / 上传 zip）
  ↓
接入层：CLI / HTTP API / Streamlit / Docker
  ↓
编排层：Multi-Agent Orchestrator（七阶段流水线 + 状态留痕、可中断恢复）
  ↓
Agent 层：Planner / Retriever / Reviewer / Security / Tester / Refactor / Reporter
  ↓
工具层：MCP / Terminal（沙箱）/ StaticAnalyzer / TestRunner / RAG / Memory
  ↓
存储层：Qdrant（可选）/ 进程内向量库 / JSONL 记忆与笔记
  ↓
模型层：任意 OpenAI 兼容 API
```

**技术栈**：Python 3.10+（开发环境 3.13.2）· 自研 Agent 框架 · ReAct / Plan-and-Solve / Reflection ·
MCP（JSON-RPC 2.0 over stdio）/ A2A · RAG + Qdrant · FastAPI · Streamlit · Docker · ruff / pytest

设计决策与逐项取舍记录在 [docs/architecture.md](docs/architecture.md)。

## ⚠️ 已知限制

- 目前只支持 **Python**。
- **不产出补丁**：Refactor 只给重构计划，不生成 diff、不修改代码（写操作须由人确认后执行）。
- Tester 产出测试代码文本并可运行仓库**现有**测试，但不会把生成的测试写进仓库。
- HTTP API 无鉴权、单线程串行、作业只存内存（重启即丢），适合内网或本机调用。
- ReAct Reviewer 在被强制收敛时（约 5% 概率）可能输出无法解析的正文，该阶段会被记为 `failed`
  并在报告中标注降级；已加入"再问一次只要 JSON"的补救，但仍属未根治的已知问题。

## 📁 项目结构

```text
CodeAgentX/
├── src/codeagentx/
│   ├── cli.py         # 统一入口：codeagentx review <本地目录 | owner/repo>
│   ├── core/          # LLM / Message / Agent / 日志 / 异常
│   ├── agents/        # 三种单 Agent 范式 + 七角色 + 结论数据结构
│   ├── orchestrator/  # 七阶段流水线编排与状态管理
│   ├── tools/         # 工具系统与安全沙箱
│   ├── rag/           # 代码分块 / Embedding / 向量库 / 检索
│   ├── context/       # GSSC 上下文工程
│   ├── protocols/     # MCP / A2A / GitHub 客户端
│   ├── evaluation/    # 评估指标与实验
│   ├── api/ ui/       # FastAPI 服务 与 Streamlit 界面
│   └── memory/ prompts/
├── data/              # 样例仓库与评测集
├── docs/              # 架构与设计决策记录
├── examples/          # 可运行示例
└── tests/             # 单元测试
```

## 🛣️ Roadmap

- [ ] 支持 JavaScript / Java
- [ ] GitHub Action 自动 PR 评论
- [ ] 多仓库对比分析
- [ ] Agentic-RL 微调小模型

## 🤝 贡献

欢迎提交 Issue 和 PR。提交前请确保：

```bash
ruff check src tests
pytest
```

## 📄 许可证

MIT License

## 👤 作者

- GitHub: [@xuanbaoouo-beep](https://github.com/xuanbaoouo-beep)
- 项目链接: [CodeAgentX](https://github.com/xuanbaoouo-beep/CodeAgentX)

## 🙏 致谢

感谢 Datawhale 社区和 Hello-Agents 项目提供的 Agent 学习体系与设计参考。
