# CodeAgentX

> 多智能体代码审查与重构助手：给它一个本地目录或 GitHub 仓库，输出一份带文件、行号与修复建议的 Markdown 审查报告。

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

先选一条路，两条都能完整使用：

| 你想要什么 | 走哪条路 | 需要准备 |
| --- | --- | --- |
| 只想拿它审自己的代码，不想在本地放源码 | **路径 A：不克隆**（Docker 或 pip） | Docker，或 Python 3.10+ |
| 想跑示例、看实现、改代码、跑测试 | **路径 B：下载到本地** | Python 3.10+ |

两条路都只需要一样东西：**一个 OpenAI 兼容接口的模型 Key**（下文用 `sk-xxx` 占位）。

### 路径 A：不克隆到本地

#### A-1 用 Docker（推荐，连 Python 都不用装）

```powershell
# ① Docker 自己从 GitHub 取代码并构建镜像 —— 你本地不需要有源码
docker build -t codeagentx:latest https://github.com/xuanbaoouo-beep/CodeAgentX.git

# ② 准备密钥文件（这不是克隆仓库，就 3 行）。新建 .env 写入：
#    LLM_API_KEY=sk-xxx
#    LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
#    LLM_MODEL_ID=Qwen/Qwen2.5-72B-Instruct

# ③ 起服务（前台运行，日志刷在这个终端，Ctrl+C 停止）
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env codeagentx:latest
```

> 这个服务是纯接口、**没有首页**：浏览器打开 `http://127.0.0.1:8000/` 看到 `Not Found` 是正常的，
> 要看接口文档请开 <http://127.0.0.1:8000/docs>。想要网页界面见下面「想要网页界面」。

服务起来后，另开一个终端提交审查任务。

提交只是**排队**（一次审查要跑几分钟，不占着 HTTP 连接），拿到 `job_id` 后轮询取结果：

```powershell
# 镜像里自带演示项目 /app/data/sample_repo，可直接用它试（前提是第 ② 步的 Key 已配好）
curl.exe -X POST http://127.0.0.1:8000/review -H "Content-Type: application/json" -d '{\"target\":\"/app/data/sample_repo\"}'
# → {"job_id":"xxxx","status":"queued","target":"/app/data/sample_repo",...}

# 轮询：queued → running → done / failed；done 时多出 result 字段（含 markdown 报告与结构化问题）
curl.exe http://127.0.0.1:8000/review/xxxx
```

没配 Key 时 `POST /review` 会返回 503 并说明原因（接口不接受调用方传 Key，密钥只在服务端配置）。

**审自己电脑上的代码**：把代码目录挂进容器，`target` 写容器内的路径。

```powershell
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v ${PWD}/my-project:/work codeagentx:latest
# 提交时 target 填 /work（或 /work/app/api.py 只审一个文件）
```

**想要网页界面**（界面与接口二选一，换 CMD 即可）：

```powershell
docker run --rm -p 127.0.0.1:8501:8501 --env-file .env codeagentx:latest `
  streamlit run src/codeagentx/ui/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
# 打开 http://127.0.0.1:8501
```

#### A-2 用 pip 装命令行（不想装 Docker，但想要 `codeagentx` 命令）

```powershell
pip install "https://github.com/xuanbaoouo-beep/CodeAgentX/archive/refs/heads/main.zip"
```

这种安装方式下程序**不读当前目录的 `.env`**（配置路径按包安装位置解析），所以用环境变量传密钥：

```powershell
# Windows PowerShell
$env:LLM_API_KEY="sk-xxx"
$env:LLM_BASE_URL="https://api-inference.modelscope.cn/v1/"
$env:LLM_MODEL_ID="Qwen/Qwen2.5-72B-Instruct"

# Linux / macOS
export LLM_API_KEY=sk-xxx
export LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
export LLM_MODEL_ID=Qwen/Qwen2.5-72B-Instruct

# 然后就能直接审任意目录
codeagentx review C:\path\to\your-project --out report.md
```

> 这种方式装出来的只有命令行工具，没有仓库里的 `examples/`、`data/` 与测试。要看示例或改代码，走路径 B。

### 路径 B：下载到本地

```powershell
# ① 取代码：有 git 用 clone；没有 git 就在 GitHub 页面点 Code → Download ZIP 解压
git clone https://github.com/xuanbaoouo-beep/CodeAgentX.git
cd CodeAgentX

# ② 建虚拟环境并安装
python -m venv .venv
.\.venv\Scripts\activate                # Linux / macOS：source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# ③ 配置：从模板复制，填好 LLM 三项（其余可留空）
copy .env.example .env                  # Linux / macOS：cp .env.example .env

# ④ 跑一次（先拿自带的演示项目试）
codeagentx review data\sample_repo --out review.md
```

`--out review.md` 把报告写进文件（不写就打印到终端）。**接着可以做的：**

```powershell
streamlit run src/codeagentx/ui/streamlit_app.py    # 网页界面：本地路径 / owner/repo / 上传 zip
python -m uvicorn codeagentx.api.main:app --port 8000 # 当接口服务跑
docker compose up --build                            # 一键起接口(8000) + 界面(8501)
python examples\review_workflow.py --mock data\sample_repo   # 离线示例，不需要 Key
pytest                                               # 跑测试
```

### 配置清单（两条路通用）

| 变量 | 是否必需 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | ✅ | 模型服务的 API Key |
| `LLM_BASE_URL` | ✅ | OpenAI 兼容端点，如 `https://api-inference.modelscope.cn/v1/` |
| `LLM_MODEL_ID` | ✅ | 模型名，如 `deepseek-flash` |
| `EMBEDDING_*` | 可选 | 不配则检索退化为离线词法匹配（可用，但没有语义能力） |
| `VECTOR_BACKEND` / `QDRANT_*` | 可选 | 不配则用进程内向量库；配了可让索引落盘、跨进程复用 |
| `GITHUB_TOKEN` | 可选 | 读私有仓库或提高额度；公开仓库匿名可读（60 次/小时） |

`.env` 放哪：**路径 B 放在仓库根目录**；路径 A-1 用 `--env-file` 传给容器；路径 A-2 用环境变量。
全部变量与默认值见 [.env.example](.env.example)。

## 📖 CLI 用法

```powershell
codeagentx review <目标> [参数]
```

目标支持三种写法：

| 目标 | 例子 | 说明 |
| --- | --- | --- |
| 本地目录 | `codeagentx review ./my-project` | 审查整个目录 |
| 单个文件 | `codeagentx review ./my-project/app/api.py` | 只审这一个文件，同目录其他文件仅作上下文 |
| GitHub 仓库 | `codeagentx review owner/repo`、`owner/repo@ref`、仓库 URL | 自动下载到临时工作区，审完删除 |

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--out report.md` | 把 Markdown 报告写入文件 |
| `--enable-test` | 跑目标仓库自带的测试用例（默认关：会执行别人的代码，慎用） |
| `--enable-refactor` | 增加重构规划阶段（只出计划，不改代码） |
| `--reflect` | 主审查角色改用 Reflection 范式（默认 ReAct） |
| `--state s.json --resume` | 状态落盘 / 断点续跑，不重复跑已完成阶段 |
| `--keep-workdir` | 保留远端仓库下载下来的临时工作区，便于排查 |
| `--max-files` / `--max-mb` | 限制远端归档解压的文件数与体积 |

退出码：`0` 全部阶段成功 / `1` 有阶段失败或报告被标记降级 / `2` 目标或配置有问题（此时不会调用模型）。

实测（`pypa/sampleproject`，未配 `GITHUB_TOKEN`）：

```text
[远端] 已下载 pypa/sampleproject@main → C:\...\Temp\codeagentx-remote-ebvu2570（12 个文件 / 13 KB）
[配置] 模型=deepseek-flash | 目标=pypa/sampleproject@main | 来源=GitHub | 测试阶段=关 | 重构规划=关 | 审查范式=react
[阶段] pending 0 | running 0 | done 5 | failed 0 | skipped 2
  plan     done     25.32s  4 条子任务
  retrieve done      4.62s  11 条证据 / 4 次查询
  review   done     38.38s  5 条问题
  security done     28.35s  2 条问题
  test     skipped   0.00s  未启用（enable_test=False）
  refactor skipped   0.00s  未启用（enable_refactor=False）
  report   done      0.00s  6 条问题
[用量] LLM 调用 15 次 | token 92076 | 耗时 96.683s | 整体成功=True
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

七个阶段每一步都留下状态与耗时（`failed` 与 `skipped` 分开统计）：

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

也可以在代码里直接调用（模板见 [`examples/review_workflow.py`](examples/review_workflow.py)）：

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
