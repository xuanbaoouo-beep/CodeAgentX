# CodeAgentX - 多智能体代码审查与重构助手

> 基于自研 Agent 框架（HelloAgents 设计风格）的智能代码审查系统，支持仓库级理解、RAG 检索、多智能体协作、Reflection 自我优化和评估闭环。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-%E5%8F%AF%E7%94%A8-brightgreen)

## 📝 项目简介

CodeAgentX 是一个面向 Python 项目的智能代码审查助手。
输入本地代码库或 GitHub 仓库地址，系统自动完成：

- 代码结构分析
- 静态问题检测
- 安全风险扫描
- 相关代码检索
- 多智能体审查
- 重构建议生成
- 结构化报告输出

它解决三个核心问题：

1. 人工审查耗时，容易遗漏问题
2. 传统静态工具只能看语法，看不懂语义
3. 普通 LLM 审查缺少上下文、证据和评估

## ✨ 核心功能

- AST 智能分块（保留函数/类边界）+ 语义/词法混合检索（内存向量库，Qdrant 为可选后端）
- 仓库级代码理解（读远端 GitHub 仓库树/文件/PR diff，转成带位置的证据）
- ReAct / Plan-and-Solve / Reflection 三种范式
- 多智能体协作：Planner / Retriever / Reviewer / Security / Tester / Refactor / Reporter
- MCP 工具调用：文件系统、GitHub（自研 JSON-RPC 2.0 over stdio，只读工具）
- TerminalTool 安全沙箱
- NoteTool 长期任务笔记 + 工作/情景记忆
- ContextBuilder 上下文工程（GSSC）+ 四级压缩（Token 预算硬约束）
- 评估体系：F1、误报率、LLM Judge、Win Rate（6 仓库 / 47 文件 / 36 条标注 × **3 次重复**）
- 三种运行方式：CLI（`codeagentx review`）/ Web UI / Docker，也可起 HTTP API 供别的程序调用

## 🛠️ 技术栈

- Python 3.10+（开发环境实测 3.13.2）
- 自研 Agent 框架（LLM / Message / Agent / ToolRegistry）
- ReAct / Plan-and-Solve / Reflection
- MCP / A2A
- RAG + Qdrant
- FastAPI
- Streamlit
- Docker
- ruff / pylint / bandit / pytest

## 🏗️ 系统架构

```text
用户输入
  ↓
接入层：CLI / Streamlit / GitHub Action
  ↓
编排层：Multi-Agent Orchestrator
  ↓
Agent 层：Planner / Retriever / Reviewer / Security / Tester / Refactor / Reporter
  ↓
工具层：MCP / Terminal / Git / StaticAnalyzer / TestRunner / RAG / Memory
  ↓
存储层：Qdrant / SQLite / Notes
  ↓
模型层：LLM API / 本地模型
```

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. 配置环境变量

```bash
# Windows
copy .env.example .env
# Linux / macOS
cp .env.example .env
```

编辑 `.env`（至少填写 LLM 三项）：

```env
LLM_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
QDRANT_URL=http://localhost:6333
GITHUB_TOKEN=your_github_token
```

### 可选服务怎么拿（都不配也能跑）

三项都是**可选增强**，不配就是如实降级（日志里会写明降到哪一档），不会假装成功。

**Embedding（决定检索是否具备语义能力）**——不配时用离线 `HashEmbedder` + BM25 词法，
仍能靠标识符命中，但没有语义。任选一条：

| 路线 | 获取入口 | `.env` 要填 |
| --- | --- | --- |
| 阿里云百炼（模板默认） | bailian.console.aliyun.com → 开通 → 「API-KEY 管理」 | `EMBEDDING_API_KEY=sk-...`（`text-embedding-v3` 是 1024 维，与 `EMBEDDING_DIM` 一致） |
| SiliconFlow | cloud.siliconflow.cn → 「API 密钥」 | `EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1`、`EMBEDDING_MODEL_ID=BAAI/bge-m3`、`EMBEDDING_DIM=1024`、`EMBEDDING_API_KEY=sk-...` |
| 本地 Ollama（零成本、不联网） | `ollama pull nomic-embed-text` | `EMBEDDING_BASE_URL=http://localhost:11434/v1`、`EMBEDDING_MODEL_ID=nomic-embed-text`、`EMBEDDING_DIM=768`、`EMBEDDING_API_KEY=ollama`（本地不校验，但不能为空） |

只填了 `LLM_API_KEY`、且 `EMBEDDING_BASE_URL` 与 `LLM_BASE_URL` 相同时会自动复用该 Key；
换 embedding 模型后维度会变，**必须清空旧集合并重建索引**（维度不一致会被提前拦下并提示）。

**Qdrant 向量库**——不是 Key，而是一个服务。不配时用 `VECTOR_BACKEND=memory`（进程内，
零依赖）。想"索引落盘、跨进程复用"才需要它：

```powershell
# 本地 Docker（不需要 API Key）
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/data/qdrant_storage:/qdrant/storage qdrant/qdrant
```

```env
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

云版（cloud.qdrant.io 免费层）才有 API Key：把集群 URL 与 Key 填进同两项即可。

**GitHub Token**——只在要用 `--live` 读真实远端仓库时需要：github.com/settings/tokens?type=beta
→ Fine-grained token → 只勾目标仓库 → `Contents: Read-only`、`Pull requests: Read-only` → 填 `GITHUB_TOKEN`。
**公开仓库不带 Token 也能读**（匿名 60 次/小时，实测可用），Token 用于私有仓库与更高额度。

> 若 `--live` 报 `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`：
> 多半是 **GitHub 加速器（Steam++ / Watt Toolkit 等）在做 TLS 中间人**——它的根证书只装进 Windows
> 证书库，而后端 Python 用 certifi 的根证书包。**退出加速器**即可直连（实测 `GET /rate_limit` → HTTP 200）；
> 或者把该 CA 导出为 PEM 交给 httpx。离线 fixture 模式不受此影响。

### 3. 运行示例（离线可跑，无需 API Key）

```powershell
.\.venv\Scripts\python.exe examples\simple_agent.py --mock "你好"                          # 最小对话 Agent
.\.venv\Scripts\python.exe examples\rag_search.py "用户登录逻辑在哪"                        # 代码检索
.\.venv\Scripts\python.exe examples\review_code.py --mock data\sample_repo                 # 代码审查（ReAct / Reflection / Plan-and-Solve）
.\.venv\Scripts\python.exe examples\review_workflow.py --mock data\sample_repo            # 多 Agent 流水线（七角色协作）
.\.venv\Scripts\python.exe examples\github_context.py                                     # 读 GitHub 仓库 → 生成上下文
```

```powershell
# 在标注集上真实评估（**没有 --mock**：脚本化输出算出来的 F1 是自证，不是指标）
# --repeat 3 = 同一「数据集 × 方案」跑 3 次取均值，并报出重复之间的波动
.\.venv\Scripts\python.exe examples\evaluate_sample_repo.py --judge --repeat 3
```

> 以上都是**示例脚本**（可 `--mock` 离线跑通）。要做一次**真实审查**，用统一入口
> `codeagentx review`，见下一节。

### 3.5 统一入口：`codeagentx review`（本地目录 / GitHub 仓库）

```powershell
# 本地目录
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo

# 单个文件：沙箱根仍取父目录，但**只审这一个文件**（范围外的文件只当上下文）
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo\app\api\posts.py

# GitHub 仓库：自动下载到临时工作区 → 审查 → 删掉工作区
.\.venv\Scripts\python.exe -m codeagentx.cli review pypa/sampleproject
.\.venv\Scripts\python.exe -m codeagentx.cli review pypa/sampleproject --ref main
.\.venv\Scripts\python.exe -m codeagentx.cli review https://github.com/pypa/sampleproject

# 常用开关
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo --out review.md       # 落 Markdown 报告
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo --enable-refactor     # 加重构规划（只出计划）
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo --state .wf.json --resume
.\.venv\Scripts\python.exe -m codeagentx.cli review acme/demo --keep-workdir              # 保留下载下来的工作区
```

安装为命令后可直接写 `codeagentx review ...`（`pip install -e .` 会注册
`[project.scripts]` 里的 `codeagentx` 入口）。

退出码：`0` 全部阶段成功、`1` 有阶段失败或报告被标记降级、`2` 目标/配置有问题（此时不会开始烧 Token）。

实测（`pypa/sampleproject`，`deepseek-flash`，未配 `GITHUB_TOKEN`）：

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

> 远端仓库按 **zipball 归档一次请求**下载（不是逐文件 API，也不是 `git clone`）：
> 未认证时配额只有 60 次/时，逐文件读几十个文件就打满，而本机可能根本没有 git。
> 归档解压按**不可信输入**处理（剥顶层目录、拒符号链接与 `..`、体积/文件数上限），
> 详见 `docs/architecture.md` AD-85~AD-87。

### 4. 当作服务跑：HTTP API

不需要把仓库拉到自己电脑上，也不需要在调用方装 Python 环境——**服务端起一个进程，别的程序用 HTTP 调它**。

```powershell
# 服务端（本机，默认只绑 127.0.0.1，接口本身没有鉴权，别直接暴露到公网）
.\.venv\Scripts\python.exe -m uvicorn codeagentx.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
# 调用方：提交一个作业（同步校验通过后立刻返回 202 + job_id，不会把连接吊住几分钟）
curl.exe -X POST http://127.0.0.1:8000/review -H "Content-Type: application/json" `
  -d '{\"target\":\"data/sample_repo\"}'
# {"job_id":"a0960192457a","status":"running","target":"data/sample_repo","poll":"/review/a0960192457a"}
# 注：只有一把工作线程，提交得够快时它会立刻开跑，所以 status 也可能直接是 running 而不是 queued。

# 之后轮询取结果（queued → running → done / failed）
curl.exe http://127.0.0.1:8000/review/a0960192457a
```

`GET /health` 用来自检服务端状态（是否配了模型密钥、模型名、队列里有多少作业）：

```json
{"status":"ok","llm_configured":true,"model":"deepseek-flash","github_token":false,"jobs":0,"by_status":{}}
```

请求体可选字段：`ref`（远端分支/标签）、`paths`（只审这几个文件/目录）、`enable_refactor`、`reflect`、`max_files`、`max_bytes`。
接口的**三条边界**（与 CLI 的差异，都是刻意设计的）：

1. **不暴露 `enable_test`**：沙箱没有容器隔离，让 HTTP 调用方一句话就跑目标仓库自带的测试，等于把"执行别人的代码"的决定权交给网络。传了会直接 **422**（请求体 `extra="forbid"`，不静默忽略）。
2. **密钥只从服务端 `.env` 读**：接口不接受调用方传入密钥；服务端没配 `LLM_API_KEY` 时 `POST /review` 立刻返回 **503** 并说明原因——不会让你轮询几分钟才发现。
3. **作业只存内存、单线程串行**：进程重启历史作业即丢；同时只能跑一个审查（向量库是全局单集合，并发会互相检索到对方的代码）。一次审查要几分钟，早提交、稍后再取。

实测（`deepseek-flash`，两次真实运行，都不是构造出来的数据）：

```text
# ① 成功的一次（2026-09-18）：审 data/sample_repo，去重合并后 14 条问题，无降级
job=a0960192457a  status=done  duration=90.6s  calls=13  tokens=93131(prompt 77556 + completion 15575)
  plan     done      9.59s  5 条子任务
  retrieve done      6.01s  11 条证据 / 5 次查询
  review   done     35.95s  12 条问题
  security done     39.06s   7 条问题
  report   done      0.00s  14 条问题（reviewer 7 + 两边都报的合并 5 + security 2）
  → success=True, degraded=False

# ② 出现过一次降级（2026-09-18 更早）：Reviewer 阶段整段丢失
job=df78d010ae16  status=done  duration=94.9s  calls=13  tokens=95107  success=False(有阶段降级)
  plan     done     10.05s  5 条子任务
  retrieve done      5.07s  12 条证据 / 5 次查询
  review   failed   39.42s  0 条问题        ⟵ 模型输出未解析成 JSON，按设计判该阶段失败
  security done     40.36s  8 条问题
  report   done      0.00s  8 条问题（含降级标记）
```

> 报告里出现 `failed` 阶段时正文会带降级标记，**别把"解析失败"读成"没有问题"**：
> ② 里 8 条问题全部来自 security 角色，Reviewer 阶段空手而归（原因与已做的补救见
> 「多 Agent 流水线」一节的**能力边界**）。这类失败是**偶发**的：日志里 38 次 `review`
> 阶段执行中有 2 次如此——① 就是同样命令下正常跑通的样子。

### 5. 运行 Web UI

同一条服务链路的图形界面（本地路径 / `owner/repo` / 上传 zip 三种输入）：

```bash
streamlit run src/codeagentx/ui/streamlit_app.py
```

打开 http://localhost:8501 ，填目标后点"开始审查"，页面会显示各阶段进度与最终报告表格。
**不点"开始审查"就不会跑审查**（也不会烧 Token）。

### 6. Docker 一键启动

不需要在目标机器上装 Python 依赖：

```powershell
# 先写好 .env（模型密钥），再一键起 API(8000) + UI(8501)
docker compose up --build
```

```powershell
# 或者只起 API，并把密钥在运行时注入（镜像里不带任何密钥）
docker build -t codeagentx:latest .
docker run --rm -p 8000:8000 --env-file .env codeagentx:latest
```

验证容器可用（不带密钥时 `llm_configured` 会是 `false`，但服务本身的健康检查应通过）：

```powershell
curl.exe http://127.0.0.1:8000/health
# {"status":"ok","llm_configured":false,"model":"Qwen/Qwen2.5-72B-Instruct",...}
```

> 镜像基于 `python:3.13-slim`，以非 root 用户（uid 10001）运行，只装 `[serve]` 依赖并带上
> `data/sample_repo` 作为演示目标；`.dockerignore` 排除 `.env`，密钥只能靠 `--env-file` 在运行时注入。

## 🔍 代码检索示例

仓库自带一个演示用的示例项目 `data/sample_repo`（一个"用户认证 + 登录接口"的小服务，
缺陷是故意保留的，见其 README），可直接体验"输入问题 → 返回相关代码片段"：

```powershell
.\.venv\Scripts\python.exe examples\rag_search.py "用户登录逻辑在哪"
.\.venv\Scripts\python.exe examples\rag_search.py --repo data\sample_repo --brief "SQL 注入风险"
.\.venv\Scripts\python.exe examples\rag_search.py --interactive    # 交互式连续检索
```

实测输出（**离线降级模式**，未配置 `EMBEDDING_API_KEY`，靠 BM25 词法 + 哈希向量召回）：

```text
[索引] 索引完成：9 个文件、22 个代码片段，向量维度 512，耗时 0.004s（当前为离线降级向量化，无语义能力）

检索「用户登录逻辑在哪」命中 3 个代码片段：

[1] app/auth/service.py:22-31  function login  命中：semantic+lexical
def login(username, password):
    """用户登录：校验用户名与密码，成功则返回会话 token。
    ...
```

> 配置了 `EMBEDDING_API_KEY` 时会自动改用真实语义向量；未配置则降级为
> 哈希向量（无语义能力）+ BM25 词法检索，**离线也能复现上述结果**。

## 🧭 代码审查示例

`examples/review_code.py` 提供三种单 Agent 范式，**离线可跑**（`--mock` 用脚本化 MockLLM 演示输出格式）：

```powershell
# ReAct：先取证再下结论（可调用 code_search / static_analyzer / terminal）
.\.venv\Scripts\python.exe examples\review_code.py --mock data/sample_repo/app/auth/service.py

# Reflection：初稿 → 自我批判 → 修订，可指定反思轮数
.\.venv\Scripts\python.exe examples\review_code.py --mock --mode reflection --reflection-rounds 1 data/sample_repo

# Plan-and-Solve：先出重构计划，再逐步执行
.\.venv\Scripts\python.exe examples\review_code.py --mock --mode plan-refactor data/sample_repo

# 真实调用：在 .env 填好 LLM_API_KEY 后去掉 --mock；未配置密钥时脚本会明确报错退出，不会假装成功
.\.venv\Scripts\python.exe examples\review_code.py data/sample_repo
```

ReAct 模式实测输出（**离线演示模式**，结论由 MockLLM 脚本生成，仅示范格式）：

```text
[配置] 模式=react | 模型=mock-llm | 工具=code_search, static_analyzer, terminal
审查目标：data/sample_repo/app/auth/service.py
问题总数：3（high 2 / medium 1 / low 0）
摘要：该登录模块存在两个高危问题：密钥硬编码与 SQL 语句字符串拼接；另有异常处理泄露内部信息。

[1] [high][security] SECRET_KEY 硬编码在源码中 @ app/auth/service.py:8
    说明：全局常量 SECRET_KEY 直接写在源码里，任何拿到仓库的人都能伪造会话 token。
    建议：改为从环境变量读取：SECRET_KEY = os.environ["SECRET_KEY"]，并在启动时校验其存在；同时轮换已泄露的密钥。
...
[统计] LLM 调用 1 次 | token 486 | 耗时 0.0s | 工具调用 0 次 | 收敛=True
```

Plan-and-Solve 模式实测输出：

```text
# 重构计划

**目标**：消除登录模块中的硬编码密钥与 SQL 注入风险，且不改变对外行为

## 步骤

1. **[done]** 把 SECRET_KEY 改为从环境变量读取并补充启动校验
   - 涉及文件：`app/auth/service.py`
   - 理由：密钥必须与代码分离；缺失时快速失败优于静默使用默认值
   - 结论：已按该步骤完成等价重构：... 验证方式：运行 python -m pytest tests/ 全部通过
2. **[done]** 把 build_user_query 改为参数化查询
...
## 风险
## 验证方式
```

> 三种范式都以结构化结论为准：**输出无法解析成 JSON 契约时不会被当成"没有问题"**，
> 而是标记 `parse_error` 并让 `success=False`；ReAct 达到迭代上限未收敛、Plan-and-Solve 有步骤失败，
> 同样如实上报失败，不做乐观包装。审查结论的数据结构（`Finding` / `ReviewReport`）
> 与提示词契约（`prompts/review.py::REPORT_CONTRACT`）字段一一对应。

## 🤖 多 Agent 流水线

`examples/review_workflow.py` 把七个角色串成一条可观测、可恢复的流水线：

```text
plan → retrieve → review → security →（test）→（refactor）→ report
```

```powershell
# 离线演示（无需 API Key）：检索与汇总真实执行，结论由脚本化 LLM 给出
.\.venv\Scripts\python.exe examples\review_workflow.py --mock data\sample_repo

# 七阶段全开：测试生成 + 重构规划
.\.venv\Scripts\python.exe examples\review_workflow.py --mock --enable-test --enable-refactor

# 真实审查：在 .env 填好 LLM_API_KEY 后去掉 --mock
.\.venv\Scripts\python.exe examples\review_workflow.py data\sample_repo

# 状态落盘 + 中断恢复：第二次运行会跳过已完成的阶段，不重复烧 Token
.\.venv\Scripts\python.exe examples\review_workflow.py --state .workflow_state.json data\sample_repo
.\.venv\Scripts\python.exe examples\review_workflow.py --state .workflow_state.json --resume data\sample_repo
```

实测输出（七阶段全开，`--mock`，节选）：

```text
[计划] 4 条子任务（检索阶段会逐条当作查询词）
  1. [security] @security 检查 login / hash_password / SECRET_KEY：口令校验与密钥管理是否安全
  ...
[证据] 检索命中 13 条片段（来自真实索引，可回溯到文件与行号）
  - app/api/routes.py:9-16  score=0.032787
[重构] 4 步（仅规划，未改动任何代码）
[测试] 现有用例：未通过（exit_code=5）
[阶段状态] pending 0 | running 0 | done 7 | failed 0 | skipped 0
  plan     done      0.02s  4 条子任务
  retrieve done      0.05s  13 条证据 / 4 次查询
  review   done      0.00s  2 条问题
  security done      0.00s  4 条问题
  test     done      0.95s  已生成复现测试，并运行现有用例
  refactor done      0.00s  4 步（仅规划）
  report   done      0.00s  5 条问题
[结论] 问题 5 条（high 4 / medium 1 / low 0） | 降级=False | 整体成功=True
[来源] reviewer 2 条、security 4 条
```

上面这次运行里，两个角色都报了「调试入口回显内部异常」这同一条问题，
Reporter 把它们合并成一条：取更严重的等级、置信度取高者、来源写成 `reviewer+security`，
所以「6 条原始结论 → 5 条去重后的问题」。

也支持在代码里直接调用：

```python
from codeagentx.orchestrator import CodeReviewWorkflow

workflow = CodeReviewWorkflow(root="./data/sample_repo", enable_test=False, enable_refactor=True)
result = workflow.run(resume=False, reset=False)

print(result.report.to_markdown())   # 可交付报告（Markdown，不含时间戳，可复现）
print(result.state.describe())       # plan=done | retrieve=done | ... | report=done
print(result.success)                # 有阶段失败或报告降级时为 False
```

设计上的四条硬约束（详见 [docs/architecture.md](docs/architecture.md)）：

1. **每步留痕**：每个阶段都写进 `WorkflowState`，`done / failed / skipped` 三者可区分——
   「没跑」和「跑了但失败」不能混为一谈。
2. **失败不静默**：单个阶段异常只把该阶段记为 `failed`，流水线继续，`report` 恒执行；
   最终报告会在摘要顶部打印降级警告，警告读者「没写的问题不等于没有问题」。
3. **恢复要真恢复**：各阶段产物（计划 / 证据 / 各角色结论）与状态一起落盘（写临时文件 + 原子替换），
   `--resume` 时复用已完成的阶段。落盘失败只记 WARNING，不让审查任务因此失败。
4. **用量可查**：各角色用量累加进 `state.metadata["usage"]`；另行记录注入的 LLM 实例总调用数
   （含重构规划这类不经过 `AgentResult` 的直连调用）。

> 能力边界（如实说明）：
> - 被审查的仓库**既可以是本地目录，也可以是 GitHub 仓库**：`codeagentx review owner/repo`
>   会先把仓库下载到临时工作区（zipball 一次请求）、以它为 root 跑完整七阶段、结束后删掉工作区
>   （见「统一入口」一节的实测输出，决策见 `docs/architecture.md` AD-85~AD-87）；
>   本地证据之外，远端 GitHub 仍可作为**额外证据来源**接入检索阶段（见 `github_context.py`）；
> - **不产出补丁**：沙箱里没有写文件工具，Refactor **只产出计划，不生成 diff、不改动代码**——
>   重构是写操作，必须由人确认后再执行（2026-09-18 范围决定，见 AD-88）；
> - Tester 产出**测试代码文本**并可选地运行仓库**现有**测试套件；沙箱里没有写文件工具，
>   所以它不会把生成的测试写进仓库，报告也不会把「我写了测试」说成「我跑了测试」。
> - **可以当作服务跑**（HTTP API / Streamlit / Docker，见「当作服务跑」一节）：
>   接口**无鉴权**（默认只绑 `127.0.0.1`）、**不接受调用方传入密钥**、**不暴露 `enable_test`**、
>   作业**只存内存**（重启即丢）、**单线程串行**（同时只跑一个审查）。这些取舍的理由见
>   `docs/architecture.md` AD-90~AD-95。
> - **已知稳定性缺口（如实说明）**：ReAct Reviewer 打满 `max_iterations` 被**强制收敛**时，
>   偶尔会输出一段无法解析成 JSON 的正文；此时编排层按设计把 `review` 阶段记为 `failed`、
>   报告顶部打降级标记，**该阶段的问题会整段丢失**（`security` 等其他角色不受影响，报告仍然可用）。
>   全部日志里 **38 次 `review` 阶段执行中有 2 次**如此（约 5%，`logs/codeagentx.log` 中
>   `stage=review status=failed` 的 `error=模型输出中未找到合法 JSON`）。
>   **已减轻、未根治**：解析失败后会**用同一段对话历史再问一次、这次只要 JSON**（见 `architecture.md` AD-96），
>   成功则整段结论照常保留（`report.metadata.json_repaired=true`），仍失败才判失败——
>   也就是说**降级行为没有被取消，只是多了一次把结论要回来的机会**。这次补救只在离线单测里验证过，
>   真实运行中还没赶上过一次（该故障约 5% 才出现，要复现得跑很多次；2026-09-18 那次成功的
>   `a0960192457a` 走的是正常路径，日志里 `review_json_repair*` 一条也没有）。日志关键字：
>   `review_json_repair_requested` / `review_json_repaired` / `review_json_repair_failed`。

## 📦 上下文工程与外部接入

`examples/github_context.py` 演示「给一个 GitHub 仓库 → 自动读出代码 → 生成预算内的上下文」：

```powershell
.\.venv\Scripts\python.exe examples\github_context.py                                    # 离线演示（默认）
.\.venv\Scripts\python.exe examples\github_context.py --prefix src/payments --max-files 4
.\.venv\Scripts\python.exe examples\github_context.py --paths README.md src/payments/auth.py --budget 600
.\.venv\Scripts\python.exe examples\github_context.py --live owner/repo                   # 读真实仓库（只读，需网络与 GITHUB_TOKEN）
```

实测输出（**离线演示**：HTTP 层被替换为按官方文档字段构造的内存响应，
`GitHubClient` 的协议代码、路径校验、base64 解码与整条 GSSC 流水线都是真实执行的）：

```text
[取回] 按仓库树挑选（上限 12）
  - README.md:1-8  296 字节
  - docs/architecture.md:1-13  361 字节
  - pyproject.toml:1-8  157 字节
  - src/payments/__init__.py:1-5  94 字节
  - src/payments/api.py:1-31  930 字节
  - src/payments/auth.py:1-21  522 字节

[GSSC 阶段]
  gather       0    9    1135    1.8ms  github 8 / note 1
  select       9    9    1135    0.2ms  上限 24 条 / 每文件 4 条，丢弃 0 条
  structure    9    9    1261    0.2ms  3 节：任务 / 代码证据 / 约束与要求
  compress     9    7     899    2.2ms  1135 → 774 token（降幅 31.8%），迭代 1 轮

[通过] 仓库证据非空：证据节 6 条
[通过] 渲染后不超预算：899 / 1000 token
```

四项能力与它们的设计边界：

1. **ContextBuilder（GSSC 流水线）**：`gather → select → structure → compress` 四阶段皆为公开方法
   （便于单测与消融实验）。渲染后的 token 预算是**硬约束**（`within_budget`），超限先压缩、
   再截断丢弃，绝不假装放得下；仓库内容一律当**不可信输入**加提示，防注释里的注入指令。
2. **压缩器四级**：去重 → 合并相邻片段 → 长文截断（保头部与尾部）→ 预算装箱，
   每级都留下 `tokens_in/out` 与说明，降幅可逐级归因（上例 31.8%）。
3. **MCP（自研 JSON-RPC 2.0 over stdio）**：`protocols/mcp_client.py` 起子进程通信，
   内置两个**只读**服务端——文件系统（`read_text_file / list_directory / directory_tree / search_files / get_file_info`）
   与 GitHub（`get_repository / list_tree / read_file / list_pull_requests / get_pull_request / get_pull_request_files / get_pull_request_diff`）。
   工具**自身**执行失败（含上游 404）一律回 `isError=true` 的正常响应，连接不断；
   `GITHUB_TOKEN` 只走环境变量传给子进程，不进命令行。
4. **GitHub REST 与 A2A**：`GitHubClient` 是只读 REST 客户端（注入 `httpx.Client` 即可离线回放），
   固定 API 版本、路径越界拒绝、401/403/404/429/5xx 分级映射；
   `protocols/a2a.py` 提供 Agent 信封（对齐 A2A 的 Part/Message）、名片（按能力路由）与进程内路由。

> 全部外部读取都只做**读**操作；远端证据与本地检索证据按「路径 + 行号」去重后并入同一条证据链
> （`CodeReviewWorkflow(github=GitHubSource(...))`），远端不可达时只降级、不让审查失败。
>
> **`--live` 真实仓库实测**（2026-09-17，公开仓库**不带 Token** 也可读，匿名 60 次/小时）：
> `octocat/Spoon-Knife` → EXIT=0，取回 README.md / index.html / styles.css，证据 3 条 / 542 token；
> `octocat/Hello-World` → EXIT=0（该仓库唯一的文件是无后缀的 `README`，靠下面这条规则才取得到）。
>
> **"算不算文本"由白名单决定**（`GITHUB_TEXT_SUFFIXES` + `GITHUB_TEXT_NAMES`）：后缀在白名单内，
> 或者**没有后缀**且文件名属于 README / LICENSE / Makefile / Dockerfile 这类约定文本；
> `a.out`、`data` 这种无从判断的一律不收。证据为 0 时示例会多打一条 `[诊断]`，
> 明说是白名单过滤掉的、还是这个范围里本就没有文件 —— 免得被误读成"live 链路坏了"。

## 📊 评估结果

**最新真实数字**（2026-09-17，`deepseek-flash`，标注集：**6 仓库 / 47 文件 / 36 条真实缺陷**；
每个「数据集 × 方案」**重复 3 次**，共 **54 次运行全部有效、0 失败**）

```powershell
# 默认跑 data/evaluation/*/labels.json 全部数据集，每个组合重复 3 次
python examples/evaluate_sample_repo.py --judge --repeat 3
```

跨仓库同权平均（宏平均，仓库内再对 3 次重复取均值）：

| 方案 | Precision | Recall | F1 | 误报率 | 定位准确率 | 判官均分 | Token | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| react（ReAct + 三工具） | **0.7815** | 0.9722 | **0.8590** | **0.2185** | **1.0000** | **4.322** | **450,373** | **422.6s** |
| reflection（Reflection） | 0.7062 | 0.9630 | 0.8075 | 0.2938 | 0.9809 | 4.289 | 849,488 | 1,638.2s |
| workflow（七阶段流水线） | 0.5024 | **1.0000** | 0.6653 | 0.4976 | 0.9907 | 3.911 | 1,458,608 | 1,316.4s |

> **先说清楚不确定性**：同一「数据集 × 方案」重复 3 次，**单组极差平均 0.1320（最大 0.2321）**——
> 也就是说**小于 ~0.13 的方案差距是噪声，不是方案差异**。react vs reflection 差 0.0515、
> 逐仓库 5:1 胜，也只能说"跨 6 个仓库整体看 react 略优"，不能说"react 更好"。
>
> **react 同时是最省的**：token 只有 reflection 的 53%、耗时只有 26%——"略优"之外的这一点是实打实的。
>
> **多 Agent 不是最优，短板在去重**：workflow 是三者里唯一查全率 100% 的（36 条标注全中、0 漏报），
> 代价是 token 为 react 的 **3.24 倍**、误报率 49.1%。去重**远没做干净**：18 次运行里还剩 **29 组
> "同文件 + 同一行"的重复条目**未被合并，判官在 **17/18 次**运行里仍把"重复"列为扣分点——
> 这条反例如实保留：**多角色协作的收益要靠合并与去重兑现，目前只兑现了一部分**。

## ✅ 真实验证记录

> **真实模型联调（deepseek-flash）**：最小对话、三种单 Agent 范式、七阶段流水线（含
> `--enable-test --enable-refactor --reflect`）、GitHub 上下文示例全部实跑，退出码 0、**无降级**；
> 七阶段全开实测 19 条问题、18 次调用 / 98607 token / 199s。
> 联调暴露的核心层缺陷已修：工具契约与平台差异、轮次耗尽后的强制收敛、截断显式暴露、
> 裸控制字符 JSON 容错、修订稿坏掉时退回初稿（决策见 `docs/architecture.md` AD-68~AD-72）。
> **Qdrant 后端真机验证**同日完成：本地 Docker 起 Qdrant → 索引 22 个代码块 → **换新进程**
> 检索，仍读回 22 条并命中 `app/auth/service.py`（跨进程持久化成立）；
> 集合维度不一致会被提前拦下并给出修复提示（AD-73）。
> **Embedding 已接入真实语义向量**（百炼 `text-embedding-v3`，1024 维）：跨语言提问也能命中——
> 「注销后旧会话是否还能继续使用」top1 是 `logout`（代码里没有对应字面词），
> 纯英文「how does authentication work」top3 全部落在 `app/auth/*`。

## 🎯 项目亮点

- 多智能体协作，职责分离
- RAG + 上下文工程，控制 Token
- Reflection 自我优化，降低误报
- MCP 工具调用，自动读仓库
- 完整评估体系，有数字可验证
- Docker 部署，可复现

## 📁 项目结构

```text
CodeAgentX/
├── src/codeagentx/
│   ├── cli.py         # 统一入口：codeagentx review <本地目录 | owner/repo>
│   ├── core/          # LLM / Message / Agent / 日志 / 异常
│   ├── agents/        # 三种单 Agent 范式 + 七角色（Planner/Retriever/Reviewer/Security/Tester/Refactor/Reporter）+ 结论数据结构
│   ├── prompts/       # 提示词模板与输出契约
│   ├── orchestrator/  # 七阶段流水线编排与状态管理（可观测、可中断恢复）
│   ├── tools/         # 工具系统与安全沙箱
│   ├── rag/           # 代码分块 / Embedding / 向量库 / 检索
│   ├── memory/        # 工作记忆 / 情景记忆 / 笔记
│   ├── context/       # GSSC 上下文工程
│   ├── protocols/     # MCP / A2A / GitHub 客户端与归档解压
│   ├── evaluation/    # 评估指标与实验
│   ├── api/           # FastAPI 服务
│   └── ui/            # Streamlit 界面
├── data/              # 样例仓库 / 评估集 / 输出
├── tests/             # 单元测试
├── docs/              # 架构与设计决策记录
└── examples/          # 可运行示例
```

## 🔮 未来计划

- [ ] 支持 JavaScript / Java
- [ ] GitHub Action 自动 PR 评论
- [ ] Agentic-RL 微调小模型
- [ ] 多仓库对比分析
- [ ] 团队知识库

## 🤝 贡献指南

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
