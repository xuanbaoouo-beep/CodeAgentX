# CodeAgentX - 多智能体代码审查与重构助手

> 基于自研 Agent 框架（HelloAgents 设计风格）的智能代码审查系统，支持仓库级理解、RAG 检索、多智能体协作、Reflection 自我优化和评估闭环。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-%E5%8F%AF%E7%94%A8-brightgreen)

## 📝 项目简介

CodeAgentX 是一个面向 Python 项目的智能代码审查助手：给它一个本地目录或 GitHub 仓库地址，
它会自动完成结构分析、静态检测、安全扫描、语义检索与多智能体审查，最后输出一份
**带文件与行号、可交付的 Markdown 报告**。

它针对三个现实问题：

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

- **语言与依赖**：Python 3.10+（开发环境实测 3.13.2）、openai / pydantic / httpx / rich
- **范式与协议**：自研 Agent 框架（LLM / Message / Agent / ToolRegistry）、ReAct / Plan-and-Solve / Reflection、MCP（JSON-RPC 2.0 over stdio）、A2A
- **检索与服务**：RAG + Qdrant（可选后端）、FastAPI、Streamlit、Docker
- **质量工具**：ruff / pylint / bandit / pytest

## 🏗️ 系统架构

```text
用户输入（本地目录 / GitHub 仓库 / 上传 zip）
  ↓
接入层：CLI / HTTP API / Streamlit / Docker
  ↓
编排层：Multi-Agent Orchestrator（七阶段流水线 + 状态留痕、可中断恢复）
  ↓
Agent 层：Planner / Retriever / Reviewer / Security / Tester / Refactor / Reporter
  ↓
工具层：MCP / Terminal（沙箱）/ Git / StaticAnalyzer / TestRunner / RAG / Memory
  ↓
存储层：Qdrant（可选）/ 进程内向量库 / JSONL 记忆与笔记
  ↓
模型层：任意 OpenAI 兼容 API
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

只跑 HTTP API / Web UI 的话 `pip install -e ".[serve]"` 即可；静态分析工具链在 `.[tools]` 里。

### 2. 配置 `.env`

```bash
# Windows
copy .env.example .env
# Linux / macOS
cp .env.example .env
```

至少填 LLM 三项：

```env
LLM_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api-inference.modelscope.cn/v1/
QDRANT_URL=http://localhost:6333
GITHUB_TOKEN=your_github_token
```

其余三项都是**可选增强**，不配也能跑——如实降级（日志里会写明降到哪一档），不会假装成功。

**Embedding（决定检索是否具备语义能力）**：不配时用离线 `HashEmbedder` + BM25 词法，
仍能靠标识符命中，但没有语义。任选一条：

| 路线 | 获取入口 | `.env` 要填 |
| --- | --- | --- |
| 阿里云百炼（模板默认） | bailian.console.aliyun.com → 开通 → 「API-KEY 管理」 | `EMBEDDING_API_KEY=sk-...`（`text-embedding-v3` 是 1024 维，与 `EMBEDDING_DIM` 一致） |
| SiliconFlow | cloud.siliconflow.cn → 「API 密钥」 | `EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1`、`EMBEDDING_MODEL_ID=BAAI/bge-m3`、`EMBEDDING_DIM=1024`、`EMBEDDING_API_KEY=sk-...` |
| 本地 Ollama（零成本、不联网） | `ollama pull nomic-embed-text` | `EMBEDDING_BASE_URL=http://localhost:11434/v1`、`EMBEDDING_MODEL_ID=nomic-embed-text`、`EMBEDDING_DIM=768`、`EMBEDDING_API_KEY=ollama`（本地不校验，但不能为空） |

只填了 `LLM_API_KEY`、且 `EMBEDDING_BASE_URL` 与 `LLM_BASE_URL` 相同时会自动复用该 Key；
换 embedding 模型后维度会变，**必须清空旧集合并重建索引**（维度不一致会被提前拦下并提示）。

**Qdrant 向量库**：它不是 Key 而是一个服务。不配时用 `VECTOR_BACKEND=memory`（进程内、零依赖），
想"索引落盘、跨进程复用"才需要它：

```powershell
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/data/qdrant_storage:/qdrant/storage qdrant/qdrant
```

```env
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

云版（cloud.qdrant.io 免费层）才有 API Key，填进同样两项即可。

**GitHub Token**：读私有仓库或需要更高额度时才要。公开仓库**匿名也能读**（60 次/小时，实测可用）。
在 github.com/settings/tokens?type=beta 建 Fine-grained token → 只勾目标仓库 →
`Contents: Read-only`、`Pull requests: Read-only` → 填 `GITHUB_TOKEN`。

### 3. 命令行

```powershell
# 本地目录
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo

# 单个文件：沙箱根仍取父目录，但只审这一个文件（范围外的文件只当上下文）
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo\app\auth\service.py

# GitHub 仓库：自动下载到临时工作区 → 审查 → 删掉工作区
.\.venv\Scripts\python.exe -m codeagentx.cli review pypa/sampleproject
.\.venv\Scripts\python.exe -m codeagentx.cli review https://github.com/pypa/sampleproject

# 常用开关
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo --out review.md       # 落 Markdown 报告
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo --enable-refactor     # 加重构规划（只出计划）
.\.venv\Scripts\python.exe -m codeagentx.cli review data\sample_repo --state .wf.json --resume
.\.venv\Scripts\python.exe -m codeagentx.cli review acme/demo --keep-workdir              # 保留下载下来的工作区
```

`pip install -e .` 会注册 `codeagentx` 命令（`[project.scripts]`），装好后可直接写 `codeagentx review ...`。

退出码：`0` 全部阶段成功 / `1` 有阶段失败或报告被标记降级 / `2` 目标或配置有问题（此时不会开始烧 Token）。

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

> 远端仓库按 **zipball 归档一次请求**下载（不是逐文件 API，也不是 `git clone`）：未认证时配额只有
> 60 次/时，逐文件读几十个文件就打满，而本机可能根本没有 git。归档解压按**不可信输入**处理
> （剥顶层目录、拒符号链接与 `..`、体积/文件数上限），详见 `docs/architecture.md` AD-85~AD-87。

### 4. 跑成服务：HTTP API

不需要把仓库拉到自己电脑上，也不需要在调用方装 Python 环境——**服务端起一个进程，别的程序用 HTTP 调它**。

```powershell
# 服务端（默认只绑 127.0.0.1；接口本身没有鉴权，别直接暴露到公网）
.\.venv\Scripts\python.exe -m uvicorn codeagentx.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
# 提交作业：同步校验通过后立刻返回 202 + job_id，不会把连接吊住几分钟
curl.exe -X POST http://127.0.0.1:8000/review -H "Content-Type: application/json" -d '{\"target\":\"data/sample_repo\"}'
# {"job_id":"a0960192457a","status":"running","target":"data/sample_repo","poll":"/review/a0960192457a"}

# 之后轮询取结果（queued → running → done / failed）
curl.exe http://127.0.0.1:8000/review/a0960192457a
```

`GET /health` 用来自检服务端状态：

```json
{"status":"ok","llm_configured":true,"model":"deepseek-flash","github_token":false,"jobs":0,"by_status":{}}
```

请求体可选字段：`ref`（远端分支/标签）、`paths`（只审这几个文件/目录）、`enable_refactor`、`reflect`、`max_files`、`max_bytes`。

接口的**三条边界**（与 CLI 的差异，都是刻意设计的）：

1. **不暴露 `enable_test`**：沙箱没有容器隔离，让 HTTP 调用方一句话就跑目标仓库自带的测试，等于把"执行别人的代码"的决定权交给网络。传了会直接 **422**（请求体 `extra="forbid"`，不静默忽略）。
2. **密钥只从服务端 `.env` 读**：接口不接受调用方传入密钥；服务端没配 `LLM_API_KEY` 时 `POST /review` 立刻返回 **503** 并说明原因——不会让你轮询几分钟才发现。
3. **作业只存内存、单线程串行**：进程重启历史作业即丢；同时只能跑一个审查（向量库是全局单集合，并发会互相检索到对方的代码）。一次审查要几分钟，建议早提交、稍后再取。

实测（`deepseek-flash`，审 `data/sample_repo`，去重合并后 14 条问题、无降级）：

```text
job=a0960192457a  status=done  duration=90.6s  calls=13  tokens=93131(prompt 77556 + completion 15575)
  plan     done      9.59s  5 条子任务
  retrieve done      6.01s  11 条证据 / 5 次查询
  review   done     35.95s  12 条问题
  security done     39.06s   7 条问题
  report   done      0.00s  14 条问题（reviewer 7 + 两边都报的合并 5 + security 2）
  → success=True, degraded=False
```

### 5. 网页界面（Streamlit）

同一条服务链路的图形界面，支持本地路径 / `owner/repo` / 上传 zip 三种输入：

```bash
streamlit run src/codeagentx/ui/streamlit_app.py
```

打开 http://localhost:8501 ，填目标后点"开始审查"，页面会显示各阶段进度与最终报告表格。
**不点"开始审查"就不会跑审查**（也不会烧 Token）；上传的 zip 解压到临时目录，跑完自动删掉。

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

验证容器可用（不带密钥时 `llm_configured` 为 `false`，服务本身的健康检查应通过）：

```powershell
curl.exe http://127.0.0.1:8000/health
# {"status":"ok","llm_configured":false,"model":"Qwen/Qwen2.5-72B-Instruct",...}
```

> 镜像基于 `python:3.13-slim`，以非 root 用户（uid 10001）运行，只装 `[serve]` 依赖并带上
> `data/sample_repo` 作为演示目标；`.dockerignore` 排除 `.env`，密钥只能靠 `--env-file` 在运行时注入。

## 📄 输出示例

报告是一份 Markdown，每条问题都带**文件与行号、等级、证据和修复建议**。下面是离线演示
（`--mock`，结论由脚本化 MockLLM 生成，仅示范格式；真实运行的格式相同）：

```text
审查目标：data/sample_repo/app/auth/service.py
问题总数：3（high 2 / medium 1 / low 0）
摘要：该登录模块存在两个高危问题：密钥硬编码与 SQL 语句字符串拼接；另有异常处理泄露内部信息。

[1] [high][security] SECRET_KEY 硬编码在源码中 @ app/auth/service.py:8
    说明：全局常量 SECRET_KEY 直接写在源码里，任何拿到仓库的人都能伪造会话 token。
    建议：改为从环境变量读取：SECRET_KEY = os.environ["SECRET_KEY"]，并在启动时校验其存在；同时轮换已泄露的密钥。
...
```

多智能体流水线（`examples/review_workflow.py --mock`，节选）——七个阶段每一步都留下状态与耗时：

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
[来源] reviewer 2 条、security 4 条
```

上面这次运行里，两个角色都报了「调试入口回显内部异常」这同一条问题，Reporter 把它们合并成一条
（取更严重的等级、置信度取高者、来源写成 `reviewer+security`），所以是「6 条原始结论 → 5 条问题」。

想在代码里直接调用：

```python
from codeagentx.orchestrator import CodeReviewWorkflow

workflow = CodeReviewWorkflow(root="./data/sample_repo", enable_test=False, enable_refactor=True)
result = workflow.run(resume=False, reset=False)

print(result.report.to_markdown())   # 可交付报告（Markdown，不含时间戳，可复现）
print(result.state.describe())       # plan=done | retrieve=done | ... | report=done
print(result.success)                # 有阶段失败或报告降级时为 False
```

还想单独体验检索 / 单 Agent 范式，仓库自带可离线跑通的示例脚本：

```powershell
.\.venv\Scripts\python.exe examples\rag_search.py "用户登录逻辑在哪"                          # 代码检索
.\.venv\Scripts\python.exe examples\review_code.py --mock data\sample_repo                  # 三种单 Agent 范式
.\.venv\Scripts\python.exe examples\review_workflow.py --mock data\sample_repo              # 多 Agent 流水线
.\.venv\Scripts\python.exe examples\github_context.py                                       # 读 GitHub 仓库 → 生成上下文
.\.venv\Scripts\python.exe examples\evaluate_sample_repo.py --judge --repeat 3              # 在标注集上评估（需 Key）
```

## ⚙️ 设计约束与能力边界

四条硬约束（详见 [docs/architecture.md](docs/architecture.md)）：

1. **每步留痕**：每个阶段都写进 `WorkflowState`，`done / failed / skipped` 三者可区分——「没跑」和「跑了但失败」不能混为一谈。
2. **失败不静默**：单个阶段异常只把该阶段记为 `failed`，流水线继续，`report` 恒执行；最终报告会在摘要顶部打印降级警告，提醒读者「没写的问题不等于没有问题」。
3. **恢复要真恢复**：各阶段产物（计划 / 证据 / 各角色结论）与状态一起落盘（写临时文件 + 原子替换），`--resume` 时复用已完成的阶段；落盘失败只记 WARNING，不让审查任务因此失败。
4. **用量可查**：各角色用量累加进 `state.metadata["usage"]`，另记注入的 LLM 实例总调用数（含重构规划这类不经过 `AgentResult` 的直连调用）。

能力边界（如实说明）：

- **输入**：本地目录 / 单个文件 / GitHub 仓库（zipball 落临时工作区，跑完删除）；目前只支持 **Python**。
- **不产出补丁**：沙箱里没有写文件工具，Refactor **只产出计划，不生成 diff、不改动代码**——重构是写操作，必须由人确认后再执行（见 AD-88）。
- **Tester 不写测试**：它产出测试代码**文本**，并可运行仓库**现有**测试套件；报告不会把「我写了测试」说成「我跑了测试」。
- **服务化边界**：接口无鉴权（默认只绑 `127.0.0.1`）、不接受调用方传入密钥、不暴露 `enable_test`、作业只存内存（重启即丢）、单线程串行。
- **已知稳定性缺口**：ReAct Reviewer 打满 `max_iterations` 被强制收敛时，偶尔会输出无法解析成 JSON 的正文；此时该阶段按设计记为 `failed`、报告顶部打降级标记，**该阶段的问题会整段丢失**（`security` 等其他角色不受影响，报告仍然可用）。日志里 38 次 `review` 阶段执行中有 2 次如此（约 5%）。**已减轻、未根治**：解析失败后会沿用同一段对话历史再问一次、这次只要 JSON（AD-96），成功则整段结论照常保留（`report.metadata.json_repaired=true`），仍失败才判失败；这次补救**只在离线单测里验证过**，真实运行中尚未赶上过一次。

## 🔌 外部接入

除了本地证据，也能把远端 GitHub 接进来当证据来源——全部只做**读**操作：

- **GitHub**：`codeagentx review owner/repo` 走 zipball 归档；`examples/github_context.py --live owner/repo` 用只读 REST 读仓库树/文件/PR diff 并生成预算内上下文。
- **MCP**（自研 JSON-RPC 2.0 over stdio）：内置文件系统与 GitHub 两个**只读**服务端，工具**自身**执行失败（含上游 404）一律回 `isError=true` 的正常响应，连接不断；`GITHUB_TOKEN` 只走环境变量传给子进程，不进命令行。
- **A2A**：Agent 信封（对齐 A2A 的 Part/Message）、按能力路由的名片与进程内路由。
- **上下文工程（GSSC）**：`gather → select → structure → compress`；渲染后的 token 预算是**硬约束**，超限先压缩、再截断丢弃，绝不假装放得下；仓库内容一律当**不可信输入**加提示，防注释里的注入指令。

> 已被验证的确定性：`--live` 实测 `octocat/Spoon-Knife`、`octocat/Hello-World` 均 EXIT=0；
> Qdrant 后端**换新进程**仍能读回索引（跨进程持久化成立）；Embedding 接入真实语义向量后，
> 跨语言提问也能命中（「注销后旧会话是否还能继续使用」top1 是 `logout`，而代码里没有对应字面词）。

## 📊 评估结果

在自建标注集（**6 仓库 / 47 文件 / 36 条真实缺陷**）上真实跑过：每个「数据集 × 方案」**重复 3 次**，
共 **54 次运行全部有效、0 失败**。跨仓库同权平均（宏平均，仓库内再对 3 次重复取均值）：

```powershell
# 复现命令
python examples/evaluate_sample_repo.py --judge --repeat 3
```

| 方案 | Precision | Recall | F1 | 误报率 | 定位准确率 | 判官均分 | Token | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| react（ReAct + 三工具） | **0.7815** | 0.9722 | **0.8590** | **0.2185** | **1.0000** | **4.322** | **450,373** | **422.6s** |
| reflection（Reflection） | 0.7062 | 0.9630 | 0.8075 | 0.2938 | 0.9809 | 4.289 | 849,488 | 1,638.2s |
| workflow（七阶段流水线） | 0.5024 | **1.0000** | 0.6653 | 0.4976 | 0.9907 | 3.911 | 1,458,608 | 1,316.4s |

> **先说清楚不确定性**：同一「数据集 × 方案」重复 3 次，**单组极差平均 0.1320（最大 0.2321）**——
> 也就是说**小于 ~0.13 的方案差距是噪声，不是方案差异**。react 与 reflection 只差 0.0515，
> 只能说"跨 6 个仓库整体看 react 略优"，不能说"react 更好"；
> 但 react 同时是**最省**的：token 只有 reflection 的 53%、耗时只有 26%——这一点是实打实的。
>
> **多 Agent 不是最优，短板在去重**：workflow 是三者里唯一查全率 100% 的（36 条标注全中、0 漏报），
> 代价是 token 为 react 的 **3.24 倍**、误报率 49.1%；判官在 **17/18 次**运行里仍把"重复"列为扣分点。
> 这条反例如实保留：**多角色协作的收益要靠合并与去重兑现，目前只兑现了一部分**。

## 📁 项目结构

```text
CodeAgentX/
├── src/codeagentx/
│   ├── cli.py         # 统一入口：codeagentx review <本地目录 | owner/repo>
│   ├── core/          # LLM / Message / Agent / 日志 / 异常
│   ├── agents/        # 三种单 Agent 范式 + 七角色 + 结论数据结构
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
├── data/              # 样例仓库 / 评估集
├── tests/             # 单元测试
├── docs/              # 架构与设计决策记录
└── examples/          # 可运行示例脚本
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
