# 架构设计（Architecture）

> 状态：v1.8。评估体系在 **6 个标注仓库 × 3 次重复**（54 次运行）上产出真实数字，
> 含重复上报去重与波动披露；单仓库单次的结论先后被"扩容"与"重复运行"两次推翻，
> 说明单仓库单次的排序不稳健。逐项决策与取舍见下方决策记录（AD-1 ~ AD-97）。

## 1. 分层视图

```text
┌──────────────────────────────────────────────────────┐
│ 接入层   CLI  /  FastAPI  /  Streamlit  /  GitHub Action │
├──────────────────────────────────────────────────────┤
│ 编排层   Orchestrator（workflow + state）               │
├──────────────────────────────────────────────────────┤
│ 能力层   ReAct / Plan-and-Solve / Reflection / GSSC     │
├──────────────────────────────────────────────────────┤
│ Agent 层 Planner·Retriever·Reviewer·Security·Tester·   │
│          Refactor·Reporter                             │
├──────────────────────────────────────────────────────┤
│ 工具层   Terminal / Git / StaticAnalyzer / TestRunner / │
│          RAG / Memory / Note                            │
├──────────────────────────────────────────────────────┤
│ 协议层   MCP（stdio 子进程） / GitHub REST / A2A         │
├──────────────────────────────────────────────────────┤
│ 存储层   Qdrant（向量） / SQLite（记忆元数据） / Notes    │
├──────────────────────────────────────────────────────┤
│ 模型层   LLM API（OpenAI 兼容） / 本地模型（可选）        │
└──────────────────────────────────────────────────────┘
```

## 2. 依赖方向（强约束）

```text
core  ←  tools  ←  rag  ←  memory  ←  context  ←  agents  ←  orchestrator  ←  api/ui
core  ←  protocols（MCP / GitHub REST / A2A）
core  ←  agents  ←  evaluation（旁路：只读被评估方，谁都不依赖它）
```

规则：

- **只允许向左依赖**，禁止反向 import（例如 `core` 不得 import `agents`）。
- `evaluation` 是**旁路层**：只依赖 `core`（含 `core.agent` 的 `AgentResult`）与
  `agents.schemas`（解析报告契约），**不 import `orchestrator`**（编排结果按鸭子类型适配，见 AD-79）；
  项目内没有任何模块 import 它——评测不该反过来改变被测系统（AD-76 / AD-80）。
- `core` 是对外唯一稳定的基础层：`LLM` / `Message` / `Agent` / `Config` / `Logger` / `exceptions`。
- `protocols` 与 `tools` 同为底层能力，**只依赖 `core`**：它讲"怎么和外部说话"（MCP 报文、
  GitHub REST、A2A 信封），不含审查业务语义；`context` 接远端代码时只按**鸭子类型**
  要求 `list_tree` / `read_file`，不 import `protocols`。真实客户端由 `orchestrator` 注入。
- `agents` 只依赖 `core + tools + rag + memory + context`，不直接读写文件，一切外部副作用经工具层。
- `orchestrator` 是唯一允许"认识所有 Agent"的模块。

## 3. 模块职责

| 模块 | 职责 | 关键类 |
| --- | --- | --- |
| `core` | LLM 调用、消息抽象、Agent 基类、配置、日志、异常 | `CodeAgentXLLM`、`Message`、`Agent`、`Config` |
| `tools` | 工具定义、注册发现、安全沙箱 | `BaseTool`、`ToolRegistry`、`TerminalTool` |
| `rag` | 代码分块、向量化、检索 | `CodeChunker`、`Embedder`、`VectorStore`、`Retriever` |
| `memory` | 短期/长期记忆与笔记 | `WorkingMemory`、`EpisodicMemory`、`NoteTool` |
| `context` | GSSC 上下文流水线与压缩、远端代码源接入 | `ContextBuilder`、`ContextCompressor`、`ContextDocument`、`GitHubSource`、`documents_from_github` |
| `protocols` | 对外协议：MCP（客户端/服务端骨架/内置服务端）、GitHub REST 只读客户端、A2A 信封与路由 | `MCPClient`、`StdioTransport`、`MCPServerBase`、`FilesystemMCPServer`、`GitHubMCPServer`、`GitHubClient`、`A2ANetwork`、`A2AMessage` |
| `agents` | 审查 Agent、三种范式、七角色、结论数据结构、工具集工厂 | `ReActReviewer`、`PlanSolveRefactor`、`ReflectionReviewer`、`PlannerAgent`、`RetrieverAgent`、`FocusedReviewAgent`、`TesterAgent`、`RefactorAgent`、`ReporterAgent`、`ReviewReport`、`build_review_toolkit` |
| `prompts` | 提示词模板与输出契约 | `REPORT_CONTRACT`、`REACT_REVIEW_SYSTEM`、`PLANNER_CONTRACT`、`REVIEWER_SYSTEM`、`SECURITY_SYSTEM`、`TESTER_SYSTEM`、`PLAN_SYSTEM` |
| `orchestrator` | 七阶段流水线编排、状态持久化与中断恢复 | `CodeReviewWorkflow`、`WorkflowState`、`StageState` |
| `evaluation` | 标注集与指标（P/R/F1、误报率、定位准确率）、运行器（跑一次 + 落盘）、LLM Judge（评报告质量）、Win Rate（汇聚与两两比较） | `EvaluationDataset`、`LabeledDefect`、`evaluate`、`run_evaluation`、`EvaluationRun`、`LLMJudge`、`compare_runs` |
| `api` / `ui` | 对外服务与界面 | `main.py`、`streamlit_app.py` |

## 4. 核心数据流（一次完整审查）

```text
1. 输入        用户给定本地路径或远端 GitHub 仓库（`owner/repo`）
2. 采集        GitTool 读本地；MCP 子进程 / GitHub REST 客户端读远端 → 目录树 + 文件清单
3. 规划        PlannerAgent 产出 3~5 个子任务（解析失败 → 兜底计划并标记该阶段失败）
4. 分块+索引   CodeChunker 按 AST 切分 → Embedder → 向量库（整条流水线只建一次索引）
5. 检索        RetrieverAgent 把子任务逐条当查询词（确定性，不调 LLM）→ 带位置的证据；
               远端证据（GitHub）在同一阶段按"路径 + 行号"去重后并入证据链
6. 审查        ReviewerAgent 结合证据产出问题列表（必要时自行调 StaticAnalyzer 取证）
7. 安全        SecurityAgent 单独一轮（注入 / 凭据 / 鉴权 / 信息泄露）
8. 验证（可选）TesterAgent 生成复现测试代码，并运行仓库现有用例作为回归基线
9. 重构（可选）RefactorAgent 产出重构**计划**（不生成 diff、不改代码，需人确认后执行）
10. 汇总       ReporterAgent 确定性合并去重 → Markdown 报告；降级时摘要顶部带警告
11. 留痕       每阶段状态与产物写入 WorkflowState 并原子落盘，支持中断恢复
```

阶段顺序即 :data:`~codeagentx.orchestrator.state.STAGES`：
``plan → retrieve → review → security →（test）→（refactor）→ report``。
括号内的两个阶段默认 ``skipped``，``report`` 恒执行。

## 5. 关键设计决策

| 编号 | 决策 | 理由 |
| --- | --- | --- |
| AD-01 | 自研 `core` 层而非直接依赖 `hello-agents` 包 | 该包对 Python 3.13 兼容性未知；自研接口风格对齐 HelloAgents，零框架风险且便于面试讲解 |
| AD-02 | `Config` 放在 `codeagentx/config.py`（包根） | 与目录结构规划一致；`core` 内各模块统一 `from codeagentx.config import Config`，避免重复定义 |
| AD-03 | 使用 `src/` 布局 | 强制走安装后的包路径，避免"本地目录意外遮蔽包名"的经典坑 |
| AD-04 | 向量库抽象为 `VectorStore` 接口，Qdrant 为首选实现 | Qdrant 需要外部服务；提供本地内存/文件实现作为降级，保证无 Docker 环境下测试可跑 |
| AD-05 | 工具一律返回结构化结果而非裸字符串 | 便于 Orchestrator 判定成败、便于评估指标统计、便于日志追踪 |
| AD-06 | 所有 LLM 调用必须经过 `CodeAgentXLLM`，禁止业务层直接调 SDK | 统一统计 Token、统一重试与超时、统一 Mock 切换 |
| AD-07 | `Config` 字段名 `.upper()` 即为环境变量名 | 新增配置字段零改造成本，不需要维护"字段 → 环境变量"映射表 |
| AD-08 | 依赖方向硬约束，禁止反向 import（见第 2 节） | 防止底层被上层需求污染，保证 `core` 可独立复用与测试 |
| AD-09 | `MockLLM` 放在主库 `core/llm.py` 而非 tests | CI、无密钥演示、离线单测三处共用同一个实现 |
| AD-10 | `UsageStats.failed_calls` = 失败尝试次数（含重试后成功的） | `calls + failed_calls` 才是实际请求数，重试率可被准确统计 |
| AD-11 | `BaseTool.run()` 统一"校验 → 执行 → 计时 → 异常归一化" | 工具永不抛异常、只返回 `ToolResult`，编排层不会被工具异常击穿 |
| AD-12 | `Agent._history` 只存不含 system prompt 的对话 | 多次 `run` 复用上下文时不重复注入系统提示 |
| AD-13 | `tool_loop()` 是基类能力，子类只实现 `run()` | 三种范式共享同一套工具调用循环，且循环有硬性轮次上限 |
| AD-14 | `_translate_error` 对项目内异常直接透传 | 避免异常语义被二次包装（如 `LLMAuthError` 变成泛化的 `LLMError`） |
| AD-15 | 沙箱威胁模型只防"误用与常见破坏性操作" | 路径逃逸/危险命令/无限执行是目标；强隔离交给 W10 的容器方案 |
| AD-16 | `run_sandboxed()` 始终 `shell=False`；可执行文件解析 PATH → 解释器同级目录 | 消除 shell 注入面；venv 的 `Scripts` 常不在 PATH，`shutil.which` 会误判 |
| AD-17 | Windows 缺失的 `cat/head/tail/ls/dir` 由 Python 只读兜底 | 仅在系统确实找不到该命令时启用，Linux 上仍走真实二进制 |
| AD-18 | 命令输出统一由 `CommandOutcome.to_text()/to_meta()` 整形 | 终端类工具共用一套渲染，避免各自重复实现 |
| AD-19 | `ToolResult.success` 只表示"工具是否正常执行完" | 命令自身非零退出码（如 grep 无匹配）不算工具故障，交由模型判断 |
| AD-20 | 校验顺序硬约定：参数/安全校验 → 环境可用性 → 执行 | 安全拦截必须确定性生效，不能因为工具没装就绕过检查 |
| AD-21 | 外部依赖缺失统一降级为 `ToolResult.fail(error_type="ToolUnavailable")` | 缺 ruff/git 时应"跳过并说明"，而不是让整次审查失败 |
| AD-22 | `TestRunner` 必须带 `--override-ini=addopts=` | 目标项目的 addopts 与工具的 `-q` 叠加成 `-qq` 会让 pytest 不打印摘要，结果无法解析 |
| AD-23 | 检索语料与 embedding 输入必须逐字一致（共用 `format_chunk_header()`） | 否则 BM25 与向量两路"看的不是同一份文本"，融合失去意义 |
| AD-24 | 向量库默认内存实现，Qdrant 为可选后端；point id 用 UUID5 映射 | 零依赖即可跑通全链路；Qdrant 只接受 int/UUID 作 id，业务 `chunk_id` 存进 payload |
| AD-25 | 离线降级 `HashEmbedder(is_semantic=False)` 必须显式标注 | 绝不把"无语义向量"伪装成真 embedding（评估结论会因此失真） |
| AD-26 | 混合检索用 RRF（k=60）而非加权求和 | 语义分与 BM25 分量纲不同、不可直接相加，RRF 只看名次 |
| AD-27 | 全链路排序必须确定性（BM25 同分按 doc_id、RRF 按 `(-score, id)`、分块按路径） | 否则"可复现"无法保证，评估数字会飘 |
| AD-28 | `WorkingMemory` 以**消息单元**为单位裁剪，且最新一条永不裁剪 | `assistant(tool_calls)` 与配对 `tool` 结果必须整体进出，否则产生非法消息序列 |
| AD-29 | `memory` 可依赖 `rag`（复用 BM25 做召回），`rag` 绝不反向依赖 `memory` | 保持单向依赖，避免循环 |
| AD-30 | 记忆/笔记落盘失败只记 WARNING 不抛异常 | 记忆丢失属于降级，不该让审查任务失败 |
| AD-31 | `WorkingMemory(reserve_tokens=None)` 按 `min(1000, max_tokens // 5)` 自适应 | 写死预留量会让 `max_tokens=1000` 这类小窗口一构造就报错 |
| AD-32 | 审查结论用两级结构：`Finding`（单条问题）+ `ReviewReport`（一次结论） | 编排层、评估层、UI 层消费同一份结构；Markdown 渲染不带时间戳以保证可复现 |
| AD-33 | 模型输出的 JSON 解析容错（围栏、夹在文字中、截取最外层括号）；解析失败**必须带标记** | `metadata["parse_error"]` 存在即表示结论不可用，绝不能表现为"没有发现问题" |
| AD-34 | 字段归一化在构造函数里完成（严重度/分类/置信度/路径） | 模型写法千奇百怪（`critical`/`80%`/反斜杠路径），丢格式可以，丢问题不行 |
| AD-35 | 提示词契约集中在 `prompts/`，与 `agents.schemas` 字段一一对应 | 改契约只改一处；避免提示词散落在各 Agent 里逐渐漂移 |
| AD-36 | ReAct 终止条件 = 模型不再请求工具 **或** 达到 `max_iterations` | 后者显式返回 `success=False`，由上层决定重试，绝不假装收敛 |
| AD-37 | Plan-and-Solve 规划阶段不挂工具，执行阶段逐步骤独立 `reset` + 工具循环 | 规划时不该动代码；步骤状态由代码写回，模型不能自称"已完成" |
| AD-38 | Reflection 三条出口：`accepted` / `no_change` / `max_rounds` | 没有终止条件的 Reflection 是纯粹的烧钱机器；评分可作 W8 消融实验的量化指标 |
| AD-39 | `agents/toolkit.py` 统一装配工具集，所有工具共用同一个 `SandboxPolicy` | 避免"路径能读但 RAG 不能索引"这类权限口径不一致 |
| AD-40 | 七角色名固定为 `plan/retrieve/review/security/test/refactor/report`，与 `STAGES` 一一对应 | 角色名即阶段名：注入替身、打印状态表、写文档只有一套词汇，不会出现"角色叫 A、阶段叫 B" |
| AD-41 | Retriever 与 Reporter **不调用 LLM**（确定性角色，用量记 `ZERO_USAGE`） | 检索与合并本身是确定性操作；让模型复述一遍只会引入失真、多烧 Token，并让"来源可追溯"和"同输入同输出"失效 |
| AD-42 | Reviewer / Security 共用 `FocusedReviewAgent` 基类，只换系统提示与 `focus` | 复用 W5 的两条硬约束（未收敛即 `success=False`、解析失败带 `parse_error`），避免某个角色的失败被静默吞掉 |
| AD-43 | Planner 解析失败 → `FALLBACK_TASKS` 兜底，同时 `success=False` + `metadata["fallback"]` | 兜底计划让流水线不至于空转，但它不是模型的规划能力，必须与"规划成功"区分，否则 W8 会把兜底算成规划准确率 |
| AD-44 | 阶段状态 `pending/running/done/failed/skipped`，"跳过"与"失败"严格区分（`SETTLED_STATUSES` 只含 done/skipped） | 报告读者必须能看出"哪一步没跑、为什么没跑"；两者混同会让"没跑"被误读成"跑过且没问题" |
| AD-45 | 单阶段异常只影响该阶段（`except Exception` 记 `failed` 后继续），`report` 阶段恒执行 | 宁可给出"部分结论 + 明确的降级标记"，也不要因为一步失败丢掉整次审查 |
| AD-46 | 报告降级 ≠ 汇总阶段失败：`metadata["degraded"]` 时该阶段仍记 `done`，detail 写"含降级环节" | 降级信息由 `degraded` 与 `failed_stages` 表达；把降级当失败会让状态表自相矛盾 |
| AD-47 | `WorkflowState.snapshot()`（不含 artifacts）与 `to_dict()`（含 artifacts）分离 | 报告里会内嵌一份状态快照；若快照带着 artifacts，而 artifacts 里又存着这份报告，就形成循环引用，`json.dumps` 直接失败、状态文件写不出来（W6 测试暴露的真 bug） |
| AD-48 | `report` 阶段结束后**补一次快照** | 阶段状态在报告产出之后才最终确定，否则报告里写着 `report=pending`，看起来像这一步没跑 |
| AD-49 | 落盘用"写临时文件 + 原子替换"，失败只记 WARNING 不抛异常 | 半截 JSON 会让恢复彻底失效；状态丢失属于降级，不该让审查任务失败（与 AD-30 一致） |
| AD-50 | `_stage_test` 的 detail 按"是否真的跑了现有用例"生成措辞 | "写了测试"与"跑了测试"是证据完全不同的两件事，措辞含糊等于给读者错误的确定感 |
| AD-51 | Tester 只产出测试代码文本、不写文件；未挂载 `test_runner` 时 `available=False` | 沙箱没有写文件工具（写操作需人确认）；`available=False` 表示"未验证"，绝不能当成"通过" |
| AD-52 | Refactor 阶段只出计划（`plan_from_findings`），不执行、不生成 diff | 重构是写操作的前置，必须由人确认后再触发；"先看方案再动手"比自动改代码安全 |
| AD-53 | `parse_review_report(source=...)` 把角色名写进报告 `metadata["agent"]`，Reporter 据此统计各角色贡献 | 否则多角色报告的"参与角色与问题条数"全是 `unknown`，"来源可追溯"成了一句空话（W6 端到端示例运行验证时发现） |
| AD-54 | `context` 用**鸭子类型**接收远端源（只要求 `list_tree` / `read_file`），`ContextDocument → Evidence` 的转换放在编排层 | 若 `context` 直接 import `agents.schemas.Evidence` 就形成 `context → agents` 反向依赖；转换放唯一"认识所有 Agent"的编排层，两端互不反向依赖 |
| AD-55 | `GitHubSource` 只按 `path_prefix` 前缀语义选文件，并**跳过后缀不在白名单内**的二进制/媒体文件 | 空 `path_prefix` = 全仓；把 PNG 之类当文本读出来只会污染上下文，白名单（`GITHUB_TEXT_SUFFIXES`）比黑名单更难漏。**2026-09-17 补充**：只按后缀判断会漏掉 `README` / `LICENSE` / `Makefile` / `Dockerfile` 这类**无后缀的入口文件**（实测 `octocat/Hello-World` 全部文件据此被过滤、证据 0 条），故白名单扩为两条路——后缀在白名单内，或**无后缀且文件名**在 `GITHUB_TEXT_NAMES`（业内约定为文本的名字，大小写不敏感）；`a.out` / `data` 这类无从判断仍不收 |
| AD-56 | 远端读取走**真实 GitHub REST 协议**，`httpx.Client` 可注入 | 本机曾无法访问 `api.github.com`（证书校验失败）；`httpx.MockTransport` 回放按官方文档字段构造的响应，URL 拼装／请求头／错误映射照样真跑，离线只影响数据来源。**2026-09-17 查明真因并已直连复验**：并非网络不可达，而是本机 GitHub 加速器（Steam++ / Watt Toolkit）对 github 域名做 TLS 中间人，其根证书只装在 **Windows 证书库**，而 httpx 默认用 **certifi** → `unable to get local issuer certificate`；退出加速器后 `GET /rate_limit` → HTTP 200（证书签发者回到 Sectigo），`--live` 实测 EXIT=0 |
| AD-57 | `GitHubClient.read_file` 未显式给 `ref` 时返回的 `ref` 为 `""` | 不假装知道读的是哪个分支——证据里写上错误的版本比留空更糟 |
| AD-58 | MCP 服务端两类失败严格分开：**包内**校验失败（缺参/类型错/仓库不在白名单）→ `isError=true` 的正常响应；**报文层**问题（未知工具名、`arguments` 不是对象）→ JSON-RPC `error` | 调用方必须能区分"API 用错了"（改代码）与"这次调用没成"（记录后继续走） |
| AD-59 | `TOOL_FAILURES` 用 `ProtocolError` 而不仅是 `MCPError` | 工具内部再调外部服务失败（如 `GitHubNotFoundError`）也属"这次没成"；若漏掉，它会逃到兜底分支被误升级成 `-32603`，调用方就分不清"协议不会用"和"上游挂了"（W7 补测试时暴露） |
| AD-60 | MCP 自研 JSON-RPC 2.0 封装（不依赖 `mcp` 包），传输层抽象为 `MCPTransport` | 该包未安装；自研后测试可注入进程内假传输，协议解析、超时与"跳过无关报文"逻辑照样被测到，不必真起进程 |
| AD-61 | MCP 服务端一律**只读**工具，令牌经环境变量传给子进程、绝不进命令行 | 命令行参数会出现在进程列表与日志里；只读工具集从根上消除"模型改坏了别人仓库"的可能 |
| AD-62 | `mcp_github` 的数值上限一律**夹取**（`max(min, min(n, max))`）而非报错，`path_prefix` 前缀语义含该目录自身 | 上限是保护而非契约，夹取让模型不必猜准数字；"含自身"必须写清，否则 `src/payments` 会漏掉该目录下的直接子文件 |
| AD-63 | 上下文证据去重按**位置**（`path` + 行号区间）而非内容哈希，本地来源优先 | 同一文件同一行段从 MCP 与 GitHub 两条链路各回一次是常态；按位置去重才能既去掉重复、又保留"同一文件不同行段"的多条证据 |
| AD-64 | 远端证据获取失败只**降级**：阶段仍记 `done`，detail 写明失败原因，证据链保留本地部分 | GitHub 不可达是外部依赖问题，与本次本地审查无关，不该让整次审查失败（同 AD-21 / AD-30 / AD-45） |
| AD-65 | `ContextBuilder` 把"渲染后 token ≤ 预算"作为**硬断言**（`within_budget` 必须为真），超限先压缩再截断 | 预算失效意味着流水线随时可能在真实模型上报 400；宁可截断并按 AD-25 的思路如实标注，也不假装放得下 |
| AD-66 | A2A 只实现"信封 + 名片 + 进程内路由"，失败码用业务字符串枚举而非复用 JSON-RPC 数字码 | 跨进程/HTTP 传输与信封路由正交，等真有需求再换 `send` 的实现即可；A2A 的失败是业务级（找不到 Agent），混用数字码只会让两边都难读 |
| AD-67 | `AgentCard` 的技能必须显式声明，不从 Agent 类里"猜" | 名片写错比没有更糟——路由会据此派错活（与 AD-35 提示词契约同一思路） |
| AD-68 | `tool_loop` 轮次耗尽后**无条件**补一次不挂工具的收敛调用（`CONVERGENCE_PROMPT`），被强制的记 `metadata["forced_convergence"]` | 最后一轮仍在请求工具时，刚取回的证据模型还没读到就退出，等于白扔一轮并交出空结论；且不能只判"正文为空"——真实模型常一边说"我再看看 X"一边继续调工具，那种过程话不是结论 |
| AD-69 | `finish_reason == "length"` 必须显式暴露（`metadata["truncated"]` + `llm_output_truncated` 警告） | 被 `max_tokens` 截断的半个 JSON 只会表现为"解析失败"，让人误判成模型不会按契约输出，排查方向完全相反（已两次踩到） |
| AD-70 | `parse_json_payload` 对**裸控制字符**放行（每个候选切片先严格、再 `strict=False`），尾逗号等结构性错误仍判失败 | 真实模型写长 JSON（尤其带代码片段）常留未转义的换行；语义可完整还原就该救回来，而放行控制字符不会把缺括号/尾逗号当成合法 JSON |
| AD-71 | Reflection 的修订稿不可解析、而初稿可解析时**退回初稿**交付，并记 `report.metadata["revision_rejected"]` + `rejected_output` | 反思不是免费的：一次坏修订不该让一份合格报告归零；但"退回"必须留痕，否则 W8 会把负收益的反思记成正收益 |
| AD-72 | 工具描述必须写明**平台差异**（`ls` 不支持 `-R`、`python` 禁 `-c/-m`、`find` 在 Windows 落到语义不同的 `find.exe`），并说明路径基准是审查目标根目录 | 工具的可用性只有描述这一个出口；真实模型会照着描述里的例子去试，描述写错就是成轮次地空跑（W5 首跑 6 轮 0 结论的直接原因） |
| AD-73 | Qdrant 后端在集合**已存在**时必须校验向量维度，不一致就抛带修复提示的 `RAGError` | 换 embedding 模型后维度会变；旧集合沿用旧维度时，Qdrant 要到**写入那一刻**才抛晦涩的底层错误，且报错位置离真正原因（配置换了、索引没重建）很远 |
| AD-74 | 评估匹配两轮且**位置偏差不算误报**：先「文件 + 行号（±3 行）」，再「文件 + 关键词」兜底；定位质量另用 `location_precision`（行号精确命中 / TP）单独表达 | 同一问题换个说法、行号差几行，本质是"找对了但定位糙"；把它算成误报会同时抹黑查准率与查全率。分开报才能区分"内容不可用"与"需要人工再定位一次" |
| AD-75 | 每条标注缺陷**只认一次**，同一缺陷被报第二遍计 **FP** | 重复条目对使用者就是噪声（要人工读两遍、还可能给出互相矛盾的严重度）；若按"只要提到就算命中"，多角色流水线会靠复述同一句话刷高查全率 |
| AD-76 | 解析失败 → 该次运行 **`valid=False`**；无效运行单独计数（`invalid_runs`）且**不进任何均分** | "模型没按契约输出"与"零问题、零误报"在数字上完全同形，混在一起会把一次失败伪装成一次完美表现；把它当 0 分又会伪造出"某方案更差" |
| AD-77 | LLM Judge **只评报告本身**（正确性/具体性/可执行性/证据/严重度标定），**不喂人工标注**，并就地截取目标代码片段供核对 | 把标注当参考答案递给判官，等于让它对着答案打分，分数再高也不能说明系统的发现能力；给代码片段是为了让"事实性错误"有可核对的依据 |
| AD-78 | 胜率只在**双方都跑过且都有效**的同一数据集上两两比；并列各记半场；`ComparisonResult` 强制自带 `caveat` 写明样本量 | 一边跑 3 个仓库、另一边跑 1 个，平均值不可比；单仓库单次的"胜率 100%"没有统计意义，结论必须把样本量一起贴出来，不能只给一个百分比 |
| AD-79 | 评估层**不导入编排层**：`outcome_from_workflow` 按鸭子类型读 `.report` / `.state` | 评估要能评估任意形态的审查流程（单 Agent / ReAct / 七阶段 / 将来的 MCP 流程）；一旦反向依赖编排层，评估层就被绑死在一种实现上，也会形成层次倒置 |
| AD-80 | 成本只报 token 数与调用次数，**不做单价换算**；评估脚本**不提供 `--mock`**，未配密钥直接退出 | 单价随模型与渠道变化，换算是使用方的事，写死金额很快就会过期；而用脚本化输出算出来的 F1 是"对着标注自证"，比没有指标更危险 |
| AD-81 | 评估时**先统一建一次索引**，并把"索引已就绪、路径基准是仓库根"作为所有方案一致的提示 | 否则"谁先跑谁付建索引成本"，对比不公平；且真实模型会自己再 index 一次、并把"相对项目根"的显示路径当参数传下去（沙箱根是仓库目录本身），白跑一轮并连带拉低该方案的表现 |
| AD-82 | 多数据集评估时，**每个仓库跑之前重建索引**（而非全程只建一次）；跨仓库汇总同时报**宏平均**与微平均 | 向量库是**全局单集合**（`codeagentx_code`），`reset=True` 会清空整个集合——只建一次的话，第二个仓库开始检索到的还是第一个仓库的代码；宏/微平均口径不同（前者每个仓库同权、后者被上报条数多的仓库主导），只报一个等于替读者挑了好看的那个 |
| AD-83 | 重复上报去重分两道：精确键 `(文件, 行号, 分类, 小写标题)` 之外，加「同文件 + 行号相差 ≤6 + 标题**换了措辞**（重叠系数 ≥0.5）」；**标题逐字相同者一律不合并**；标题特征太少（<4）不参与近似合并 | workflow 的误报主因就是同一缺陷被不同角色换措辞报多遍（判官独立点名）。只按精确键去重会漏掉"换了措辞"的重复；但近似合并放宽会**吞掉真缺陷**（同一行上常有"缺鉴权"与"`post_id` 未校验"两个不同问题），因此阈值刻意保守、且要求"措辞变了"才算重复——同一句话出现在不同行号，更可能是同一问题在两处不同位置。特征数阈值是为杜绝"问题 A"与"问题 B"只差一个字符被判成完全相似 |
| AD-84 | 重复运行（`--repeat N`）时，**两两胜率先把同一「数据集 × 方案」的 N 次取均值再比**，汇总表也用均值；并另打印**波动表**（同组极差均值/最大），`caveat` 必须披露重复次数 | 不取均值的话，胜负会由"哪一次恰好留在映射里"决定，重复运行等于白跑；而只报均值差、不报波动，读者会把纯随机差异当成方案差异——实测单组极差平均 **0.1320**（最大 0.2321），**小于这个数的方案差距根本不该判方向**（上一版"react 与 reflection 差 0.0029 实质打平"的结论就是被这条推翻的） |
| AD-85 | 远端仓库按 **zipball 归档一次请求**落到临时工作区（`/repos/{owner}/{repo}/zipball/{ref}`），不用 git clone，也不用逐文件 Contents API | 逐文件读要 N+1 次请求，未认证配额只有 **60 次/时**，几十个文件的仓库一轮就打满；而本机/CI 容器**未必有 git**（本仓库 4 个 skip 里 2 个就是 git 相关用例）。归档接口一次拿整棵树，代价（多下几个无关文件）远小于配额耗尽 |
| AD-86 | 归档解压一律按**不可信输入**处理：剥掉 zipball 的单层顶层目录、拒绝符号链接与 `..`/盘符冒号条目、`resolve()` 后再校验一次"没逃出工作区"、单文件/总字节/文件数三重上限（前两者超限跳过并记账，文件数与总字节超限**显式失败**）、`.git` 等 VCS 目录不入工作区 | zip 来自外部，路径逃逸能把文件写到工作区之外（zip-slip）；上限是防止"一个仓库 zip 写满磁盘"这类不可控后果。**超限必须报错而不是默默截断**——半个仓库被当成"全部代码"审完，结论是假的 |
| AD-87 | 远端工作区的生命周期由 CLI 独占：`tempfile.mkdtemp("codeagentx-remote-")` 建、成功/下载失败/流水线抛异常**三条路径都删**，只有显式 `--keep-workdir` 才保留并打印路径 | 别人的整棵仓库留在磁盘上既占空间又绕过"审查完即离开"的预期；下载失败时更容易留下半截目录被下次误用 |
| AD-88 | **范围决定（2026-09-18，用户拍板）：不做"可验证的修复补丁"。** 沙箱**不新增写文件工具**，Refactor 保持"只出计划不执行"、Tester 保持"只给测试代码文本 + 跑现有用例"（与 AD-51 / AD-52 一致）；系统对外承诺收敛为**输入本地代码库或 GitHub 仓库 → 输出结构化审查报告 + 风险定位 + 重构建议** | 自动改代码并要求"补丁可通过编译与测试"意味着把写操作与执行权都交给 Agent，与 AD-15 的威胁模型（只防误用，不做强隔离）不匹配；在没有人复核的前提下产出补丁，收益（省一次手改）远小于风险（静默改坏别人的仓库）。**因此 `requirements.md` 的"修复通过率"指标不再适用**，评估口径以 AD-74~AD-84 为准 |
| AD-89 | 「只审一个文件」= 沙箱根**仍取父目录**，但审查**范围**（`CodeReviewWorkflow.paths`）限定到该文件：规划只拿到范围内的文件清单、检索证据与最终报告都剔除范围外内容；范围路径不存在或跳出根目录时**直接报错** | RAG 索引与路径守卫都要求根是目录，把根换成文件要改动 `rag`/`tools` 两层的语义；"范围"只是编排层的过滤，改动面小得多。三处都要过滤是因为只过滤一处就会漏：规划不限 → 子任务跑去审别的文件；证据不限 → 模型拿着别处的片段下结论；报告不限 → 用户会看到"这个文件"的标题下挂着整个仓库的问题。范围外的代码仍可被工具读作上下文，但**不作为上报对象**；静默退化成"审全仓库"比报错更贵，所以越界/不存在一律报错 |
| AD-90 | 对外服务用**「提交 + 轮询」作业模型**：`POST /review` 同步完成校验后返回 `202 + job_id`，`GET /review/{id}` 取状态与结果；作业与结果**只存内存**，进程重启即丢 | 一次审查要几分钟（实测 `data/sample_repo` 94.9s），同步等着返回会让 HTTP 连接被占住、被网关/代理掐断，调用方也拿不到"进度到哪了"。作业不落库是刻意的：这是**单人自托管**服务，没有多实例、没有恢复需求，引入持久化只会换来一套要维护的迁移与清理逻辑。代价（重启丢失历史作业）如实写在接口文档与断言里，不假装成"作业永不丢" |
| AD-91 | 服务端**单工作线程串行**执行审查（`MAX_CONCURRENT_REVIEWS = 1`），作业只排队不并发 | 向量库是**全局单集合**（AD-82），两个审查并发跑会互相检索到**对方的代码**并以此当证据下结论——给出"用别人代码当证据"的审查报告比排队等待坏得多。要做并发必须先给每个作业独立 collection 与独立索引，那是另一个量级的改动；W10 的目标是"别人能用上"，不是"能扛并发" |
| AD-92 | 「目标 → 工作区」抽成公共层 `src/codeagentx/workspace.py`（`prepare_target()` 返回 `PreparedTarget`：`root` / `display` / `paths` / `workdir` / `cleanup()`），CLI 与 HTTP API **共用同一份** | 单文件审查（AD-89）与远端仓库"用完即删"（AD-87）都是**两条入口都必须遵守**的规则；各写一份的结果就是规则迟早分叉（CLI 会过滤范围、API 不会），而这类分叉不会被任何单侧测试发现。共用后测试也随之只有一份（目标判定用例整体从 `test_cli.py` 搬到 `test_workspace.py`） |
| AD-93 | 对外接口**不暴露 `enable_test`**，且请求体 `extra="forbid"` | 沙箱没有容器隔离（AD-15），让 HTTP 调用方一句话就跑目标仓库自带测试，等于把"执行别人的代码"的决定权交到网络上；CLI 保留该开关是因为它跑在**自己的机器、自己的目标**上，风险归本人。禁掉多余字段是让"传了却没生效"变成 **422** 而不是静默忽略——静默忽略会让人以为测试跑过了 |
| AD-94 | 配置类错误在**提交时同步报出**：`ReviewService.submit()` 未配置 `LLM_API_KEY` 直接抛 `ConfigError` → HTTP **503**（带中文说明），不进入队列 | 否则调用方拿到 202、轮询几分钟，最后才发现"服务端根本没配密钥"——错误发现得越晚越贵。同理，密钥只从服务端 `.env` 读，**接口不接受调用方传入密钥**（否则等于开了个免费代理） |
| AD-95 | 镜像**不内置任何密钥**：`.env` 靠 `env_file` / `--env-file` 在运行时注入，`.dockerignore` 排除 `.env`；容器以非 root 用户（uid 10001）运行 | 密钥打进镜像层就等于进了 registry 与本地缓存，删文件也删不掉（层还在）；`COPY .env` 这种写法一旦养成习惯，迟早推到公开仓库。镜像内只装 `[serve]` 依赖与 `data/sample_repo`（演示目标），保证"不装 Python 环境也能验证服务能不能起" |
| AD-96 | ReAct 角色（Reviewer / Security 共用 `ReActReviewer.run`）的结论**解析不了时，用同一段对话历史再补问一次、这次只要 JSON**（`JSON_REPAIR_PROMPT`，不挂工具）；成功则采纳新结论并把 `result.output` 换成被采纳的那份，失败则维持原判（`parse_error` → 阶段 failed）。**只补问一次，不重试整个阶段** | 真实故障：Reviewer 打满 `max_iterations` 被强制收敛（AD-70）时，手里工具结果已经很多，容易顺手写出一大段"总结式"自由文本而不是 JSON——全部日志里 37 次 `review` 阶段执行有 2 次如此，**整段结论直接丢失**（`security` 不受影响，所以报告看起来"只有一半问题"）。三条候选里：重跑整个 review 阶段要再花一整轮取证（十几倍贵）；放宽 JSON 解析器（容尾逗号/未转义引号）会把坏 JSON 当好 JSON，与既有"结构错误必须拒"的决定冲突；**补问一次只多一次调用**（仅在解析失败时触发，实测约 5% 的执行会走到），且不改变"解析失败=该次审查无效"的既有语义——失败仍然判失败 |
| AD-97 | 补问那一次调用的用量**合并进该角色的 `AgentResult.usage`**（`_merge_usage`），并打 `review_json_repair_requested` / `review_json_repaired` / `review_json_repair_failed` 三个日志事件 | 补救调用是真花钱的 LLM 调用：不并进用量，"token 成本"指标就会**低估**这条路径的代价，等于用记账漏洞掩盖成本；日志事件是这次修复唯一的线上观测手段——故障本身约 5% 才复现一次，没有日志就永远不知道补救到底有没有生效 |

## 6. 安全设计要点

| 层次 | 措施 |
| --- | --- |
| 命令执行 | 白名单命令 + 参数黑名单（`rm -rf`、`curl | sh` 等） + 超时 + 工作目录限制 |
| 文件访问 | 所有路径经 `resolve()` 后校验必须位于允许的根目录内，拒绝符号链接逃逸 |
| 归档解压 | 远端 zip 按不可信输入处理：剥顶层目录、拒符号链接与 `..`/盘符条目、`resolve()` 后二次校验未逃出工作区、单文件/总字节/文件数上限（超限显式失败）（AD-86） |
| 网络 | MCP/GitHub 仅允许白名单域名；Token 只从环境变量读取，不写入日志 |
| LLM | 提示词中注入的代码内容做分隔标记，降低提示注入风险 |
| 输出 | 报告中的文件路径一律相对化，避免泄露本机绝对路径 |

## 7. 可观测性

- 日志双通道：控制台（rich 美化）+ 文件（JSON Lines，便于分析）。
- 每次 LLM 调用记录：模型、输入/输出 token、耗时、是否重试。
- 每次工具调用记录：工具名、参数摘要、耗时、是否被安全策略拦截。
- 每个阶段记录 `stage_finished`（阶段名 / 状态 / detail / error，`failed` 走 WARNING 级别）；
  阶段内抛异常记 `stage_crashed`，并只影响该阶段。
- 阶段状态与产物写入状态文件（`state_save_failed` / `workflow_resume_unavailable` 等异常路径也有日志）；
  一行摘要 `WorkflowState.describe()` 进日志，轻量快照进最终报告的 `metadata.workflow`。
- 一次审查生成一个 `run_id`，贯穿全部日志与产出文件。
