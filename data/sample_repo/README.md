# 示例仓库（sample_repo）

这是 CodeAgentX 用来做**离线演示与效果验证**的小型 Python 项目，
模拟一个"用户认证 + 登录接口"的服务，规模刻意做小（4 个模块），
便于人工核对"检索到的片段到底对不对"。

## 目录结构

```
app/
  auth/service.py      用户登录/登出与会话逻辑（检索演示的主要目标）
  auth/models.py       用户与会话数据模型
  api/routes.py        HTTP 路由层，调用认证服务
  db/repository.py     数据访问层（内存表，接口对齐真实项目）
```

## 请勿"修好"这里的缺陷

本仓库**故意保留了几处典型缺陷**，供后续周次的审查 Agent 识别
（不要把它们当成需要修复的 bug，它们是测试数据）：

| id | 位置 | 缺陷 |
| --- | --- | --- |
| `hardcoded-secret` | `auth/service.py` | 密钥硬编码在源码里 |
| `weak-password-hash` | `auth/service.py` | `hash_password` 用无盐 sha256（弱哈希） |
| `missing-input-validation` | `auth/service.py` | `login` 未校验 `username` / `password` 是否为空 |
| `logout-noop` | `auth/service.py` | `logout` 是空实现，token 实际未失效 |
| `exception-leak` | `api/routes.py` | 把内部异常信息直接返回给客户端（信息泄露） |
| `no-login-rate-limit` | `api/routes.py` | 没有登录失败次数限制 |
| `sql-string-concat` | `db/repository.py` | 用字符串拼接构造 SQL（注入风险） |

> 上表的 `id` 与评估标注 `data/evaluation/sample_repo/labels.json` 一一对应：
> 标注只允许来自这张表，表里没有的一律不算 ground truth（防止把模型事后发现的问题补进标签）。

## 检索演示

```powershell
.\.venv\Scripts\python.exe examples\rag_search.py --repo data\sample_repo "用户登录逻辑在哪"
.\.venv\Scripts\python.exe examples\rag_search.py --repo data\sample_repo "SQL 注入风险"
```
