# blog_api（评估样本仓库）

这是 CodeAgentX 评估用的**样本仓库**，模拟一个"博客 HTTP 接口 + 认证层"的服务，
刻意做小（分层：api 路由层 / services 业务层 / db 数据访问层 / models 数据结构），
便于人工核对"审查 Agent 报出来的缺陷到底对不对"。

> **警告：本仓库里存在刻意植入的缺陷，它们是测试数据，请勿"修好"。**
> 这些缺陷被下方的缺陷表逐条记录，是评估时唯一的 ground truth 来源。

## 关于"Flask 风格"

真实项目里这一层通常用 Flask 写，但本机没有安装 flask，因此这里用**纯标准库做等价简化**：

- 不 `import flask`，路由就是普通函数，命名成 `handle_xxx(request)`；
- `request` 是已解析好的请求字典（例如 `{"username": "alice", "page_size": 50}`）；
- 返回值 `{"status": ..., "body": ...}` 对应真实框架里的状态码与响应体。

除了没有框架胶水代码，分层、调用关系与真实项目一致，因此审查结论可以迁移。

## 目录结构

```
app/
  __init__.py
  api/
    __init__.py
    auth.py             认证层：签发会话令牌、核对口令
    posts.py            HTTP 路由层：登录、帖子列表、按标签查询、删帖
  services/
    __init__.py
    post_service.py     业务层：分页取帖、按标签筛选、删除帖子
  db/
    __init__.py
    store.py            数据访问层（内存表，接口对齐真实数据库）
  models/
    __init__.py
    post.py             帖子与用户的数据模型
README.md
```

## 请勿"修好"这里的缺陷

下表是本仓库**刻意植入并被文档记录的缺陷**，供审查 Agent 识别。
它们不是待修复的 bug，而是评估用的测试数据。

| id | 位置 | 缺陷 | severity |
| --- | --- | --- | --- |
| `hardcoded-jwt-secret` | `app/api/auth.py` | 会话/JWT 签名密钥硬编码在源码里 | high |
| `missing-admin-authz` | `app/api/posts.py` | 管理端删帖接口没有任何鉴权/权限校验，任何调用者都能删 | high |
| `sql-string-concat` | `app/db/store.py` | 按标签查询帖子时用字符串拼接构造 SQL（注入风险） | high |
| `unbounded-page-size` | `app/api/posts.py` | 分页参数 `page_size` 不设上限，可被用来拖垮服务 | medium |
| `idor-delete-any-post` | `app/services/post_service.py` | 删除帖子不校验帖子归属者（越权删除他人帖子） | high |
| `plaintext-password-compare` | `app/api/auth.py` | 口令用普通 `==` 比对且直接比较明文（未哈希、非常量时间比较） | medium |

> 上表的 `id` 与评估标注 `data/evaluation/blog_api/labels.json` 一一对应：
> 标注只允许来自这张表，表里没有的一律不算 ground truth
> （防止把模型事后发现的其它问题补进标签，那会污染评测结果）。

## 怎么跑起来看

本仓库只依赖标准库，可以直接 import 观察各层行为（`-B` 是为了不生成 `__pycache__`）：

```powershell
.\.venv\Scripts\python.exe -B -c "import sys; sys.path.insert(0, 'data/evaluation/blog_api/target'); from app.db import store; from app.models.post import Post, User; store.save_user(User('alice', 's3cret')); store.insert_post(Post(1, 'hello', 'alice', ['python'])); from app.api.posts import handle_list_posts, handle_list_posts_by_tag, handle_delete_post; print('list =', handle_list_posts({'page_size': 100000000})['status'], '| by_tag =', handle_list_posts_by_tag({'tag': 'python'})['status'], '| delete =', handle_delete_post({'post_id': 1}))"
```

上面这条会打印 `list = 200 | by_tag = 200 | delete = {'status': 200, 'body': {'deleted': 1}}`——
即分页不设上限照样返回、按标签查询照常工作、未鉴权也能删帖成功。

也可以只做语法/导入检查（不产生 `__pycache__`）：

```powershell
.\.venv\Scripts\python.exe -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('data/evaluation/blog_api/target').rglob('*.py')]; print('syntax ok')"
```
