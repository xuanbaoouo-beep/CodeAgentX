# 评估样本仓库：url_shortener

这是 **CodeAgentX 评估用样本仓库**。仓库里的缺陷是**刻意植入的测试数据**，
用于离线验证审查 Agent 的检索与定位效果，**请勿修复**。

被审目标是一个最小的短链服务（只用标准库，**不依赖 `requests`**：
需要发起 HTTP 的地方以占位函数 + 注释说明"此处等价于 HTTP 请求"），
规模刻意做小，便于人工核对"标注的行号区间里到底是不是那段代码"。

## 目录结构

```
app/
  __init__.py     包初始化
  store.py        短链存储与点击计数（无缺陷）
  shortener.py    短码生成与创建短链
  validators.py   URL 校验
  preview.py      链接预览（按 URL 抓取标题）
  api.py          HTTP 接口层：创建短链、跳转与统计
```

## 请勿"修好"这里的缺陷

本仓库**故意保留了几处典型缺陷**，供评估使用
（不要把它们当成需要修复的 bug，它们是测试数据）：

| id | 位置 | 缺陷 |
| --- | --- | --- |
| `predictable-short-code` | `app/shortener.py` | 短码用自增计数器生成，可被枚举 → 任意人可遍历他人短链 |
| `open-redirect` | `app/api.py` | 跳转前不校验目标域名白名单，可被用作钓鱼跳板 |
| `ssrf-preview-fetch` | `app/preview.py` | 服务端根据用户传入的 URL 去抓取预览，未做内网地址 / 协议拦截（SSRF） |
| `missing-scheme-validation` | `app/validators.py` | 只检查字符串非空就当作合法 URL，`javascript:` / `file:` 也能通过 |
| `unauth-stats-endpoint` | `app/api.py` | 统计接口无任何鉴权，任意调用者能拿到全站数据 |

> 上表的 `id` 与评估标注 `data/evaluation/url_shortener/labels.json` 一一对应：
> 标注只允许来自这张表，表里没有的一律不算 ground truth
> （防止把模型事后发现的问题补进标签，那是对评测结果的污染）。

## 运行说明

仓库不依赖任何第三方包，直接 `import app.api` 即可。
本仓库**不提供**可执行入口与测试，仅作为静态审查的素材。
