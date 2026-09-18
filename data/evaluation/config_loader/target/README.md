# 配置加载器（config_loader）

这是 CodeAgentX 评估用样本仓库，模拟一个**配置加载与缓存**的小库。
缺陷是**刻意植入的测试数据**，请勿修复——它们是评估的 ground truth。

为了不引入第三方依赖（本机没装 PyYAML），本库只解析 `json` 配置，
接口与真实项目保持一致，属于**等价简化**：真实项目可能换成 YAML/TOML/配置中心，
但"解析、校验、缓存、读取失败处理"这几个环节的缺陷形态是相同的。

## 目录结构

```
configlib/
  loader.py        加载入口：解析配置值并写日志（缺陷：eval 求值、敏感值落日志）
  schema.py        配置结构：期望类型与取值（缺陷：内置默认口令、不校验类型）
  cache.py         配置缓存（缺陷：缓存键不含来源/环境）
  sources.py       配置来源读取（缺陷：读失败静默回退默认配置）
```

## 请勿"修好"这里的缺陷

下表的 `id` 与评估标注 `data/evaluation/config_loader/labels.json` 一一对应：
标注只允许来自这张表，表里没有的一律不算 ground truth
（防止把模型事后发现的问题补进标签，污染评测结果）。

| id | 位置 | 缺陷 |
| --- | --- | --- |
| `eval-config-value` | `configlib/loader.py` | 用 `eval()` 解析配置值，配置文件可执行任意代码 |
| `default-admin-password` | `configlib/schema.py` | 内置默认管理员口令，未配置时静默使用 |
| `cache-key-ignores-source` | `configlib/cache.py` | 缓存键只用文件名、不含来源/环境，多环境配置互相串味 |
| `no-type-validation` | `configlib/schema.py` | 配置值不做类型校验，字符串被当整数用 |
| `sensitive-value-logging` | `configlib/loader.py` | 加载配置时把整份配置（含口令/密钥）打进日志 |
| `silent-default-on-error` | `configlib/sources.py` | 读取失败时静默回退到默认配置，不抛错也不告警 |
