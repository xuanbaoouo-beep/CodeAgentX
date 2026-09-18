# 文件服务（file_service）

这是 CodeAgentX 评估用样本仓库，模拟一个**文件上传/下载服务**。
缺陷是**刻意植入的测试数据**，请勿修复——它们是评估的 ground truth。

为了不引入第三方依赖（本机没装 Flask/FastAPI），接口层用普通函数**等价简化**：
`fileservice/api.py` 里的 `download` / `upload` 就对应真实项目里的路由处理函数，
请求与响应都退化成普通字典，其余逻辑（路径拼接、存储、上传、临时文件）保持一致。

## 目录结构

```
fileservice/
  api.py           接口层：下载/上传端点的等价简化实现（缺陷：越权下载）
  paths.py         路径解析：把用户相对路径映射到存储根目录（缺陷：路径穿越）
  storage.py       存储层：落盘与调用外部转换工具（缺陷：命令注入）
  upload.py        上传处理：读取上传流与类型校验（缺陷：无大小限制、类型可伪造）
  tempfiles.py     临时文件工具（缺陷：临时文件与文件句柄均不清理）
```

## 请勿"修好"这里的缺陷

下表的 `id` 与评估标注 `data/evaluation/file_service/labels.json` 一一对应：
标注只允许来自这张表，表里没有的一律不算 ground truth
（防止把模型事后发现的问题补进标签，污染评测结果）。

| id | 位置 | 缺陷 |
| --- | --- | --- |
| `path-traversal-download` | `fileservice/paths.py` | 下载路径直接与根目录拼接，未 `resolve()` + 前缀校验（路径穿越） |
| `no-authz-on-download` | `fileservice/api.py` | 下载接口不校验文件归属者，任意用户可下载他人文件 |
| `unbounded-upload-size` | `fileservice/upload.py` | 上传不限制大小，一次性读进内存 |
| `unvalidated-content-type` | `fileservice/upload.py` | 只信客户端声明的 content-type，不校验真实内容（可传可执行脚本） |
| `shell-injection-filename` | `fileservice/storage.py` | `shell=True` 且把用户文件名拼进命令（命令注入） |
| `temp-file-leak` | `fileservice/tempfiles.py` | 临时文件用完不清理，`mkstemp` 的文件句柄也不关闭 |
