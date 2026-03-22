# Train Guard Agent 对接文档

本文档以 `../train-guard-dashboard` 当前实现为准。

## 1. 配置拉取

当 `TrainingAgent` 拿到以下四元组时，会自动拉取远程配置：

- `backend_base_url`
- `access_key_id`
- `secret_key`
- `project_id`

缺任何一个都会抛出：

```python
ValueError("后端自动拉取配置要求 backend_base_url、access_key_id、secret_key、project_id 同时提供")
```

也可以通过本地配置提供：

- `backend.base_url`
- `backend.access_key_id`
- `backend.secret_key`
- `backend.project_id`

请求定义：

- Method: `GET`
- URL: `{backend_base_url}/api/agent/config`
- Headers:

```http
X-Access-Key-Id: <access_key_id>
X-Secret-Key: <secret_key>
X-Project-Id: <project_id>
```

成功状态码要求：

- `200`

推荐响应：

```json
{
  "code": 0,
  "data": {
    "config": {
      "server": {
        "url": "https://your-domain.com/api/metrics/ingest",
        "timeout": 10,
        "retry_count": 3
      },
      "agent": {
        "upload_frequency": "epoch",
        "upload_interval": 1,
        "enable_async": true
      }
    }
  }
}
```

## 2. 指标上传

最终上报地址优先级：

1. `TrainingAgent(server_url=...)`
2. 远程配置里的 `server.url`
3. 本地配置里的 `server.url`

请求定义：

- Method: `POST`
- URL: `{server.url}`
- Headers:

```http
X-Access-Key-Id: <access_key_id>
X-Secret-Key: <secret_key>
X-Project-Id: <project_id>
Content-Type: application/json
```

成功状态码：

- `200`
- `201`
- `202`

当前 agent 上报的是 dashboard 友好的扁平事件，例如：

```json
{
  "train_id": "demo_001",
  "epoch": 3,
  "batch": 20,
  "step": 120,
  "timestamp": 1770000000.12,
  "loss": 0.245,
  "accuracy": 0.91,
  "learning_rate": 0.0003,
  "batch_time": 0.13,
  "epoch_time": 12.8,
  "system_cpu_percent": 42.5
}
```

这类结构可以被 dashboard worker 直接解析成时序点。

## 3. Python 最小示例

```python
from train_guard.agent import TrainingAgent

agent = TrainingAgent(
    backend_base_url="http://localhost:8000",
    access_key_id="ak_xxx",
    secret_key="sk_xxx",
    project_id="project_abcd1234",
    train_id="demo_001",
)
```
