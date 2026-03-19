# PyTorch Training Agent 对接协议文档

本文档用于后端联调，重点说明两部分：

1. 训练指标上报接口（Agent -> 指标接收服务）
2. 配置下发接口（Agent -> 配置后端）

---

## 1. 指标上报接口

### 1.1 基本信息

- 触发方：训练进程中的 `TrainingAgent`
- 目标地址：`server.url`
- 请求方法：`POST`
- 超时时间：`server.timeout`（默认 `10` 秒）
- 重试次数：`server.retry_count`（默认 `3` 次）
- 退避策略：指数退避，等待时间依次为 `1s`, `2s`, `4s`...

### 1.2 成功判定

Agent 仅将以下状态码视为上传成功：

- `200`
- `201`
- `202`

其他状态码（包括 `204`）都会被判定为失败并进入重试逻辑。

### 1.3 请求体外层格式（固定）

```json
{
  "timestamp": "2026-03-19T13:21:02.527408",
  "train_id": "integration_test",
  "metrics": {
    "...": "..."
  }
}
```

字段说明：

- `timestamp`: ISO 8601 时间字符串（由 Agent 生成）
- `train_id`: 训练任务标识
- `metrics`: 指标对象（不同上传模式结构不同）

---

## 2. metrics 字段格式

`metrics` 的结构取决于上传模式。

### 2.1 Epoch 模式（`upload_frequency=epoch`）

触发时机：`with agent.epoch_context():` 退出时。

示例：

```json
{
  "epoch": 5,
  "timestamp": 1760000000.123,
  "loss_avg": 0.234,
  "loss_min": 0.120,
  "loss_max": 0.450,
  "accuracy_avg": 0.912,
  "accuracy_min": 0.800,
  "accuracy_max": 0.950,
  "learning_rate_avg": 0.0005,
  "learning_rate_min": 0.0005,
  "learning_rate_max": 0.001,
  "batch_time_avg": 0.041,
  "batch_time_min": 0.030,
  "batch_time_max": 0.070,
  "epoch_time_avg": 12.83,
  "epoch_time_min": 12.83,
  "epoch_time_max": 12.83,
  "f1_score_avg": 0.88,
  "f1_score_min": 0.86,
  "f1_score_max": 0.90,
  "system_info": {
    "timestamp": 1760000000.125,
    "cpu_percent": 33.4,
    "memory_percent": 58.2,
    "memory_available_mb": 10240.5,
    "gpu_allocated_mb": 2048.0,
    "gpu_reserved_mb": 3072.0,
    "gpu_count": 1
  }
}
```

说明：

- 每个已采集指标都会展开为三元组：`{name}_avg`, `{name}_min`, `{name}_max`
- `system_info` 由 `metrics.include_system_info` 控制
- Epoch 上传成功后，Agent 会清空内部 buffer，进入下一轮统计

### 2.2 Batch 模式（`upload_frequency=batch`）

触发时机：每 `upload_interval` 个 batch 上传一次。

示例：

```json
{
  "epoch": 3,
  "batch": 120,
  "timestamp": 1760000000.567,
  "loss": {
    "current": 0.21,
    "mean": 0.33,
    "min": 0.18,
    "max": 0.62,
    "count": 120
  },
  "accuracy": {
    "current": 0.94,
    "mean": 0.90,
    "min": 0.72,
    "max": 0.96,
    "count": 120
  },
  "learning_rate": {
    "current": 0.0005,
    "mean": 0.0007,
    "min": 0.0005,
    "max": 0.001,
    "count": 120
  },
  "batch_time": {
    "current": 0.038,
    "mean": 0.041,
    "min": 0.029,
    "max": 0.070,
    "count": 120
  },
  "system_info": {
    "timestamp": 1760000000.570,
    "cpu_percent": 31.2,
    "memory_percent": 57.8
  }
}
```

说明：

- 每个指标是对象格式：`current / mean / min / max / count`
- 自定义指标（`agent.record_metric("name", value)`）也遵循同样结构

### 2.3 手动上传（`manual_upload`）

手动上传使用与 Batch 模式相同的 `metrics` 结构，并额外包含可选字段：

```json
{
  "description": "checkpoint_epoch_10"
}
```

该字段位于 `metrics` 对象内部。

---

## 3. 配置下发接口（后端配置服务）

### 3.1 触发条件

Agent 初始化时，只要 `backend_api + app_id + app_secret` 三项同时具备，就会请求后端拉取配置。

- 三项都存在：发起请求
- 三项只存在部分：直接抛错（不会请求）

### 3.2 请求格式

```http
POST {backend_api}
X-App-Id: {app_id}
X-App-Secret: {app_secret}
Content-Type: application/json

{
  "app_id": "{app_id}",
  "app_secret": "{app_secret}"
}
```

说明：

- 超时使用 `backend.timeout`（默认 `10` 秒）
- 成功状态码必须是 `200` 或 `201`

### 3.3 响应解析规则

响应必须是 JSON 对象，Agent 会按以下规则提取配置：

1. 优先尝试包装结构中的 `config`：
   - `response.config`
   - `response.data.config`
   - `response.result.config`
2. 若不存在，再尝试 `response.data` 或 `response.result`
3. 若仍不匹配，尝试 `response` 顶层对象

当对象中出现以下任一键时，会被识别为“可用配置”：

- `backend`
- `server`
- `agent`
- `metrics`
- `model`
- `training`
- `server_url`

### 3.4 支持的响应样式

#### 样式 A：推荐完整结构

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "config": {
      "server": {
        "url": "https://metrics.example.com/v1/collect",
        "timeout": 8,
        "retry_count": 2
      },
      "agent": {
        "upload_frequency": "epoch",
        "upload_interval": 1,
        "enable_async": true
      },
      "metrics": {
        "include_system_info": true,
        "monitor": {
          "loss": true,
          "accuracy": true,
          "learning_rate": true,
          "batch_time": true,
          "epoch_time": true,
          "model_size": true
        }
      }
    }
  }
}
```

#### 样式 B：最简结构（只下发上报地址）

```json
{
  "data": {
    "server_url": "https://metrics.example.com/v1/collect"
  }
}
```

#### 样式 C：直接返回配置

```json
{
  "server": {
    "url": "https://metrics.example.com/v1/collect"
  },
  "agent": {
    "upload_frequency": "batch",
    "upload_interval": 10
  }
}
```

### 3.5 `server_url` 兼容规则

若响应中包含 `server_url`，Agent 会自动标准化为：

```json
{
  "server": {
    "url": "..."
  }
}
```

若同时存在 `server.url` 与 `server_url`，以 `server.url` 为准。

---

## 4. 配置合并优先级

最终生效配置优先级（低 -> 高）：

1. Agent 内置默认配置
2. 本地用户配置文件
3. 后端下发配置
4. 初始化参数 `server_config / agent_config / metrics_config`
5. 初始化参数 `server_url`（对 `server.url` 最终覆盖）

---

## 5. 后端对接建议

### 5.1 指标接收接口建议

- 始终返回 `200/201/202` 之一
- 支持幂等：建议按 `(train_id, metrics.epoch, timestamp)` 去重
- 对 `metrics` 做 schema 宽松校验（允许自定义指标）

### 5.2 配置下发接口建议

- 返回样式 A（`data.config`）可读性最好
- 至少保证提供可用的 `server.url` 或 `server_url`
- 如需灰度控制，可按 `app_id` 返回不同采样频率与监控项

### 5.3 联调检查清单

- [ ] 配置接口返回码是否为 `200` 或 `201`
- [ ] 配置接口响应是否为 JSON 对象
- [ ] 返回体中是否包含可识别配置键（如 `server` 或 `server_url`）
- [ ] 指标接口是否接受 `metrics` 为动态结构
- [ ] 指标接口是否返回 `200/201/202`

---

## 6. 快速联调样例

### 6.1 配置接口样例（伪代码）

```python
@app.post("/api/agent/config")
def get_agent_config(body, headers):
    app_id = body.get("app_id")
    app_secret = body.get("app_secret")

    # TODO: 校验 app_id / app_secret

    return {
        "code": 0,
        "data": {
            "config": {
                "server": {
                    "url": "https://metrics.example.com/v1/collect",
                    "timeout": 10,
                    "retry_count": 3
                },
                "agent": {
                    "upload_frequency": "epoch",
                    "upload_interval": 1,
                    "enable_async": True
                }
            }
        }
    }, 200
```

### 6.2 指标接口样例（伪代码）

```python
@app.post("/v1/collect")
def collect_metrics(body):
    train_id = body.get("train_id")
    metrics = body.get("metrics", {})
    # TODO: 写入数据库 / 消息队列
    return {"code": 0, "message": "ok"}, 200
```

---

如需我再补一版“数据库建表建议（MySQL/ClickHouse）+ 指标查询 API 规范”，我可以直接在这个文档后追加。