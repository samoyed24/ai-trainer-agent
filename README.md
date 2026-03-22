# Train Guard Agent

面向 `train-guard-dashboard` 的训练侧 Agent。

它负责两件事：

- 从 dashboard 后端拉取当前项目的有效配置
- 在训练过程中持续上报扁平指标事件到 `/api/metrics/ingest`

## 快速开始

### 1. 安装

```bash
uv sync
```

### 2. 用 dashboard 凭证初始化

```python
from train_guard.agent import TrainingAgent

agent = TrainingAgent(
    backend_base_url="http://localhost:8000",
    access_key_id="ak_xxx",
    secret_key="sk_xxx",
    project_id="project_abcd1234",
    train_id="demo-run-001",
)
```

### 3. 集成到训练循环

```python
for epoch in range(num_epochs):
    with agent.epoch_context():
        for inputs, targets in train_loader:
            with agent.batch_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                agent.record_loss(loss.item())
                agent.record_accuracy(accuracy)
                agent.record_learning_rate(optimizer.param_groups[0]["lr"])

agent.close()
```

## Dashboard 协议

Agent 会按 `../train-guard-dashboard` 的约定工作：

- 拉配置：`GET /api/agent/config`
- 上报指标：`POST /api/metrics/ingest`
- 鉴权头：
  - `X-Access-Key-Id`
  - `X-Secret-Key`
  - `X-Project-Id`

配置接口返回的 `server.url` 会被直接用于上报。

## 本地 CLI 配置

如果你不想在代码里显式传凭证，可以先配置一次：

```bash
train-guard-config set backend.base_url http://localhost:8000 --type str
train-guard-config set backend.access_key_id ak_xxx --type str
train-guard-config set backend.secret_key sk_xxx --type str
train-guard-config set backend.project_id project_abcd1234 --type str
```

之后代码里可以直接：

```python
from train_guard.agent import TrainingAgent

agent = TrainingAgent(train_id="demo-run-001")
```

## 指标事件格式

当前 agent 会尽量发送 dashboard 友好的扁平事件，例如：

```json
{
  "train_id": "demo-run-001",
  "epoch": 3,
  "batch": 20,
  "step": 120,
  "timestamp": 1770000000.12,
  "loss": 0.245,
  "accuracy": 0.91,
  "learning_rate": 0.0003,
  "batch_time": 0.13,
  "system_cpu_percent": 42.5
}
```

这样 dashboard 的时序消费端可以直接抽取数值指标。

## 常用 API

```python
agent.record_loss(value)
agent.record_accuracy(value)
agent.record_learning_rate(value)
agent.record_metric("f1_score", value)
agent.manual_upload(description="checkpoint")
summary = agent.get_summary()
```

## 验证

```bash
uv run python -m unittest -v tests.test_agent
```
