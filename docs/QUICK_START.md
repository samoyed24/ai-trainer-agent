# Train Guard Agent Quick Start

## 1. 安装

```bash
uv sync
```

## 2. 配置 dashboard 凭证

代码中直接传：

```python
from train_guard.agent import TrainingAgent

agent = TrainingAgent(
    backend_base_url="http://localhost:8000",
    access_key_id="ak_xxx",
    secret_key="sk_xxx",
    project_id="project_abcd1234",
)
```

或者先用 CLI 保存：

```bash
train-guard-config set backend.base_url http://localhost:8000 --type str
train-guard-config set backend.access_key_id ak_xxx --type str
train-guard-config set backend.secret_key sk_xxx --type str
train-guard-config set backend.project_id project_abcd1234 --type str
```

## 3. 集成训练循环

```python
for epoch in range(num_epochs):
    with agent.epoch_context():
        for inputs, targets in train_loader:
            with agent.batch_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                agent.record_loss(loss.item())
                agent.record_accuracy(accuracy)
                agent.record_learning_rate(optimizer.param_groups[0]["lr"])

agent.close()
```

## 4. Agent 与 dashboard 的交互

- 拉配置：`GET /api/agent/config`
- 上报指标：`POST /api/metrics/ingest`
- 鉴权头：
  - `X-Access-Key-Id`
  - `X-Secret-Key`
  - `X-Project-Id`

## 5. CLI 速查

```bash
train-guard-config path
train-guard-config show
train-guard-config get backend.base_url
train-guard-config set server.url http://localhost:8000/api/metrics/ingest --type str
train-guard-config unset server.url
train-guard-config reset --yes
```
