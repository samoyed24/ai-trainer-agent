# PyTorch Training Agent

一个功能强大的PyTorch训练监控agent，**只需填入后端API、APP_ID、APP_SECRET**，就能自动拉取配置、实时收集训练指标并上传。

## ⚡ 最简启动

```python
from ai_trainer_agent.agent import TrainingAgent

# 仅需三个参数即可启动！
agent = TrainingAgent(
    backend_api="https://backend.example.com/api/agent/config",
    app_id="your_app_id",
    app_secret="your_app_secret",
)
```

## 功能特性

✨ **核心功能**
- 🎯 实时指标收集：Loss、Accuracy、Learning Rate、Batch Time等
- 📤 灵活的上传策略：支持Epoch级、Batch级、自定义时间间隔上传
- 🔄 异步上传：非阻塞上传，不影响训练速度
- 🔧 自动重试机制：上传失败自动重试（指数退避）
- 📊 系统信息监控：GPU内存、CPU使用率、内存占用
- 🎛️ 后端托管配置：从后端自动拉取完整配置，零配置启动
- 🌐 易于集成：简洁的API接口，最少代码改动

## 目录结构

```
ai-trainer-agent/
├── ai_trainer_agent/
│   ├── agent.py                 # 主Agent模块
│   ├── metrics_collector.py      # 指标收集器
│   ├── uploader.py              # 上传器
│   └── config.py                # 配置文件
├── examples/
│   ├── quick_start.py
│   ├── advanced_examples.py
│   └── example_training.py      # 示例训练脚本
├── tests/
│   └── test_agent.py
├── docs/
│   ├── QUICK_START.md
│   ├── INTEGRATION_PROTOCOL.md  # 对接协议（上传格式 + 配置下发格式）
│   ├── PROJECT_OVERVIEW.md
│   └── CHEATSHEET.py
├── pyproject.toml           # uv项目配置（依赖源）
├── uv.lock                  # uv锁文件
├── requirements.txt         # 兼容导出依赖（可选）
└── README.md                # 本文件
```


## 快速开始（3步）

### 第一步：安装
```bash
uv sync
# 或用 pip install -e .
```

### 第二步：配置后端凭证

**方式A：代码中直接传参（推荐首次测试）**
```python
from ai_trainer_agent.agent import TrainingAgent

agent = TrainingAgent(
    backend_api="https://backend.example.com/api/agent/config",
    app_id="your_app_id",
    app_secret="your_app_secret",
    train_id="my_training_001"
)
```

**方式B：CLI 配置一次，代码零参数（推荐生产）**
```bash
# 仅需配置一次
ai-trainer-config set backend.api https://backend.example.com/api/agent/config --type str
ai-trainer-config set backend.app_id your_app_id --type str
ai-trainer-config set backend.app_secret your_app_secret --type str
```

```python
# 之后每次启动都无需参数
from ai_trainer_agent.agent import TrainingAgent
agent = TrainingAgent()  # 自动从后端拉取配置
```

### 第三步：集成到训练代码

```python
# 在训练循环中使用
for epoch in range(num_epochs):
    with agent.epoch_context():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 训练代码
            output = model(inputs)
            loss = criterion(output, targets)
            
            # 记录指标（3行代码）
            agent.record_loss(loss.item())
            agent.record_accuracy(accuracy)
            agent.record_learning_rate(optimizer.param_groups[0]['lr'])

# 关闭 Agent
agent.close()
```

---

## 后端 API 约定

Agent 初始化时会自动调用后端拉取完整配置。

**请求格式**
```
POST {backend_api}
Headers:
  X-App-Id: {app_id}
  X-App-Secret: {app_secret}
Body:
  {
    "app_id": "{app_id}",
    "app_secret": "{app_secret}"
  }
```

**响应格式**（支持以下任意一种）  

1. 完整方式：
```json
{
  "code": 0,
  "data": {
    "config": {
      "server": {"url": "https://your-server.com/api/metrics"},
      "agent": {"upload_frequency": "epoch"},
      "metrics": {...}
    }
  }
}
```

2. 简化方式（仅需上报）：
```json
{
  "data": {
    "server_url": "https://your-server.com/api/metrics"
  }
}
```

3. 直接返回配置：
```json
{
  "server": {"url": "https://your-server.com/api/metrics"},
  "agent": {"upload_frequency": "epoch"}
}
```

---

## 安装及配置方式

### 通过 pip 安装

```bash
# 标准安装
pip install .

# 开发模式（推荐）
pip install -e .

# 从 Git 安装
pip install "git+https://github.com/<owner>/<repo>.git@dev"
```

### CLI 工具使用

安装后自动获得 `ai-trainer-config` 命令：

```bash
# 查看配置路径
ai-trainer-config path

# 查看完整配置
ai-trainer-config show

# 查看单个键
ai-trainer-config get backend.api

# 设置配置
ai-trainer-config set backend.api https://backend.example.com/api/agent/config --type str
ai-trainer-config set server.url https://your-server.com/api/metrics --type str

# 删除配置
ai-trainer-config unset backend.api

# 重置为默认
ai-trainer-config reset --yes
```

配置键为点路径格式，可用顶层键：`backend`、`server`、`agent`、`metrics`、`model`、`training`

#### 运行方式

请在项目根目录执行（包含 `pyproject.toml` 的目录）：

```bash
# 推荐：模块方式运行
uv run -m examples.quick_start
uv run -m tests.test_agent

# 备选：直接脚本运行（需加 PYTHONPATH）
cd /Users/samoyed24/dev/ai-trainer/ai-trainer-agent
PYTHONPATH=. .venv/bin/python examples/example_training.py
```

## 交替使用方式

### 直接上报（不用后端时）

如果你只想простой上报到某个 URL，不使用后端托管：

```python
agent = TrainingAgent(
    server_url="https://your-server.com/api/metrics"
)
```

或者 CLI：
```bash
ai-trainer-config set server.url https://your-server.com/api/metrics --type str
```

### 完整自定义（高级场景）

```python
from ai_trainer_agent.config import SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG
from ai_trainer_agent.agent import TrainingAgent

agent = TrainingAgent(
    server_config=SERVER_CONFIG,
    agent_config=AGENT_CONFIG,
    metrics_config=METRICS_CONFIG,
    train_id="my_training_001"
)
```

### 记录自定义指标

```python
# 记录任意自定义指标
agent.record_metric('f1_score', 0.92)
agent.record_metric('precision', 0.95)
agent.record_metric('recall', 0.89)
```

### 手动上传

```python
# 在任何时刻手动上传当前指标
agent.manual_upload(description="checkpoint_epoch_5")
```

### 系统信息监控

配置是否包含系统信息（GPU、CPU、内存）：

```python
METRICS_CONFIG = {
    "include_system_info": True,  # 包含系统信息
}
```

### 上下文管理器用法

```python
# 使用with语句自动处理资源
with TrainingAgent(SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG) as agent:
    for epoch in range(num_epochs):
        with agent.epoch_context():
            # 训练代码
            pass
# Agent会自动关闭
```

## 配置详解

### BACKEND_CONFIG

```python
{
    "api": "https://backend.example.com/api/agent/config",
    "app_id": "your_app_id",
    "app_secret": "your_app_secret",
    "timeout": 10,             # 拉取远程配置的超时时间（秒）
}
```

后端响应支持以下格式之一：

- `{"config": {...}}`
- `{"data": {"config": {...}}}`
- `{"data": {...}}`（直接返回配置对象）
- `{"server_url": "https://..."}`（简化模式）

## 高级功能

### 上传策略

#### Epoch级上传（推荐）

在每个epoch结束时上传聚合的指标：

```python
agent = TrainingAgent(
    backend_api="...",
    app_id="...",
    app_secret="...",
)
# 默认就是 epoch 模式，无需额外配置
```

#### Batch级上传

每N个batch上传一次指标：

```bash
# 通过 CLI 配置
ai-trainer-config set agent.upload_frequency batch --type str
ai-trainer-config set agent.upload_interval 10 --type int
```

### 记录自定义指标

```python
agent.record_metric('f1_score', 0.92)
agent.record_metric('precision', 0.95)
agent.record_metric('validation_loss', 0.25)
```

### 手动上传

```python
# 在任何时刻手动上传当前指标
agent.manual_upload(description="best_model_checkpoint")
```

### 系统信息监控

配置是否包含系统信息（GPU、CPU、内存）：

```bash
ai-trainer-config set metrics.include_system_info true --type bool
```

### 上下文管理器

```python
# 使用 with 语句自动处理资源关闭
with TrainingAgent(backend_api="...", app_id="...", app_secret="...") as agent:
    for epoch in range(num_epochs):
        with agent.epoch_context():
            # 训练代码
            pass
# 当 with 块退出时，Agent 会自动关闭
```

---

## 配置详解

### BACKEND_CONFIG（后端托管配置）

```python
{
    "api": "https://backend.example.com/api/agent/config",
    "app_id": "your_app_id",
    "app_secret": "your_app_secret",
    "timeout": 10,  # 拉取远程配置的超时时间（秒）
}
```

通过 CLI 配置：

```bash
ai-trainer-config set backend.api https://backend.example.com/api/agent/config --type str
ai-trainer-config set backend.app_id your_app_id --type str
ai-trainer-config set backend.app_secret your_app_secret --type str
ai-trainer-config set backend.timeout 10 --type int
```

### SERVER_CONFIG（服务器上报配置）

```python
{
    "url": "https://your-server.com/api/metrics",  # 上传目标URL
    "timeout": 10,                                   # 上传超时时间（秒）
    "retry_count": 3,                                # 上传失败重试次数
}
```

通过 CLI 配置（不使用后端托管时）：

```bash
ai-trainer-config set server.url https://your-server.com/api/metrics --type str
ai-trainer-config set server.timeout 10 --type int
ai-trainer-config set server.retry_count 3 --type int
```

### AGENT_CONFIG（上传策略配置）

```python
{
    "upload_frequency": "epoch",  # 上传频率：'epoch' 或 'batch'
    "upload_interval": 1,         # batch 模式时，每 N 个 batch 上传一次
    "buffer_size": 100,           # 缓冲区大小
    "enable_async": True,         # 是否异步上传（推荐 True，不阻塞训练）
}
```

### METRICS_CONFIG（指标监控配置）

```python
{
    "monitor": {
        "loss": True,              # 监控损失
        "accuracy": True,          # 监控准确率
        "learning_rate": True,     # 监控学习率
        "batch_time": True,        # 监控批次耗时
        "epoch_time": True,        # 监控epoch耗时
        "model_size": True,        # 监控模型大小
    },
    "include_system_info": True,   # 包含系统信息（CPU/GPU/内存）
}
```

---

## 上传数据格式

### Agent 发送给后端的数据格式

每个 epoch 或 batch 结束时，Agent 会发送以下格式的 POST 请求到 `server.url`：

```json
{
    "timestamp": "2024-03-18T10:30:45.123456",
    "train_id": "abc12345",
    "metrics": {
        "epoch": 5,
        "loss": {
            "current": 0.234,
            "mean": 0.345,
            "min": 0.123,
            "max": 0.567,
            "count": 156
        },
        "accuracy": {
            "current": 0.92,
            "mean": 0.87,
            "min": 0.75,
            "max": 0.95,
            "count": 156
        },
        "system_info": {
            "cpu_percent": 45.2,
            "memory_percent": 52.1,
            "gpu_allocated_mb": 4096,
            "gpu_reserved_mb": 5120,
            "gpu_count": 1
        }
    }
}
```

---

## 运行示例

### 快速示例

```bash
uv run -m examples.quick_start
```

### 高级功能演示

```bash
uv run -m examples.advanced_examples
```

### 完整CIFAR-10训练

```bash
uv run -m examples.example_training
```

此示例会：
- 加载CIFAR-10数据集（首次下载 ~170MB）
- 运行完整CNN训练循环
- 每个epoch自动上传指标

### 运行测试

```bash
uv run -m tests.test_agent
```

---

## API 速查

### TrainingAgent 初始化

```python
class TrainingAgent:
    def __init__(
        backend_api=None,          # 后端配置 API 地址
        app_id=None,               # 后端应用 ID
        app_secret=None,           # 后端应用密钥
        server_url=None,           # 快速覆盖上报URL
        server_config=None,        # 完整服务器配置
        agent_config=None,         # 完整 agent 配置
        metrics_config=None,       # 完整指标配置
        train_id=None,             # 训练ID
    )
```

### 主要方法

```python
agent.record_loss(loss_value)              # 记录损失
agent.record_accuracy(accuracy_value)      # 记录准确率
agent.record_learning_rate(lr_value)       # 记录学习率
agent.record_metric(name, value)           # 记录自定义指标
agent.manual_upload(description)           # 手动上传
agent.get_summary()                        # 获取训练总结
agent.close()                              # 关闭 Agent
```

### 上下文管理

```python
with agent.epoch_context():    # Epoch 上下文
    with agent.batch_context():  # Batch 上下文
        # 训练代码

## 常见问题

### Q: 后端三参数 backend_api、app_id、app_secret 都是必须的吗？

A: 如果要使用后端托管模式，这三个参数必须同时提供。如果只提供其中一两个，Agent 会报错以避免部分配置丢失。

### Q: 后端接口需要返回什么格式？

A: Agent 支持多种响应格式（见"后端 API 约定"部分）。只要响应中包含 `server` 配置对象或 `server_url` 字段，Agent 都能解析。

### Q: 可以同时使用后端配置和本地参数覆盖吗？

A: 可以。配置的优先级是：函数参数 > 后端拉取配置 > 本地保存配置 > 默认配置。

### Q: 异步上传会不会丢数据？

A: 不会。Agent 在 `close()` 时会等待所有异步上传完成。记得在 finally 块中调用 `agent.close()`。

### Q: 会影响训练速度吗？

A: 不会。异步上传在后台线程进行，不阻塞主训练循环。整个收集和上传过程开销 <1ms/batch。

### Q: 上传失败会自动重试吗？

A: 会。Agent 默认重试3次，使用指数退避策略，既不会无限重试也不会立即放弃。

### Q: 支持分布式训练吗？

A: 支持。为每个进程提供不同的 `train_id` 即可区分。

### Q: 没有 GPU 也能用吗？

A: 可以。Agent 会自动使用 CPU，GPU 信息字段会显示为空。

### Q: 报错 `ModuleNotFoundError: No module named 'ai_trainer_agent'` 怎么办？

A: 确保在项目根目录运行，使用 `uv run` 或加上 `PYTHONPATH` 环境变量。

## 性能指标

- **收集指标开销**: <1ms/batch
- **上传延迟**: <100ms（异步）
- **内存占用**: <10MB
- **支持GPU监控**: 支持NVIDIA GPU

## 日志输出示例

```
2024-03-18 10:30:45 - [INFO] - 🚀 训练Agent初始化完成 (Train ID: abc12345)
2024-03-18 10:30:45 - [INFO] -    上传策略: epoch
2024-03-18 10:30:45 - [INFO] -    服务器: https://webhook.site/...
2024-03-18 10:30:50 - [INFO] - ✓ 指标上传成功 (epoch=1, status=200)
```

## 注意事项

⚠️ **重要提示**
- 使用后端托管模式时，必须同时提供 `backend_api`、`app_id`、`app_secret`
- 确保最终生效的 `server.url` 可用（可来自本地配置或后端配置）
- 首次运行示例时会下载CIFAR-10数据集，需要网络连接
- 异步上传情况下，主程序退出前要调用`agent.close()`确保所有数据上传
- GPU监控需要PyTorch配置CUDA支持

## 扩展功能

你可以基于此框架进行以下扩展：

- 🔐 添加认证机制（如API Key、OAuth）
- 📊 集成更多可视化工具（如TensorBoard、Weights&Biases）
- 💾 本地指标缓存（网络中断时降级处理）
- 🌐 支持多种协议（gRPC、WebSocket等）
- 🎯 条件上传（如仅上传满足条件的指标）

## License

MIT License

## 联系方式

如有问题或建议，欢迎反馈！
