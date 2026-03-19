# 🚀 PyTorch Training Agent - 快速上手

## 30 秒快速开始

### 1. 安装
```bash
uv sync
```

### 2. 代码中启动
```python
from train_guard.agent import TrainingAgent

# 仅 3 个参数即可启动！
agent = TrainingAgent(
    backend_api="https://backend.example.com/api/agent/config",
    app_id="your_app_id",
    app_secret="your_app_secret",
)

# 在训练循环中使用
for epoch in range(num_epochs):
    with agent.epoch_context():
        # 你的训练代码
        agent.record_loss(loss_value)
        agent.record_accuracy(accuracy)

agent.close()
```

**完成！** 训练指标已自动上传到后端服务器。

---

## 📋 详细讲解（5 分钟）

### 场景一：后端托管配置（推荐，零配置文件）

最简单的方式——代码中直接传入后端凭证：

```python
from train_guard.agent import TrainingAgent

agent = TrainingAgent(
    backend_api="https://backend.example.com/api/agent/config",  # 后端配置 API
    app_id="your_app_id",                                        # 应用 ID
    app_secret="your_app_secret",                                # 应用密钥
)
```

初始化时，Agent 会自动：
1. 请求后端拉取完整配置（上报地址、上传频率等）
2. 验证配置有效性
3. 准备开始上报

**优势：** 无需在客户端维护配置文件，所有配置由后端统一管理。

---

### 场景二：本地配置（CLI 配置一次，代码无参数）

#### 第一步：通过 CLI 配置（仅需一次）

```bash
# 仅配置一次后端凭证
train-guard-config set backend.api https://backend.example.com/api/agent/config --type str
train-guard-config set backend.app_id your_app_id --type str
train-guard-config set backend.app_secret your_app_secret --type str
```

#### 第二步：代码中无需传参

```python
from train_guard.agent import TrainingAgent

# 自动从本地配置读取后端凭证，然后拉取远程配置
agent = TrainingAgent()
```

**优势：** 代码简洁，配置持久化在本地，便于多个项目复用。

---

### 场景三：直接上报（仅需上报到某个 URL，不用后端）

如果你只是想简单地把指标上报到某个 URL，无需后端配置系统：

```python
from train_guard.agent import TrainingAgent

agent = TrainingAgent(
    server_url="https://your-server.com/api/metrics"
)
```

或 CLI：
```bash
train-guard-config set server.url https://your-server.com/api/metrics --type str
```

---

## 🛠️ 安装方式

### 使用 uv（推荐）

```bash
cd /path/to/train-guard
uv sync
```

### 使用 pip

```bash
pip install -e .
```

---

## 📖 集成到现有代码

### 基本改动（5 行代码）

```python
# 导入
from train_guard.agent import TrainingAgent

# 在训练开始前初始化
agent = TrainingAgent(
    backend_api="https://backend.example.com/api/agent/config",
    app_id="your_app_id",
    app_secret="your_app_secret",
)

try:
    # 在 epoch 循环外包装 with
    for epoch in range(num_epochs):
        with agent.epoch_context():  # ← 加一行
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # 你的训练代码...
                output = model(inputs)
                loss = criterion(output, targets)
                
                # 记录指标（加 3 行）
                agent.record_loss(loss.item())           # ← 加
                agent.record_accuracy(accuracy)          # ← 加
                agent.record_learning_rate(lr)           # ← 加
finally:
    agent.close()  # ← 加一行
```

就这样！总共只改动 **5 行代码**。

---

## ⚙️ 后端 API 约定

Agent 在初始化时会调用你的后端服务以下：

### 请求

```
POST {backend_api}

Headers:
  X-App-Id: {app_id}
  X-App-Secret: {app_secret}

Body: JSON
{
  "app_id": "{app_id}",
  "app_secret": "{app_secret}"
}
```

### 响应（支持多种格式）

只要响应中包含以下信息，Agent 都能解析：

**格式一：完整配置**
```json
{
  "code": 0,
  "data": {
    "config": {
      "server": {"url": "https://your-server.com/api/metrics"},
      "agent": {"upload_frequency": "epoch"}
    }
  }
}
```

**格式二：简化配置（仅上报）**
```json
{
  "data": {
    "server_url": "https://your-server.com/api/metrics"
  }
}
```

**格式三：直接返回**
```json
{
  "server": {"url": "https://your-server.com/api/metrics"}
}
```

---

## 🎯 常见用法

### 自定义指标

```python
agent.record_metric('f1_score', 0.92)
agent.record_metric('precision', 0.95)
```

### 手动上传（体检点）

```python
# 在任何时刻手动上传当前指标
agent.manual_upload(description="checkpoint_epoch_10")
```

### 获取训练总结

```python
summary = agent.get_summary()
print(f"Train ID: {summary['train_id']}")
print(f"Current metrics: {summary['current_metrics']}")
```

---

## 🔍 CLI 工具速查

安装后自动获得 `train-guard-config` 命令：

```bash
# 查看配置路径
train-guard-config path

# 查看完整配置
train-guard-config show

# 查看单个键
train-guard-config get backend.api

# 设置配置
train-guard-config set backend.api https://... --type str

# 删除配置
train-guard-config unset backend.api

# 重置为默认
train-guard-config reset --yes
```

可用顶层配置键：`backend`、`server`、`agent`、`metrics`、`model`、`training`

---

## 运行示例
```

### 使用with语句自动管理
```python
with TrainingAgent(...) as agent:
    # 训练代码
    pass
# Agent自动关闭
```

---

## 🎓 学习路径

### 初级（30分钟）
1. ✅ 阅读本文件（QUICK_START.md）
2. ✅ 运行 `uv run -m examples.quick_start` 查看输出
3. ✅ 查看代码理解基本用法

### 中级（1小时）
1. ✅ 运行 `uv run -m examples.advanced_examples` 学习高级功能
2. ✅ 阅读 [README.md](../README.md) 详细文档
3. ✅ 查看 `train_guard/agent.py` 源码理解架构

### 高级（2小时+）
1. ✅ 运行 `uv run -m examples.example_training` 完整示例
2. ✅ 阅读 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) 架构详解
3. ✅ 修改 `train_guard/config.py` 自定义配置
4. ✅ 将Agent集成到自己的训练代码
5. ✅ 查看上传到服务器的实际数据

---

## 🔍 查看上传的数据

```bash
uv run -m examples.quick_start          # 快速演示
uv run -m examples.advanced_examples    # 高级功能
uv run -m examples.example_training     # 完整示例
uv run -m tests.test_agent              # 运行测试
```

---

## ❓ 常见问题

**Q: 三个参数 backend_api、app_id、app_secret 都是必须的吗？**  
A: 是的。只要提供任何一个，Agent 就会认为你要用后端模式。如果参数不全会报错。

**Q: 后端返回的配置格式有限制吗？**  
A: 没有。Agent 支持多种格式（详见"后端 API 约定"部分），会自动提取可用配置。

**Q: 可以只上报到某个 URL，不用后端吗？**  
A: 可以，用 `TrainingAgent(server_url="...")` 或通过 CLI 配置 `server.url`。

**Q: 异步上传会丢数据吗？**  
A: 不会。Agent 在关闭前会等待所有异步上传完成，所以记得调用 `agent.close()`。

**Q: 支持分布式训练吗？**  
A: 支持。为每个进程赋予不同的 `train_id` 即可区分。

**Q: 没有 GPU 也能用吗？**  
A: 可以。Agent 会自动使用 CPU，GPU 字段会显示 0。

---

## 🔧 配置键参考

通过 CLI 配置任何键（使用点路径）：

```bash
# 后端
train-guard-config set backend.api https://... --type str
train-guard-config set backend.app_id your_app_id --type str
train-guard-config set backend.app_secret your_secret --type str
train-guard-config set backend.timeout 10 --type int

# 服务器
train-guard-config set server.url https://... --type str
train-guard-config set server.timeout 10 --type int
train-guard-config set server.retry_count 3 --type int

# Agent
train-guard-config set agent.upload_frequency epoch --type str
train-guard-config set agent.upload_interval 1 --type int
train-guard-config set agent.enable_async true --type bool

# 指标
train-guard-config set metrics.include_system_info true --type bool
```

---

## 🎯 下一步

- 📖 查看 [README.md](../README.md) 了解完整功能
- 🔬 运行 `uv run -m examples.advanced_examples` 学习高级特性
- 💾 阅读 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) 理解架构
- ✅ 运行 `uv run -m tests.test_agent` 验证环境

---

**祝你使用愉快！** 🚀
