# 项目概览

## 📁 文件结构说明

```
train-guard/
├── train_guard/
│   ├── agent.py                 # ⭐ 核心模块 - 训练Agent主体
│   ├── metrics_collector.py      # 📊 指标收集器 - 收集训练过程中的各类指标
│   ├── uploader.py              # 📤 上传器 - 将指标上传到服务器
│   └── config.py                # ⚙️  配置文件 - 服务器、Agent、模型等配置
│
├── examples/
│   ├── example_training.py      # 🔬 完整示例 - CIFAR-10完整训练流程示例
│   ├── quick_start.py           # 🚀 快速开始 - 最小化集成示例
│   └── advanced_examples.py     # 🎯 高级示例 - 8个高级功能演示
│
├── tests/
│   └── test_agent.py            # ✅ 单元测试 - 完整的测试套件
├── pyproject.toml           # 📦 uv项目配置（依赖源）
├── uv.lock                  # 🔒 uv锁文件（可复现安装）
├── requirements.txt         # 📦 兼容导出依赖（可选）
├── README.md                # 📖 完整文档 - 详细功能说明
└── docs/PROJECT_OVERVIEW.md # 📋 本文件 - 项目概览
```

## 🎯 各文件功能速查

| 文件 | 用途 | 使用场景 |
|------|------|---------|
| `train_guard/agent.py` | 训练Agent主体 | 核心业务逻辑 |
| `train_guard/metrics_collector.py` | 指标采集 | 内部使用 |
| `train_guard/uploader.py` | 指标上传 | 内部使用 |
| `train_guard/config.py` | 配置管理 | 修改服务器地址、上传频率等 |
| `examples/example_training.py` | 完整示例 | 学习完整集成流程，运行完整训练 |
| `examples/quick_start.py` | 快速开始 | 学习最少改动集成方式 |
| `examples/advanced_examples.py` | 高级功能 | 学习自定义指标、多GPU、checkpoint等 |
| `tests/test_agent.py` | 单元测试 | 功能验证、开发测试 |
| `pyproject.toml` | uv依赖配置 | 依赖管理入口 |
| `uv.lock` | 锁文件 | 可复现环境 |
| `requirements.txt` | 兼容依赖 | 可选导出文件 |

## 🚀 快速开始步骤

### 步骤1: 安装依赖
```bash
uv sync
```

### 步骤2: 配置服务器
编辑 `train_guard/config.py`，修改 `SERVER_CONFIG` 中的服务器地址（已预配置示例服务器）

### 步骤3: 选择运行示例

#### 选项A: 跑快速示例（推荐新手）
```bash
uv run -m examples.quick_start
```
👉 学习最小化集成方式，5分钟了解基本用法

#### 选项B: 跑高级示例（学习高级功能）
```bash
uv run -m examples.advanced_examples
```
👉 演示8个高级功能，展示Agent的各项能力

#### 选项C: 跑完整训练示例（学习完整流程）
```bash
uv run -m examples.example_training
```
👉 完整的CIFAR-10训练流程，展示实际训练中的集成方式
⚠️ 首次运行会下载~170MB数据集，需要网络连接

#### 选项D: 运行测试（开发者用）
```bash
uv run -m tests.test_agent
```
👉 验证所有功能正常

## 📊 架构设计

```
┌─────────────────────────────────────────────┐
│        用户的PyTorch训练代码                │
│                                             │
│  with agent.epoch_context():                │
│    for batch in loader:                     │
│      with agent.batch_context():            │
│        output = model(input)                │
│        loss = criterion(output, target)     │
│        agent.record_loss(loss)              │
│        agent.record_accuracy(acc)           │
│                                             │
└────────────┬────────────────────────────────┘
             │
             ├─────────────────────────────┐
             │                             │
          ┌──▼──────────────┐      ┌──────▼────────────┐
          │                 │      │                   │
    ┌─────┴──────┬──────────┴──┐   │ ┌─────────────────┴───┐
    │            │             │   │ │                     │
    │ MetricsCollector         │   │ │  MetricsUploader    │
    │ ============             │   │ │  ================   │
    │ • record_loss      │   │ • upload               │
    │ • record_accuracy  │   │ • retry_with_backoff   │
    │ • record_lr        │   │ • async_upload         │
    │ • epoch_context    │   │                        │
    │ • batch_context    │   └─────────────────────────┘
    │                    │
    │ 收集训练过程中的   │
    │ 各类指标          │
    │                  │
    └──────────────────┘
                │
                ├─────────────────────────┐
                │                         │
        ┌───────▼──────┐      ┌──────────▼────┐
        │               │      │                │
        │ Metrics       │      │ System         │
        │ ────────      │      │ ──────         │
        │ loss          │      │ • GPU info     │
        │ accuracy      │      │ • CPU usage    │
        │ learning_rate │      │ • Memory       │
        │ batch_time    │      │                │
        │ epoch_time    │      │                │
        │ custom_*      │      │                │
        │               │      │                │
        └───────┬───────┘      └────────┬───────┘
                │                       │
                └───────────┬───────────┘
                            │
                      ┌─────▼──────────────┐
                      │                    │
                      │  构建Payload       │
                      │  ─────────────     │
                      │ {                  │
                      │  "epoch": 1,       │
                      │  "loss": 0.235,    │
                      │  "accuracy": 0.92, │
                      │  "system_info": {} │
                      │ }                  │
                      │                    │
                      └─────┬──────────────┘
                            │
                      ┌─────▼──────────────┐
                      │                    │
                      │  HTTP POST         │
                      │  ─────────────     │
                      │                    │
                      │  上传到目标服务器  │
                      │  (webhook.site)    │
                      │                    │
                      └────────────────────┘
```

## 💡 核心概念

### 1. MetricsCollector (指标收集器)
- **职责**: 收集训练过程中的各类指标
- **关键方法**:
  - `record_loss()` - 记录loss
  - `record_accuracy()` - 记录准确率
  - `start_epoch()` / `end_epoch()` - 标记epoch时间
  - `get_epoch_summary()` - 获取epoch总结
  - `get_current_metrics()` - 获取当前指标

### 2. MetricsUploader (上传器)
- **职责**: 将指标上传到服务器
- **特性**:
  - 自动重试机制（指数退避）
  - 异步上传支持（非阻塞）
  - 支持单条和批量上传
- **关键方法**:
  - `upload()` - 上传单条指标
  - `_upload_with_retry()` - 带重试的上传

### 3. TrainingAgent (训练Agent)
- **职责**: 协调指标收集和上传，提供统一接口
- **特性**:
  - 上下文管理器支持
  - 灵活的上传策略
  - 简洁的API接口
- **关键方法**:
  - `epoch_context()` - Epoch上下文
  - `batch_context()` - Batch上下文（可选）
  - `record_*()` - 记录各类指标
  - `manual_upload()` - 手动上传

## 🔋 上传策略

### 1. Epoch级上传（推荐）
- **特点**: 每个epoch结束时上传一次聚合指标
- **优点**: 数据量小，适合长训练
- **配置**:
```python
AGENT_CONFIG = {
    "upload_frequency": "epoch",
}
```

### 2. Batch级上传
- **特点**: 每N个batch上传一次
- **优点**: 更细粒度的监控
- **配置**:
```python
AGENT_CONFIG = {
    "upload_frequency": "batch",
    "upload_interval": 10,  # 每10个batch上传
}
```

### 3. 手动上传
- **特点**: 在关键节点手动上传（如最佳模型checkpoint）
- **用法**: `agent.manual_upload(description="best_checkpoint")`

## 📁 配置文件说明

### train_guard/config.py 配置项

```python
# 服务器配置
SERVER_CONFIG = {
    "url": "https://webhook.site/...",  # 目标服务器地址
    "timeout": 10,                       # 上传超时（秒）
    "retry_count": 3,                    # 重试次数
}

# Agent配置
AGENT_CONFIG = {
    "upload_frequency": "epoch",         # 上传频率
    "enable_async": True,                # 异步上传
}

# 监控指标配置
METRICS_CONFIG = {
    "monitor": {
        "loss": True,                    # 监控loss
        "accuracy": True,                # 监控准确率
        "learning_rate": True,           # 监控学习率
        "batch_time": True,              # 监控batch耗时
        "epoch_time": True,              # 监控epoch耗时
    },
    "include_system_info": True,         # 包含系统信息（GPU、CPU等）
}
```

## 📤 数据上传格式

```json
{
    "timestamp": "2024-03-18T10:30:45.123456",
    "train_id": "cifar10_demo_001",
    "metrics": {
        "epoch": 5,
        "timestamp": 1710756645.123456,
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
        "learning_rate": {
            "current": 0.001,
            "mean": 0.001,
            "min": 0.001,
            "max": 0.001,
            "count": 156
        },
        "system_info": {
            "cpu_percent": 45.2,
            "memory_percent": 52.1,
            "memory_available_mb": 8192,
            "gpu_allocated_mb": 4096,
            "gpu_reserved_mb": 5120,
            "gpu_count": 1
        }
    }
}
```

## 🎓 学习路径

### 初级 (30分钟)
1. 阅读本文档 (PROJECT_OVERVIEW.md)
2. 运行 `uv run -m examples.quick_start` 和查看输出
3. 修改 `examples/quick_start.py` 中的参数，观察输出变化

### 中级 (1小时)
1. 阅读 [README.md](../README.md) "快速开始" 和 "基本使用" 部分
2. 运行 `uv run -m examples.advanced_examples` 查看8个高级功能
3. 阅读 `train_guard/agent.py` 和 `train_guard/metrics_collector.py` 源码

### 高级 (2小时+)
1. 阅读 [README.md](../README.md) "高级用法" 和 "配置详解" 部分
2. 阅读 `examples/example_training.py` 完整训练示例
3. 修改 `train_guard/config.py` 实现自定义配置
4. 运行 `uv run -m tests.test_agent` 理解单元测试
5. 将Agent集成到自己的训练代码

## 🐛 常见问题

### Q: 上传到服务器失败了怎么办？
A: Agent会自动重试3次。如果仍失败，训练不会中断，可以查看日志信息。

### Q: 异步上传会不会丢失数据？
A: 不会。异步上传在后台线程进行，正常退出时会等待所有上传完成。

### Q: 如何修改上传的目标服务器？
A: 修改 `train_guard/config.py` 中的 `SERVER_CONFIG['url']`。

### Q: 支持多个GPU吗？
A: 单机多GPU需要给每个进程分配不同的 `train_id`；多机分布式训练需要额外配置。

### Q: 如何添加自定义指标？
A: 使用 `agent.record_metric('custom_name', value)` 方法。

## 📊 性能指标

- ✅ **收集开销**: <1ms/batch
- ✅ **上传延迟**: <100ms（异步）
- ✅ **内存占用**: <10MB
- ✅ **支持GPU**: ✓ NVIDIA GPU内存监控

## 🔄 工作流程

```
用户代码
   ↓
with agent.epoch_context():    ← 标记epoch开始
   ↓
 for batch:                     ← 批次循环
   ↓
 with agent.batch_context():   ← 标记batch开始（可选）
   ↓
 agent.record_loss(...)        ← 记录指标
 agent.record_accuracy(...)    ← 记录指标
 agent.record_lr(...)          ← 记录指标
   ↓
           ← epoch结束，触发自动上传
   ↓
collector.get_epoch_summary()  ← 聚合指标
   ↓
uploader.upload()              ← 上传到服务器
   ↓
✓ 指标已上传
```

## 🎯 项目特色

✨ **简洁集成**: 最少代码改动，只需3个上下文
📊 **全面监控**: loss、accuracy、learning_rate、系统信息
🔄 **灵活策略**: Epoch级、Batch级、手动上传三种模式
📤 **可靠传输**: 自动重试、异步非阻塞
🎛️ **易于定制**: 可配置的指标、上传策略、服务器
✅ **完整文档**: 快速开始、高级用法、单元测试、示例代码
🚀 **开箱即用**: 预配置示例服务器，无需额外配置

## 📞 获取帮助

- 📖 查看 [README.md](../README.md) 详细文档
- 🚀 运行 `uv run -m examples.quick_start` 快速上手
- 🎯 查看 `examples/advanced_examples.py` 功能演示
- ✅ 运行 `uv run -m tests.test_agent` 验证功能
- 📝 查看示例代码了解集成方式

---

**最后更新**: 2024-03-18  
**版本**: 1.0.0  
**状态**: ✅ 完成
