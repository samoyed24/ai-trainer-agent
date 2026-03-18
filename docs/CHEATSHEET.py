"""
快速参考卡片 / Cheatsheet
常见用法速查表
"""

# ============================================================================
# 1. 基础集成 - 最少代码改动
# ============================================================================

from ai_trainer_agent.agent import TrainingAgent
from ai_trainer_agent.config import SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG

# 初始化Agent
agent = TrainingAgent(SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG)

try:
    # 训练循环
    for epoch in range(num_epochs):
        with agent.epoch_context():  # ← 用这个包装epoch
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                with agent.batch_context():  # ← 用这个包装batch（可选）
                    # 正常训练代码
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # 记录指标（只需要加这3行）
                    agent.record_loss(loss.item())
                    agent.record_accuracy(accuracy)
                    agent.record_learning_rate(optimizer.param_groups[0]['lr'])
finally:
    agent.close()  # ← 记得关闭


# ============================================================================
# 2. 快速启动 - 使用with语句自动管理
# ============================================================================

with TrainingAgent(SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG) as agent:
    with agent.epoch_context():
        agent.record_loss(0.5)
        agent.record_accuracy(0.85)
    # Agent会自动关闭


# ============================================================================
# 3. 记录指标 - 各类数据的记录方法
# ============================================================================

# 标准指标
agent.record_loss(loss_value)
agent.record_accuracy(acc_value)
agent.record_learning_rate(lr_value)

# 自定义指标
agent.record_metric('precision', 0.95)
agent.record_metric('recall', 0.89)
agent.record_metric('f1_score', 0.92)
agent.record_metric('validation_loss', 0.25)


# ============================================================================
# 4. 上传策略 - 三种上传模式
# ============================================================================

# 策略1: Epoch级上传（推荐）
config = AGENT_CONFIG.copy()
config['upload_frequency'] = 'epoch'
agent = TrainingAgent(SERVER_CONFIG, config, METRICS_CONFIG)

# 策略2: Batch级上传
config = AGENT_CONFIG.copy()
config['upload_frequency'] = 'batch'
config['upload_interval'] = 10  # 每10个batch上传一次

# 策略3: 手动上传
agent.manual_upload(description="checkpoint_epoch_5")


# ============================================================================
# 5. 系统监控 - 启用/禁用系统信息
# ============================================================================

# 启用系统监控（默认启用）
metrics_config = METRICS_CONFIG.copy()
metrics_config['include_system_info'] = True  # 包含GPU、CPU、内存信息

# 禁用系统监控
metrics_config = METRICS_CONFIG.copy()
metrics_config['include_system_info'] = False


# ============================================================================
# 6. 选择性监控 - 启用/禁用特定指标
# ============================================================================

metrics_config = METRICS_CONFIG.copy()
metrics_config['monitor'] = {
    'loss': True,
    'accuracy': True,
    'learning_rate': True,
    'batch_time': True,
    'epoch_time': True,
}


# ============================================================================
# 7. 获取信息 - 查询训练状态
# ============================================================================

# 获取当前指标
metrics = agent.collector.get_current_metrics()

# 获取epoch总结
summary = agent.collector.get_epoch_summary()

# 获取完整训练总结
full_summary = agent.get_summary()
print(f"Train ID: {full_summary['train_id']}")
print(f"Current Epoch: {full_summary['current_metrics']['epoch']}")


# ============================================================================
# 8. 高级配置 - 自定义服务器和行为
# ============================================================================

# 自定义服务器配置
custom_server_config = {
    "url": "https://your-server.com/api/metrics",
    "timeout": 15,          # 增加超时时间
    "retry_count": 5,       # 增加重试次数
}

# 自定义Agent配置
custom_agent_config = {
    "upload_frequency": "batch",
    "upload_interval": 5,   # 每5个batch上传
    "enable_async": True,   # 异步上传
    "buffer_size": 200,     # 增加缓冲区
}

agent = TrainingAgent(
    custom_server_config,
    custom_agent_config,
    METRICS_CONFIG,
    train_id="custom_training_001"
)


# ============================================================================
# 9. 多GPU/多进程 - 分布式场景
# ============================================================================

# 为不同GPU/进程分配不同的train_id
import torch.distributed as dist

if dist.is_initialized():
    rank = dist.get_rank()
    train_id = f"gpu_{rank}_training"
else:
    train_id = "single_gpu_training"

agent = TrainingAgent(
    SERVER_CONFIG,
    AGENT_CONFIG,
    METRICS_CONFIG,
    train_id=train_id
)


# ============================================================================
# 10. Checkpoint保存 - 在关键点上传
# ============================================================================

for epoch in range(num_epochs):
    with agent.epoch_context():
        # 训练代码
        pass
    
    # 定期保存checkpoint并上传
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
        agent.manual_upload(description=f"checkpoint_epoch_{epoch}")


# ============================================================================
# 11. 条件记录 - 根据条件记录指标
# ============================================================================

with agent.epoch_context():
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        with agent.batch_context():
            # 训练代码
            pass
            
            # 仅每10个batch记录一次
            if batch_idx % 10 == 0:
                agent.record_learning_rate(optimizer.param_groups[0]['lr'])


# ============================================================================
# 12. 错误处理 - 优雅的错误处理
# ============================================================================

try:
    with TrainingAgent(SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG) as agent:
        with agent.epoch_context():
            # 训练代码
            pass
except Exception as e:
    print(f"训练出错: {e}")
    # Agent会在finally中自动关闭
finally:
    pass  # Agent已自动关闭


# ============================================================================
# 13. 动态配置 - 运行时修改参数
# ============================================================================

# 修改学习率并记录
for param_group in optimizer.param_groups:
    param_group['lr'] = new_learning_rate
agent.record_learning_rate(new_learning_rate)

# 动态记录检查点信息
if is_best_model:
    agent.record_metric('is_best', 1.0)
agent.manual_upload(description="model_checkpoint")


# ============================================================================
# 14. 完整示例代码模板
# ============================================================================

"""
#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from ai_trainer_agent.agent import TrainingAgent
from ai_trainer_agent.config import SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    with TrainingAgent(SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG) as agent:
        for epoch in range(num_epochs):
            with agent.epoch_context():
                total_loss = 0
                correct = 0
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    with agent.batch_context():
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        agent.record_loss(loss.item())
                        agent.record_learning_rate(optimizer.param_groups[0]['lr'])
                        
                        _, preds = outputs.max(1)
                        accuracy = preds.eq(targets).sum().item() / len(targets)
                        agent.record_accuracy(accuracy)
                        
                        total_loss += loss.item()
                        correct += preds.eq(targets).sum().item()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}")

if __name__ == '__main__':
    train()
"""


# ============================================================================
# 15. 常用代码片段
# ============================================================================

# 计算和记录准确率
_, preds = outputs.max(1)
accuracy = (preds == targets).sum().item() / len(targets)
agent.record_accuracy(accuracy)

# 获取当前学习率
lr = optimizer.param_groups[0]['lr']
agent.record_learning_rate(lr)

# 调整学习率并记录
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1
agent.record_learning_rate(optimizer.param_groups[0]['lr'])

# 计算多指标
metrics = {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'val_acc': val_acc,
}
for name, value in metrics.items():
    agent.record_metric(name, value)


# ============================================================================
# 日志配置 - 调整日志级别
# ============================================================================

import logging

# 获取Agent的logger
logger = logging.getLogger('agent')

# 设置日志级别
logger.setLevel(logging.INFO)      # INFO级别
logger.setLevel(logging.DEBUG)     # DEBUG级别（更详细）
logger.setLevel(logging.WARNING)   # WARNING及以上


# ============================================================================
# 配置修改速查
# ============================================================================

# 修改服务器URL
from ai_trainer_agent.config import SERVER_CONFIG
SERVER_CONFIG['url'] = "https://new-server.com/metrics"

# 修改上传频率
from ai_trainer_agent.config import AGENT_CONFIG
AGENT_CONFIG['upload_frequency'] = 'batch'
AGENT_CONFIG['upload_interval'] = 20

# 修改监控指标
from ai_trainer_agent.config import METRICS_CONFIG
METRICS_CONFIG['monitor']['loss'] = True
METRICS_CONFIG['monitor']['accuracy'] = True
METRICS_CONFIG['include_system_info'] = True


# ============================================================================
# 官方文档链接
# ============================================================================
"""
📖 完整文档:       README.md
📋 项目概览:       PROJECT_OVERVIEW.md
🚀 快速开始:       uv run -m examples.quick_start
🎯 高级示例:       uv run -m examples.advanced_examples
🔬 完整示例:       uv run -m examples.example_training
✅ 单元测试:       uv run -m tests.test_agent
💻 本文件:         CHEATSHEET.py
"""

print(__doc__)
