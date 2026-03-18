"""
示例：使用PyTorch Training Agent进行CIFAR-10分类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from tqdm import tqdm

from ai_trainer_agent.config import (
    SERVER_CONFIG,
    AGENT_CONFIG,
    METRICS_CONFIG,
    MODEL_CONFIG,
    TRAINING_CONFIG,
)
from ai_trainer_agent.agent import TrainingAgent


# 简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data(batch_size=32):
    """加载CIFAR-10数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 下载并加载数据（首次运行会下载数据）
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, agent):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="训练中", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        with agent.batch_context():
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录指标
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # 上传到Agent
            agent.record_loss(loss.item())
            agent.record_learning_rate(optimizer.param_groups[0]['lr'])
            
            # 计算批次准确率
            batch_accuracy = predicted.eq(targets).sum().item() / targets.size(0)
            agent.record_accuracy(batch_accuracy)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{batch_accuracy:.2%}'
            })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, agent):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="验证中", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            # 记录验证指标
            agent.record_metric('val_loss', loss.item())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def main():
    """主训练函数"""
    
    # 获取配置
    device = torch.device(TRAINING_CONFIG['device'])
    num_epochs = TRAINING_CONFIG['num_epochs']
    batch_size = TRAINING_CONFIG['batch_size']
    learning_rate = TRAINING_CONFIG['learning_rate']
    
    print(f"🔧 配置信息:")
    print(f"   设备: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    
    # 初始化Agent
    agent = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=AGENT_CONFIG,
        metrics_config=METRICS_CONFIG,
        train_id="cifar10_demo_001"
    )
    
    try:
        # 加载数据
        print("📥 加载CIFAR-10数据集...")
        train_loader, val_loader = load_data(batch_size)
        print(f"   训练样本数: {len(train_loader.dataset)}")
        print(f"   验证样本数: {len(val_loader.dataset)}")
        
        # 初始化模型
        print("🏗️  初始化模型...")
        model = SimpleCNN(num_classes=MODEL_CONFIG['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 计算模型大小
        model_params = sum(p.numel() for p in model.parameters())
        agent.record_metric('model_params', float(model_params))
        print(f"   模型参数数: {model_params:,.0f}")
        
        # 训练循环
        print(f"\n{'='*60}")
        print("🚀 开始训练")
        print(f"{'='*60}\n")
        
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            print(f"\n📍 Epoch [{epoch+1}/{num_epochs}]")
            
            with agent.epoch_context():
                # 训练
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, agent
                )
                
                # 验证
                val_loss, val_acc = validate(model, val_loader, criterion, device, agent)
                
                # 记录epoch级别的指标
                agent.record_metric('val_accuracy', val_acc)
                agent.record_metric('val_loss_epoch', val_loss)
                
                print(f"✓ 训练完成")
                print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2%}")
                print(f"  验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2%}")
                
                # 保存最佳模型
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    torch.save(model.state_dict(), 'best_model.pth')
                    agent.record_metric('checkpoint_saved', 1.0)
                    print(f"  💾 保存最佳模型 (Accuracy: {best_accuracy:.2%})")
        
        # 最终上传
        print(f"\n{'='*60}")
        print("✓ 训练完成！")
        print(f"{'='*60}")
        print(f"最佳准确率: {best_accuracy:.2%}")
        
        # 手动上传最终结果
        agent.manual_upload(description="Training completed")
        
        # 打印总结
        summary = agent.get_summary()
        print(f"\n📋 训练总结:")
        print(f"   Train ID: {summary['train_id']}")
        print(f"   上传策略: {summary['upload_frequency']}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise
    
    finally:
        agent.close()


if __name__ == '__main__':
    main()
