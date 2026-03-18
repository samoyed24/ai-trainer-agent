"""
快速示例 - 最小化集成
演示如何在现有代码中快速集成Training Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 导入Agent相关模块
from ai_trainer_agent.agent import TrainingAgent


def simple_training_example():
    """
    最简单的集成示例
    
    这个例子展示了如何在最少代码改动的情况下集成Training Agent
    """
    
    # ===== 你的训练代码 =====
    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 示例数据
    X = torch.randn(100, 10).to(device)
    y = torch.randint(0, 2, (100,)).to(device)
    
    # ===== 集成Training Agent =====
    # 最简初始化：读取本地配置（若已配置 backend.* 会自动从后端拉取）
    agent = TrainingAgent(train_id="quick_demo_001")
    
    try:
        # 训练循环
        num_epochs = 3
        for epoch in range(num_epochs):
            # 用 with 语句包装epoch
            with agent.epoch_context():
                total_loss = 0
                correct = 0
                
                for i in range(0, len(X), 10):
                    batch_x = X[i:i+10]
                    batch_y = y[i:i+10]
                    
                    # 用 with 语句包装batch（可选）
                    with agent.batch_context():
                        # 您的正常训练代码
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # 记录指标（只需要加这些行）
                        agent.record_loss(loss.item())
                        _, preds = outputs.max(1)
                        accuracy = preds.eq(batch_y).sum().item() / len(batch_y)
                        agent.record_accuracy(accuracy)
                        agent.record_learning_rate(optimizer.param_groups[0]['lr'])
                        
                        total_loss += loss.item()
                        correct += preds.eq(batch_y).sum().item()
                
                # Epoch结束时，agent自动上传指标
                avg_loss = total_loss / (len(X) // 10)
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={correct/len(X):.2%}")
        
        print("✓ 训练完成！指标已自动上传到服务器")
        
    finally:
        # 关闭Agent
        agent.close()


def quick_api_reference():
    """快速API参考"""
    
    print("""
    ========== Training Agent 快速API参考 ==========
    
    # 1. 初始化
    agent = TrainingAgent()

    # 或者：后端托管模式
    agent = TrainingAgent(
        backend_api="https://backend.example.com/api/agent/config",
        app_id="your_app_id",
        app_secret="your_app_secret",
    )
    
    # 2. 使用epoch上下文（自动在epoch结束时上传指标）
    with agent.epoch_context():
        # 训练代码
        pass
    
    # 3. 使用batch上下文（可选，用于batch级别统计）
    with agent.batch_context():
        # 单个batch的训练代码
        pass
    
    # 4. 记录指标
    agent.record_loss(loss_value)
    agent.record_accuracy(acc_value)
    agent.record_learning_rate(lr_value)
    agent.record_metric('custom_metric', value)
    
    # 5. 手动上传
    agent.manual_upload(description="checkpoint_epoch_5")
    
    # 6. 获取总结
    summary = agent.get_summary()
    
    # 7. 关闭Agent
    agent.close()
    
    ==================================================
    
    📝 注意事项：
    - epoch_context 会自动在结束时上传指标
    - batch_context 是可选的，用于batch级统计
    - 异步上传不会阻塞训练
    - 记得在最后调用 close() 确保所有数据上传完成
    """)


def integration_tips():
    """集成提示"""
    
    print("""
    ========== 集成到现有代码的步骤 ==========
    
    1. 在文件顶部导入Agent：
         from ai_trainer_agent.agent import TrainingAgent
    
    2. 在训练初始化代码后初始化Agent：
         agent = TrainingAgent()
         # 或后端托管模式：
         # agent = TrainingAgent(backend_api=..., app_id=..., app_secret=...)
    
    3. 用 with agent.epoch_context(): 包装epoch循环
    
    4. 在计算loss和accuracy后记录它们：
       agent.record_loss(loss.item())
       agent.record_accuracy(accuracy)
       agent.record_learning_rate(optimizer.param_groups[0]['lr'])
    
    5. 在try-finally中确保关闭Agent：
       finally:
           agent.close()
    
    💡 修改最少的情况下，只需改动这些：
    - 添加导入语句
    - 初始化Agent
    - 在epoch循环外层加 with agent.epoch_context():
    - 加3行 agent.record_*() 调用
    - 加 agent.close() 调用
    
    =======================================
    """)


if __name__ == '__main__':
    print("🚀 PyTorch Training Agent - 快速示例\n")
    
    # 打印API参考
    quick_api_reference()
    
    # 打印集成提示
    integration_tips()
    
    # 运行示例
    print("\n运行简单训练示例...\n")
    simple_training_example()
