"""
高级示例 - 更多功能演示
展示Training Agent的高级用法和定制化场景
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ai_trainer_agent.config import SERVER_CONFIG, AGENT_CONFIG, METRICS_CONFIG
from ai_trainer_agent.agent import TrainingAgent


def advanced_example_1_custom_metrics():
    """示例1: 记录自定义指标"""
    print("\n========== 示例1: 自定义指标 ==========\n")
    
    agent = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=AGENT_CONFIG,
        metrics_config=METRICS_CONFIG,
        train_id="advanced_custom_metrics"
    )
    
    try:
        with agent.epoch_context():
            # 记录标准指标
            agent.record_loss(0.234)
            agent.record_accuracy(0.92)
            
            # 记录自定义指标
            agent.record_metric('precision', 0.95)
            agent.record_metric('recall', 0.89)
            agent.record_metric('f1_score', 0.92)
            agent.record_metric('validation_loss', 0.245)
            
            print("✓ 已记录自定义指标: precision, recall, f1_score, validation_loss")
            print("✓ 这些指标将在epoch结束时与标准指标一起上传")
    
    finally:
        agent.close()


def advanced_example_2_different_upload_strategies():
    """示例2: 不同的上传策略"""
    print("\n========== 示例2: 不同的上传策略 ==========\n")
    
    # 策略1: Epoch级上传（推荐，当前示例的默认策略）
    print("策略1: Epoch级上传")
    config = AGENT_CONFIG.copy()
    config['upload_frequency'] = 'epoch'
    
    agent1 = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=config,
        metrics_config=METRICS_CONFIG,
        train_id="strategy_epoch"
    )
    print(f"  ✓ 每个epoch结束时上传指标")
    agent1.close()
    
    # 策略2: Batch级上传
    print("\n策略2: Batch级上传")
    config = AGENT_CONFIG.copy()
    config['upload_frequency'] = 'batch'
    config['upload_interval'] = 10  # 每10个batch上传一次
    
    agent2 = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=config,
        metrics_config=METRICS_CONFIG,
        train_id="strategy_batch"
    )
    print(f"  ✓ 每10个batch上传一次指标")
    agent2.close()


def advanced_example_3_manual_uploads():
    """示例3: 手动上传（checkpoint保存点）"""
    print("\n========== 示例3: 手动上传（Checkpoints） ==========\n")
    
    agent = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=AGENT_CONFIG,
        metrics_config=METRICS_CONFIG,
        train_id="advanced_manual_upload"
    )
    
    try:
        # 模拟训练流程
        for epoch in range(3):
            with agent.epoch_context():
                agent.record_loss(1.0 - 0.1 * epoch)
                agent.record_accuracy(0.5 + 0.1 * epoch)
            
            # 每个epoch结束后自动上传
            print(f"  📤 Epoch {epoch+1} 自动上传完成")
            
            # 在特定条件下手动上传（如最佳模型保存）
            if epoch == 1:
                agent.manual_upload(description="Best model checkpoint")
                print(f"  📤 手动上传: 最佳模型checkpoint")
    
    finally:
        agent.close()


def advanced_example_4_system_monitoring():
    """示例4: 系统信息监控"""
    print("\n========== 示例4: 系统信息监控 ==========\n")
    
    # 启用系统监控
    metrics_config = METRICS_CONFIG.copy()
    metrics_config['include_system_info'] = True
    
    agent = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=AGENT_CONFIG,
        metrics_config=metrics_config,
        train_id="advanced_system_monitoring"
    )
    
    try:
        with agent.epoch_context():
            agent.record_loss(0.5)
            agent.record_accuracy(0.85)
            print("✓ 当前指标已记录")
            print("✓ 系统信息（GPU内存、CPU使用率等）将在上传时一起发送")
    
    finally:
        agent.close()


def advanced_example_5_context_manager():
    """示例5: 使用with语句自动管理Agent生命周期"""
    print("\n========== 示例5: 上下文管理器 ==========\n")
    
    # 使用with语句，Agent会自动初始化和关闭
    with TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=AGENT_CONFIG,
        metrics_config=METRICS_CONFIG,
        train_id="advanced_context_manager"
    ) as agent:
        print("✓ Agent已初始化")
        
        with agent.epoch_context():
            agent.record_loss(0.3)
            agent.record_accuracy(0.88)
            print("✓ 指标已记录并将上传")
        
        print("✓ Epoch上下文已退出，指标已上传")
    
    print("✓ Agent已自动关闭")


def advanced_example_6_multi_gpu_simulation():
    """示例6: 多GPU场景模拟（分布式训练模拟）"""
    print("\n========== 示例6: 多GPU场景模拟 ==========\n")
    
    # 模拟多个GPU上的训练进程
    for gpu_id in range(2):
        train_id = f"gpu_{gpu_id}_training"
        
        agent = TrainingAgent(
            server_config=SERVER_CONFIG,
            agent_config=AGENT_CONFIG,
            metrics_config=METRICS_CONFIG,
            train_id=train_id
        )
        
        try:
            with agent.epoch_context():
                # 不同GPU上的loss会略有不同
                loss = 0.5 + 0.05 * gpu_id
                agent.record_loss(loss)
                agent.record_accuracy(0.85 + 0.02 * gpu_id)
                print(f"  GPU{gpu_id} - Loss: {loss:.3f}")
        finally:
            agent.close()
    
    print("✓ 所有GPU的指标已上传，各有不同的train_id区分")


def advanced_example_7_get_summary():
    """示例7: 获取训练总结"""
    print("\n========== 示例7: 获取训练总结 ==========\n")
    
    agent = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=AGENT_CONFIG,
        metrics_config=METRICS_CONFIG,
        train_id="advanced_summary"
    )
    
    try:
        with agent.epoch_context():
            # 记录一些指标
            for i in range(5):
                agent.record_loss(0.5 - 0.05 * i)
                agent.record_accuracy(0.7 + 0.03 * i)
            
            # 获取当前总结
            summary = agent.get_summary()
            
            print("📋 训练总结:")
            print(f"  Train ID: {summary['train_id']}")
            print(f"  上传策略: {summary['upload_frequency']}")
            
            metrics = summary['current_metrics']
            print(f"  当前Epoch: {metrics['epoch']}")
            print(f"  当前Batch: {metrics['batch']}")
            
            if 'loss' in metrics:
                loss_info = metrics['loss']
                print(f"  Loss - 平均值: {loss_info['mean']:.4f}, "
                      f"最小值: {loss_info['min']:.4f}, "
                      f"最大值: {loss_info['max']:.4f}")
    
    finally:
        agent.close()


def advanced_example_8_batch_level_tracking():
    """示例8: Batch级别的细粒度跟踪"""
    print("\n========== 示例8: Batch级别跟踪 ==========\n")
    
    # 配置为batch级上传
    config = AGENT_CONFIG.copy()
    config['upload_frequency'] = 'batch'
    config['upload_interval'] = 2  # 每2个batch上传一次
    
    agent = TrainingAgent(
        server_config=SERVER_CONFIG,
        agent_config=config,
        metrics_config=METRICS_CONFIG,
        train_id="advanced_batch_tracking"
    )
    
    try:
        with agent.epoch_context():
            # 模拟10个batch
            for batch_idx in range(10):
                with agent.batch_context():
                    loss = 1.0 - 0.05 * batch_idx
                    agent.record_loss(loss)
                    agent.record_accuracy(0.5 + 0.02 * batch_idx)
                
                print(f"  Batch {batch_idx+1}: ", end='')
                if (batch_idx + 1) % 2 == 0:
                    print("✓ 上传")
                else:
                    print("")
    
    finally:
        agent.close()


def main():
    """运行所有高级示例"""
    
    print("""
    ╔════════════════════════════════════════════════╗
    ║     PyTorch Training Agent - 高级示例集        ║
    ╚════════════════════════════════════════════════╝
    """)
    
    examples = [
        ("自定义指标", advanced_example_1_custom_metrics),
        ("上传策略", advanced_example_2_different_upload_strategies),
        ("手动上传", advanced_example_3_manual_uploads),
        ("系统监控", advanced_example_4_system_monitoring),
        ("上下文管理器", advanced_example_5_context_manager),
        ("多GPU模拟", advanced_example_6_multi_gpu_simulation),
        ("训练总结", advanced_example_7_get_summary),
        ("Batch级跟踪", advanced_example_8_batch_level_tracking),
    ]
    
    for idx, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"❌ 示例 {idx} ({name}) 出错: {e}")
    
    print("\n" + "="*50)
    print("✓ 所有高级示例演示完成！")
    print("="*50)


if __name__ == '__main__':
    main()
