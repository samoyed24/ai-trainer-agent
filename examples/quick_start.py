"""面向 train-guard-dashboard 的快速示例。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from train_guard.agent import TrainingAgent


def simple_training_example():
    model = nn.Linear(10, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = torch.randn(100, 10).to(device)
    targets = torch.randint(0, 2, (100,)).to(device)

    agent = TrainingAgent(
        backend_base_url="http://localhost:8000",
        access_key_id="ak_demo",
        secret_key="sk_demo",
        project_id="project_demo",
        train_id="quick_demo_001",
    )

    try:
        for epoch in range(3):
            with agent.epoch_context():
                correct = 0
                for start in range(0, len(inputs), 10):
                    batch_x = inputs[start : start + 10]
                    batch_y = targets[start : start + 10]

                    with agent.batch_context():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        _, preds = outputs.max(1)
                        accuracy = preds.eq(batch_y).sum().item() / len(batch_y)

                        agent.record_loss(loss.item())
                        agent.record_accuracy(accuracy)
                        agent.record_learning_rate(optimizer.param_groups[0]["lr"])
                        correct += preds.eq(batch_y).sum().item()

                print(f"Epoch {epoch + 1}: accuracy={correct / len(inputs):.2%}")
    finally:
        agent.close()


def quick_api_reference():
    print(
        """
========== Train Guard Agent 快速 API ==========

agent = TrainingAgent(
    backend_base_url="http://localhost:8000",
    access_key_id="ak_xxx",
    secret_key="sk_xxx",
    project_id="project_abcd1234",
)

with agent.epoch_context():
    with agent.batch_context():
        agent.record_loss(loss_value)
        agent.record_accuracy(acc_value)
        agent.record_learning_rate(lr_value)
        agent.record_metric("f1_score", 0.92)

agent.manual_upload(description="checkpoint_epoch_5")
summary = agent.get_summary()
agent.close()
================================================
"""
    )


if __name__ == "__main__":
    quick_api_reference()
    simple_training_example()
