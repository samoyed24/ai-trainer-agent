"""高级示例。"""

from __future__ import annotations

from copy import deepcopy

from train_guard.agent import TrainingAgent
from train_guard.config import AGENT_CONFIG, METRICS_CONFIG, SERVER_CONFIG


def _demo_agent(**overrides):
    server_config = deepcopy(SERVER_CONFIG)
    agent_config = deepcopy(AGENT_CONFIG)
    metrics_config = deepcopy(METRICS_CONFIG)

    server_config["url"] = "http://localhost:8000/api/metrics/ingest"
    agent_config["enable_async"] = False

    if "server_config" in overrides:
        server_config.update(overrides.pop("server_config"))
    if "agent_config" in overrides:
        agent_config.update(overrides.pop("agent_config"))
    if "metrics_config" in overrides:
        metrics_config.update(overrides.pop("metrics_config"))

    return TrainingAgent(
        server_config=server_config,
        agent_config=agent_config,
        metrics_config=metrics_config,
        train_id=overrides.pop("train_id", "advanced-demo"),
        **overrides,
    )


def custom_metrics_example():
    with _demo_agent(train_id="advanced_custom_metrics") as agent:
        with agent.epoch_context():
            agent.record_loss(0.234)
            agent.record_accuracy(0.92)
            agent.record_metric("precision", 0.95)
            agent.record_metric("recall", 0.89)
            agent.record_metric("f1_score", 0.92)


def batch_upload_example():
    with _demo_agent(
        train_id="advanced_batch_tracking",
        agent_config={"upload_frequency": "batch", "upload_interval": 2},
    ) as agent:
        with agent.epoch_context():
            for batch_idx in range(4):
                with agent.batch_context():
                    agent.record_loss(1.0 - 0.05 * batch_idx)
                    agent.record_accuracy(0.5 + 0.05 * batch_idx)


def system_metrics_example():
    with _demo_agent(
        train_id="advanced_system_metrics",
        metrics_config={"include_system_info": True},
    ) as agent:
        with agent.epoch_context():
            agent.record_loss(0.5)
            agent.record_accuracy(0.85)


if __name__ == "__main__":
    custom_metrics_example()
    batch_upload_example()
    system_metrics_example()
