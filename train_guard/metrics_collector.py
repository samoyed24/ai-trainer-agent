"""训练指标收集器。"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import psutil
import torch


class MetricsCollector:
    """收集训练过程中的标量指标。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.metrics_buffer: defaultdict[str, list[float]] = defaultdict(list)
        self.epoch_start_time: float | None = None
        self.batch_start_time: float | None = None
        self.current_epoch = 0
        self.current_batch = 0
        self.global_step = 0

    def start_epoch(self) -> None:
        self.epoch_start_time = time.time()
        self.current_batch = 0

    def end_epoch(self) -> None:
        if self.epoch_start_time and self.config["monitor"].get("epoch_time"):
            self.metrics_buffer["epoch_time"].append(time.time() - self.epoch_start_time)
        self.current_epoch += 1

    def start_batch(self) -> None:
        self.batch_start_time = time.time()

    def end_batch(self) -> None:
        if self.batch_start_time and self.config["monitor"].get("batch_time"):
            self.metrics_buffer["batch_time"].append(time.time() - self.batch_start_time)
        self.current_batch += 1
        self.global_step += 1

    def record_loss(self, loss: float) -> None:
        if self.config["monitor"].get("loss"):
            self.metrics_buffer["loss"].append(float(loss))

    def record_accuracy(self, accuracy: float) -> None:
        if self.config["monitor"].get("accuracy"):
            self.metrics_buffer["accuracy"].append(float(accuracy))

    def record_learning_rate(self, learning_rate: float) -> None:
        if self.config["monitor"].get("learning_rate"):
            self.metrics_buffer["learning_rate"].append(float(learning_rate))

    def record_custom_metric(self, name: str, value: float) -> None:
        self.metrics_buffer[name].append(float(value))

    def build_event(self, summary_mode: bool = False) -> dict[str, Any]:
        event: dict[str, Any] = {
            "epoch": self.current_epoch,
            "batch": self.current_batch,
            "step": self.global_step,
            "timestamp": time.time(),
        }

        for key, values in self.metrics_buffer.items():
            if not values:
                continue
            event[key] = (sum(values) / len(values)) if summary_mode else values[-1]

        if self.config.get("include_system_info"):
            event.update(self._get_system_metrics())

        return event

    def get_current_metrics(self) -> dict[str, Any]:
        metrics = self.build_event(summary_mode=False)
        for key, values in self.metrics_buffer.items():
            if values:
                metrics[f"{key}_stats"] = {
                    "current": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        return metrics

    def get_epoch_summary(self) -> dict[str, Any]:
        summary = self.build_event(summary_mode=True)
        for key, values in self.metrics_buffer.items():
            if values:
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_count"] = len(values)
        return summary

    def clear_buffer(self) -> None:
        self.metrics_buffer.clear()

    def _get_system_metrics(self) -> dict[str, Any]:
        prefix = str(self.config.get("system_info_prefix", "system")).strip() or "system"
        info: dict[str, Any] = {}

        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
            memory = psutil.virtual_memory()
            info[f"{prefix}_cpu_percent"] = cpu_percent
            info[f"{prefix}_memory_percent"] = memory.percent
            info[f"{prefix}_memory_available_mb"] = memory.available / (1024 * 1024)
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                info[f"{prefix}_gpu_allocated_mb"] = gpu_memory["allocated_bytes.current"] / (1024 * 1024)
                info[f"{prefix}_gpu_reserved_mb"] = gpu_memory["reserved_bytes.current"] / (1024 * 1024)
                info[f"{prefix}_gpu_count"] = torch.cuda.device_count()
            except Exception:
                pass

        return info

    def reset(self) -> None:
        self.metrics_buffer.clear()
        self.current_epoch = 0
        self.current_batch = 0
        self.global_step = 0
        self.epoch_start_time = None
        self.batch_start_time = None
