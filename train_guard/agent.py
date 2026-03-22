"""Train Guard Agent 主模块。"""

from __future__ import annotations

import copy
import logging
import uuid
from contextlib import contextmanager
from typing import Any

from .config import (
    _build_agent_headers,
    fetch_remote_config,
    load_config,
    merge_config,
)
from .metrics_collector import MetricsCollector
from .uploader import MetricsUploader


logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class TrainingAgent:
    """收集并上报训练指标到 Train Guard Dashboard。"""

    def __init__(
        self,
        server_config: dict[str, Any] | None = None,
        agent_config: dict[str, Any] | None = None,
        metrics_config: dict[str, Any] | None = None,
        server_url: str | None = None,
        train_id: str | None = None,
        backend_base_url: str | None = None,
        access_key_id: str | None = None,
        secret_key: str | None = None,
        project_id: str | None = None,
    ):
        loaded = load_config()
        backend_local = loaded.get("backend", {})

        resolved_backend_base_url = backend_base_url or backend_local.get("base_url", "")
        resolved_access_key_id = access_key_id or backend_local.get("access_key_id", "")
        resolved_secret_key = secret_key or backend_local.get("secret_key", "")
        resolved_project_id = project_id or backend_local.get("project_id", "")
        resolved_backend_timeout = int(backend_local.get("timeout", 10))

        backend_values = [
            resolved_backend_base_url,
            resolved_access_key_id,
            resolved_secret_key,
            resolved_project_id,
        ]
        has_any_backend_field = any(backend_values)
        has_complete_backend_field = all(backend_values)

        if has_any_backend_field and not has_complete_backend_field:
            raise ValueError("后端自动拉取配置要求 backend_base_url、access_key_id、secret_key、project_id 同时提供")

        self.auth_headers: dict[str, str] = {}
        if has_complete_backend_field:
            remote_config = fetch_remote_config(
                base_url=resolved_backend_base_url,
                access_key_id=resolved_access_key_id,
                secret_key=resolved_secret_key,
                project_id=resolved_project_id,
                timeout=resolved_backend_timeout,
            )
            loaded = merge_config(loaded, remote_config)
            self.auth_headers = _build_agent_headers(
                access_key_id=resolved_access_key_id,
                secret_key=resolved_secret_key,
                project_id=resolved_project_id,
            )
            logger.info("已从后端拉取配置")

        resolved_server_config = copy.deepcopy(loaded["server"])
        resolved_agent_config = copy.deepcopy(loaded["agent"])
        resolved_metrics_config = copy.deepcopy(loaded["metrics"])

        if server_config:
            resolved_server_config = merge_config(resolved_server_config, server_config)
        if agent_config:
            resolved_agent_config = merge_config(resolved_agent_config, agent_config)
        if metrics_config:
            resolved_metrics_config = merge_config(resolved_metrics_config, metrics_config)
        if server_url:
            resolved_server_config["url"] = server_url

        if not resolved_server_config.get("url"):
            raise ValueError("server_config.url 不能为空")

        self.train_id = train_id or str(uuid.uuid4())
        self.server_config = resolved_server_config
        self.agent_config = resolved_agent_config
        self.metrics_config = resolved_metrics_config
        self.collector = MetricsCollector(self.metrics_config)
        self.uploader = MetricsUploader(self.server_config, self.agent_config, auth_headers=self.auth_headers)
        self.upload_frequency = self.agent_config.get("upload_frequency", "epoch")
        self.upload_interval = self.agent_config.get("upload_interval", 1)
        self.batch_count_since_upload = 0

        logger.info("训练 Agent 初始化完成 (train_id=%s)", self.train_id)
        logger.info("上传策略: %s", self.upload_frequency)
        logger.info("服务器: %s", self.server_config["url"])

    @contextmanager
    def epoch_context(self):
        self.collector.start_epoch()
        try:
            yield
        finally:
            self.collector.end_epoch()
            if self.upload_frequency == "epoch":
                self._upload_epoch_metrics()

    @contextmanager
    def batch_context(self):
        self.collector.start_batch()
        try:
            yield
        finally:
            self.collector.end_batch()
            self.batch_count_since_upload += 1
            if self.upload_frequency == "batch" and self.batch_count_since_upload >= self.upload_interval:
                self._upload_batch_metrics()
                self.batch_count_since_upload = 0

    def record_loss(self, loss: float) -> None:
        self.collector.record_loss(loss)

    def record_accuracy(self, accuracy: float) -> None:
        self.collector.record_accuracy(accuracy)

    def record_learning_rate(self, learning_rate: float) -> None:
        self.collector.record_learning_rate(learning_rate)

    def record_metric(self, name: str, value: float) -> None:
        self.collector.record_custom_metric(name, value)

    def _build_payload(self, metrics: dict[str, Any]) -> dict[str, Any]:
        payload = dict(metrics)
        payload["train_id"] = self.train_id
        return payload

    def _upload_epoch_metrics(self) -> None:
        payload = self._build_payload(self.collector.get_epoch_summary())
        success = self.uploader.upload(payload)
        if success:
            self.collector.clear_buffer()

    def _upload_batch_metrics(self) -> None:
        payload = self._build_payload(self.collector.build_event(summary_mode=False))
        self.uploader.upload(payload)

    def manual_upload(self, description: str | None = None) -> None:
        payload = self._build_payload(self.collector.build_event(summary_mode=False))
        if description:
            payload["description"] = description
        self.uploader.upload(payload)

    def get_summary(self) -> dict[str, Any]:
        return {
            "train_id": self.train_id,
            "current_metrics": self.collector.get_current_metrics(),
            "upload_frequency": self.upload_frequency,
            "server_url": self.server_config["url"],
        }

    def close(self) -> None:
        self.uploader.close()
        self.collector.reset()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
