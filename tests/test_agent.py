"""Train Guard Agent 测试。"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from train_guard.agent import TrainingAgent
from train_guard.config import AGENT_CONFIG, METRICS_CONFIG, SERVER_CONFIG, fetch_remote_config
from train_guard.metrics_collector import MetricsCollector
from train_guard.uploader import MetricsUploader


def _isolated_runtime_config():
    return {
        "backend": {
            "base_url": "",
            "access_key_id": "",
            "secret_key": "",
            "project_id": "",
            "timeout": 10,
        },
        "server": {
            "url": "http://localhost:8000/api/metrics/ingest",
            "timeout": 10,
            "retry_count": 3,
        },
        "agent": {
            "upload_frequency": "epoch",
            "upload_interval": 1,
            "enable_async": False,
        },
        "metrics": {
            "monitor": {
                "loss": True,
                "accuracy": True,
                "learning_rate": True,
                "batch_time": True,
                "epoch_time": True,
            },
            "include_system_info": True,
            "system_info_prefix": "system",
        },
    }


def _isolated_server_config():
    return _isolated_runtime_config()["server"]


def _isolated_agent_config():
    return _isolated_runtime_config()["agent"]


def _isolated_metrics_config():
    return _isolated_runtime_config()["metrics"]


class TestMetricsCollector(unittest.TestCase):
    def setUp(self):
        self.collector = MetricsCollector(_isolated_metrics_config())

    def test_record_loss(self):
        self.collector.record_loss(0.5)
        self.collector.record_loss(0.3)
        metrics = self.collector.get_current_metrics()
        self.assertEqual(metrics["loss"], 0.3)
        self.assertEqual(metrics["loss_stats"]["count"], 2)
        self.assertAlmostEqual(metrics["loss_stats"]["mean"], 0.4)

    def test_record_accuracy(self):
        self.collector.record_accuracy(0.8)
        self.collector.record_accuracy(0.9)
        metrics = self.collector.get_current_metrics()
        self.assertEqual(metrics["accuracy"], 0.9)
        self.assertAlmostEqual(metrics["accuracy_stats"]["mean"], 0.85)

    def test_epoch_context(self):
        self.collector.start_epoch()
        time.sleep(0.1)
        self.collector.end_epoch()
        metrics = self.collector.get_current_metrics()
        self.assertIn("epoch_time", metrics)
        self.assertGreater(metrics["epoch_time"], 0.05)

    def test_batch_context(self):
        self.collector.start_batch()
        time.sleep(0.05)
        self.collector.end_batch()
        metrics = self.collector.get_current_metrics()
        self.assertIn("batch_time", metrics)
        self.assertGreater(metrics["batch_time"], 0.03)

    def test_clear_buffer(self):
        self.collector.record_loss(0.5)
        self.collector.clear_buffer()
        metrics = self.collector.get_current_metrics()
        self.assertNotIn("loss", metrics)
        self.assertNotIn("loss_stats", metrics)

    def test_system_info(self):
        metrics = self.collector.get_current_metrics()
        self.assertIn("system_cpu_percent", metrics)


class TestMetricsUploader(unittest.TestCase):
    def setUp(self):
        self.uploader = MetricsUploader(
            SERVER_CONFIG,
            {**AGENT_CONFIG, "enable_async": False},
            auth_headers={
                "X-Access-Key-Id": "ak_001",
                "X-Secret-Key": "sk_001",
                "X-Project-Id": "project_001",
            },
        )

    def tearDown(self):
        self.uploader.close()

    @patch("requests.Session.post")
    def test_successful_upload(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_post.return_value = mock_response

        result = self.uploader._upload_with_retry({"train_id": "run_1", "loss": 0.5, "epoch": 1, "step": 2})

        self.assertTrue(result)
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["X-Project-Id"], "project_001")

    @patch("requests.Session.post")
    def test_upload_retry(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "server error"
        mock_post.return_value = mock_response

        result = self.uploader._upload_with_retry({"train_id": "run_1", "loss": 0.5})

        self.assertFalse(result)
        self.assertEqual(mock_post.call_count, 3)


class TestRemoteConfigFetch(unittest.TestCase):
    @patch("train_guard.config.requests.get")
    def test_fetch_remote_config_from_data_config(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "config": {
                    "server": {"url": "https://dashboard.example.com/api/metrics/ingest"},
                    "agent": {"upload_frequency": "batch", "upload_interval": 2},
                }
            },
        }
        mock_get.return_value = mock_response

        config = fetch_remote_config(
            base_url="https://dashboard.example.com",
            access_key_id="ak_001",
            secret_key="sk_001",
            project_id="project_001",
            timeout=8,
        )

        self.assertEqual(config["server"]["url"], "https://dashboard.example.com/api/metrics/ingest")
        self.assertEqual(config["agent"]["upload_frequency"], "batch")
        mock_get.assert_called_once_with(
            "https://dashboard.example.com/api/agent/config",
            headers={
                "X-Access-Key-Id": "ak_001",
                "X-Secret-Key": "sk_001",
                "X-Project-Id": "project_001",
            },
            timeout=8,
        )

    @patch("train_guard.config.requests.get")
    def test_fetch_remote_config_accepts_api_base_url(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"server": {"url": "https://dashboard.example.com/api/metrics/ingest"}}
        mock_get.return_value = mock_response

        config = fetch_remote_config(
            base_url="https://dashboard.example.com/api",
            access_key_id="ak_001",
            secret_key="sk_001",
            project_id="project_001",
        )

        self.assertEqual(config["server"]["url"], "https://dashboard.example.com/api/metrics/ingest")


class TestTrainingAgent(unittest.TestCase):
    def setUp(self):
        self.load_config_patcher = patch("train_guard.agent.load_config", return_value=_isolated_runtime_config())
        self.load_config_patcher.start()

        self.session_post_patcher = patch("requests.Session.post")
        self.mock_session_post = self.session_post_patcher.start()
        mock_response = MagicMock()
        mock_response.status_code = 202
        self.mock_session_post.return_value = mock_response

        self.agent = TrainingAgent(
            server_config=_isolated_server_config(),
            agent_config=_isolated_agent_config(),
            metrics_config=_isolated_metrics_config(),
            train_id="test_001",
        )

    def tearDown(self):
        self.agent.close()
        self.session_post_patcher.stop()
        self.load_config_patcher.stop()

    def test_initialization(self):
        self.assertEqual(self.agent.train_id, "test_001")
        self.assertIsNotNone(self.agent.collector)
        self.assertIsNotNone(self.agent.uploader)

    @patch("train_guard.agent.fetch_remote_config")
    def test_minimal_initialization_with_defaults(self, mock_fetch_remote_config):
        agent = TrainingAgent(server_url="http://localhost:8000/api/metrics/ingest", train_id="minimal_default")
        try:
            self.assertEqual(agent.train_id, "minimal_default")
            mock_fetch_remote_config.assert_not_called()
        finally:
            agent.close()

    @patch("train_guard.agent.fetch_remote_config")
    def test_minimal_initialization_with_backend_credentials(self, mock_fetch_remote_config):
        mock_fetch_remote_config.return_value = {
            "server": {"url": "https://dashboard.example.com/api/metrics/ingest", "timeout": 15},
            "agent": {"upload_frequency": "batch", "upload_interval": 3},
        }

        agent = TrainingAgent(
            backend_base_url="https://dashboard.example.com",
            access_key_id="ak_001",
            secret_key="sk_001",
            project_id="project_001",
            train_id="backend_mode",
        )
        try:
            self.assertEqual(agent.server_config["url"], "https://dashboard.example.com/api/metrics/ingest")
            self.assertEqual(agent.upload_frequency, "batch")
            self.assertEqual(agent.auth_headers["X-Project-Id"], "project_001")
        finally:
            agent.close()

        mock_fetch_remote_config.assert_called_once_with(
            base_url="https://dashboard.example.com",
            access_key_id="ak_001",
            secret_key="sk_001",
            project_id="project_001",
            timeout=10,
        )

    def test_backend_credentials_must_be_complete(self):
        with self.assertRaises(ValueError):
            TrainingAgent(
                backend_base_url="https://dashboard.example.com",
                access_key_id="ak_001",
                train_id="invalid_backend",
            )

    def test_record_metrics(self):
        self.agent.record_loss(0.5)
        self.agent.record_accuracy(0.85)
        self.agent.record_learning_rate(0.001)
        metrics = self.agent.collector.get_current_metrics()
        self.assertAlmostEqual(metrics["loss"], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.85)
        self.assertAlmostEqual(metrics["learning_rate"], 0.001)

    def test_epoch_context(self):
        with self.agent.epoch_context():
            self.agent.record_loss(0.5)
            self.agent.record_accuracy(0.85)
        self.assertEqual(self.agent.collector.current_epoch, 1)

    def test_batch_context(self):
        with self.agent.batch_context():
            self.agent.record_loss(0.5)
        self.assertEqual(self.agent.collector.current_batch, 1)
        self.assertEqual(self.agent.collector.global_step, 1)

    def test_custom_metric(self):
        self.agent.record_metric("f1_score", 0.92)
        metrics = self.agent.collector.get_current_metrics()
        self.assertAlmostEqual(metrics["f1_score"], 0.92)

    def test_get_summary(self):
        with self.agent.epoch_context():
            self.agent.record_loss(0.5)
        summary = self.agent.get_summary()
        self.assertEqual(summary["train_id"], "test_001")
        self.assertIn("current_metrics", summary)
        self.assertIn("server_url", summary)

    @patch.object(TrainingAgent, "_upload_epoch_metrics")
    def test_epoch_upload(self, mock_upload):
        self.agent.upload_frequency = "epoch"
        with self.agent.epoch_context():
            self.agent.record_loss(0.5)
        mock_upload.assert_called_once()

    def test_context_manager(self):
        with TrainingAgent(
            server_config=_isolated_server_config(),
            agent_config=_isolated_agent_config(),
            metrics_config=_isolated_metrics_config(),
        ) as agent:
            agent.record_loss(0.5)
            self.assertIsNotNone(agent)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.load_config_patcher = patch("train_guard.agent.load_config", return_value=_isolated_runtime_config())
        self.load_config_patcher.start()

    def tearDown(self):
        self.load_config_patcher.stop()

    @patch("requests.Session.post")
    def test_full_training_workflow(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_post.return_value = mock_response

        with TrainingAgent(
            server_config=_isolated_server_config(),
            agent_config=_isolated_agent_config(),
            metrics_config=_isolated_metrics_config(),
            train_id="integration_test",
        ) as agent:
            for epoch in range(2):
                with agent.epoch_context():
                    for batch in range(5):
                        with agent.batch_context():
                            loss = 1.0 - 0.1 * epoch - 0.01 * batch
                            accuracy = 0.5 + 0.05 * epoch + 0.01 * batch
                            agent.record_loss(loss)
                            agent.record_accuracy(accuracy)
                            agent.record_learning_rate(0.001)

        self.assertGreaterEqual(mock_post.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
