"""
单元测试 - 验证Training Agent的各项功能
"""

import unittest
import time
import json
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

from ai_trainer_agent.agent import TrainingAgent
from ai_trainer_agent.metrics_collector import MetricsCollector
from ai_trainer_agent.uploader import MetricsUploader
from ai_trainer_agent.config import (
    AGENT_CONFIG,
    METRICS_CONFIG,
    SERVER_CONFIG,
    fetch_remote_config,
)


def _isolated_runtime_config():
    """返回隔离后的运行配置，确保测试不会读取用户本地后端凭证。"""
    return {
        "backend": {
            "api": "",
            "app_id": "",
            "app_secret": "",
            "timeout": 10,
        },
        "server": {
            "url": "http://localhost:8000/metrics",
            "timeout": 10,
            "retry_count": 3,
        },
        "agent": {
            "upload_frequency": "epoch",
            "upload_interval": 1,
            "buffer_size": 100,
            "enable_async": False,
        },
        "metrics": {
            "monitor": {
                "loss": True,
                "accuracy": True,
                "learning_rate": True,
                "batch_time": True,
                "epoch_time": True,
                "model_size": True,
            },
            "include_system_info": True,
        },
    }


def _isolated_server_config():
    return _isolated_runtime_config()["server"]


def _isolated_agent_config():
    return _isolated_runtime_config()["agent"]


def _isolated_metrics_config():
    return _isolated_runtime_config()["metrics"]


class TestMetricsCollector(unittest.TestCase):
    """测试指标收集器"""
    
    def setUp(self):
        self.collector = MetricsCollector(METRICS_CONFIG)
    
    def test_record_loss(self):
        """测试loss记录"""
        self.collector.record_loss(0.5)
        self.collector.record_loss(0.3)
        
        metrics = self.collector.get_current_metrics()
        self.assertIn('loss', metrics)
        self.assertEqual(metrics['loss']['count'], 2)
        self.assertAlmostEqual(metrics['loss']['mean'], 0.4)
    
    def test_record_accuracy(self):
        """测试准确率记录"""
        self.collector.record_accuracy(0.8)
        self.collector.record_accuracy(0.9)
        
        metrics = self.collector.get_current_metrics()
        self.assertIn('accuracy', metrics)
        self.assertEqual(metrics['accuracy']['count'], 2)
        self.assertAlmostEqual(metrics['accuracy']['mean'], 0.85)
    
    def test_epoch_context(self):
        """测试epoch时间计算"""
        self.collector.start_epoch()
        time.sleep(0.1)
        self.collector.end_epoch()
        
        metrics = self.collector.get_current_metrics()
        self.assertIn('epoch_time', metrics)
        self.assertGreater(metrics['epoch_time']['current'], 0.05)
    
    def test_batch_context(self):
        """测试batch时间计算"""
        self.collector.start_batch()
        time.sleep(0.05)
        self.collector.end_batch()
        
        metrics = self.collector.get_current_metrics()
        self.assertIn('batch_time', metrics)
        self.assertGreater(metrics['batch_time']['current'], 0.03)
    
    def test_clear_buffer(self):
        """测试buffer清空"""
        self.collector.record_loss(0.5)
        self.collector.record_accuracy(0.8)
        
        self.collector.clear_buffer()
        metrics = self.collector.get_current_metrics()
        
        # batch和epoch计数器应该保留，但metrics应该为空
        self.assertNotIn('loss', metrics)
        self.assertNotIn('accuracy', metrics)
    
    def test_system_info(self):
        """测试系统信息收集"""
        metrics = self.collector.get_current_metrics()
        self.assertIn('system_info', metrics)
        info = metrics['system_info']
        self.assertIn('cpu_percent', info)


class TestMetricsUploader(unittest.TestCase):
    """测试上传器"""
    
    def setUp(self):
        self.uploader = MetricsUploader(SERVER_CONFIG, AGENT_CONFIG)
    
    def tearDown(self):
        self.uploader.close()
    
    @patch('requests.Session.post')
    def test_successful_upload(self, mock_post):
        """测试成功上传"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        metrics = {'epoch': 1, 'loss': 0.5}
        result = self.uploader._upload_with_retry(metrics)
        
        self.assertTrue(result)
        self.assertTrue(mock_post.called)
    
    @patch('requests.Session.post')
    def test_upload_retry(self, mock_post):
        """测试重试机制"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        metrics = {'epoch': 1, 'loss': 0.5}
        result = self.uploader._upload_with_retry(metrics)
        
        # 应该重试3次
        self.assertFalse(result)
        self.assertEqual(mock_post.call_count, 3)


class TestRemoteConfigFetch(unittest.TestCase):
    """测试后端配置拉取"""

    @patch('ai_trainer_agent.config.requests.post')
    def test_fetch_remote_config_from_data_config(self, mock_post):
        """支持从 data.config 提取完整配置"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": 0,
            "data": {
                "config": {
                    "server": {"url": "https://remote.example.com/metrics"},
                    "agent": {"upload_frequency": "batch", "upload_interval": 2},
                }
            },
        }
        mock_post.return_value = mock_response

        config = fetch_remote_config(
            backend_api="https://backend.example.com/config",
            app_id="app_001",
            app_secret="secret_001",
            timeout=8,
        )

        self.assertEqual(config["server"]["url"], "https://remote.example.com/metrics")
        self.assertEqual(config["agent"]["upload_frequency"], "batch")
        self.assertEqual(config["agent"]["upload_interval"], 2)
        mock_post.assert_called_once_with(
            "https://backend.example.com/config",
            json={"app_id": "app_001", "app_secret": "secret_001"},
            headers={"X-App-Id": "app_001", "X-App-Secret": "secret_001"},
            timeout=8,
        )

    @patch('ai_trainer_agent.config.requests.post')
    def test_fetch_remote_config_support_server_url_shortcut(self, mock_post):
        """支持后端简化字段 server_url"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "server_url": "https://remote.example.com/webhook"
            }
        }
        mock_post.return_value = mock_response

        config = fetch_remote_config(
            backend_api="https://backend.example.com/config",
            app_id="app_001",
            app_secret="secret_001",
        )

        self.assertEqual(config["server"]["url"], "https://remote.example.com/webhook")


class TestTrainingAgent(unittest.TestCase):
    """测试训练Agent"""
    
    def setUp(self):
        self.load_config_patcher = patch(
            'ai_trainer_agent.agent.load_config',
            return_value=_isolated_runtime_config(),
        )
        self.load_config_patcher.start()

        self.session_post_patcher = patch('requests.Session.post')
        self.mock_session_post = self.session_post_patcher.start()
        mock_response = MagicMock()
        mock_response.status_code = 200
        self.mock_session_post.return_value = mock_response

        self.agent = TrainingAgent(
            server_config=_isolated_server_config(),
            agent_config=_isolated_agent_config(),
            metrics_config=_isolated_metrics_config(),
            train_id="test_001"
        )
    
    def tearDown(self):
        self.agent.close()
        self.session_post_patcher.stop()
        self.load_config_patcher.stop()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.agent.train_id, "test_001")
        self.assertIsNotNone(self.agent.collector)
        self.assertIsNotNone(self.agent.uploader)

    @patch('ai_trainer_agent.agent.fetch_remote_config')
    def test_minimal_initialization_with_defaults(self, mock_fetch_remote_config):
        """测试最简初始化（不传配置）"""
        agent = TrainingAgent(train_id="minimal_default")
        try:
            self.assertEqual(agent.train_id, "minimal_default")
            self.assertIsNotNone(agent.server_config.get("url"))
            mock_fetch_remote_config.assert_not_called()
        finally:
            agent.close()

    @patch('ai_trainer_agent.agent.fetch_remote_config')
    def test_minimal_initialization_with_server_url(self, mock_fetch_remote_config):
        """测试最简初始化（仅传server_url）"""
        custom_url = "https://example.com/webhook"
        agent = TrainingAgent(server_url=custom_url, train_id="minimal_url")
        try:
            self.assertEqual(agent.server_config.get("url"), custom_url)
            mock_fetch_remote_config.assert_not_called()
        finally:
            agent.close()

    @patch('ai_trainer_agent.agent.fetch_remote_config')
    def test_minimal_initialization_with_backend_credentials(self, mock_fetch_remote_config):
        """测试仅传后端凭证自动拉取配置"""
        mock_fetch_remote_config.return_value = {
            "server": {
                "url": "https://remote.example.com/metrics",
                "timeout": 15,
            },
            "agent": {
                "upload_frequency": "batch",
                "upload_interval": 3,
            },
        }

        agent = TrainingAgent(
            backend_api="https://backend.example.com/config",
            app_id="app_001",
            app_secret="secret_001",
            train_id="backend_mode",
        )
        try:
            self.assertEqual(agent.server_config.get("url"), "https://remote.example.com/metrics")
            self.assertEqual(agent.upload_frequency, "batch")
            self.assertEqual(agent.upload_interval, 3)
        finally:
            agent.close()

        mock_fetch_remote_config.assert_called_once_with(
            backend_api="https://backend.example.com/config",
            app_id="app_001",
            app_secret="secret_001",
            timeout=10,
        )

    def test_backend_credentials_must_be_complete(self):
        """测试后端凭证必须完整提供"""
        with self.assertRaises(ValueError):
            TrainingAgent(
                backend_api="https://backend.example.com/config",
                app_id="app_001",
                train_id="invalid_backend",
            )
    
    def test_record_metrics(self):
        """测试记录指标"""
        self.agent.record_loss(0.5)
        self.agent.record_accuracy(0.85)
        self.agent.record_learning_rate(0.001)
        
        metrics = self.agent.collector.get_current_metrics()
        self.assertAlmostEqual(metrics['loss']['current'], 0.5)
        self.assertAlmostEqual(metrics['accuracy']['current'], 0.85)
        self.assertAlmostEqual(metrics['learning_rate']['current'], 0.001)
    
    def test_epoch_context(self):
        """测试epoch上下文"""
        with self.agent.epoch_context():
            self.agent.record_loss(0.5)
            self.agent.record_accuracy(0.85)
        
        # epoch应该在上下文退出时递增
        self.assertEqual(self.agent.collector.current_epoch, 1)
    
    def test_batch_context(self):
        """测试batch上下文"""
        with self.agent.batch_context():
            self.agent.record_loss(0.5)
        
        self.assertEqual(self.agent.collector.current_batch, 1)
    
    def test_custom_metric(self):
        """测试自定义指标"""
        self.agent.record_metric('f1_score', 0.92)
        self.agent.record_metric('precision', 0.95)
        
        metrics = self.agent.collector.get_current_metrics()
        self.assertAlmostEqual(metrics['f1_score']['current'], 0.92)
        self.assertAlmostEqual(metrics['precision']['current'], 0.95)
    
    def test_get_summary(self):
        """测试获取总结"""
        with self.agent.epoch_context():
            self.agent.record_loss(0.5)
        
        summary = self.agent.get_summary()
        self.assertEqual(summary['train_id'], "test_001")
        self.assertIn('current_metrics', summary)
        self.assertIn('upload_frequency', summary)
    
    @patch.object(TrainingAgent, '_upload_epoch_metrics')
    def test_epoch_upload(self, mock_upload):
        """测试epoch级上传"""
        self.agent.upload_frequency = 'epoch'
        
        with self.agent.epoch_context():
            self.agent.record_loss(0.5)
        
        # 应该调用上传方法
        mock_upload.assert_called_once()
    
    def test_context_manager(self):
        """测试with语句使用"""
        with TrainingAgent(
            server_config=_isolated_server_config(),
            agent_config=_isolated_agent_config(),
            metrics_config=_isolated_metrics_config()
        ) as agent:
            self.assertIsNotNone(agent)
            agent.record_loss(0.5)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        self.load_config_patcher = patch(
            'ai_trainer_agent.agent.load_config',
            return_value=_isolated_runtime_config(),
        )
        self.load_config_patcher.start()

    def tearDown(self):
        self.load_config_patcher.stop()
    
    @patch('ai_trainer_agent.agent.fetch_remote_config')
    @patch('requests.Session.post')
    def test_full_training_workflow(self, mock_post, mock_fetch_remote_config):
        """测试完整训练流程"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        with TrainingAgent(
            server_config=_isolated_server_config(),
            agent_config=_isolated_agent_config(),
            metrics_config=_isolated_metrics_config(),
            train_id="integration_test"
        ) as agent:
            # 模拟多个epoch的训练
            for epoch in range(2):
                with agent.epoch_context():
                    for batch in range(5):
                        with agent.batch_context():
                            loss = 1.0 - 0.1 * epoch - 0.01 * batch
                            accuracy = 0.5 + 0.05 * epoch + 0.01 * batch
                            
                            agent.record_loss(loss)
                            agent.record_accuracy(accuracy)
                            agent.record_learning_rate(0.001)
        
        # 应该至少上传2次（每个epoch）
        self.assertGreaterEqual(mock_post.call_count, 2)
        mock_fetch_remote_config.assert_not_called()


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsUploader))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════╗
    ║   PyTorch Training Agent - 单元测试   ║
    ╚════════════════════════════════════════╝
    """)
    
    success = run_tests()
    
    if success:
        print("\n✓ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败")
        sys.exit(1)
