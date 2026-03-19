"""
PyTorch训练Agent - 主模块
管理指标收集和上传的完整流程
"""

import copy
import logging
import uuid
from typing import Dict, Any, Optional
from contextlib import contextmanager

from .config import fetch_remote_config, load_config, merge_config
from .metrics_collector import MetricsCollector
from .uploader import MetricsUploader


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingAgent:
    """PyTorch训练Agent - 负责收集和上传训练指标"""
    
    def __init__(
        self,
        server_config: Optional[Dict[str, Any]] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        metrics_config: Optional[Dict[str, Any]] = None,
        server_url: Optional[str] = None,
        train_id: Optional[str] = None,
        backend_api: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
    ):
        """
        初始化训练Agent
        
        Args:
            server_config: 服务器配置（可选，默认读取用户配置）
            agent_config: Agent配置（可选，默认读取用户配置）
            metrics_config: 指标监控配置（可选，默认读取用户配置）
            server_url: 快速覆盖上报地址（只上报场景可仅传这个参数）
            train_id: 训练ID（自动生成如果不提供）
            backend_api: 配置后端 API 地址（提供后将自动拉取配置）
            app_id: 后端应用 ID
            app_secret: 后端应用密钥
        """
        loaded = load_config()
        backend_local = loaded.get("backend", {})

        resolved_backend_api = backend_api or backend_local.get("api", "")
        resolved_app_id = app_id or backend_local.get("app_id", "")
        resolved_app_secret = app_secret or backend_local.get("app_secret", "")
        resolved_backend_timeout = int(backend_local.get("timeout", 10))

        has_any_backend_field = any([resolved_backend_api, resolved_app_id, resolved_app_secret])
        has_complete_backend_field = all([resolved_backend_api, resolved_app_id, resolved_app_secret])

        if has_any_backend_field and not has_complete_backend_field:
            raise ValueError("后端自动拉取配置要求 backend_api、app_id、app_secret 同时提供")

        if has_complete_backend_field:
            remote_config = fetch_remote_config(
                backend_api=resolved_backend_api,
                app_id=resolved_app_id,
                app_secret=resolved_app_secret,
                timeout=resolved_backend_timeout,
            )
            loaded = merge_config(loaded, remote_config)
            logger.info("✓ 已从后端拉取配置")

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

        self.train_id = train_id or str(uuid.uuid4())[:8]
        self.server_config = resolved_server_config
        self.agent_config = resolved_agent_config
        self.metrics_config = resolved_metrics_config
        
        # 初始化组件
        self.collector = MetricsCollector(self.metrics_config)
        self.uploader = MetricsUploader(self.server_config, self.agent_config)
        
        # 配置上传策略
        self.upload_frequency = self.agent_config.get('upload_frequency', 'epoch')
        self.upload_interval = self.agent_config.get('upload_interval', 1)
        self.batch_count_since_upload = 0
        
        logger.info(f"🚀 训练Agent初始化完成 (Train ID: {self.train_id})")
        logger.info(f"   上传策略: {self.upload_frequency}")
        logger.info(f"   服务器: {self.server_config['url']}")
    
    @contextmanager
    def epoch_context(self):
        """
        Epoch上下文管理器
        
        使用方法:
            with agent.epoch_context():
                # 训练代码
        """
        self.collector.start_epoch()
        try:
            yield
        finally:
            self.collector.end_epoch()
            if self.upload_frequency == 'epoch':
                self._upload_epoch_metrics()
    
    @contextmanager
    def batch_context(self):
        """
        Batch上下文管理器
        
        使用方法:
            with agent.batch_context():
                # 批次训练代码
        """
        self.collector.start_batch()
        try:
            yield
        finally:
            self.collector.end_batch()
            self.batch_count_since_upload += 1
            
            # 根据配置决定是否上传
            if self.upload_frequency == 'batch':
                if self.batch_count_since_upload >= self.upload_interval:
                    self._upload_batch_metrics()
                    self.batch_count_since_upload = 0
    
    def record_loss(self, loss: float):
        """记录训练损失"""
        self.collector.record_loss(loss)
    
    def record_accuracy(self, accuracy: float):
        """记录准确率"""
        self.collector.record_accuracy(accuracy)
    
    def record_learning_rate(self, learning_rate: float):
        """记录学习率"""
        self.collector.record_learning_rate(learning_rate)
    
    def record_metric(self, name: str, value: float):
        """记录自定义指标"""
        self.collector.record_custom_metric(name, value)
    
    def _upload_epoch_metrics(self):
        """上传epoch级别的指标"""
        metrics = self.collector.get_epoch_summary()
        logger.debug(f"📊 Epoch {metrics['epoch']} 指标统计: loss_avg={metrics.get('loss_avg', 'N/A'):.4f}")
        
        success = self.uploader.upload(metrics, self.train_id)
        
        if success:
            # 清空buffer以准备下一个epoch
            self.collector.clear_buffer()
    
    def _upload_batch_metrics(self):
        """上传batch级别的指标"""
        metrics = self.collector.get_current_metrics()
        logger.debug(f"📊 Batch {metrics['batch']} 指标上传")
        
        self.uploader.upload(metrics, self.train_id)
    
    def manual_upload(self, description: Optional[str] = None):
        """
        手动上传当前指标
        
        Args:
            description: 上传的描述信息
        """
        metrics = self.collector.get_current_metrics()
        if description:
            metrics['description'] = description
        
        logger.info(f"📤 手动上传指标: {description or 'checkpoint'}")
        self.uploader.upload(metrics, self.train_id)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取当前训练总结"""
        summary = {
            'train_id': self.train_id,
            'current_metrics': self.collector.get_current_metrics(),
            'upload_frequency': self.upload_frequency,
        }
        return summary
    
    def close(self):
        """关闭Agent"""
        logger.info(f"👋 关闭训练Agent (Train ID: {self.train_id})")
        self.uploader.close()
        self.collector.reset()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()
