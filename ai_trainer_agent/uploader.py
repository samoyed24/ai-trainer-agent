"""
上传器 - 将收集的指标上传到目标服务器
"""

import requests
import json
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import time


logger = logging.getLogger(__name__)


class MetricsUploader:
    """负责将指标上传到目标服务器"""
    
    def __init__(self, server_config: Dict[str, Any], agent_config: Dict[str, Any]):
        """
        初始化上传器
        
        Args:
            server_config: 服务器配置（URL、超时等）
            agent_config: Agent配置（异步上传等）
        """
        self.server_url = server_config['url']
        self.timeout = server_config.get('timeout', 10)
        self.retry_count = server_config.get('retry_count', 3)
        self.enable_async = agent_config.get('enable_async', True)
        self.session = requests.Session()
        
    def upload(self, metrics: Dict[str, Any], train_id: Optional[str] = None) -> bool:
        """
        上传指标
        
        Args:
            metrics: 要上传的指标字典
            train_id: 训练ID（用于识别不同的训练任务）
            
        Returns:
            是否上传成功
        """
        payload = {
            'timestamp': datetime.now().isoformat(),
            'train_id': train_id or 'default',
            'metrics': metrics,
        }
        
        if self.enable_async:
            # 异步上传
            thread = threading.Thread(
                target=self._upload_with_retry,
                args=(payload,),
                daemon=True
            )
            thread.start()
            return True
        else:
            # 同步上传
            return self._upload_with_retry(payload)
    
    def _upload_with_retry(self, payload: Dict[str, Any]) -> bool:
        """
        带重试机制的上传
        
        Args:
            payload: 要上传的数据
            
        Returns:
            是否上传成功
        """
        for attempt in range(self.retry_count):
            try:
                response = self.session.post(
                    self.server_url,
                    json=payload,
                    timeout=self.timeout,
                )
                
                if response.status_code in [200, 201, 202]:
                    # 兼容两种格式：{"metrics": {...}} 或直接指标字典
                    epoch = None
                    metrics = payload.get('metrics')
                    if isinstance(metrics, dict):
                        epoch = metrics.get('epoch')
                    if epoch is None:
                        epoch = payload.get('epoch')

                    logger.info(
                        f"✓ 指标上传成功 (epoch={epoch}, "
                        f"status={response.status_code})"
                    )
                    return True
                else:
                    logger.warning(
                        f"⚠ 上传失败 (status={response.status_code}): {response.text[:100]}"
                    )
                    
            except requests.Timeout:
                logger.warning(f"⚠ 上传超时 (尝试 {attempt + 1}/{self.retry_count})")
                
            except requests.ConnectionError as e:
                logger.warning(f"⚠ 连接失败 (尝试 {attempt + 1}/{self.retry_count}): {str(e)[:50]}")
                
            except Exception as e:
                logger.error(f"✗ 上传出错: {str(e)[:100]}")
                return False
            
            # 重试前等待
            if attempt < self.retry_count - 1:
                wait_time = (2 ** attempt)  # 指数退避
                logger.debug(f"等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
        
        logger.error(f"✗ 上传失败（已重试 {self.retry_count} 次）")
        return False
    
    def upload_batch(self, metrics_list: list, train_id: Optional[str] = None) -> bool:
        """
        批量上传指标
        
        Args:
            metrics_list: 指标列表
            train_id: 训练ID
            
        Returns:
            是否上传成功
        """
        payload = {
            'timestamp': datetime.now().isoformat(),
            'train_id': train_id or 'default',
            'batch': metrics_list,
        }
        
        return self._upload_with_retry(payload)
    
    def close(self):
        """关闭连接"""
        self.session.close()
