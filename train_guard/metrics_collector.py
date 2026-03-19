"""
指标收集器 - 负责收集训练过程中的各类指标
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional, List
from collections import defaultdict


class MetricsCollector:
    """收集训练过程中的各类指标"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化指标收集器
        
        Args:
            config: 监控指标配置
        """
        self.config = config
        self.metrics_buffer = defaultdict(list)
        self.epoch_start_time = None
        self.batch_start_time = None
        self.current_epoch = 0
        self.current_batch = 0
        
    def start_epoch(self):
        """标记epoch开始"""
        self.epoch_start_time = time.time()
        self.current_batch = 0
        
    def end_epoch(self):
        """标记epoch结束"""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            if self.config['monitor'].get('epoch_time'):
                self.metrics_buffer['epoch_time'].append(epoch_time)
        self.current_epoch += 1
        
    def start_batch(self):
        """标记batch开始"""
        self.batch_start_time = time.time()
        
    def end_batch(self):
        """标记batch结束"""
        if self.batch_start_time:
            batch_time = time.time() - self.batch_start_time
            if self.config['monitor'].get('batch_time'):
                self.metrics_buffer['batch_time'].append(batch_time)
        self.current_batch += 1
        
    def record_loss(self, loss: float):
        """记录损失值"""
        if self.config['monitor'].get('loss'):
            self.metrics_buffer['loss'].append(float(loss))
            
    def record_accuracy(self, accuracy: float):
        """记录准确率"""
        if self.config['monitor'].get('accuracy'):
            self.metrics_buffer['accuracy'].append(float(accuracy))
            
    def record_learning_rate(self, learning_rate: float):
        """记录学习率"""
        if self.config['monitor'].get('learning_rate'):
            self.metrics_buffer['learning_rate'].append(float(learning_rate))
            
    def record_custom_metric(self, name: str, value: float):
        """记录自定义指标"""
        self.metrics_buffer[name].append(float(value))
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前的聚合指标"""
        metrics = {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'timestamp': time.time(),
        }
        
        # 聚合buffer中的指标
        for key, values in self.metrics_buffer.items():
            if values:
                metrics[key] = {
                    'current': values[-1],
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                }
        
        # 添加系统信息
        if self.config.get('include_system_info'):
            metrics['system_info'] = self._get_system_info()
            
        return metrics
    
    def get_epoch_summary(self) -> Dict[str, Any]:
        """获取epoch总结"""
        summary = {
            'epoch': self.current_epoch,
            'timestamp': time.time(),
        }
        
        # 计算每个指标的平均值
        for key, values in self.metrics_buffer.items():
            if values and key not in ['epoch', 'batch']:
                summary[f'{key}_avg'] = sum(values) / len(values)
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
        
        # 添加系统信息
        if self.config.get('include_system_info'):
            summary['system_info'] = self._get_system_info()
            
        return summary
    
    def clear_buffer(self):
        """清空metrics buffer"""
        self.metrics_buffer.clear()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'timestamp': time.time(),
        }
        
        # CPU和内存信息
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            info['cpu_percent'] = cpu_percent
            info['memory_percent'] = memory.percent
            info['memory_available_mb'] = memory.available / (1024 * 1024)
        except Exception as e:
            info['system_info_error'] = str(e)
        
        # GPU信息
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                info['gpu_allocated_mb'] = gpu_memory['allocated_bytes.current'] / (1024 * 1024)
                info['gpu_reserved_mb'] = gpu_memory['reserved_bytes.current'] / (1024 * 1024)
                info['gpu_count'] = torch.cuda.device_count()
            except Exception as e:
                info['gpu_info_error'] = str(e)
        
        return info
    
    def reset(self):
        """重置收集器"""
        self.metrics_buffer.clear()
        self.current_epoch = 0
        self.current_batch = 0
        self.epoch_start_time = None
        self.batch_start_time = None
