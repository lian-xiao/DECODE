# filepath: e:\BaiduSyncdisk\Code\pythonProject\Mol_Image_omics\utils\memory_monitor.py
"""
内存监控工具
"""

import psutil
import torch
import gc
import logging

logger = logging.getLogger(__name__)

def get_memory_usage():
    """获取当前内存使用情况"""
    # CPU内存
    process = psutil.Process()
    cpu_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # GPU内存
    gpu_memory_mb = 0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    
    return cpu_memory_mb, gpu_memory_mb

def log_memory_usage(prefix=""):
    """记录内存使用情况"""
    cpu_mem, gpu_mem = get_memory_usage()
    
    if torch.cuda.is_available():
        logger.info(f"{prefix}Memory usage - CPU: {cpu_mem:.1f} MB, GPU: {gpu_mem:.1f} MB")
    else:
        logger.info(f"{prefix}Memory usage - CPU: {cpu_mem:.1f} MB")

def cleanup_memory():
    """清理内存"""
    # 强制垃圾回收
    gc.collect()
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.debug("Memory cleanup completed")

def check_memory_threshold(threshold_mb=8000):
    """检查是否超过内存阈值"""
    cpu_mem, gpu_mem = get_memory_usage()
    total_mem = cpu_mem + gpu_mem
    
    if total_mem > threshold_mb:
        logger.warning(f"Memory usage ({total_mem:.1f} MB) exceeds threshold ({threshold_mb} MB)")
        return True
    return False