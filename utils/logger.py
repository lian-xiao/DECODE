"""
Logging utilities for MMDP-VAE project.

This module provides centralized logging configuration and utilities
for consistent logging across the entire project.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logging(
    log_level: str = 'INFO',
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    format_style: str = 'detailed'
) -> None:
    """
    Set up logging configuration for the entire project.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_dir: Directory to save log files (default: './logs')
        log_filename: Name of log file (default: auto-generated with timestamp)
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        format_style: Logging format style ('simple', 'detailed', 'json')
    """
    # Create log directory if it doesn't exist
    if log_dir is None:
        log_dir = './logs'
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'mmdp_vae_{timestamp}.log'
    
    log_file_path = log_dir / log_filename
    
    # Define logging formats
    formats = {
        'simple': '%(levelname)s - %(name)s - %(message)s',
        'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        'json': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
    }
    
    # Choose format
    log_format = formats.get(format_style, formats['detailed'])
    
    # Create formatters
    console_formatter = logging.Formatter(
        fmt=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        fmt=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if file_output:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup completed - Level: {log_level}")
    if file_output:
        logger.info(f"Log file: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log message
        """
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """
    Custom filter to add context information to log records.
    """
    
    def __init__(self, context: Dict[str, Any]):
        """
        Initialize with context information.
        
        Args:
            context: Dictionary of context information to add to logs
        """
        super().__init__()
        self.context = context
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add context information to log record.
        
        Args:
            record: Log record to modify
            
        Returns:
            True (always allow the record)
        """
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class ExperimentLogger:
    """
    Specialized logger for experiment tracking and metrics.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = './logs/experiments',
        auto_timestamp: bool = True
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for experiment logs
            auto_timestamp: Whether to add timestamp to experiment name
        """
        self.experiment_name = experiment_name
        if auto_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"{experiment_name}_{timestamp}"
        
        self.log_dir = Path(log_dir) / self.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up experiment-specific logger
        self.logger = logging.getLogger(f"experiment.{self.experiment_name}")
        
        # Create experiment log file
        log_file = self.log_dir / 'experiment.log'
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize metrics file
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        
        self.logger.info(f"Experiment '{self.experiment_name}' started")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {config_file}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = 'train'
    ) -> None:
        """
        Log training/validation metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Current training step
            epoch: Current epoch
            phase: Training phase ('train', 'val', 'test')
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'metrics': metrics
        }
        
        if step is not None:
            log_entry['step'] = step
        if epoch is not None:
            log_entry['epoch'] = epoch
        
        # Append to metrics file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to experiment logger
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f"[{phase.upper()}] {metrics_str}")
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Log model architecture and parameter information.
        
        Args:
            model_info: Dictionary containing model information
        """
        model_file = self.log_dir / 'model_info.json'
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        self.logger.info(f"Model information saved to {model_file}")
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]) -> None:
        """
        Log checkpoint save event.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            metrics: Metrics at checkpoint time
        """
        checkpoint_info = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': checkpoint_path,
            'metrics': metrics
        }
        
        checkpoints_file = self.log_dir / 'checkpoints.jsonl'
        with open(checkpoints_file, 'a') as f:
            f.write(json.dumps(checkpoint_info) + '\n')
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log error with context information.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        errors_file = self.log_dir / 'errors.jsonl'
        with open(errors_file, 'a') as f:
            f.write(json.dumps(error_info) + '\n')
        
        self.logger.error(f"Error occurred: {error}", exc_info=True)
    
    def finalize(self, status: str = 'completed') -> None:
        """
        Finalize experiment logging.
        
        Args:
            status: Final status of experiment ('completed', 'failed', 'interrupted')
        """
        self.logger.info(f"Experiment '{self.experiment_name}' {status}")
        
        # Save final summary
        summary = {
            'experiment_name': self.experiment_name,
            'status': status,
            'start_time': None,  # Could be extracted from first log entry
            'end_time': datetime.now().isoformat(),
            'log_dir': str(self.log_dir)
        }
        
        summary_file = self.log_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


class ProgressLogger:
    """
    Logger for tracking training progress with periodic updates.
    """
    
    def __init__(
        self,
        total_steps: int,
        log_interval: int = 100,
        logger_name: str = 'progress'
    ):
        """
        Initialize progress logger.
        
        Args:
            total_steps: Total number of training steps
            log_interval: Interval for logging progress
            logger_name: Name of the logger
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.logger = logging.getLogger(logger_name)
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
    
    def log_progress(
        self,
        current_step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log training progress.
        
        Args:
            current_step: Current training step
            metrics: Optional metrics to include
        """
        if current_step % self.log_interval == 0 or current_step == self.total_steps:
            current_time = datetime.now()
            elapsed = current_time - self.start_time
            progress_pct = (current_step / self.total_steps) * 100
            
            # Calculate ETA
            if current_step > 0:
                avg_time_per_step = elapsed.total_seconds() / current_step
                remaining_steps = self.total_steps - current_step
                eta_seconds = remaining_steps * avg_time_per_step
                eta = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
            else:
                eta = "unknown"
            
            # Format message
            message = f"Step {current_step}/{self.total_steps} ({progress_pct:.1f}%) - ETA: {eta}"
            
            if metrics:
                metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
                message += f" - {metrics_str}"
            
            self.logger.info(message)
            self.last_log_time = current_time


# Utility functions
def configure_pytorch_lightning_logging(log_level: str = 'INFO') -> None:
    """
    Configure PyTorch Lightning logging to integrate with project logging.
    
    Args:
        log_level: Logging level for PyTorch Lightning
    """
    pl_loggers = [
        'pytorch_lightning',
        'pytorch_lightning.core',
        'pytorch_lightning.trainer',
        'pytorch_lightning.callbacks'
    ]
    
    for logger_name in pl_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))


def suppress_warnings() -> None:
    """Suppress common warnings that may clutter logs."""
    import warnings
    
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress PyTorch Lightning warnings
    import pytorch_lightning as pl
    pl.utilities.warnings.disable_warnings()


def log_system_info() -> None:
    """Log system and environment information."""
    logger = logging.getLogger(__name__)
    
    try:
        import platform
        import psutil
        import torch
        
        # System info
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        # Hardware info
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # GPU info
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            logger.info("CUDA: Not available")
    
    except ImportError as e:
        logger.warning(f"Could not log system info: {e}")


# Initialize module-level logger
_module_logger = logging.getLogger(__name__)