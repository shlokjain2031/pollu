#!/usr/bin/env python3
"""
Logging utility for capturing output to both console and files.

Each data stream (pipeline, landsat8, sentinel2, etc.) gets its own log file
organized by timestamp in logs/{stream_name}/{timestamp}.log at project root.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(stream_name: str, logs_root: Path = Path("logs")) -> logging.Logger:
    """
    Set up logging for a data stream.
    
    Creates logs/{stream_name}/{timestamp}.log and returns configured logger
    that outputs to both console and file.
    
    Args:
        stream_name: Name of the data stream (e.g., 'landsat8', 'sentinel2', 'pipeline')
        logs_root: Root directory for logs (default: logs/ at project root)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory structure
    logs_dir = logs_root / stream_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp-based log filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger(stream_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler (DEBUG level - capture everything)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (INFO level - user-facing)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log that logging started
    logger.info(f"Logging initialized for '{stream_name}' stream")
    logger.debug(f"Log file: {log_file}")
    
    return logger


def print_to_logger(logger: logging.Logger, message: str, level: str = "info"):
    """
    Convenience function to log messages at specified level.
    
    Args:
        logger: Logger instance from setup_logging()
        message: Message to log
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
    """
    level_map = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
        "critical": logger.critical,
    }
    log_func = level_map.get(level.lower(), logger.info)
    log_func(message)
