"""
Logging Configuration Module.

Configures application logging with both console and file handlers,
ensuring consistent formatting and rotation without leaking sensitive client data.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from app.core.config import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Initialize root and application loggers.

    Args:
        log_level: Optional log level override (e.g. DEBUG, INFO).
        log_dir: Optional log directory override.

    Returns:
        The configured root logger instance.
    """
    settings = get_settings()
    level_name = (log_level or settings.LOG_LEVEL).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    target_log_dir = log_dir or settings.logs_dir
    target_log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = target_log_dir / "app.log"

    # Formatter for structured readable logs
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Rotating File Handler (max 10 MB per file, up to 5 backups)
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Root Logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicates during re-initialization
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger("call_audit")
    logger.info("Logging initialized at level %s. Log file: %s", level_name, log_file_path)
    return logger
