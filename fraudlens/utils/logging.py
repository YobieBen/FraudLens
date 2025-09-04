"""
Logging utilities for FraudLens.

Author: Yobie Benjamin
Date: 2025-08-26 18:34:00 PDT
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def get_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format: Optional[str] = None,
) -> logger:
    """
    Get configured logger instance.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        rotation: Log rotation size/time
        retention: Log retention period
        format: Log format string

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Set format
    if format is None:
        format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    # Bind context if name provided
    if name:
        return logger.bind(name=name)

    return logger
