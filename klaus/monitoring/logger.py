"""Loguru-based logging setup for Klaus."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from klaus.config.settings import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "data" / "logs"


def setup_logger(level: str = "INFO") -> None:
    """Configure loguru with console + file sinks."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console: coloured, concise
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File: full detail, rotating daily
    logger.add(
        LOG_DIR / "klaus_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
    )

    # Separate error log
    logger.add(
        LOG_DIR / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="60 days",
    )

    logger.info("Klaus logger initialised")
