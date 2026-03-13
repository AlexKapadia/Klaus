"""Loguru-based logging setup for Step Sister."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from stepsister.config.settings import STEPSISTER_ROOT

LOG_DIR = STEPSISTER_ROOT / "data" / "logs"


def setup_fx_logger(level: str = "INFO") -> None:
    """Configure loguru with console + file sinks for the forex platform."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

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
        LOG_DIR / "stepsister_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
    )

    # Separate error log
    logger.add(
        LOG_DIR / "fx_errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="60 days",
    )

    logger.info("Step Sister logger initialised")
