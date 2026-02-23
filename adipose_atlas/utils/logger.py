"""Prepare logging."""

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configures logging settings. Sets up the logger to write to STDOUT with a
    standardized format including time, level, module, line number, and the
    message.

    Args:
        level: The logging level to use ("DEBUG", "INFO", "ERROR"). Defaults to
            "INFO".
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "{name}:{line} - <level>{message}</level>",
        diagnose=False,
    )
