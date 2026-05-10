"""
Centralized logging setup for the Water Potability Prediction System.

Provides a factory function that creates loggers with both console and file handlers.
Use training.log for pipeline logs and api.log for API request/response logs.
"""

import logging
import os
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Create and configure a logger with StreamHandler and optional FileHandler.

    Args:
        name:     Logger name (usually __name__ of the calling module).
        log_file: Filename to write logs to inside the logs/ directory.
                  If None, only console logging is configured.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler — DEBUG and above
    if log_file:
        try:
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler(os.path.join("logs", log_file), encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError as exc:
            logger.warning(
                "File logging disabled because the log file could not be opened: %s",
                exc,
            )

    return logger
