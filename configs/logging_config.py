"""Application-wide logging configuration.

This module provides a single reusable entry point, :func:`get_logger`,
for obtaining a configured logger instance. Other modules should never
configure logging themselves (e.g., by calling
``logging.basicConfig``); they should simply call ``get_logger(__name__)``
at the top of the module.
"""

import logging
import sys
from pathlib import Path

from configs.config import settings
from configs.paths import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FILE_NAME = "project.log"


def _build_console_handler() -> logging.Handler:
    """Create a stream handler that writes readable logs to the console.

    Returns:
        A configured :class:`logging.StreamHandler` instance.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _build_file_handler(log_file_path: Path) -> logging.Handler:
    """Create a file handler that persists logs to disk.

    Args:
        log_file_path: Path to the log file to write to.

    Returns:
        A configured :class:`logging.FileHandler` instance.
    """
    handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def get_logger(name: str, *, enable_file_logging: bool = True) -> logging.Logger:
    """Return a configured logger for the given module name.

    This is the single entry point for logging configuration across the
    project. Calling it multiple times with the same ``name`` is safe
    and will not duplicate log handlers.

    Args:
        name: The name of the logger, typically ``__name__`` of the
            calling module.
        enable_file_logging: Whether logs should also be written to a
            shared log file under the project's ``logs`` directory.
            Defaults to True.

    Returns:
        A configured :class:`logging.Logger` instance ready for use.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured (e.g., re-imported module); avoid
        # attaching duplicate handlers which would cause repeated logs.
        return logger

    logger.setLevel(settings.logging_level)
    logger.propagate = False

    logger.addHandler(_build_console_handler())

    if enable_file_logging:
        log_file_path = LOGS_DIR / _LOG_FILE_NAME
        logger.addHandler(_build_file_handler(log_file_path))

    return logger
