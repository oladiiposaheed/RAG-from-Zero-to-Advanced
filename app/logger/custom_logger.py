"""
Custom logging configuration.

Provides a setup_logging function to configure the root logger,
and get_logger to retrieve module-specific loggers.
"""

import logging
from pathlib import Path
from app.config.config import config


def setup_logging(log_file: str | Path | None = None, level: str | None = None):
    """Configure the root logger with console and optional file handler."""
    # Determine logging level
    level = level or config.log_level

    # Adjust level based on environment
    if config.environment == 'development':
        if level.upper() == 'INFO':
            level = 'DEBUG'
    elif config.environment == 'production':
        if level.upper() in ('INFO', 'DEBUG'):
            level = 'WARNING'

    # Get root logger and set level (NOW outside the if/elif)
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given name."""
    return logging.getLogger(name)