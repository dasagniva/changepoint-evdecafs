"""Logging configuration for changepoint-evdecafs.

Provides a factory function that creates loggers writing to both the console
(INFO level) and a timestamped file in the project logs/ directory (DEBUG level).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Create and return a named logger with console + file handlers.

    Parameters
    ----------
    name:
        Logger name, typically the calling module's ``__name__``.
    log_dir:
        Directory where log files are written.  Created if it does not exist.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = setup_logger(__name__, log_dir="logs")
    >>> logger.info("Pipeline started")
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{name.replace('.', '_')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if the logger already exists
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # File handler — DEBUG and above
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
