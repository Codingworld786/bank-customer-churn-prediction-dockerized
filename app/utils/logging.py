"""Logging configuration."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB
    backup_count: int = 5,
) -> None:
    """Configure application logging with console + rotating file handler."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger("app")
    root.setLevel(level)
    root.handlers.clear()

    # Console handler for realtime debugging
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(handler)

    # Rotating file handler for persisted logs under logs/
    if log_dir and log_file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(f"app.{name}")
