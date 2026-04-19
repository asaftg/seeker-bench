"""
Central logging configuration.

Every module calls `get_logger(__name__)` instead of using the
root logger directly. Logs go to stdout (for dev) AND to
`logs/seeker.log` with rotation (for post-mortem debugging
after a field test).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def _project_root() -> Path:
    """Return the project root directory.

    Works both in dev (running from source) and inside a
    PyInstaller folder dist (sys._MEIPASS-relative layout).
    """
    if getattr(sys, "frozen", False):
        # PyInstaller: logs go next to the exe, not into _MEIPASS (read-only)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def configure(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> None:
    """Configure root logging. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(name)-28s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file
    log_dir_path = Path(log_dir) if log_dir else _project_root() / "logs"
    log_dir_path.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_dir_path / "seeker.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Quiet down chatty third-party libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger. Auto-configures on first call."""
    if not _CONFIGURED:
        configure()
    return logging.getLogger(name)
