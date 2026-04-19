"""
Config loader for `config/app_config.yaml`.

Every module that needs a tunable calls `load_config()` and
reads from the returned dict. We intentionally don't use a
dataclass schema here — the config is user-editable YAML and
the surface is still evolving in Phase A.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


_CACHE: Dict[str, Any] | None = None


def load_config(reload: bool = False) -> Dict[str, Any]:
    """Load and cache the YAML config."""
    global _CACHE
    if _CACHE is not None and not reload:
        return _CACHE

    cfg_path = _project_root() / "config" / "app_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        _CACHE = yaml.safe_load(f) or {}
    return _CACHE
