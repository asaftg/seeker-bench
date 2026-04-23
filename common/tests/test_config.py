"""Tests for common/config.load_config.

The loader is tiny but load-bearing — every module calls it. Cover:
  - cached return on second call (no re-read)
  - reload=True forces re-read
  - malformed YAML raises cleanly rather than returning junk
  - real project config loads and has the expected top-level keys
"""
from __future__ import annotations

import pytest
import yaml

from common import config as config_mod


def _reset_cache():
    config_mod._CACHE = None


def test_load_config_returns_dict_with_expected_sections():
    _reset_cache()
    try:
        cfg = config_mod.load_config()
    finally:
        _reset_cache()
    assert isinstance(cfg, dict)
    # These sections are referenced all over the codebase; make sure
    # the real app_config.yaml at least parses into a dict with them.
    for key in ("thermal", "gui"):
        assert key in cfg, f"missing top-level config section {key!r}"


def test_load_config_caches(monkeypatch, tmp_path):
    _reset_cache()
    # Point loader at a temp file
    tmp_cfg = tmp_path / "config" / "app_config.yaml"
    tmp_cfg.parent.mkdir()
    tmp_cfg.write_text("a: 1\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "_project_root", lambda: tmp_path)

    first = config_mod.load_config()
    assert first == {"a": 1}

    # Mutate the file on disk; cache should still return the old value.
    tmp_cfg.write_text("a: 2\n", encoding="utf-8")
    second = config_mod.load_config()
    assert second == {"a": 1}

    # reload=True picks up the new value.
    third = config_mod.load_config(reload=True)
    assert third == {"a": 2}

    _reset_cache()


def test_load_config_empty_file_returns_empty_dict(monkeypatch, tmp_path):
    _reset_cache()
    tmp_cfg = tmp_path / "config" / "app_config.yaml"
    tmp_cfg.parent.mkdir()
    tmp_cfg.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_mod, "_project_root", lambda: tmp_path)
    try:
        cfg = config_mod.load_config(reload=True)
        assert cfg == {}
    finally:
        _reset_cache()


def test_load_config_malformed_yaml_raises(monkeypatch, tmp_path):
    _reset_cache()
    tmp_cfg = tmp_path / "config" / "app_config.yaml"
    tmp_cfg.parent.mkdir()
    # Intentionally busted YAML
    tmp_cfg.write_text("a: [unterminated\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "_project_root", lambda: tmp_path)
    try:
        with pytest.raises(yaml.YAMLError):
            config_mod.load_config(reload=True)
    finally:
        _reset_cache()
