"""Persistent extrinsic-calibration store.

Why a separate JSON file (not back into ``app_config.yaml``):
  * YAML is the source of truth for code-defined defaults — checked in,
    code-reviewed, opinionated. Calibration is per-deployment, per-mount,
    changes every time you bolt the rig to a new tripod, and should NEVER
    overwrite the engineer-curated defaults file.
  * JSON is dirt-cheap to load/save, no third-party deps, and the file
    can be wiped to restore "no bias" (the YAML defaults take over).
  * The .gitignore for this project excludes ``calibration.json`` so
    user mounts don't pollute commits.

File layout (all keys optional — anything missing → falls back to YAML
default → falls back to 0.0):

    {
      "radar":   {"az_bias_deg": 0.0, "el_bias_deg": 0.0},
      "thermal": {"az_bias_deg": 0.0, "el_bias_deg": 0.0},
      "_meta":   {"saved_utc": "2026-04-24T10:33:01Z"}
    }

The ``_meta`` block is informational only — the GUI shows it next to
the SAVE button so the user knows how stale the file is.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Module-level so callers (main.py at startup, app.py at SAVE click)
# resolve the SAME file. We anchor at the project root via the package
# location — works under PyInstaller too because common/ is bundled.
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "calibration.json"


def path() -> Path:
    """Return the canonical calibration.json path. Public so tests can monkey-patch."""
    return _DEFAULT_PATH


def load() -> dict:
    """Read calibration.json. Returns an empty dict if missing or unreadable.

    Never raises — startup must continue even if the file is corrupt.
    A warning is logged so the issue is visible in seeker_bench.log.
    """
    p = path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning("calibration.json was not a dict — ignoring")
            return {}
        return data
    except Exception as e:  # corrupt JSON, IO error, encoding mismatch…
        log.warning("Failed to read %s: %s — ignoring", p, e)
        return {}


def save(*,
         radar_az: Optional[float] = None,
         radar_el: Optional[float] = None,
         thermal_az: Optional[float] = None,
         thermal_el: Optional[float] = None) -> Path:
    """Write the four extrinsic biases to calibration.json atomically.

    Atomic = write temp file in same dir + os.replace. Avoids the
    "save-button click during a power glitch leaves a 0-byte JSON"
    failure mode that would silently zero the user's hard-won
    calibration on next boot.

    Any None argument leaves that channel's previous saved value
    untouched — caller can save a partial update if it wants. In
    practice the GUI always sends all four, but this keeps the API
    flexible for future "save thermal only" UI.

    Returns the resolved file path so the caller can show it to the user.
    """
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Merge over whatever's already on disk, so partial saves don't drop
    # the other sensor's bias.
    existing = load()
    radar = dict(existing.get("radar") or {})
    thermal = dict(existing.get("thermal") or {})

    if radar_az is not None:
        radar["az_bias_deg"] = float(radar_az)
    if radar_el is not None:
        radar["el_bias_deg"] = float(radar_el)
    if thermal_az is not None:
        thermal["az_bias_deg"] = float(thermal_az)
    if thermal_el is not None:
        thermal["el_bias_deg"] = float(thermal_el)

    payload = {
        "radar": radar,
        "thermal": thermal,
        "_meta": {
            "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    # Atomic write: temp file in the same directory (rename across
    # filesystems isn't atomic on Windows), then os.replace.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".calibration.", suffix=".json.tmp", dir=str(p.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, p)
    except Exception:
        # Best-effort cleanup; never let cleanup raise over the original error.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    log.info("Saved extrinsic calibration → %s", p)
    return p


def apply_to_managers(radar_manager=None, fusion_manager=None) -> dict:
    """Load calibration.json and push values into the live managers.

    Call this AFTER the managers exist but BEFORE start(), or any time
    you want to re-sync from disk. Returns the dict that was actually
    applied (so main.py can log it).

    Missing keys are skipped — they keep the YAML default that was
    already loaded by the manager constructor. The user can wipe
    calibration.json to revert to YAML defaults without code changes.
    """
    data = load()
    radar = (data.get("radar") or {})
    thermal = (data.get("thermal") or {})

    applied: dict = {}
    if radar_manager is not None:
        kw = {}
        if "az_bias_deg" in radar:
            kw["az_bias_deg"] = float(radar["az_bias_deg"])
        if "el_bias_deg" in radar:
            kw["el_bias_deg"] = float(radar["el_bias_deg"])
        if kw:
            try:
                radar_manager.set_extrinsic(**kw)
                applied["radar"] = kw
            except Exception as e:
                log.warning("Failed to apply radar calibration: %s", e)

    if fusion_manager is not None:
        kw = {}
        if "az_bias_deg" in thermal:
            kw["thermal_az_bias_deg"] = float(thermal["az_bias_deg"])
        if "el_bias_deg" in thermal:
            kw["thermal_el_bias_deg"] = float(thermal["el_bias_deg"])
        if kw:
            try:
                fusion_manager.set_extrinsic(**kw)
                applied["thermal"] = kw
            except Exception as e:
                log.warning("Failed to apply thermal calibration: %s", e)

    return applied
