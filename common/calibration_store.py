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

Two file layouts are supported. v1 (legacy, biases only) is read-only-
compatible — the runtime falls back to FOV-only projection if no v2
fields are present. v2 adds full intrinsics + 6-DoF stereo extrinsics
for EO and thermal:

    v1:
    {
      "radar":   {"az_bias_deg": 0.0, "el_bias_deg": 0.0},
      "thermal": {"az_bias_deg": 0.0, "el_bias_deg": 0.0},
      "_meta":   {"saved_utc": "..."}
    }

    v2:
    {
      "version": 2,
      "eo":      {"K": [[...]], "dist": [...]},
      "thermal": {"K": [[...]], "dist": [...],
                  "R_to_eo": [[...]], "t_to_eo": [tx, ty, tz],
                  "az_bias_deg": 0.0, "el_bias_deg": 0.0},
      "radar":   {"az_bias_deg": 0.0, "el_bias_deg": 0.0},
      "_meta":   {"saved_utc": "...", "calib_session_id": "..."}
    }

Targeted save APIs (``save_intrinsic``, ``save_extrinsic_6dof``,
``save_bias``) preserve the partial-merge semantics: each one reads the
current file, modifies only its own slice, and atomically rewrites.
That lets the EO-intrinsic capture pipeline and the thermal/stereo
solver write to the same file without clobbering each other (as long
as they don't write the same slice at the same instant — which the
GUI flow and the capture scripts don't).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

log = logging.getLogger(__name__)

# Module-level so callers (main.py at startup, app.py at SAVE click)
# resolve the SAME file. We anchor at the project root via the package
# location — works under PyInstaller too because common/ is bundled.
_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "calibration.json"

_CURRENT_VERSION = 2


def path() -> Path:
    """Return the canonical calibration.json path. Public so tests can monkey-patch."""
    return _DEFAULT_PATH


def load() -> dict:
    """Read calibration.json. Returns an empty dict if missing or unreadable.

    Never raises — startup must continue even if the file is corrupt.
    A warning is logged so the issue is visible in seeker_bench.log.

    The returned dict is the raw on-disk form (v1 or v2). Consumers
    decide how to handle missing fields. ``apply_to_managers`` does
    the v1-vs-v2 dispatch.
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


def _atomic_write(payload: dict) -> Path:
    """Atomic write to calibration.json. Used by every save_* API.

    Atomic = write temp file in same dir + os.replace. Avoids the
    "save during a power glitch leaves a 0-byte JSON" failure mode
    that would silently zero the user's hard-won calibration on next
    boot.
    """
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)

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

    return p


def _stamp_meta(payload: dict) -> dict:
    """Update _meta.saved_utc in-place and return the same payload."""
    meta = dict(payload.get("_meta") or {})
    meta["saved_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["_meta"] = meta
    return payload


def save(*,
         radar_az: Optional[float] = None,
         radar_el: Optional[float] = None,
         thermal_az: Optional[float] = None,
         thermal_el: Optional[float] = None) -> Path:
    """Legacy bias-only save. Kept for back-compat with the GUI's existing
    ``extrinsic_save`` flow.

    Any None argument leaves that channel's previous saved value
    untouched — caller can save a partial update if it wants. In
    practice the GUI always sends all four, but this keeps the API
    flexible for future "save thermal only" UI.

    New code should prefer ``save_bias(sensor, az, el)`` which is
    per-sensor and composes cleanly with intrinsic/extrinsic saves.

    Returns the resolved file path so the caller can show it to the user.
    """
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

    # Preserve any v2 fields (K, dist, R_to_eo, t_to_eo, eo block)
    # that may have been written by the calibration scripts.
    payload = dict(existing)
    payload["radar"] = radar
    payload["thermal"] = thermal
    _stamp_meta(payload)

    p = _atomic_write(payload)
    log.info("Saved extrinsic calibration → %s", p)
    return p


def save_bias(sensor: str, *,
              az_bias_deg: Optional[float] = None,
              el_bias_deg: Optional[float] = None) -> Path:
    """Per-sensor bias save. v2-friendly alternative to ``save()``.

    Only the named sensor's bias fields are touched; everything else
    in the file (intrinsics, extrinsics, other sensors) is preserved.
    """
    if sensor not in ("radar", "thermal", "eo"):
        raise ValueError(f"unknown sensor: {sensor!r}")
    existing = load()
    block = dict(existing.get(sensor) or {})
    if az_bias_deg is not None:
        block["az_bias_deg"] = float(az_bias_deg)
    if el_bias_deg is not None:
        block["el_bias_deg"] = float(el_bias_deg)
    payload = dict(existing)
    payload[sensor] = block
    _stamp_meta(payload)
    return _atomic_write(payload)


def save_intrinsic(sensor: str, *,
                   K: Sequence[Sequence[float]],
                   dist: Sequence[float],
                   reproj_rms_px: Optional[float] = None) -> Path:
    """Write camera intrinsics for ``sensor`` ('eo' or 'thermal').

    K is the 3x3 camera matrix; dist is the OpenCV distortion vector
    (5 params for standard model, 8 for rational model). Both are
    serialized as nested lists for human-readable JSON.

    ``reproj_rms_px`` is stashed in ``_meta`` as a calibration quality
    indicator; the runtime ignores it but it shows up in logs.

    Marks the file as version 2.
    """
    if sensor not in ("eo", "thermal"):
        raise ValueError(f"intrinsic save unsupported for sensor: {sensor!r}")
    existing = load()
    block = dict(existing.get(sensor) or {})
    block["K"] = [[float(v) for v in row] for row in K]
    block["dist"] = [float(v) for v in dist]
    payload = dict(existing)
    payload["version"] = _CURRENT_VERSION
    payload[sensor] = block
    if reproj_rms_px is not None:
        meta = dict(payload.get("_meta") or {})
        meta[f"{sensor}_intrinsic_rms_px"] = float(reproj_rms_px)
        payload["_meta"] = meta
    _stamp_meta(payload)
    p = _atomic_write(payload)
    log.info("Saved %s intrinsics → %s (rms=%s)", sensor, p, reproj_rms_px)
    return p


def save_extrinsic_6dof(sensor: str, *,
                        R_to_eo: Sequence[Sequence[float]],
                        t_to_eo: Sequence[float],
                        stereo_rms_px: Optional[float] = None) -> Path:
    """Write 6-DoF extrinsics from ``sensor`` to EO frame.

    R_to_eo is a 3x3 rotation; t_to_eo is a length-3 translation in
    metres. Convention: a point P in the sensor's own frame maps to
    EO via ``P_eo = R_to_eo @ P_sensor + t_to_eo``.

    Currently only valid for ``sensor='thermal'`` (radar 6-DoF is out
    of scope per the calibration plan; radar uses bias only).
    """
    if sensor != "thermal":
        raise ValueError(
            f"6-DoF extrinsic save unsupported for sensor: {sensor!r} "
            "(only 'thermal' has stereo extrinsics in the current scope)"
        )
    existing = load()
    block = dict(existing.get(sensor) or {})
    block["R_to_eo"] = [[float(v) for v in row] for row in R_to_eo]
    block["t_to_eo"] = [float(v) for v in t_to_eo]
    payload = dict(existing)
    payload["version"] = _CURRENT_VERSION
    payload[sensor] = block
    if stereo_rms_px is not None:
        meta = dict(payload.get("_meta") or {})
        meta[f"{sensor}_stereo_rms_px"] = float(stereo_rms_px)
        payload["_meta"] = meta
    _stamp_meta(payload)
    p = _atomic_write(payload)
    log.info("Saved %s 6-DoF extrinsics → %s (rms=%s)", sensor, p, stereo_rms_px)
    return p


def apply_to_managers(radar_manager=None, fusion_manager=None) -> dict:
    """Load calibration.json and push values into the live managers.

    Call this AFTER the managers exist but BEFORE start(), or any time
    you want to re-sync from disk. Returns the dict that was actually
    applied (so main.py can log it).

    v1 path: pushes az/el biases into RadarManager + FusionManager via
    ``set_extrinsic``. Unchanged from the legacy behavior.

    v2 path: additionally constructs ``Camera`` objects from the EO
    and thermal K/dist/R/t blocks and pushes them via
    ``set_calibrated_cameras`` (if the manager exposes it). v2 loading
    is best-effort — if the manager doesn't expose the new API, the
    v1 bias path still applies and the runtime stays on the FOV-only
    projection. This keeps the legacy code path byte-identical for
    the replay regression guard described in the calibration plan.
    """
    data = load()
    radar = (data.get("radar") or {})
    thermal = (data.get("thermal") or {})
    eo = (data.get("eo") or {})

    applied: dict = {}

    # ---- v1 bias path (unchanged) ----
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

    # ---- v2 calibrated-camera path (additive) ----
    # Only fires if the file actually has v2 content AND the manager
    # exposes set_calibrated_cameras. Either condition missing → silent
    # fallback to v1 path above.
    if fusion_manager is not None and hasattr(fusion_manager, "set_calibrated_cameras"):
        eo_K = eo.get("K")
        eo_dist = eo.get("dist")
        thr_K = thermal.get("K")
        thr_dist = thermal.get("dist")
        thr_R = thermal.get("R_to_eo")
        thr_t = thermal.get("t_to_eo")
        # All four (eo K/dist + thermal K/dist + thermal R/t) must be
        # present to enable v2 — partial config falls back to v1.
        if all(v is not None for v in (eo_K, eo_dist, thr_K, thr_dist, thr_R, thr_t)):
            try:
                # Local import to avoid pulling numpy into modules that
                # only need calibration_store at import time.
                import numpy as np
                from fusion.projection import Camera

                eo_cam = Camera(
                    K=np.asarray(eo_K, dtype=np.float64),
                    dist=np.asarray(eo_dist, dtype=np.float64),
                    R_to_eo=np.eye(3, dtype=np.float64),
                    t_to_eo=np.zeros(3, dtype=np.float64),
                )
                thr_cam = Camera(
                    K=np.asarray(thr_K, dtype=np.float64),
                    dist=np.asarray(thr_dist, dtype=np.float64),
                    R_to_eo=np.asarray(thr_R, dtype=np.float64),
                    t_to_eo=np.asarray(thr_t, dtype=np.float64),
                )
                fusion_manager.set_calibrated_cameras(eo=eo_cam, thermal=thr_cam)
                applied["v2_cameras"] = {"eo": True, "thermal": True}
            except Exception as e:
                log.warning("Failed to apply v2 calibrated cameras: %s", e)

    return applied
