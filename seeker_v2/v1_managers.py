"""seeker_v2 ↔ v1 manager bridge.

V2 keeps its multi-process EO/thermal/inference architecture but reuses
v1's RadarManager + GimbalManager + JSONLRecorder in-process. These
classes are large, well-tested, and re-implementing them in V2 would
take days. They publish to v1's FrameBus singleton (which lives in V2
main's process space when this bridge is loaded).

V2's WS handler then reads from v1's BUS (radar / gimbal) and from V2's
own queues (eo / thermal / inference / fusion) and merges into a single
v1-compatible WS envelope (see seeker_v2/wire_v1.py).

Construction is a near-copy of the relevant slice of v1's main.py:
  - RadarManager(cli_port=..., data_port=..., cfg_path=..., ...)
    publishes radar.RadarFrame on Topic.RADAR
  - GimbalManager() publishes gimbal.GimbalState on Topic.GIMBAL_STATE
  - JSONLRecorder subscribes to topics and writes JSONL on demand
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger("seeker_v2.v1_managers")


def _load_yaml_section(yaml_path: str, section: str) -> dict:
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get(section, {}) or {})
    except Exception as e:
        log.warning("yaml load %s.%s failed: %r", yaml_path, section, e)
        return {}


# ── Radar ─────────────────────────────────────────────────────────────

def start_radar(yaml_path: Optional[str] = None) -> Any:
    """Start v1's RadarManager. Returns the manager (or None on failure).

    Reads the same config keys v1's main.py reads from
    config/app_config.yaml (radar.*). Falls back to sensible defaults
    if YAML missing.
    """
    try:
        from radar.radar_manager import RadarManager
        from radar.clustering import ClusterParams
    except Exception as e:
        log.error("radar import failed: %r", e); return None

    radar_cfg = _load_yaml_section(yaml_path, "radar") if yaml_path else {}
    trk = (radar_cfg.get("tracker") or {})
    ext = (radar_cfg.get("extrinsic") or {})

    cluster_defaults = ClusterParams()
    cluster_params = ClusterParams(
        eps_pos_m=float((radar_cfg.get("clustering") or {}).get("eps_pos_m", cluster_defaults.eps_pos_m)),
        eps_dop_mps=float((radar_cfg.get("clustering") or {}).get("eps_dop_mps", cluster_defaults.eps_dop_mps)),
        min_samples=int((radar_cfg.get("clustering") or {}).get("min_samples", cluster_defaults.min_samples)),
    )

    try:
        rm = RadarManager(
            cli_port=radar_cfg.get("cli_port", "/dev/seeker_radar_cli"),
            data_port=radar_cfg.get("data_port", "/dev/seeker_radar_data"),
            cfg_path=radar_cfg.get("cfg_path", "radar/cfg/awr2944P_unified.cfg"),
            cli_baud=int(radar_cfg.get("cli_baud", 115200)),
            data_baud=int(radar_cfg.get("data_baud", 921600)),
            snr_min_db=float(radar_cfg.get("snr_min_db", 12.0)),
            max_range_m=float(radar_cfg.get("max_range_m", 250.0)),
            az_half_deg=float(radar_cfg.get("az_half_deg", 60.0)),
            speed_min_mps=float(radar_cfg.get("speed_min_mps", 0.0)),
            range_min_m=float(radar_cfg.get("range_min_m", 0.0)),
            profile_name=str(radar_cfg.get("profile_name", "awr2944p_ddm")),
            stream_timeout_s=float(radar_cfg.get("stream_timeout_s", 3.0)),
            cluster_params=cluster_params,
            az_bias_deg=float(ext.get("az_bias_deg", 0.0)),
            el_bias_deg=float(ext.get("el_bias_deg", 0.0)),
        )
        rm.start()
        log.info("RadarManager started (cli=%s, data=%s)",
                 rm.cli_port, rm.data_port)
        return rm
    except Exception:
        log.exception("RadarManager construction failed")
        return None


# ── Gimbal ────────────────────────────────────────────────────────────

def start_gimbal(yaml_path: Optional[str] = None) -> Any:
    """Start v1's GimbalManager. Returns the manager (or None)."""
    try:
        from gimbal.gimbal_manager import GimbalManager
    except Exception as e:
        log.error("gimbal import failed: %r", e); return None
    try:
        gm = GimbalManager()
        gm.start()
        log.info("GimbalManager started")
        return gm
    except Exception:
        log.exception("GimbalManager construction failed")
        return None


# ── Recorder ──────────────────────────────────────────────────────────

def start_recorder(yaml_path: Optional[str] = None) -> Any:
    """Start v1's JSONLRecorder. Returns the recorder."""
    try:
        from common.frame_bus import BUS
        from recording.jsonl_recorder import JSONLRecorder
    except Exception as e:
        log.error("recorder import failed: %r", e); return None
    try:
        rec_cfg = _load_yaml_section(yaml_path, "recording") if yaml_path else {}
        out_dir = rec_cfg.get("out_dir", "/var/log/seeker")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rec = JSONLRecorder(BUS, out_dir=out_dir)
        log.info("JSONLRecorder ready (out_dir=%s)", out_dir)
        return rec
    except Exception:
        log.exception("JSONLRecorder construction failed")
        return None


# ── BUS topic getters (read-only, used by WS sender) ──────────────────

def bus_get_radar() -> Any:
    """Latest RadarFrame from v1's BUS, or None."""
    try:
        from common.frame_bus import BUS
        from common.frames import Topic
        return BUS.get_latest(Topic.RADAR)
    except Exception:
        return None


def bus_get_gimbal_state() -> Any:
    """Latest GimbalState from v1's BUS, or None."""
    try:
        from common.frame_bus import BUS
        from common.frames import Topic
        return BUS.get_latest(Topic.GIMBAL_STATE)
    except Exception:
        return None


def bus_get_radar_aa() -> Any:
    """Latest RadarFrame from raw-ADC pipeline (Phase 3), or None."""
    try:
        from common.frame_bus import BUS
        from common.frames import Topic
        return BUS.get_latest(Topic.RADAR_AA)
    except Exception:
        return None
