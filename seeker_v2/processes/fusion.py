"""Fusion process — cross-sensor association + tracking.

Reads:
  - eo_det_q (from Inference)        — EO YOLO detections
  - thermal_det_q (from Inference)   — thermal HV YOLO detections
  - radar_targets_q (from Radar)     — radar tracked targets
  - thermal frame ring (for MOSSE)   — needed to update MOSSE tracker
                                        per thermal frame

Produces:
  - fused_q (to GUI/WS server)       — list of FusedTrack dicts per
                                        WS broadcast tick

Algorithm (same as v1 fusion_manager):
  1. Cross-sensor association (EO ↔ thermal by angular IoU)
  2. Radar attach (separate gate, lower threshold)
  3. Persistent track matcher (world-frame angular distance)
  4. MOSSE tracker bridges classifier-tick gaps for thermal H/V

This process is intentionally NOT GPU-bound — it's pure CPU Python.
Runs at ~25 Hz (the fusion tick rate).
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from seeker_v2.processes.ipc import FrameRing

log = logging.getLogger("seeker_v2.fusion")


@dataclass
class FusionConfig:
    rate_hz: float = 25.0
    xsensor_iou_thresh: float = 0.15
    radar_iou_thresh: float = 0.05
    track_match_gate_deg: float = 1.5
    track_min_hits: int = 2
    track_max_misses: int = 8
    # Sensor angular extents (mirrored from v1 config)
    eo_hfov_deg: float = 11.1
    eo_vfov_deg: float = 9.23
    thermal_hfov_deg: float = 75.0
    thermal_vfov_deg: float = 60.0
    # MOSSE
    mosse_enabled: bool = True
    mosse_psr_lost: float = 5.0
    mosse_lost_frames: int = 6
    # Thermal frame shared-memory (for MOSSE per-frame update)
    thermal_shm_name: str = "seeker_thermal_bgr"
    thermal_n_slots: int = 4
    thermal_width: int = 640
    thermal_height: int = 512


def _drain_queue_to_dict(q, key="frame_id") -> dict:
    """Drain everything currently on q into a dict keyed by frame_id."""
    out: dict = {}
    while True:
        try:
            msg = q.get_nowait()
        except Exception:
            break
        if msg is None:
            continue
        out[msg.get(key, -1)] = msg
    return out


def _angular_iou(a, b) -> float:
    """1D-AABB-style angular IoU between two (az, el, w, h) tuples, deg."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax - aw / 2, bx - bw / 2)
    y1 = max(ay - ah / 2, by - bh / 2)
    x2 = min(ax + aw / 2, bx + bw / 2)
    y2 = min(ay + ah / 2, by + bh / 2)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-9)


def _bbox_to_angular(det: dict, sensor_w_px: int, sensor_h_px: int,
                     hfov: float, vfov: float):
    """Convert pixel bbox to angular (az, el, ang_w, ang_h) in degrees.
    Right-positive azimuth, up-positive elevation."""
    cx = det["x"] + det["w"] / 2.0
    cy = det["y"] + det["h"] / 2.0
    # Normalize to [-0.5, 0.5]
    nx = cx / sensor_w_px - 0.5
    ny = 0.5 - cy / sensor_h_px  # invert y: up positive
    az = nx * hfov
    el = ny * vfov
    ang_w = (det["w"] / sensor_w_px) * hfov
    ang_h = (det["h"] / sensor_h_px) * vfov
    return (az, el, ang_w, ang_h)


def _radar_to_angular(t: dict):
    """Compute radar angular pos from x,y,z (radar frame).

    Convention: radar x = right, y = up, z = forward (NB: in our
    coordinate system, conventional radar puts y=up. Adjust if not.)
    Returns (az_deg, el_deg, ang_w, ang_h) in degrees. Bbox extent is
    a fixed estimate at typical target dim / range.
    """
    x, y, z = t["x"], t["y"], t["z"]
    # Range and angles
    rng = max(0.5, (x * x + y * y + z * z) ** 0.5)
    # If z is forward distance:
    forward = z if z > 0.1 else 1.0
    az = np.degrees(np.arctan2(x, forward))
    el = np.degrees(np.arctan2(y, forward))
    # Assume 1.5m wide x 1.8m tall typical target at range
    ang_w = np.degrees(2 * np.arctan2(0.75, rng))
    ang_h = np.degrees(2 * np.arctan2(0.9, rng))
    return (float(az), float(el), float(ang_w), float(ang_h))


def run(cfg: FusionConfig, ctrl_q, eo_det_q, thermal_det_q,
        radar_targets_q, fused_q, stats_q) -> int:
    """Process entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _stop = {"flag": False}

    def _on_sig(*_):
        _stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    # Try to attach thermal ring for MOSSE
    thermal_ring = None
    try:
        thermal_ring = FrameRing.attach(
            cfg.thermal_shm_name,
            n_slots=cfg.thermal_n_slots,
            frame_bytes=cfg.thermal_width * cfg.thermal_height * 3,
        )
    except Exception as e:
        log.warning("thermal ring attach failed (MOSSE disabled): %r", e)

    # Persistent track state (simplified for Phase 2.1)
    persistent: list = []  # list of dicts
    next_track_id = 1
    fusion_frame_id = 0

    # MOSSE pool — only initialize if vision module exists in v1 tree
    mosse_pool = None
    if cfg.mosse_enabled and thermal_ring is not None:
        try:
            from vision.correlation_tracker_set import (
                CorrelationTrackerSet, CorrelationTrackerSetConfig,
            )
            mosse_pool = CorrelationTrackerSet(
                CorrelationTrackerSetConfig(
                    enabled=True,
                    psr_lost=cfg.mosse_psr_lost,
                    lost_frames=cfg.mosse_lost_frames,
                )
            )
            log.info("MOSSE pool active")
        except Exception as e:
            log.warning("MOSSE init failed (continuing without): %r", e)

    period_s = 1.0 / max(cfg.rate_hz, 1.0)
    last_stats_emit = time.monotonic()
    n_ticks = 0
    n_radar_targets_total = 0
    last_thermal_seq = 0

    try:
        while not _stop["flag"]:
            tick_t0 = time.monotonic()

            try:
                while True:
                    cmd = ctrl_q.get_nowait()
                    if cmd is None or cmd[0] == "shutdown":
                        _stop["flag"] = True
                        break
            except Exception:
                pass
            if _stop["flag"]:
                break

            # ── Drain inputs ─────────────────────────────────────────
            eo_msgs = _drain_queue_to_dict(eo_det_q)
            thermal_msgs = _drain_queue_to_dict(thermal_det_q)
            radar_msgs = _drain_queue_to_dict(radar_targets_q)

            # Take the latest of each
            eo_dets = []
            if eo_msgs:
                latest_fid = max(eo_msgs.keys())
                eo_dets = eo_msgs[latest_fid].get("detections", [])
            thermal_dets = []
            if thermal_msgs:
                latest_fid = max(thermal_msgs.keys())
                thermal_dets = thermal_msgs[latest_fid].get("detections", [])
            radar_targets = []
            if radar_msgs:
                latest_fid = max(radar_msgs.keys())
                radar_targets = radar_msgs[latest_fid].get("targets", [])

            n_radar_targets_total += len(radar_targets)

            # ── MOSSE update (if thermal frame available) ─────────────
            if mosse_pool is not None and thermal_ring is not None:
                tdesc, tseq = thermal_ring.latest()
                if tdesc is not None and tseq != last_thermal_seq:
                    last_thermal_seq = tseq
                    try:
                        view = thermal_ring.reader_view(tdesc.slot_idx)
                        thermal_bgr = np.frombuffer(
                            bytes(view), dtype=np.uint8
                        ).reshape(tdesc.height, tdesc.width, 3)
                        # MOSSE pool update would go here. The v1 API
                        # is on_detector_tick + per-frame update; we
                        # simplify for Phase 2.1.
                    except Exception as e:
                        log.debug("MOSSE update failed: %r", e)

            # ── Convert to angular space + cross-sensor associate ────
            eo_obs = []
            for d in eo_dets:
                az, el, aw, ah = _bbox_to_angular(
                    d, sensor_w_px=2472, sensor_h_px=2064,
                    hfov=cfg.eo_hfov_deg, vfov=cfg.eo_vfov_deg,
                )
                eo_obs.append({
                    **d, "az": az, "el": el, "ang_w": aw, "ang_h": ah,
                    "sensor": "eo",
                })

            thermal_obs = []
            for d in thermal_dets:
                az, el, aw, ah = _bbox_to_angular(
                    d, sensor_w_px=cfg.thermal_width,
                    sensor_h_px=cfg.thermal_height,
                    hfov=cfg.thermal_hfov_deg, vfov=cfg.thermal_vfov_deg,
                )
                thermal_obs.append({
                    **d, "az": az, "el": el, "ang_w": aw, "ang_h": ah,
                    "sensor": "thermal",
                })

            radar_obs = []
            for t in radar_targets:
                az, el, aw, ah = _radar_to_angular(t)
                radar_obs.append({
                    **t, "az": az, "el": el, "ang_w": aw, "ang_h": ah,
                    "sensor": "radar",
                })

            # Cross-sensor association: EO+thermal by IoU, then radar
            candidates: list = []
            used_t = [False] * len(thermal_obs)
            for e in eo_obs:
                best_i, best_iou = -1, 0.0
                for i, t in enumerate(thermal_obs):
                    if used_t[i] or t.get("class") != e.get("class"):
                        continue
                    iou = _angular_iou(
                        (e["az"], e["el"], e["ang_w"], e["ang_h"]),
                        (t["az"], t["el"], t["ang_w"], t["ang_h"]),
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                if best_i >= 0 and best_iou >= cfg.xsensor_iou_thresh:
                    used_t[best_i] = True
                    fused = {
                        "az": e["az"], "el": e["el"],
                        "ang_w": e["ang_w"], "ang_h": e["ang_h"],
                        "class": e["class"],
                        "conf": max(e["conf"], thermal_obs[best_i]["conf"]),
                        "sensors": ["eo", "thermal"],
                        "eo_id": e.get("track_id"),
                        "thermal_id": thermal_obs[best_i].get("track_id"),
                    }
                else:
                    fused = {
                        "az": e["az"], "el": e["el"],
                        "ang_w": e["ang_w"], "ang_h": e["ang_h"],
                        "class": e["class"], "conf": e["conf"],
                        "sensors": ["eo"], "eo_id": e.get("track_id"),
                    }
                candidates.append(fused)
            for i, t in enumerate(thermal_obs):
                if used_t[i]:
                    continue
                candidates.append({
                    "az": t["az"], "el": t["el"],
                    "ang_w": t["ang_w"], "ang_h": t["ang_h"],
                    "class": t["class"], "conf": t["conf"],
                    "sensors": ["thermal"], "thermal_id": t.get("track_id"),
                })

            # Radar attach
            used_c = [False] * len(candidates)
            for r in radar_obs:
                best_i, best_iou = -1, 0.0
                for i, c in enumerate(candidates):
                    if used_c[i]:
                        continue
                    iou = _angular_iou(
                        (r["az"], r["el"], r["ang_w"], r["ang_h"]),
                        (c["az"], c["el"], c["ang_w"], c["ang_h"]),
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                if best_i >= 0 and best_iou >= cfg.radar_iou_thresh:
                    used_c[best_i] = True
                    candidates[best_i]["sensors"].append("radar")
                    candidates[best_i]["radar_tid"] = r.get("tid")
                    candidates[best_i]["range_m"] = float(
                        (r["x"] ** 2 + r["y"] ** 2 + r["z"] ** 2) ** 0.5
                    )
                else:
                    candidates.append({
                        "az": r["az"], "el": r["el"],
                        "ang_w": r["ang_w"], "ang_h": r["ang_h"],
                        "class": -1,  # unknown until fused
                        "conf": 0.0,
                        "sensors": ["radar"],
                        "radar_tid": r.get("tid"),
                        "range_m": float(
                            (r["x"] ** 2 + r["y"] ** 2 + r["z"] ** 2) ** 0.5
                        ),
                    })

            # ── Persistent matcher ───────────────────────────────────
            # Greedy nearest-neighbor in angular space.
            t_used = [False] * len(persistent)
            for c in candidates:
                best_i, best_dist = -1, 999.0
                for i, p in enumerate(persistent):
                    if t_used[i]:
                        continue
                    d = ((c["az"] - p["az"]) ** 2
                         + (c["el"] - p["el"]) ** 2) ** 0.5
                    if d < best_dist and d <= cfg.track_match_gate_deg:
                        best_dist = d
                        best_i = i
                if best_i >= 0:
                    t_used[best_i] = True
                    p = persistent[best_i]
                    p["az"] = c["az"]; p["el"] = c["el"]
                    p["ang_w"] = c["ang_w"]; p["ang_h"] = c["ang_h"]
                    p["class"] = (c["class"] if c["class"] != -1
                                  else p.get("class", -1))
                    p["conf"] = c["conf"]
                    p["sensors_now"] = c["sensors"]
                    p["hits"] = p.get("hits", 0) + 1
                    p["misses"] = 0
                    p["last_seen_ts"] = time.time()
                else:
                    persistent.append({
                        "id": next_track_id,
                        "az": c["az"], "el": c["el"],
                        "ang_w": c["ang_w"], "ang_h": c["ang_h"],
                        "class": c["class"],
                        "conf": c["conf"],
                        "sensors_now": c["sensors"],
                        "hits": 1, "misses": 0,
                        "born_ts": time.time(),
                        "last_seen_ts": time.time(),
                    })
                    next_track_id += 1

            # Mark unmatched as missed
            for i, used in enumerate(t_used):
                if not used:
                    persistent[i]["misses"] = persistent[i].get("misses", 0) + 1
                    persistent[i]["sensors_now"] = []

            # Reap dead tracks
            persistent = [
                p for p in persistent
                if p.get("misses", 0) <= cfg.track_max_misses
            ]

            # Confirmed tracks (visible to GUI)
            confirmed = [
                p for p in persistent
                if p.get("hits", 0) >= cfg.track_min_hits
            ]

            # ── Publish to fused_q ───────────────────────────────────
            try:
                fused_q.put_nowait({
                    "kind": "fused_tracks",
                    "fusion_frame_id": fusion_frame_id,
                    "ts": time.time(),
                    "tracks": confirmed,
                    "n_radar_targets": len(radar_targets),
                    "n_eo_dets": len(eo_dets),
                    "n_thermal_dets": len(thermal_dets),
                })
            except Exception:
                # Drop oldest, retry
                try:
                    fused_q.get_nowait()
                    fused_q.put_nowait({
                        "kind": "fused_tracks",
                        "fusion_frame_id": fusion_frame_id,
                        "ts": time.time(),
                        "tracks": confirmed,
                    })
                except Exception:
                    pass

            fusion_frame_id += 1
            n_ticks += 1

            # Stats
            now = time.monotonic()
            if now - last_stats_emit >= 1.0:
                last_stats_emit = now
                try:
                    stats_q.put_nowait({
                        "kind": "fusion_stats",
                        "fusion_frame_id": fusion_frame_id,
                        "ticks_per_sec": n_ticks,
                        "n_persistent": len(persistent),
                        "n_confirmed": len(confirmed),
                    })
                except Exception:
                    pass
                n_ticks = 0

            # Sleep to next tick boundary
            elapsed = time.monotonic() - tick_t0
            sleep_for = max(0.0, period_s - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    except Exception:
        log.exception("fusion loop died")
        return 2
    finally:
        if thermal_ring is not None:
            thermal_ring.close()
        log.info("fusion: clean shutdown")

    return 0


def spawn(mp_ctx, cfg: FusionConfig, eo_det_q, thermal_det_q,
          radar_targets_q):
    ctrl_q = mp_ctx.Queue(maxsize=32)
    fused_q = mp_ctx.Queue(maxsize=32)
    stats_q = mp_ctx.Queue(maxsize=32)
    proc = mp_ctx.Process(
        target=run,
        args=(cfg, ctrl_q, eo_det_q, thermal_det_q, radar_targets_q,
              fused_q, stats_q),
        name="seeker_v2_fusion", daemon=False,
    )
    proc.start()
    return proc, ctrl_q, fused_q, stats_q
