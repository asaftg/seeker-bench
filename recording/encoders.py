"""
FrameBus dataclass → JSON-safe dict encoders for the JSONL recorder.

These mirror the wire-format helpers in ``gui/sensor_bridge.py`` but
with two key differences:

1. **Higher-fidelity JPEG by default** (q=92 vs the GUI's q=82) — the
   recording is the source of truth for offline replay; we'd rather
   spend disk than throw away pixel detail. Quality is configurable.

2. **No GUI-side coupling** — the recorder doesn't care about the
   thermal bias, fused-track projections, or panel rendering. It
   captures the raw publisher output. The replay server rehydrates
   the wire shape on its way back to the browser.

Forward-compat lever: adding a new channel = add one ``encode_*``
function and register it in ``recording/jsonl_recorder.py:CHANNELS``.
No schema, no migrations, no recorder restart.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, Optional

import cv2

from common.frames import EOFrame, FusedTrack, GimbalState, RadarFrame, ThermalFrame


def _jpeg_b64(img, quality: int) -> Optional[str]:
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_thermal(tf: Optional[ThermalFrame], jpeg_quality: int = 92) -> Optional[Dict[str, Any]]:
    """Encode a ThermalFrame for JSONL.

    Returns None when ``tf`` is None — the recorder skips the line in
    that case (no point archiving "nothing was published yet").
    Disconnected sentinel frames ARE recorded so the replay knows the
    sensor went down at that timestamp.
    """
    if tf is None:
        return None
    if not tf.connected:
        return {
            "frame_id": int(tf.frame_id),
            "timestamp": float(tf.timestamp),
            "connected": False,
        }
    h, w = (tf.agc8.shape[:2] if tf.agc8 is not None else (0, 0))
    out: Dict[str, Any] = {
        "frame_id": int(tf.frame_id),
        "timestamp": float(tf.timestamp),
        "connected": True,
        "width": int(w),
        "height": int(h),
        "hfov_deg": float(tf.hfov_deg),
        "vfov_deg": float(tf.vfov_deg),
        "zoom_preset": str(tf.zoom_preset),
        "gimbal_pan_at_capture": (None if tf.gimbal_pan_at_capture is None
                                   else round(float(tf.gimbal_pan_at_capture), 4)),
        "gimbal_tilt_at_capture": (None if tf.gimbal_tilt_at_capture is None
                                    else round(float(tf.gimbal_tilt_at_capture), 4)),
        "jpeg_b64": _jpeg_b64(tf.agc8, jpeg_quality),
        "detections": [
            {
                "bbox": {"x": int(d.bbox.x), "y": int(d.bbox.y),
                         "w": int(d.bbox.w), "h": int(d.bbox.h)},
                "area_px": int(d.area_px),
                "contrast": round(float(d.contrast), 1),
                "synthetic": bool(getattr(d, "synthetic", False)),
                "track_id": getattr(d, "track_id", None),
                "classification": (
                    None if d.classification is None else {
                        "target_class": d.classification.target_class.value,
                        "confidence": round(float(d.classification.confidence), 3),
                        "classifier_used": d.classification.classifier_used,
                    }
                ),
            }
            for d in tf.detections
        ],
        "heat_tracks": [
            {
                "id": int(ht.id),
                "bbox": {"x": int(ht.bbox.x), "y": int(ht.bbox.y),
                         "w": int(ht.bbox.w), "h": int(ht.bbox.h)},
                "hits": int(ht.hits),
                "misses": int(ht.misses),
                "age": int(ht.age),
                "confirmed": bool(ht.confirmed),
                "coasting": bool(ht.coasting),
                "synthetic": bool(getattr(ht, "synthetic", False)),
            }
            for ht in (getattr(tf, "heat_tracks", None) or [])
        ],
    }
    return out


def encode_eo(ef: Optional[EOFrame], jpeg_quality: int = 92) -> Optional[Dict[str, Any]]:
    if ef is None:
        return None
    if not ef.connected:
        return {
            "frame_id": int(ef.frame_id),
            "timestamp": float(ef.timestamp),
            "connected": False,
            "initializing": bool(getattr(ef, "initializing", False)),
        }
    h, w = (ef.bgr.shape[:2] if ef.bgr is not None else (0, 0))
    return {
        "frame_id": int(ef.frame_id),
        "timestamp": float(ef.timestamp),
        "connected": True,
        "initializing": bool(getattr(ef, "initializing", False)),
        "width": int(w),
        "height": int(h),
        "hfov_deg": float(ef.hfov_deg),
        "vfov_deg": float(ef.vfov_deg),
        "source_device": ef.source_device,
        "gimbal_pan_at_capture": (None if ef.gimbal_pan_at_capture is None
                                   else round(float(ef.gimbal_pan_at_capture), 4)),
        "gimbal_tilt_at_capture": (None if ef.gimbal_tilt_at_capture is None
                                    else round(float(ef.gimbal_tilt_at_capture), 4)),
        "jpeg_b64": _jpeg_b64(ef.bgr, jpeg_quality),
        "detections": [
            {
                "bbox": {"x": int(d.bbox.x), "y": int(d.bbox.y),
                         "w": int(d.bbox.w), "h": int(d.bbox.h)},
                "track_id": d.track_id,
                "target_class": d.target_class.value,
                "confidence": round(float(d.confidence), 3),
            }
            for d in ef.detections
        ],
    }


def encode_radar(rf: Optional[RadarFrame]) -> Optional[Dict[str, Any]]:
    """Encode a RadarFrame.

    Radar coords are in the SENSOR (gimbal-mounted) frame: +x right,
    +y forward, +z up. Replay tools rotate by the recorded gimbal pan
    if/when they need a world-frame view.
    """
    if rf is None:
        return None
    if not rf.connected:
        return {
            "frame_id": int(rf.frame_id),
            "timestamp": float(rf.timestamp),
            "connected": False,
            "profile": rf.profile,
            "max_range_m": float(rf.max_range_m),
            "fov_half_deg": float(rf.fov_half_deg),
        }
    import math as _m
    def _safe(v: float) -> float:
        try:
            f = float(v)
        except Exception:
            return 0.0
        return 0.0 if _m.isnan(f) else f
    return {
        "frame_id": int(rf.frame_id),
        "timestamp": float(rf.timestamp),
        "connected": True,
        "profile": rf.profile,
        "max_range_m": float(rf.max_range_m),
        "fov_half_deg": float(rf.fov_half_deg),
        "num_points": int(rf.num_points),
        "num_targets": int(rf.num_targets),
        "gimbal_pan_at_capture": (None if rf.gimbal_pan_at_capture is None
                                   else round(float(rf.gimbal_pan_at_capture), 4)),
        "gimbal_tilt_at_capture": (None if rf.gimbal_tilt_at_capture is None
                                    else round(float(rf.gimbal_tilt_at_capture), 4)),
        "points": [
            {
                "x": round(float(d.x_m), 3),
                "y": round(float(d.y_m), 3),
                "z": round(float(d.z_m), 3),
                "v": round(float(d.doppler_mps), 2),
                "snr": round(_safe(d.snr_db), 1),
                "r": round(float(d.range_m), 2),
                "az": round(float(d.az_deg), 1),
                "el": round(float(d.el_deg), 1),
                "tid": int(d.target_id),
            }
            for d in rf.detections
        ],
        "targets": [
            {
                "tid": int(t.tid),
                "x": round(float(t.pos_x_m), 3),
                "y": round(float(t.pos_y_m), 3),
                "z": round(float(t.pos_z_m), 3),
                "vx": round(float(t.vel_x_mps), 2),
                "vy": round(float(t.vel_y_mps), 2),
                "vz": round(float(t.vel_z_mps), 2),
                "sx": round(float(t.size_x_m), 2),
                "sy": round(float(t.size_y_m), 2),
                "sz": round(float(t.size_z_m), 2),
                "conf": round(float(t.confidence), 2),
                "src": t.source,
                "np": int(t.num_points),
                "coasting": bool(t.coasting),
                "hits": int(t.hits),
                "misses": int(t.misses),
            }
            for t in rf.targets
        ],
    }


def encode_gimbal(gs: Optional[GimbalState]) -> Optional[Dict[str, Any]]:
    if gs is None:
        return None
    def _bbox_dict(bb):
        if bb is None: return None
        return {"x": int(bb.x), "y": int(bb.y),
                "w": int(bb.w), "h": int(bb.h)}
    return {
        "timestamp": float(gs.timestamp),
        "connected": bool(gs.connected),
        "pan_deg": float(gs.pan_deg),
        "tilt_deg": float(gs.tilt_deg),
        "mode": str(gs.mode),
        "target_pan_deg": float(gs.target_pan_deg),
        "target_tilt_deg": float(gs.target_tilt_deg),
        "tracked_target_id": gs.tracked_target_id,
        "error": gs.error,
        # Synth lock + LK-corrected target residual (added 2026-04-27
        # for the BB-drift fix; sensor_bridge uses these to render the
        # synth bbox at the world target's actual image position).
        "synth_world_az_deg": (None if gs.synth_world_az_deg is None
                                else float(gs.synth_world_az_deg)),
        "synth_world_el_deg": (None if gs.synth_world_el_deg is None
                                else float(gs.synth_world_el_deg)),
        "target_resid_az_deg": (None if gs.target_resid_az_deg is None
                                 else float(gs.target_resid_az_deg)),
        "target_resid_el_deg": (None if gs.target_resid_el_deg is None
                                 else float(gs.target_resid_el_deg)),
        # Lock-mode fields (gimbal.lock_mode in YAML). Recorded so a
        # future replay can analyze the lock state machine offline.
        # The lock-mode v1 retro (recordings/lock poorly.jsonl) was
        # blocked because these fields weren't serialized — fix that
        # now so v2 sessions are debuggable.
        "lock_state": getattr(gs, "lock_state", "off"),
        "lock_bbox_eo": _bbox_dict(getattr(gs, "lock_bbox_eo", None)),
        "lock_bbox_thermal": _bbox_dict(getattr(gs, "lock_bbox_thermal", None)),
        "lock_target_id": getattr(gs, "lock_target_id", None),
    }


def encode_fused(tracks: Optional[list]) -> Optional[Dict[str, Any]]:
    """Wrap a list of FusedTrack into ``{tracks: [...]}``."""
    if tracks is None:
        return None
    out = []
    for t in tracks:
        if not isinstance(t, FusedTrack):
            continue
        out.append({
            "id": int(t.id),
            "target_class": t.target_class.value,
            "confidence": round(float(t.confidence), 3),
            "sensors": list(t.sensors),
            "primary": str(t.primary),
            "az_deg": round(float(t.az_deg), 3),
            "el_deg": round(float(t.el_deg), 3),
            "ang_w_deg": round(float(t.ang_w_deg), 3),
            "ang_h_deg": round(float(t.ang_h_deg), 3),
            "hits": int(t.hits),
            "misses": int(t.misses),
            "world_az_deg": (None if t.world_az_deg is None
                              else round(float(t.world_az_deg), 3)),
            "world_el_deg": (None if t.world_el_deg is None
                              else round(float(t.world_el_deg), 3)),
        })
    return {"tracks": out}
