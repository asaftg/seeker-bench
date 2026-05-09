"""seeker_v2 → v1-compatible WebSocket wire format.

The v2 backend produces data in V2-native shapes (FrameDescriptor +
dicts off shared-memory rings + multiprocessing.Queue messages). The
*GUI* shipped with seeker is the v1 GUI: a tightly-coupled bundle of
HTML + CSS + JS that expects a very specific WebSocket envelope shape
(see v1's gui/sensor_bridge.py:build_ws_message).

This module is the adapter: it takes v2 data and emits envelopes that
look exactly like what v1's gui/sensor_bridge.py would produce, so the
v1 frontend can render unchanged on top of v2's multi-process backend.

Two output paths mirror v1 exactly:

  build_ws_message(...) -> dict
      The shared "sensors" envelope (JSON, sent at ~30 Hz over WS).
      Contains thermal/eo metadata, radar, fused, gimbal, illuminator.
      EO image bytes are NOT in this envelope (skip_jpeg=True equivalent);
      they ride the binary fast path below.

  build_eo_binary_frame(eo_hdr_dict, jpeg_bytes) -> bytes
      [4 bytes LE = header length N][N bytes UTF-8 JSON header][JPEG bytes]
      Sent at sensor-arrival cadence per EO frame.

References (v1 source of truth):
  gui/sensor_bridge.py:thermal_to_wire    (lines 103-266)
  gui/sensor_bridge.py:radar_to_wire      (lines 335-553)
  gui/sensor_bridge.py:eo_to_wire_split   (lines 556-636)
  gui/sensor_bridge.py:fused_to_wire      (lines 731-870)
  gui/sensor_bridge.py:build_ws_message   (lines 873-1019)
  gui/app.py:_eo_sender                   (binary WS frame layout)
"""
from __future__ import annotations

import base64
import json
import math
import struct
import time
from typing import Any, Dict, List, Optional, Tuple


# ── Disconnected stubs ─────────────────────────────────────────────────
# Mirrors the v1 helpers' "no connection" return values so the GUI's
# render path doesn't need any special-casing.

def thermal_disconnected_stub(hfov_deg: float = 75.0,
                              vfov_deg: float = 60.0) -> Dict[str, Any]:
    return {
        "connected": False,
        "frame_id": 0,
        "timestamp": 0.0,
        "jpeg_b64": None,
        "width": 0, "height": 0,
        "hfov_deg": hfov_deg, "vfov_deg": vfov_deg,
        "zoom_preset": "full",
        "detections": [],
        "heat_tracks": [],
    }


def eo_disconnected_stub(hfov_deg: float = 11.05,
                         vfov_deg: float = 9.23) -> Dict[str, Any]:
    return {
        "connected": False,
        "initializing": False,
        "frame_id": 0,
        "timestamp": 0.0,
        "jpeg_b64": None,
        "width": 0, "height": 0,
        "hfov_deg": hfov_deg, "vfov_deg": vfov_deg,
        "source_device": None,
        "detections": [],
    }


def radar_disconnected_stub() -> Dict[str, Any]:
    return {
        "connected": False,
        "frame_id": 0,
        "timestamp": 0.0,
        "profile": "awr2944p_unified",
        "max_range_m": 100.0,
        "fov_half_deg": 60.0,
        "num_points": 0,
        "num_targets": 0,
        "points": [],
        "targets": [],
        "detections": [],
    }


def gimbal_disconnected_stub(main_id: Optional[int] = None) -> Dict[str, Any]:
    return {
        "pan": 0.0, "tilt": 0.0,
        "mode": "auto" if main_id is not None else "manual",
        "connected": False,
        "tracked_target_id": None,
        "target_pan": 0.0, "target_tilt": 0.0,
        "error": None,
        "lock_state": "off",
        "lock_bbox_eo": None,
        "lock_bbox_thermal": None,
        "lock_target_id": None,
        "lock_solo_mode": False,
    }


# ── Per-sensor wire encoders ───────────────────────────────────────────

def thermal_to_wire(
    desc,                                  # ipc.FrameDescriptor or None
    jpeg_bytes: Optional[bytes],           # latest cached JPEG bytes
    heat_dets_v2: Optional[List[Dict[str, Any]]] = None,
    hfov_deg: float = 75.0,
    vfov_deg: float = 60.0,
    zoom_preset: str = "full",
) -> Dict[str, Any]:
    """V2 → v1 thermal envelope.

    desc.meta on the thermal ring carries `heat_dets` as a list of
    {x, y, w, h, cx, cy, area, peak} (see processes/thermal_capture.py:
    _tophat_heat_detect). v1 expects each detection wrapped as
    {bbox, area_px, contrast, classification, synthetic, track_id,
    fused_id}.
    """
    if desc is None:
        return thermal_disconnected_stub(hfov_deg, vfov_deg)

    jpeg_b64 = (base64.b64encode(jpeg_bytes).decode("ascii")
                if jpeg_bytes else None)

    raw_dets = heat_dets_v2 or (desc.meta or {}).get("heat_dets") or []
    det_list = []
    for d in raw_dets:
        det_list.append({
            "bbox": {"x": int(d.get("x", 0)), "y": int(d.get("y", 0)),
                     "w": int(d.get("w", 0)), "h": int(d.get("h", 0))},
            "area_px": int(d.get("area", 0)),
            "contrast": round(float(d.get("peak", 0)), 1),
            "classification": None,
            "synthetic": False,
            "track_id": None,
            "fused_id": None,
        })

    return {
        "connected": True,
        "frame_id": int(desc.frame_id),
        "timestamp": float(desc.mtime),
        "jpeg_b64": jpeg_b64,
        "width": int(desc.width),
        "height": int(desc.height),
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "zoom_preset": zoom_preset,
        "detections": det_list,
        "heat_tracks": [],
    }


def eo_to_wire_split(
    desc,
    jpeg_bytes: Optional[bytes],
    eo_dets_v2: Optional[List[Dict[str, Any]]] = None,
    hfov_deg: float = 11.05,
    vfov_deg: float = 9.23,
) -> Tuple[Dict[str, Any], Optional[bytes]]:
    """V2 → v1 EO envelope (split: header dict + raw JPEG bytes).

    Header has `jpeg_size` not `jpeg_b64`; the JPEG bytes ride raw on
    the binary WS frame the EO sender sends.

    eo_dets_v2: optional list of {track_id, class, conf, x, y, w, h}
    from inference.py.
    """
    if desc is None:
        hdr = eo_disconnected_stub(hfov_deg, vfov_deg)
        # Strip jpeg_b64 (None already), add jpeg_size=0
        hdr.pop("jpeg_b64", None)
        hdr["jpeg_size"] = 0
        return hdr, None

    det_list = []
    for d in (eo_dets_v2 or []):
        det_list.append({
            "bbox": {"x": int(d.get("x", 0)), "y": int(d.get("y", 0)),
                     "w": int(d.get("w", 0)), "h": int(d.get("h", 0))},
            "track_id": int(d.get("track_id", -1)) if d.get("track_id", -1) >= 0 else None,
            "fused_id": None,
            "classification": {
                "target_class": _class_name(d.get("class", -1)),
                "confidence": round(float(d.get("conf", 0.0)), 3),
                "classifier_used": "yolo_eo",
            },
        })

    hdr = {
        "connected": True,
        "initializing": False,
        "frame_id": int(desc.frame_id),
        "timestamp": float(desc.mtime),
        "jpeg_size": len(jpeg_bytes) if jpeg_bytes else 0,
        "width": int(desc.width),
        "height": int(desc.height),
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "source_device": "imx568",
        "detections": det_list,
    }
    return hdr, jpeg_bytes


def eo_to_wire(
    desc,
    eo_dets_v2: Optional[List[Dict[str, Any]]] = None,
    hfov_deg: float = 11.05,
    vfov_deg: float = 9.23,
) -> Dict[str, Any]:
    """Same as eo_to_wire_split but for the shared envelope (no JPEG)."""
    hdr, _ = eo_to_wire_split(desc, None, eo_dets_v2, hfov_deg, vfov_deg)
    hdr.pop("jpeg_size", None)
    hdr["jpeg_b64"] = None
    return hdr


def radar_to_wire(
    targets_v2: Optional[List[Dict[str, Any]]],
    last_frame_id: int = 0,
    last_ts: float = 0.0,
    profile: str = "awr2944p_unified",
    max_range_m: float = 100.0,
    fov_half_deg: float = 60.0,
) -> Dict[str, Any]:
    """V2 → v1 radar envelope.

    V2's radar_capture publishes targets like:
      {tid, x, y, z, vx, vy, vz, miss_count, hits}

    v1 needs each target shaped as:
      {tid, fused_id, x, y, z, vx, vy, vz, sx, sy, sz, conf, src,
       np, coasting, hits, misses, class}
    """
    if not targets_v2:
        stub = radar_disconnected_stub()
        stub["frame_id"] = int(last_frame_id)
        stub["timestamp"] = float(last_ts)
        stub["profile"] = profile
        stub["max_range_m"] = max_range_m
        stub["fov_half_deg"] = fov_half_deg
        # If we have targets but they're empty, still consider connected
        # only when last_frame_id > 0 (i.e. radar process at least
        # produced a frame in this session).
        stub["connected"] = last_frame_id > 0
        return stub

    targets_wire = []
    for t in targets_v2:
        targets_wire.append({
            "tid": int(t.get("tid", 0)),
            "fused_id": None,
            "x": round(float(t.get("x", 0.0)), 3),
            "y": round(float(t.get("y", 0.0)), 3),
            "z": round(float(t.get("z", 0.0)), 3),
            "vx": round(float(t.get("vx", 0.0)), 2),
            "vy": round(float(t.get("vy", 0.0)), 2),
            "vz": round(float(t.get("vz", 0.0)), 2),
            "sx": 0.5, "sy": 0.5, "sz": 0.5,
            "conf": 0.5,
            "src": "dbscan",
            "np": int(t.get("hits", 0)),
            "coasting": int(t.get("miss_count", 0)) > 0,
            "hits": int(t.get("hits", 0)),
            "misses": int(t.get("miss_count", 0)),
            "class": "radar_detection",
        })

    return {
        "connected": True,
        "frame_id": int(last_frame_id),
        "timestamp": float(last_ts),
        "profile": profile,
        "max_range_m": max_range_m,
        "fov_half_deg": fov_half_deg,
        "num_points": 0,
        "num_targets": len(targets_wire),
        "points": [],
        "targets": targets_wire,
        "detections": [],
    }


def fused_to_wire(
    tracks_v2: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """V2 fusion output → v1 fused list.

    V2 fusion publishes:
      {tid, az_deg, el_deg, sensors, cls, conf, hits, age_ms,
       eo_box?, thermal_box?}

    v1 expects per-track:
      {id, target_class, confidence, sensors, primary, az_deg, el_deg,
       ang_w_deg, ang_h_deg, hits, bbox_thermal, bbox_eo}
    """
    if not tracks_v2:
        return []
    out = []
    for t in tracks_v2:
        eo_box = t.get("eo_box")
        th_box = t.get("thermal_box")
        out.append({
            "id": int(t.get("tid", 0)),
            "target_class": str(t.get("cls", "unknown")),
            "confidence": round(float(t.get("conf", 0.0)), 3),
            "sensors": list(t.get("sensors", [])),
            "primary": (t.get("sensors") or ["radar"])[0],
            "az_deg": round(float(t.get("az_deg", 0.0)), 2),
            "el_deg": round(float(t.get("el_deg", 0.0)), 2),
            "ang_w_deg": round(float(t.get("ang_w_deg", 1.0)), 2),
            "ang_h_deg": round(float(t.get("ang_h_deg", 1.0)), 2),
            "hits": int(t.get("hits", 0)),
            "bbox_thermal": ({"x": int(th_box[0]), "y": int(th_box[1]),
                              "w": int(th_box[2]), "h": int(th_box[3])}
                             if th_box else None),
            "bbox_eo": ({"x": int(eo_box[0]), "y": int(eo_box[1]),
                         "w": int(eo_box[2]), "h": int(eo_box[3])}
                        if eo_box else None),
        })
    return out


# ── Top-level envelope ─────────────────────────────────────────────────

def build_ws_message(
    *,
    thermal_desc=None, thermal_jpeg_bytes=None, thermal_heat_dets=None,
    eo_desc=None, eo_dets=None,
    # NEW: pass-through for v1 BUS objects so we can call v1's
    # serializers verbatim and inherit all their fields.
    v1_radar_frame=None,
    v1_gimbal_state=None,
    v1_radar_aa_frame=None,
    # Fallback path (V2 native radar_capture). Used only if v1_radar_frame=None.
    radar_targets=None, radar_last_frame_id=0, radar_last_ts=0.0,
    fused_tracks=None,
    gimbal_state=None,
    eo_hfov_deg: float = 11.05, eo_vfov_deg: float = 9.23,
    thermal_hfov_deg: float = 75.0, thermal_vfov_deg: float = 60.0,
    nir_mode: str = "auto",
    main_target_id: Optional[int] = None,
    tracked_target_id: Optional[int] = None,
    tracked_heat_id: Optional[int] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Build the SHARED envelope. Type-tag set so JS demuxes correctly.

    EO JPEG bytes are NOT included; they flow on the binary fast path.
    """
    fused_wire = fused_to_wire(fused_tracks)
    top_targets = sorted(fused_wire,
                         key=lambda t: -(int(t.get("hits", 0))))[: max(0, top_n)]
    top_targets = sorted(top_targets, key=lambda t: int(t.get("id") or 0))

    # ── Radar: prefer v1's RadarFrame via v1's serializer (more
    # complete shape; the GUI's radar_view.js relies on it). Falls back
    # to V2's stub if v1 path unavailable.
    radar_payload = None
    if v1_radar_frame is not None:
        try:
            from gui.sensor_bridge import radar_to_wire as v1_radar_to_wire
            radar_payload = v1_radar_to_wire(
                v1_radar_frame,
                radar_aa_frame=v1_radar_aa_frame,
            )
        except Exception:
            radar_payload = None
    if radar_payload is None:
        radar_payload = radar_to_wire(
            radar_targets,
            last_frame_id=radar_last_frame_id,
            last_ts=radar_last_ts,
        )

    # ── Gimbal: build payload from v1's GimbalState if present.
    if v1_gimbal_state is not None:
        try:
            gs = v1_gimbal_state
            gimbal_payload = {
                "pan": round(float(getattr(gs, "pan_deg", 0.0)), 2),
                "tilt": round(float(getattr(gs, "tilt_deg", 0.0)), 2),
                "mode": getattr(gs, "mode", "manual"),
                "connected": bool(getattr(gs, "connected", False)),
                "tracked_target_id": getattr(gs, "tracked_target_id", None),
                "target_pan": round(float(getattr(gs, "target_pan_deg", 0.0)), 2),
                "target_tilt": round(float(getattr(gs, "target_tilt_deg", 0.0)), 2),
                "error": getattr(gs, "error", None),
                "lock_state": getattr(gs, "lock_state", "off"),
                "lock_bbox_eo": None,
                "lock_bbox_thermal": None,
                "lock_target_id": getattr(gs, "lock_target_id", None),
                "lock_solo_mode": False,
            }
        except Exception:
            gimbal_payload = gimbal_disconnected_stub(main_target_id)
    else:
        gimbal_payload = gimbal_state or gimbal_disconnected_stub(main_target_id)

    return {
        "type": "sensors",
        "ts": time.time(),
        "thermal": thermal_to_wire(
            thermal_desc, thermal_jpeg_bytes, thermal_heat_dets,
            hfov_deg=thermal_hfov_deg, vfov_deg=thermal_vfov_deg,
        ),
        "eo": eo_to_wire(eo_desc, eo_dets,
                          hfov_deg=eo_hfov_deg, vfov_deg=eo_vfov_deg),
        "radar": radar_payload,
        "fused": fused_wire,
        "tracks": [],
        "top_targets": top_targets,
        "main_target_id": main_target_id,
        "tracked_target_id": tracked_target_id,
        "tracked_heat_id": tracked_heat_id,
        "gimbal": gimbal_payload,
        "illuminator": {
            "state": nir_mode,
            "duty": 0.20 if nir_mode == "auto" else (1.0 if nir_mode == "on" else 0.0),
        },
    }


def build_eo_binary_frame(
    eo_hdr: Dict[str, Any],
    jpeg_bytes: Optional[bytes],
) -> bytes:
    """v1's EO binary frame: [4-byte LE hdr_len][JSON header][JPEG bytes].

    Header is a mini envelope: {type: "eo_only", ts, eo: <eo_hdr>}.
    """
    hdr_obj = {"type": "eo_only", "ts": time.time(), "eo": eo_hdr}
    hdr_bytes = json.dumps(hdr_obj, default=str).encode("utf-8")
    payload = struct.pack("<I", len(hdr_bytes)) + hdr_bytes
    if jpeg_bytes:
        payload += jpeg_bytes
    return payload


# ── Helpers ────────────────────────────────────────────────────────────

# Map ultralytics COCO class indices that we care about to the human-
# readable strings v1's GUI looks for. Anything else falls through as
# "object". Mirror seeker_v2/processes/inference.py's extraction.
_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "vehicle",
    3: "vehicle",
    5: "vehicle",
    7: "vehicle",
}


def _class_name(cls_idx: int) -> str:
    try:
        return _CLASS_NAMES.get(int(cls_idx), "object")
    except Exception:
        return "object"
