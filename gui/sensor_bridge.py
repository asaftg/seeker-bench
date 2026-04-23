"""
Serialization bridge: FrameBus dataclasses → JSON-ready dicts.

Keeps the FastAPI handler free of any numpy / cv2 knowledge and
gives us one place to adjust the wire format (bump versions,
drop fields, compress differently) without touching the JS.
"""
from __future__ import annotations

import base64
import time
from typing import Any, Dict, Optional

import cv2

from common.frame_bus import BUS
from common.frames import EOFrame, FusedTrack, GimbalState, RadarFrame, ThermalFrame, Topic
from fusion.angular import angular_bbox_visible, angular_to_bbox


def thermal_to_wire(tf: Optional[ThermalFrame], jpeg_quality: int = 80) -> Dict[str, Any]:
    """Serialize a ThermalFrame for the WebSocket.

    When `tf is None` OR `tf.connected is False`, the wire frame
    signals a disconnected state with no image payload.
    """
    if tf is None or not tf.connected:
        return {
            "connected": False,
            "frame_id": tf.frame_id if tf is not None else 0,
            "timestamp": tf.timestamp if tf is not None else 0.0,
            "jpeg_b64": None,
            "width": 0,
            "height": 0,
            "hfov_deg": 75.0,
            "vfov_deg": 60.0,
            "zoom_preset": "full",
            "detections": [],
            "heat_tracks": [],
        }

    # JPEG-encode the AGC display image
    jpeg_b64 = None
    w, h = 0, 0
    if tf.agc8 is not None:
        img = tf.agc8
        h, w = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if ok:
            jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    # Detections
    det_list = []
    for det in tf.detections:
        entry = {
            "bbox": {
                "x": det.bbox.x, "y": det.bbox.y,
                "w": det.bbox.w, "h": det.bbox.h,
            },
            "area_px": det.area_px,
            "contrast": round(float(det.contrast), 1),
            "classification": None,
            "synthetic": bool(getattr(det, "synthetic", False)),
        }
        if det.classification is not None:
            entry["classification"] = {
                "target_class": det.classification.target_class.value,
                "confidence": round(float(det.classification.confidence), 3),
                "classifier_used": det.classification.classifier_used,
            }
        det_list.append(entry)

    # Dev-mode heat-blob tracker snapshot. The GUI filters on its own
    # devMode flag; we always send it so toggling dev-mode is a pure
    # client-side operation (no round-trip).
    heat_tracks = []
    for ht in getattr(tf, "heat_tracks", None) or []:
        heat_tracks.append({
            "id": int(ht.id),
            "bbox": {"x": ht.bbox.x, "y": ht.bbox.y, "w": ht.bbox.w, "h": ht.bbox.h},
            "hits": int(ht.hits),
            "misses": int(ht.misses),
            "age": int(ht.age),
            "confirmed": bool(ht.confirmed),
            "coasting": bool(ht.coasting),
            "synthetic": bool(getattr(ht, "synthetic", False)),
        })

    return {
        "connected": True,
        "frame_id": tf.frame_id,
        "timestamp": tf.timestamp,
        "jpeg_b64": jpeg_b64,
        "width": w,
        "height": h,
        "hfov_deg": tf.hfov_deg,
        "vfov_deg": tf.vfov_deg,
        "zoom_preset": tf.zoom_preset,
        "detections": det_list,
        "heat_tracks": heat_tracks,
    }


def radar_to_wire(
    rf: Optional[RadarFrame] = None,
    max_points: int = 256,
) -> Dict[str, Any]:
    """Serialize a RadarFrame for the WebSocket.

    Called with ``rf=None`` or ``rf.connected=False`` the return value
    mirrors the Phase A stub shape so the GUI can render DISCONNECTED
    without any case-specific branches.

    ``max_points`` caps the outgoing point cloud to the top-N-by-SNR
    entries per frame — mmw_demoDDM can spit several hundred points
    in a busy scene, and full-resolution would bloat the WS envelope.
    Targets are always small (tracklets cap near ~10) so they're not
    capped.
    """
    if rf is None or not rf.connected:
        return {
            "connected": False,
            "frame_id": rf.frame_id if rf is not None else 0,
            "timestamp": rf.timestamp if rf is not None else 0.0,
            "profile": rf.profile if rf is not None else "awr2944p_ddm",
            "max_range_m": rf.max_range_m if rf is not None else 50.0,
            "num_points": 0,
            "num_targets": 0,
            "points": [],
            "targets": [],
            "detections": [],   # reserved for fusion-labeled output
        }

    # Top-N-by-SNR point cap. NaN SNR (DDM build without SideInfo TLV)
    # sorts to the end via a finite sentinel so the cap still works;
    # the JSON serializer can't emit NaN so we also coerce to 0.0 below.
    import math as _m
    def _snr_key(d):
        s = float(d.snr_db)
        return -s if not _m.isnan(s) else float("inf")
    dets = rf.detections
    if len(dets) > max_points:
        dets = sorted(dets, key=_snr_key)[:max_points]

    def _safe_snr(v: float) -> float:
        return 0.0 if _m.isnan(v) else round(v, 1)

    points_wire = [
        {
            "x": round(d.x_m, 3),
            "y": round(d.y_m, 3),
            "z": round(d.z_m, 3),
            "v": round(d.doppler_mps, 2),
            "snr": _safe_snr(float(d.snr_db)),
            "r": round(d.range_m, 2),
            "az": round(d.az_deg, 1),
            "el": round(d.el_deg, 1),
            "tid": int(d.target_id),
        }
        for d in dets
    ]

    targets_wire = [
        {
            "tid": int(t.tid),
            "x": round(t.pos_x_m, 3),
            "y": round(t.pos_y_m, 3),
            "z": round(t.pos_z_m, 3),
            "vx": round(t.vel_x_mps, 2),
            "vy": round(t.vel_y_mps, 2),
            "vz": round(t.vel_z_mps, 2),
            "sx": round(t.size_x_m, 2),
            "sy": round(t.size_y_m, 2),
            "sz": round(t.size_z_m, 2),
            "conf": round(float(t.confidence), 2),
            "src": t.source,
            "np": int(t.num_points),
            "class": "radar_detection",
        }
        for t in rf.targets
    ]

    return {
        "connected": True,
        "frame_id": rf.frame_id,
        "timestamp": rf.timestamp,
        "profile": rf.profile,
        "max_range_m": rf.max_range_m,
        "num_points": int(rf.num_points),
        "num_targets": int(rf.num_targets),
        "points": points_wire,
        "targets": targets_wire,
        "detections": [],
    }


def eo_to_wire(ef: Optional[EOFrame], jpeg_quality: int = 80) -> Dict[str, Any]:
    """Serialize an EOFrame for the WebSocket.

    Wire format matches ThermalFrame as closely as possible so the GUI
    can share rendering code:

        {connected, frame_id, timestamp, jpeg_b64, width, height,
         hfov_deg, vfov_deg, source_device, detections: [...]}

    Each detection is shaped like a thermal detection (bbox + classification)
    so ``overlays.js::drawDetectionBox`` can render EO boxes with zero
    case-specific code.
    """
    if ef is None or not ef.connected:
        return {
            "connected": False,
            "frame_id": ef.frame_id if ef is not None else 0,
            "timestamp": ef.timestamp if ef is not None else 0.0,
            "jpeg_b64": None,
            "width": 0,
            "height": 0,
            "hfov_deg": ef.hfov_deg if ef is not None else 11.05,
            "vfov_deg": ef.vfov_deg if ef is not None else 9.23,
            "source_device": None,
            "detections": [],
        }

    jpeg_b64 = None
    w, h = 0, 0
    if ef.bgr is not None:
        img = ef.bgr
        h, w = img.shape[:2]
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if ok:
            jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    det_list = []
    for det in ef.detections:
        det_list.append({
            "bbox": {
                "x": det.bbox.x, "y": det.bbox.y,
                "w": det.bbox.w, "h": det.bbox.h,
            },
            "track_id": det.track_id,
            "classification": {
                "target_class": det.target_class.value,
                "confidence": round(float(det.confidence), 3),
                "classifier_used": "yolo_eo",
            },
        })

    return {
        "connected": True,
        "frame_id": ef.frame_id,
        "timestamp": ef.timestamp,
        "jpeg_b64": jpeg_b64,
        "width": w,
        "height": h,
        "hfov_deg": ef.hfov_deg,
        "vfov_deg": ef.vfov_deg,
        "source_device": ef.source_device,
        "detections": det_list,
    }


def fused_to_wire(
    tracks: Optional[list],
    tf: Optional[ThermalFrame],
    ef: Optional[EOFrame],
) -> list[Dict[str, Any]]:
    """Serialize FusedTrack list with per-sensor pixel projections.

    Each output dict carries:
        id, target_class, confidence, sensors, primary,
        az_deg, el_deg, ang_w_deg, ang_h_deg, hits,
        bbox_thermal: {x,y,w,h} | None,   # projected into thermal pixels
        bbox_eo:      {x,y,w,h} | None,   # projected into EO pixels

    The GUI uses these to draw a single green bbox on each panel that
    represents the fused target at the same world angle.
    """
    if not tracks:
        return []

    t_w = tf.agc8.shape[1] if (tf is not None and tf.connected and tf.agc8 is not None) else 0
    t_h = tf.agc8.shape[0] if (tf is not None and tf.connected and tf.agc8 is not None) else 0
    t_hfov = tf.hfov_deg if tf is not None else 75.0
    t_vfov = tf.vfov_deg if tf is not None else 60.0

    e_w = ef.bgr.shape[1] if (ef is not None and ef.connected and ef.bgr is not None) else 0
    e_h = ef.bgr.shape[0] if (ef is not None and ef.connected and ef.bgr is not None) else 0
    e_hfov = ef.hfov_deg if ef is not None else 11.05
    e_vfov = ef.vfov_deg if ef is not None else 9.23

    out: list[Dict[str, Any]] = []
    for trk in tracks:
        if not isinstance(trk, FusedTrack):
            continue
        # Thermal projection
        bt = None
        if t_w and t_h and angular_bbox_visible(
            trk.az_deg, trk.el_deg, trk.ang_w_deg, trk.ang_h_deg, t_hfov, t_vfov
        ):
            x, y, w, h = angular_to_bbox(
                trk.az_deg, trk.el_deg, trk.ang_w_deg, trk.ang_h_deg,
                t_w, t_h, t_hfov, t_vfov,
            )
            if w > 0 and h > 0:
                bt = {"x": x, "y": y, "w": w, "h": h}
        # EO projection
        be = None
        if e_w and e_h and angular_bbox_visible(
            trk.az_deg, trk.el_deg, trk.ang_w_deg, trk.ang_h_deg, e_hfov, e_vfov
        ):
            x, y, w, h = angular_to_bbox(
                trk.az_deg, trk.el_deg, trk.ang_w_deg, trk.ang_h_deg,
                e_w, e_h, e_hfov, e_vfov,
            )
            if w > 0 and h > 0:
                be = {"x": x, "y": y, "w": w, "h": h}

        out.append({
            "id": trk.id,
            "target_class": trk.target_class.value,
            "confidence": round(float(trk.confidence), 3),
            "sensors": list(trk.sensors),
            "primary": trk.primary,
            "az_deg": round(trk.az_deg, 3),
            "el_deg": round(trk.el_deg, 3),
            "ang_w_deg": round(trk.ang_w_deg, 3),
            "ang_h_deg": round(trk.ang_h_deg, 3),
            "hits": trk.hits,
            "bbox_thermal": bt,
            "bbox_eo": be,
        })
    return out


def track_score(t: Dict[str, Any]) -> float:
    """Ranking score for the GUI's top-N target list.

    Sensor count dominates (a 2-sensor fused track is *much* more
    trustworthy than any single-sensor one), confidence breaks ties.
    With this scheme, sorting desc puts 2-sensor high-conf at the top,
    then 1-sensor high-conf, then low-conf single-sensor.
    """
    return float(len(t.get("sensors", []))) + float(t.get("confidence", 0.0))


def build_ws_message(
    tf=None,
    ef=None,
    fused=None,
    gstate: Optional[GimbalState] = None,
    jpeg_quality: int = 80,
    nir_mode: str = "auto",
    tracked_target_id: Optional[int] = None,
    tracked_heat_id: Optional[int] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Build the full WebSocket envelope.

    ``tracked_target_id`` is the user's current TRACK selection from
    the targets list. If it's still alive in the fused list it becomes
    ``main_target_id`` (green highlight + gimbal auto-track target);
    otherwise ``main_target_id`` is None and the gimbal stays manual.
    """
    fused_wire = fused_to_wire(fused, tf, ef)

    # Ranked target list — pick top-N by score, then re-sort by stable
    # key (fused track id) so rows don't shuffle as scores fluctuate
    # tick-to-tick. A row that jumps slot 3 → 1 → 2 mid-click is how
    # the TRACK button "flickers" and loses clicks.
    top_targets = sorted(fused_wire, key=track_score, reverse=True)[: max(0, int(top_n))]
    top_targets = sorted(top_targets, key=lambda t: int(t.get("id") or 0))

    # Main target = user's tracked ID, but ONLY if it's still in the
    # fused list this tick. Otherwise the lock drops (GUI clears the
    # row, gimbal logic treats this as "no target → manual").
    main_id: Optional[int] = None
    if tracked_target_id is not None:
        for t in fused_wire:
            if int(t["id"]) == int(tracked_target_id):
                main_id = int(tracked_target_id)
                break

    # Gimbal state — prefer the real GimbalManager state published on
    # the bus. Fall back to a disconnected stub so the GUI never sees
    # missing fields.
    if isinstance(gstate, GimbalState):
        gimbal_payload = {
            "pan": round(float(gstate.pan_deg), 2),
            "tilt": round(float(gstate.tilt_deg), 2),
            "mode": gstate.mode if main_id is None or gstate.mode == "auto" else "auto",
            "connected": bool(gstate.connected),
            "tracked_target_id": gstate.tracked_target_id,
            "target_pan":  round(float(gstate.target_pan_deg), 2),
            "target_tilt": round(float(gstate.target_tilt_deg), 2),
            "error": gstate.error,
        }
    else:
        gimbal_payload = {
            "pan": 0.0,
            "tilt": 0.0,
            "mode": "auto" if main_id is not None else "manual",
            "connected": False,
            "tracked_target_id": tracked_target_id,
            "target_pan": 0.0,
            "target_tilt": 0.0,
            "error": None,
        }

    return {
        "ts": time.time(),
        "thermal": thermal_to_wire(tf, jpeg_quality=jpeg_quality),
        "eo": eo_to_wire(ef, jpeg_quality=jpeg_quality),
        "radar": radar_to_wire(BUS.get_latest(Topic.RADAR)),
        "fused": fused_wire,
        "tracks": [],
        "top_targets": top_targets,
        "main_target_id": main_id,
        "tracked_target_id": tracked_target_id,
        "tracked_heat_id": tracked_heat_id,
        "gimbal": gimbal_payload,
        "illuminator": {
            "state": nir_mode,
            "duty": 0.20 if nir_mode == "auto" else (1.0 if nir_mode == "on" else 0.0),
        },
    }
