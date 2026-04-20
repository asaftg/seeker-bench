"""
Serialization bridge: FrameBus dataclasses → JSON-ready dicts.

Keeps the FastAPI handler free of any numpy / cv2 knowledge and
gives us one place to adjust the wire format (bump versions,
drop fields, compress differently) without touching the JS.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np

from common.frames import EOFrame, FusedTrack, ThermalFrame
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
        }
        if det.classification is not None:
            entry["classification"] = {
                "target_class": det.classification.target_class.value,
                "confidence": round(float(det.classification.confidence), 3),
                "classifier_used": det.classification.classifier_used,
            }
        det_list.append(entry)

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
    }


def radar_to_wire() -> Dict[str, Any]:
    """Phase A stub — radar always disconnected."""
    return {
        "connected": False,
        "points": [],
        "detections": [],
        "profile": "automotive_default",
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


def build_ws_message(
    tf=None,
    ef=None,
    fused=None,
    jpeg_quality: int = 80,
    tracker_on: bool = True,
    nir_mode: str = "auto",
    gimbal_pan: float = 0.0,
    gimbal_tilt: float = 60.0,
) -> Dict[str, Any]:
    """Build the full Phase B WebSocket envelope.

    Optional fields (eo, radar, tracks, main_target_id, gimbal, illuminator)
    follow the schema defined in Ticket 2.  Absent hardware sends its stub.
    """
    import time as _time
    fused_wire = fused_to_wire(fused, tf, ef)
    # Pick a main target: highest-confidence multi-sensor track, else
    # highest-confidence single-sensor track. This is the ID the GUI
    # lights up in green at the top and the gimbal would track.
    main_id = None
    if fused_wire:
        multi = [t for t in fused_wire if len(t["sensors"]) >= 2]
        pool = multi if multi else fused_wire
        best = max(pool, key=lambda t: (len(t["sensors"]), t["confidence"]))
        main_id = best["id"]
    return {
        "ts": _time.time(),
        "thermal": thermal_to_wire(tf, jpeg_quality=jpeg_quality),
        "eo": eo_to_wire(ef, jpeg_quality=jpeg_quality),
        "radar": radar_to_wire(),
        "fused": fused_wire,
        "tracks": [],
        "main_target_id": main_id,
        "gimbal": {
            "pan": gimbal_pan,
            "tilt": gimbal_tilt,
            "mode": "auto" if tracker_on else "manual",
        },
        "illuminator": {
            "state": nir_mode,
            "duty": 0.20 if nir_mode == "auto" else (1.0 if nir_mode == "on" else 0.0),
        },
        "tracker_on": tracker_on,
    }
