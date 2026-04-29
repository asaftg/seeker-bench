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


def thermal_to_wire(tf: Optional[ThermalFrame], jpeg_quality: int = 80,
                    gstate: Optional[GimbalState] = None) -> Dict[str, Any]:
    """Serialize a ThermalFrame for the WebSocket.

    When `tf is None` OR `tf.connected is False`, the wire frame
    signals a disconnected state with no image payload.

    When ``gstate`` carries a synthetic-target world-frame lock
    (``synth_world_az_deg / _el_deg`` set), synthetic heat-track bboxes
    are repositioned each frame to where the locked world target
    *should* appear in the thermal image, given the current gimbal
    pose and the frame's FOV. This bypasses the synth ``_Track``'s
    optical-flow propagation (which is unreliable on low-texture
    thermal scenes — observed live in thermal_test3.jsonl where the
    bbox drifted 100+ px off the target after slew completed). With
    the world-angle override, the bbox stays planted on the world
    target as long as the gimbal pose readout is correct.
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
    #
    # Synthetic-target bbox override: if the gimbal has a synth world-
    # frame lock active, recompute the synth track's bbox image
    # position from the locked world angles + current gimbal pose +
    # frame FOV. This kills the OF-propagation drift that operators
    # see when thermal texture is low.
    synth_world_az = synth_world_el = None
    target_resid_az = target_resid_el = None
    cur_pan = cur_tilt = 0.0
    if isinstance(gstate, GimbalState):
        synth_world_az = gstate.synth_world_az_deg
        synth_world_el = gstate.synth_world_el_deg
        target_resid_az = gstate.target_resid_az_deg
        target_resid_el = gstate.target_resid_el_deg
        cur_pan = float(gstate.pan_deg)
        cur_tilt = float(gstate.tilt_deg)
    have_synth_lock = (synth_world_az is not None
                       and synth_world_el is not None
                       and w > 0 and h > 0
                       and tf.hfov_deg > 0 and tf.vfov_deg > 0)
    # When LK has a fresh measurement of where the world target really
    # is in the current camera frame (target_resid_*_deg), use that —
    # it reflects the camera's PHYSICAL pose, not the controller's
    # commanded pose which can lie when the servo isn't following.
    # Fall back to the SW-pose computation when LK isn't available.
    use_lk_residual = (have_synth_lock
                       and target_resid_az is not None
                       and target_resid_el is not None)

    heat_tracks = []
    for ht in getattr(tf, "heat_tracks", None) or []:
        is_synth = bool(getattr(ht, "synthetic", False))
        bx, by, bw, bh = ht.bbox.x, ht.bbox.y, ht.bbox.w, ht.bbox.h
        if is_synth and have_synth_lock:
            if use_lk_residual:
                iaz = float(target_resid_az)
                iel = float(target_resid_el)
            else:
                iaz = synth_world_az - cur_pan
                iel = synth_world_el - cur_tilt
            nx = iaz / float(tf.hfov_deg) + 0.5
            ny = -iel / float(tf.vfov_deg) + 0.5
            cx = nx * w
            cy = ny * h
            bx = int(round(cx - bw / 2.0))
            by = int(round(cy - bh / 2.0))
        heat_tracks.append({
            "id": int(ht.id),
            "bbox": {"x": bx, "y": by, "w": int(bw), "h": int(bh)},
            "hits": int(ht.hits),
            "misses": int(ht.misses),
            "age": int(ht.age),
            "confirmed": bool(ht.confirmed),
            "coasting": bool(ht.coasting),
            "synthetic": is_synth,
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


def _radar_target_to_panel_bboxes(
    t: Any,
    tf: Optional[ThermalFrame],
    ef: Optional[EOFrame],
    az_bias_deg: float = 0.0,
    el_bias_deg: float = 0.0,
) -> tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """Project a single RadarTarget into thermal + EO pixel bboxes.

    Radar frame convention (from tlv_parser): x=right, y=forward (range axis),
    z=up. Boresight is +y. Matches fusion.angular's az-right / el-up convention.

    Returns (bbox_thermal, bbox_eo) — either may be None if the target is
    outside that sensor's FOV or that sensor's frame size is unknown.

    NOTE: assumes radar and cameras are co-located with shared boresight.
    That's good enough for the current bench setup; a real mount will need
    extrinsic calibration (translation + rotation of radar relative to EO).
    """
    import math as _m
    x, y, z = float(t.pos_x_m), float(t.pos_y_m), float(t.pos_z_m)
    # Skip targets behind the sensor — projection is meaningless there.
    if y <= 0.1:
        return None, None
    horiz = _m.sqrt(x * x + y * y)
    az_deg = _m.degrees(_m.atan2(x, y)) + float(az_bias_deg)
    el_deg = (_m.degrees(_m.atan2(z, horiz)) if horiz > 1e-6 else 0.0) + float(el_bias_deg)

    # Angular extent from cluster size at slant range. Size is in metres;
    # width uses lateral dim (sx), height uses vertical dim (sz). Floor at
    # ~0.4° so a tiny cluster doesn't render as a single-pixel dot.
    r_slant = _m.sqrt(x * x + y * y + z * z)
    sx = max(0.4, float(t.size_x_m))
    sz = max(0.4, float(t.size_z_m))
    if r_slant < 0.5:
        r_slant = 0.5
    ang_w_deg = max(0.4, 2.0 * _m.degrees(_m.atan2(sx / 2.0, r_slant)))
    ang_h_deg = max(0.4, 2.0 * _m.degrees(_m.atan2(sz / 2.0, r_slant)))

    bt = None
    if tf is not None and tf.connected and tf.agc8 is not None:
        t_h, t_w = tf.agc8.shape[:2]
        if t_w and t_h and angular_bbox_visible(
            az_deg, el_deg, ang_w_deg, ang_h_deg, tf.hfov_deg, tf.vfov_deg
        ):
            bx, by, bw, bh = angular_to_bbox(
                az_deg, el_deg, ang_w_deg, ang_h_deg,
                t_w, t_h, tf.hfov_deg, tf.vfov_deg,
            )
            if bw > 0 and bh > 0:
                bt = {"x": bx, "y": by, "w": bw, "h": bh}
    be = None
    if ef is not None and ef.connected and ef.bgr is not None:
        e_h, e_w = ef.bgr.shape[:2]
        if e_w and e_h and angular_bbox_visible(
            az_deg, el_deg, ang_w_deg, ang_h_deg, ef.hfov_deg, ef.vfov_deg
        ):
            bx, by, bw, bh = angular_to_bbox(
                az_deg, el_deg, ang_w_deg, ang_h_deg,
                e_w, e_h, ef.hfov_deg, ef.vfov_deg,
            )
            if bw > 0 and bh > 0:
                be = {"x": bx, "y": by, "w": bw, "h": bh}
    return bt, be


def radar_to_wire(
    rf: Optional[RadarFrame] = None,
    max_points: int = 256,
    tf: Optional[ThermalFrame] = None,
    ef: Optional[EOFrame] = None,
    radar_az_bias_deg: float = 0.0,
    radar_el_bias_deg: float = 0.0,
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
            "fov_half_deg": rf.fov_half_deg if rf is not None else 60.0,
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

    targets_wire = []
    for t in rf.targets:
        # Phase 2 fusion path now produces FusedTrack entries with
        # sensors=["radar"] for radar-only targets, and the fused-track
        # projection draws those onto the EO / thermal panels in the
        # canonical class-coloured style. So we no longer pre-project
        # raw radar targets into camera pixel space here — that was
        # duplicating every box once the fusion path was wired in. The
        # radar PANEL still renders this targets payload via radar_view.js.
        targets_wire.append({
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
            "coasting": bool(t.coasting),
            "hits": int(t.hits),
            "misses": int(t.misses),
            "class": "radar_detection",
        })

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


def eo_to_wire_split(
    ef: Optional[EOFrame], jpeg_quality: int = 80
) -> tuple[Dict[str, Any], Optional[bytes]]:
    """Like eo_to_wire, but returns (header_dict, jpeg_bytes) instead of
    folding base64'd JPEG into the JSON. Used by the binary WS path —
    see gui/app.py:_eo_sender.

    The header dict is identical to eo_to_wire's output EXCEPT
    `jpeg_b64` is replaced with `jpeg_size` (just the byte length, so
    the client knows what to expect). JPEG bytes go on the wire as
    raw binary — no base64 (33% bytes + CPU saved), no JSON wrap.

    Falls back to (header, None) for disconnected/empty frames.
    """
    if ef is None or not ef.connected:
        hdr = {
            "connected": False,
            "initializing": bool(getattr(ef, "initializing", False)) if ef is not None else False,
            "frame_id": ef.frame_id if ef is not None else 0,
            "timestamp": ef.timestamp if ef is not None else 0.0,
            "jpeg_size": 0,
            "width": 0,
            "height": 0,
            "hfov_deg": ef.hfov_deg if ef is not None else 11.05,
            "vfov_deg": ef.vfov_deg if ef is not None else 9.23,
            "source_device": None,
            "detections": [],
        }
        return hdr, None

    w, h = 0, 0
    if ef.bgr is not None:
        h, w = ef.bgr.shape[:2]

    # Fast path: reuse cached encode from EO process thread.
    cached = getattr(ef, "jpeg_bytes", None)
    cached_q = int(getattr(ef, "jpeg_quality", -1))
    jpeg_bytes: Optional[bytes] = None
    if cached and cached_q == int(jpeg_quality):
        jpeg_bytes = cached
    elif ef.bgr is not None:
        ok, buf = cv2.imencode(".jpg", ef.bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if ok:
            jpeg_bytes = bytes(buf)

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

    hdr = {
        "connected": True,
        "initializing": bool(getattr(ef, "initializing", False)),
        "frame_id": ef.frame_id,
        "timestamp": ef.timestamp,
        "jpeg_size": len(jpeg_bytes) if jpeg_bytes is not None else 0,
        "width": w,
        "height": h,
        "hfov_deg": ef.hfov_deg,
        "vfov_deg": ef.vfov_deg,
        "source_device": ef.source_device,
        "detections": det_list,
    }
    return hdr, jpeg_bytes


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
            "initializing": bool(getattr(ef, "initializing", False)) if ef is not None else False,
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
    # Fast path: EOManager already encoded the JPEG on its process
    # thread (see eo/eo_manager.py:_process_and_publish). Reuse those
    # bytes if the requested quality matches — this is the whole point
    # of the EOFrame.jpeg_bytes cache. Falls back to inline encode for
    # legacy EOFrames (fake source, replay) that don't carry bytes.
    cached = getattr(ef, "jpeg_bytes", None)
    cached_q = int(getattr(ef, "jpeg_quality", -1))
    if ef.bgr is not None:
        h, w = ef.bgr.shape[:2]
    if cached and cached_q == int(jpeg_quality):
        jpeg_b64 = base64.b64encode(cached).decode("ascii")
    elif ef.bgr is not None:
        ok, buf = cv2.imencode(".jpg", ef.bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
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
        "initializing": bool(getattr(ef, "initializing", False)),
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
    thermal_az_bias_deg: float = 0.0,
    thermal_el_bias_deg: float = 0.0,
) -> list[Dict[str, Any]]:
    """Serialize FusedTrack list with per-sensor pixel projections.

    Each output dict carries:
        id, target_class, confidence, sensors, primary,
        az_deg, el_deg, ang_w_deg, ang_h_deg, hits,
        bbox_thermal: {x,y,w,h} | None,   # projected into thermal pixels
        bbox_eo:      {x,y,w,h} | None,   # projected into EO pixels

    The GUI uses these to draw a single green bbox on each panel that
    represents the fused target at the same world angle.

    Why the thermal bias is SUBTRACTED here while it's ADDED in
    FusionManager._observations_from_thermal:
      - Fusion adds bias to thermal raw az to align thermal observations
        into EO's reference frame. Fused tracks therefore live in the
        EO-aligned (shared) frame.
      - To project a fused track ONTO THERMAL PIXELS we need to undo
        that mapping: shared_az -> thermal_raw_az = shared_az - bias.
      - EO is the ground truth; its raw frame == shared frame, so no
        bias correction on the EO projection.
    Without this subtraction, the projected thermal bbox is offset
    from the actual thermal detection by exactly `thermal_az_bias`
    (operator-reported 2026-04-25: EO->thermal projection misaligned).
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
        # Thermal projection — subtract bias to land in thermal's raw frame
        thr_az = trk.az_deg - float(thermal_az_bias_deg)
        thr_el = trk.el_deg - float(thermal_el_bias_deg)
        bt = None
        if t_w and t_h and angular_bbox_visible(
            thr_az, thr_el, trk.ang_w_deg, trk.ang_h_deg, t_hfov, t_vfov
        ):
            x, y, w, h = angular_to_bbox(
                thr_az, thr_el, trk.ang_w_deg, trk.ang_h_deg,
                t_w, t_h, t_hfov, t_vfov,
            )
            if w > 0 and h > 0:
                bt = {"x": x, "y": y, "w": w, "h": h}
        # EO projection — EO is ground truth, no bias correction
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
            # Pass-through EO ByteTrack id so the EO panel's raw-det
            # labeller can match by id instead of bbox-IoU (more robust
            # under EMA smoothing of the fused track's stored angles).
            "eo_track_id": getattr(trk, "eo_track_id", None),
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
    eo_jpeg_quality: Optional[int] = None,
    nir_mode: str = "auto",
    tracked_target_id: Optional[int] = None,
    tracked_heat_id: Optional[int] = None,
    top_n: int = 5,
    radar_az_bias_deg: float = 0.0,
    radar_el_bias_deg: float = 0.0,
    thermal_az_bias_deg: float = 0.0,
    thermal_el_bias_deg: float = 0.0,
) -> Dict[str, Any]:
    """Build the full WebSocket envelope.

    ``tracked_target_id`` is the user's current TRACK selection from
    the targets list. If it's still alive in the fused list it becomes
    ``main_target_id`` (green highlight + gimbal auto-track target);
    otherwise ``main_target_id`` is None and the gimbal stays manual.
    """
    fused_wire = fused_to_wire(
        fused, tf, ef,
        thermal_az_bias_deg=thermal_az_bias_deg,
        thermal_el_bias_deg=thermal_el_bias_deg,
    )

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

    # EO gets its own JPEG quality knob — a 2K mono sensor with a real
    # lens shows visible JPEG ringing/blocking on foliage and brick at
    # the thermal-grade default of 80. Mono compresses well, so a much
    # higher quality is cheap (~+15-25% bandwidth, sharper image).
    eo_q = int(eo_jpeg_quality) if eo_jpeg_quality is not None else int(jpeg_quality)
    return {
        "ts": time.time(),
        "thermal": thermal_to_wire(tf, jpeg_quality=jpeg_quality,
                                   gstate=gstate),
        "eo": eo_to_wire(ef, jpeg_quality=eo_q),
        "radar": radar_to_wire(
            BUS.get_latest(Topic.RADAR), tf=tf, ef=ef,
            radar_az_bias_deg=radar_az_bias_deg,
            radar_el_bias_deg=radar_el_bias_deg,
        ),
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
