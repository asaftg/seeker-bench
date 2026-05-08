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

from common.config import load_config
from common.frame_bus import BUS
from common.frames import EOFrame, FusedTrack, GimbalState, RadarFrame, ThermalFrame, Topic

# Load lock-mode flags once at import. Cheap; YAML re-reads on bench
# restart only. The flags travel on the gimbal payload to the GUI so
# the JS can decide whether to apply solo-rendering.
_LM_CFG = (load_config().get("gimbal", {}) or {}).get("lock_mode", {}) or {}
_LOCK_ENABLED: bool = bool(_LM_CFG.get("enabled", False))
_LOCK_SOLO_MODE: bool = bool(_LM_CFG.get("solo_mode", True))
from fusion.angular import angular_bbox_visible, angular_to_bbox


def _bbox_iou(a: dict, b: dict) -> float:
    """Pixel-bbox IoU. Used as a fallback when per-sensor id stamping
    misses (typical: ByteTrack id-swap creates one frame where the
    raw det's track_id is new but fusion hasn't yet rebuilt its
    eo_track_id link)."""
    if not a or not b:
        return 0.0
    ax2 = a["x"] + a["w"]; ay2 = a["y"] + a["h"]
    bx2 = b["x"] + b["w"]; by2 = b["y"] + b["h"]
    ix1 = max(a["x"], b["x"]); iy1 = max(a["y"], b["y"])
    ix2 = min(ax2, bx2);       iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1);  ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return float(inter) / float(ua) if ua > 0 else 0.0


def _build_fused_id_index(fused: Optional[list]) -> tuple[dict, dict, dict]:
    """Build three lookup tables: per-sensor track id → fused track id.

    Computed once per WS frame in build_ws_message and threaded into
    each ``*_to_wire`` so raw detections can be stamped with their
    fused id directly. The GUI then renders a single ``det.fused_id``
    field — no client-side matcher logic, no IoU drift, no E#vs#
    desync. This is the architectural simplification that replaces
    the per-render ``fusedIdForDet`` lookup.

    Returns (eo_map, thermal_map, radar_map). Each maps sensor's
    track id → fused track id. Keys are the per-sensor ids
    (``EODetection.track_id``, ``ThermalDetection.track_id`` /
    ``HeatTrackSnapshot.id``, ``RadarTarget.tid``); values are the
    cross-sensor ``FusedTrack.id``. Missing key = no fused link.
    """
    eo_map: dict = {}
    th_map: dict = {}
    rd_map: dict = {}
    if not fused:
        return eo_map, th_map, rd_map
    for trk in fused:
        if not isinstance(trk, FusedTrack):
            continue
        fid = int(trk.id)
        if trk.eo_track_id is not None:
            eo_map[int(trk.eo_track_id)] = fid
        if trk.thermal_heat_id is not None:
            th_map[int(trk.thermal_heat_id)] = fid
        if trk.radar_tid is not None:
            rd_map[int(trk.radar_tid)] = fid
    return eo_map, th_map, rd_map


def _fused_id_for_bbox(rawBBox: dict, fused_wire: Optional[list],
                       sideKey: str, iouMin: float = 0.30) -> Optional[int]:
    """Bbox-IoU fallback: scan fused_wire for the projected bbox that
    overlaps `rawBBox` most. Used when the per-sensor id map missed
    (typical: ByteTrack id-swap). Returns the fused track id or None.
    `sideKey` is "bbox_eo" or "bbox_thermal"."""
    if not fused_wire or not rawBBox:
        return None
    best_id = None
    best_iou = iouMin
    for t in fused_wire:
        fb = t.get(sideKey)
        if not fb:
            continue
        iou = _bbox_iou(rawBBox, fb)
        if iou >= best_iou:
            best_iou = iou
            best_id = t.get("id")
    return best_id


def thermal_to_wire(tf: Optional[ThermalFrame], jpeg_quality: int = 80,
                    gstate: Optional[GimbalState] = None,
                    fused_id_by_thermal: Optional[dict] = None,
                    fused_wire: Optional[list] = None) -> Dict[str, Any]:
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

    # JPEG-encode the AGC display image. Fast path: ThermalManager
    # already encoded the JPEG on its process thread (see
    # thermal/thermal_manager.py:_process_and_publish, mirrors EO). Reuse
    # those bytes when the cache's quality matches the requested one;
    # falls back to inline encode for legacy ThermalFrames (replay,
    # disconnect-reconnect race) that don't carry bytes.
    jpeg_b64 = None
    w, h = 0, 0
    if tf.agc8 is not None:
        img = tf.agc8
        h, w = img.shape[:2]
        cached = getattr(tf, "jpeg_bytes", None)
        cached_q = int(getattr(tf, "jpeg_quality", -1))
        if cached and cached_q == int(jpeg_quality):
            jpeg_b64 = base64.b64encode(cached).decode("ascii")
        else:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if ok:
                jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    # Detections
    det_list = []
    th_map = fused_id_by_thermal or {}
    for det in tf.detections:
        det_tid = getattr(det, "track_id", None)
        bbox = {
            "x": det.bbox.x, "y": det.bbox.y,
            "w": det.bbox.w, "h": det.bbox.h,
        }
        # Two-stage fused id lookup: id-map first (fast, exact), then
        # bbox-IoU against fused_wire's projected bbox_thermal as a
        # fallback for the brief tick after a ByteTrack id-swap.
        fused_id = (th_map.get(int(det_tid))
                     if det_tid is not None else None)
        if fused_id is None:
            fused_id = _fused_id_for_bbox(bbox, fused_wire, "bbox_thermal")
        entry = {
            "bbox": bbox,
            "area_px": det.area_px,
            "contrast": round(float(det.contrast), 1),
            "classification": None,
            "synthetic": bool(getattr(det, "synthetic", False)),
            # Per-sensor heat-tracker id stamped by DetectionTracker.
            "track_id": det_tid,
            # Fused track id, pre-resolved by the backend. None when
            # this raw det isn't yet linked to any fused track.
            "fused_id": fused_id,
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
    fused_id_by_radar: Optional[dict] = None,
    radar_aa_frame: Optional[RadarFrame] = None,
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
        # Stock TLV is offline. If the Phase 3 raw-ADC pipeline is
        # alive (LVDS still flowing even when TLV stalled), surface
        # its targets through the same radar payload so the operator
        # still sees A/A hits in the radar canvas. We synthesize a
        # minimal RadarFrame-shaped wire dict here. Fields not
        # populated by the AA pipeline (num_points, profile from Stock)
        # fall back to safe defaults.
        if (radar_aa_frame is not None
                and getattr(radar_aa_frame, "connected", False)):
            aa_targets = []
            for t in radar_aa_frame.targets:
                aa_targets.append({
                    "tid": int(t.tid) + 100000,
                    "fused_id": None,
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
                    "class": "drone" if t.source == "pmm" else "radar_detection",
                })
            # ALSO forward A/G CFAR detections (range/azimuth points)
            # so the operator sees the raw radar picture, not just
            # clustered drone targets. Previously this branch ignored
            # ``radar_aa_frame.detections`` and the GUI was empty even
            # when the pipeline was producing 15+ valid CFAR hits per
            # frame — fixed 2026-05-04.
            aa_points = []
            for d in radar_aa_frame.detections[:max_points]:
                aa_points.append({
                    "x": round(d.x_m, 3),
                    "y": round(d.y_m, 3),
                    "z": round(d.z_m, 3),
                    "v": round(d.doppler_mps, 2),
                    "snr": round(float(d.snr_db), 1),
                    "r": round(d.range_m, 2),
                    "az": round(d.az_deg, 1),
                    "el": round(d.el_deg, 1),
                    "tid": int(d.target_id),
                })
            return {
                "connected": True,    # raw-ADC IS connected
                "frame_id": radar_aa_frame.frame_id,
                "timestamp": radar_aa_frame.timestamp,
                "profile": radar_aa_frame.profile or "awr2944p_unified",
                "max_range_m": radar_aa_frame.max_range_m,
                "fov_half_deg": radar_aa_frame.fov_half_deg,
                "num_points": len(aa_points),
                "num_targets": len(aa_targets),
                "points": aa_points,
                "targets": aa_targets,
                "detections": [],
            }
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
        rd_map = fused_id_by_radar or {}
        targets_wire.append({
            "tid": int(t.tid),
            # Backend-resolved fused id for this radar Kalman track,
            # or None when fusion hasn't linked it yet.
            "fused_id": rd_map.get(int(t.tid)),
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

    # Merge raw-ADC overlay (Phase 3 A/A and A/G output). Only present
    # when the operator is in mode=aa AND the DCAPipeline is alive.
    # We tag the merged targets with src="pmm" / src="ag" so the
    # frontend can color/label them differently from Stock TLV
    # targets (src="dbscan"). Range/angle/velocity fields are the
    # same shape so radar_view.js renders them identically.
    if (radar_aa_frame is not None
            and getattr(radar_aa_frame, "connected", False)):
        for t in radar_aa_frame.targets:
            targets_wire.append({
                "tid": int(t.tid) + 100000,   # offset to avoid clash with Stock tids
                "fused_id": None,
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
                "src": t.source,           # "pmm" for A/A, "ag" for A/G
                "np": int(t.num_points),
                "coasting": bool(t.coasting),
                "hits": int(t.hits),
                "misses": int(t.misses),
                "class": "drone" if t.source == "pmm" else "radar_detection",
            })
        # A/G additionally puts CFAR detection points into .detections;
        # surface them so the operator sees the raw range/azimuth
        # picture, not just clustered targets.
        for d in radar_aa_frame.detections[:max_points]:
            points_wire.append({
                "x": round(d.x_m, 3),
                "y": round(d.y_m, 3),
                "z": round(d.z_m, 3),
                "v": round(d.doppler_mps, 2),
                "snr": round(float(d.snr_db), 1),
                "r": round(d.range_m, 2),
                "az": round(d.az_deg, 1),
                "el": round(d.el_deg, 1),
                "tid": int(d.target_id),
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
    ef: Optional[EOFrame], jpeg_quality: int = 80,
    fused_id_by_eo: Optional[dict] = None,
    fused_wire: Optional[list] = None,
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
    eo_map = fused_id_by_eo or {}
    for det in ef.detections:
        bbox = {"x": det.bbox.x, "y": det.bbox.y,
                "w": det.bbox.w, "h": det.bbox.h}
        fid = (eo_map.get(int(det.track_id))
                if det.track_id is not None else None)
        if fid is None:
            fid = _fused_id_for_bbox(bbox, fused_wire, "bbox_eo")
        det_list.append({
            "bbox": bbox,
            "track_id": det.track_id,
            "fused_id": fid,
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


def eo_to_wire(ef: Optional[EOFrame], jpeg_quality: int = 80,
               fused_id_by_eo: Optional[dict] = None,
               fused_wire: Optional[list] = None,
               skip_jpeg: bool = False) -> Dict[str, Any]:
    """Serialize an EOFrame for the WebSocket.

    Wire format matches ThermalFrame as closely as possible so the GUI
    can share rendering code:

        {connected, frame_id, timestamp, jpeg_b64, width, height,
         hfov_deg, vfov_deg, source_device, detections: [...]}

    Each detection is shaped like a thermal detection (bbox + classification)
    so ``overlays.js::drawDetectionBox`` can render EO boxes with zero
    case-specific code.

    ``skip_jpeg=True`` makes the function return ``jpeg_b64=None`` without
    paying the base64 cost. The shared GUI WS sender uses this because
    EO bytes ride the binary _eo_sender fast path — the redundant
    base64 of a 460 KB JPEG every shared-tick is what dragged all three
    sensor panels to 7-9 Hz.
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
    if ef.bgr is not None:
        h, w = ef.bgr.shape[:2]
    if not skip_jpeg:
        # Fast path: EOManager already encoded the JPEG on its process
        # thread (see eo/eo_manager.py:_process_and_publish). Reuse those
        # bytes if the requested quality matches — this is the whole
        # point of the EOFrame.jpeg_bytes cache. Falls back to inline
        # encode for legacy EOFrames (fake source, replay) that don't
        # carry bytes.
        cached = getattr(ef, "jpeg_bytes", None)
        cached_q = int(getattr(ef, "jpeg_quality", -1))
        if cached and cached_q == int(jpeg_quality):
            jpeg_b64 = base64.b64encode(cached).decode("ascii")
        elif ef.bgr is not None:
            ok, buf = cv2.imencode(".jpg", ef.bgr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if ok:
                jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    det_list = []
    eo_map = fused_id_by_eo or {}
    for det in ef.detections:
        bbox = {"x": det.bbox.x, "y": det.bbox.y,
                "w": det.bbox.w, "h": det.bbox.h}
        fid = (eo_map.get(int(det.track_id))
                if det.track_id is not None else None)
        if fid is None:
            fid = _fused_id_for_bbox(bbox, fused_wire, "bbox_eo")
        det_list.append({
            "bbox": bbox,
            "track_id": det.track_id,
            "fused_id": fid,
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

    # Per-panel pose-at-capture, for re-projecting world-frame tracks into
    # each frame's actual viewing angle. Without this the fused bbox lags
    # the image during a fast slew: fusion stamps trk.az_deg using
    # `cur_pan` at fusion-publish time (15 Hz tick), but the EO frame
    # being drawn was captured at an OLDER pose, and the thermal frame
    # at yet a third pose. At a 50 °/s slew the per-tick lag works out
    # to ~3° on EO (HFOV 11°) → ~170 px offset on the 1236-wide canvas,
    # which the operator perceives as "losing the target." When the
    # track carries world_az_deg / world_el_deg AND the panel frame
    # carries gimbal_*_at_capture, we project off (world − pose_at_capture)
    # so the bbox lands on the image's actual viewing angle.
    e_pan_cap = getattr(ef, "gimbal_pan_at_capture", None) if ef is not None else None
    e_tilt_cap = getattr(ef, "gimbal_tilt_at_capture", None) if ef is not None else None
    t_pan_cap = getattr(tf, "gimbal_pan_at_capture", None) if tf is not None else None
    t_tilt_cap = getattr(tf, "gimbal_tilt_at_capture", None) if tf is not None else None

    out: list[Dict[str, Any]] = []
    for trk in tracks:
        if not isinstance(trk, FusedTrack):
            continue
        wa = getattr(trk, "world_az_deg", None)
        we = getattr(trk, "world_el_deg", None)
        # Thermal projection — pose-sync to the thermal frame's
        # capture pose if both world angles + thermal pose-at-capture
        # are available; fall back to the legacy (camera-frame az_deg
        # at fusion-publish-pose) path when world fusion is off or
        # the thermal frame predates the pose stamp.
        if wa is not None and we is not None and t_pan_cap is not None and t_tilt_cap is not None:
            thr_az = wa - float(t_pan_cap) - float(thermal_az_bias_deg)
            thr_el = we - float(t_tilt_cap) - float(thermal_el_bias_deg)
        else:
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
        # EO projection — EO is ground truth, no bias correction.
        # Same pose-sync pattern as thermal above.
        if wa is not None and we is not None and e_pan_cap is not None and e_tilt_cap is not None:
            eo_az = wa - float(e_pan_cap)
            eo_el = we - float(e_tilt_cap)
        else:
            eo_az = trk.az_deg
            eo_el = trk.el_deg
        be = None
        if e_w and e_h and angular_bbox_visible(
            eo_az, eo_el, trk.ang_w_deg, trk.ang_h_deg, e_hfov, e_vfov
        ):
            x, y, w, h = angular_to_bbox(
                eo_az, eo_el, trk.ang_w_deg, trk.ang_h_deg,
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
            # Pass-through per-sensor tracker IDs so each panel can
            # label raw dets with the matching fused id by direct
            # equality instead of bbox-IoU. Symmetric across all three
            # sensors (Phase B1 + B2). All three are Optional[int] —
            # None means "this sensor hasn't contributed an obs yet."
            "eo_track_id":     getattr(trk, "eo_track_id",     None),
            "thermal_heat_id": getattr(trk, "thermal_heat_id", None),
            "radar_tid":       getattr(trk, "radar_tid",       None),
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

    # Resolve per-sensor track id → fused id ONCE per WS frame. The
    # individual *_to_wire helpers stamp ``fused_id`` on each raw det
    # using these maps so the GUI doesn't run a per-render matcher.
    # Single source of truth for "is this raw det fused, and into what?"
    eo_to_fused, thermal_to_fused, radar_to_fused = _build_fused_id_index(fused)

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
    def _bbox_to_dict(bb):
        if bb is None:
            return None
        return {"x": int(bb.x), "y": int(bb.y),
                "w": int(bb.w), "h": int(bb.h)}

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
            # Lock-mode fields (gimbal.lock_mode in YAML). The GUI
            # uses lock_state to decide GREEN (active) vs AMBER
            # (coasting) and lock_bbox_{eo,thermal} as the bbox to
            # draw. None when lock mode is disabled or no engagement.
            "lock_state": getattr(gstate, "lock_state", "off"),
            "lock_bbox_eo": _bbox_to_dict(getattr(gstate, "lock_bbox_eo", None)),
            "lock_bbox_thermal": _bbox_to_dict(getattr(gstate, "lock_bbox_thermal", None)),
            "lock_target_id": getattr(gstate, "lock_target_id", None),
            # Solo render flag (gimbal.lock_mode.solo_mode in YAML).
            # When true AND tracked_target_id is set, the GUI hides
            # all non-engaged red detection boxes + non-engaged
            # fused-track boxes — only the engaged target shows.
            "lock_solo_mode": _LOCK_SOLO_MODE,
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
            "lock_state": "off",
            "lock_bbox_eo": None,
            "lock_bbox_thermal": None,
            "lock_target_id": None,
            "lock_solo_mode": _LOCK_SOLO_MODE,
        }

    # EO gets its own JPEG quality knob — a 2K mono sensor with a real
    # lens shows visible JPEG ringing/blocking on foliage and brick at
    # the thermal-grade default of 80. Mono compresses well, so a much
    # higher quality is cheap (~+15-25% bandwidth, sharper image).
    eo_q = int(eo_jpeg_quality) if eo_jpeg_quality is not None else int(jpeg_quality)
    return {
        "ts": time.time(),
        "thermal": thermal_to_wire(tf, jpeg_quality=jpeg_quality,
                                   gstate=gstate,
                                   fused_id_by_thermal=thermal_to_fused,
                                   fused_wire=fused_wire),
        # skip_jpeg=True: the binary _eo_sender fast path in gui/app.py
        # delivers EO JPEG bytes at sensor-arrival cadence; the shared
        # WS message only needs metadata (size, fov, detections) for
        # fused-track bbox_eo projection. Re-base64-ing the same 460 KB
        # JPEG on every shared-tick costs ~10 ms and was the dominant
        # bottleneck dragging all panels to ~7-9 Hz.
        "eo": eo_to_wire(ef, jpeg_quality=eo_q,
                          fused_id_by_eo=eo_to_fused,
                          fused_wire=fused_wire,
                          skip_jpeg=True),
        "radar": radar_to_wire(
            BUS.get_latest(Topic.RADAR), tf=tf, ef=ef,
            radar_az_bias_deg=radar_az_bias_deg,
            radar_el_bias_deg=radar_el_bias_deg,
            fused_id_by_radar=radar_to_fused,
            # Phase 3 raw-ADC overlay. Merges PMM (A/A) hits and
            # A/G CFAR detections from radar_dca.DCAPipeline into
            # the same wire payload so the existing radar canvas
            # renders both. Pipeline only publishes when mode==aa,
            # so in stock+ag this is None and adds nothing.
            radar_aa_frame=BUS.get_latest(Topic.RADAR_AA),
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
