"""End-to-end test of the per-sensor id → fused id link chain.

Verifies that for each contributing sensor (EO, thermal, radar), the
per-sensor tracker id placed on a raw observation survives:

    EO:      EOFrame.detections[].track_id
              → _observations_from_eo
              → candidate dict (eo+thermal pair / eo-only)
              → _update_tracks (birth + update)
              → FusedTrack.eo_track_id
              → fused_to_wire["eo_track_id"]

    THERMAL: ThermalFrame.detections[].track_id
              → _observations_from_thermal
              → candidate dict (eo+thermal pair / thermal-only)
              → FusedTrack.thermal_heat_id
              → fused_to_wire["thermal_heat_id"]

    RADAR:   RadarFrame.targets[].tid
              → _observations_from_radar (radar_tid)
              → candidate dict (radar joins / radar-only)
              → FusedTrack.radar_tid
              → fused_to_wire["radar_tid"]

Run: python scripts/_smoke_id_chain.py
Exit 0 = chain intact for every sensor; non-zero = something dropped
the id at one of the stages above.
"""
from __future__ import annotations
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.frames import (BBox, ClassificationResult, EODetection, EOFrame,
                           FusedTrack, RadarFrame, RadarTarget, TargetClass,
                           ThermalDetection, ThermalFrame)
from fusion.fusion_manager import FusionManager
from gui.sensor_bridge import fused_to_wire


def fail(msg: str) -> None:
    sys.stderr.write(f"FAIL: {msg}\n")
    sys.exit(1)


def run_chain(label, ef, tf, rf, expect_eo=None, expect_thermal=None,
              expect_radar=None):
    """Run one fusion tick with the given sensor frames and assert the
    fused track exposes the expected per-sensor link ids on the wire.
    """
    fm = FusionManager()
    # _tick uses BUS.get_latest; we drive _update_tracks directly via
    # the observation builders instead so there's no thread/bus state.
    eo_obs = fm._observations_from_eo(ef) if ef is not None else []
    th_obs = fm._observations_from_thermal(tf) if tf is not None else []
    ra_obs = fm._observations_from_radar(rf) if rf is not None else []

    candidates = []
    used_t = [False] * len(th_obs)
    for e in eo_obs:
        # Find best thermal pair (mirrors _tick logic, IoU)
        best_i, best_iou = -1, 0.0
        for i, t in enumerate(th_obs):
            if used_t[i] or t["class"] != e["class"]:
                continue
            from fusion.angular import angular_iou
            iou = angular_iou(e["az"], e["el"], e["ang_w"], e["ang_h"],
                               t["az"], t["el"], t["ang_w"], t["ang_h"])
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= 0.05:
            t = th_obs[best_i]
            used_t[best_i] = True
            candidates.append({
                "sensors": ["eo", "thermal"], "primary": "eo",
                "class": e["class"],
                "az": e["az"], "el": e["el"],
                "ang_w": e["ang_w"], "ang_h": e["ang_h"],
                "conf": max(e["conf"], t["conf"]),
                "eo_track_id":     e.get("eo_track_id"),
                "thermal_heat_id": t.get("thermal_heat_id"),
            })
        else:
            candidates.append({
                "sensors": ["eo"], "primary": "eo",
                "class": e["class"],
                "az": e["az"], "el": e["el"],
                "ang_w": e["ang_w"], "ang_h": e["ang_h"],
                "conf": e["conf"],
                "eo_track_id": e.get("eo_track_id"),
            })
    for i, t in enumerate(th_obs):
        if used_t[i]:
            continue
        candidates.append({
            "sensors": ["thermal"], "primary": "thermal",
            "class": t["class"],
            "az": t["az"], "el": t["el"],
            "ang_w": t["ang_w"], "ang_h": t["ang_h"],
            "conf": t["conf"],
            "thermal_heat_id": t.get("thermal_heat_id"),
        })
    # Radar joins (simplified — gate on IoU against camera candidates)
    n_cam = len(candidates)
    used_c = [False] * n_cam
    for r in ra_obs:
        from fusion.angular import angular_iou
        best_i, best_iou = -1, 0.0
        for i in range(n_cam):
            if used_c[i]:
                continue
            c = candidates[i]
            iou = angular_iou(r["az"], r["el"], r["ang_w"], r["ang_h"],
                               c["az"], c["el"], c["ang_w"], c["ang_h"])
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= fm.radar_iou_gate:
            candidates[best_i]["sensors"].append("radar")
            if r.get("radar_tid") is not None:
                candidates[best_i]["radar_tid"] = int(r["radar_tid"])
            used_c[best_i] = True
        else:
            candidates.append({
                "sensors": ["radar"], "primary": "radar",
                "class": r["class"],
                "az": r["az"], "el": r["el"],
                "ang_w": r["ang_w"], "ang_h": r["ang_h"],
                "conf": r["conf"],
                "radar_tid": r.get("radar_tid"),
            })

    fm._update_tracks(candidates)
    fm._update_tracks(candidates)  # second tick to exercise the update path
    if not fm._tracks:
        fail(f"[{label}] no fused track born")

    trk = fm._tracks[0]
    # Build a FusedTrack like _publish would.
    ft = FusedTrack(
        id=int(trk["id"]),
        target_class=TargetClass.VEHICLE,
        confidence=float(trk["conf"]),
        sensors=list(trk["sensor_misses"].keys()),
        primary=str(trk["primary"]),
        az_deg=float(trk["az"]), el_deg=float(trk["el"]),
        ang_w_deg=float(trk["ang_w"]), ang_h_deg=float(trk["ang_h"]),
        hits=int(trk["hits"]), misses=int(trk["misses"]),
        eo_track_id=trk.get("eo_track_id"),
        thermal_heat_id=trk.get("thermal_heat_id"),
        radar_tid=trk.get("radar_tid"),
    )
    wire = fused_to_wire([ft], tf=tf, ef=ef)
    if not wire:
        fail(f"[{label}] fused_to_wire empty")
    w = wire[0]

    for fld, expected, name in [
        ("eo_track_id",     expect_eo,      "EO"),
        ("thermal_heat_id", expect_thermal, "thermal"),
        ("radar_tid",       expect_radar,   "radar"),
    ]:
        if expected is None:
            continue
        got = w.get(fld)
        if got != expected:
            fail(f"[{label}] {name} link broken: wire[{fld!r}]={got!r}, "
                 f"expected {expected!r}")
    print(f"  {label}: eo={w.get('eo_track_id')!r} "
          f"thermal={w.get('thermal_heat_id')!r} "
          f"radar={w.get('radar_tid')!r}  OK")


def main() -> int:
    print("Per-sensor id chain test:")

    bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    agc8 = np.zeros((512, 640, 3), dtype=np.uint8)

    # ── EO only ────────────────────────────────────────────────
    ef = EOFrame(
        timestamp=0.0, frame_id=1, connected=True,
        bgr=bgr,
        detections=[EODetection(
            bbox=BBox(900, 500, 120, 80),
            confidence=0.92, target_class=TargetClass.VEHICLE,
            track_id=2,
        )],
        hfov_deg=11.05, vfov_deg=9.23,
    )
    run_chain("EO only", ef, None, None, expect_eo=2)

    # ── Thermal only ──────────────────────────────────────────
    tf = ThermalFrame(
        timestamp=0.0, frame_id=1, connected=True, agc8=agc8,
        detections=[ThermalDetection(
            bbox=BBox(280, 220, 80, 60), area_px=4800, contrast=10.0,
            classification=ClassificationResult(
                target_class=TargetClass.VEHICLE,
                confidence=0.85, classifier_used="yolo_hv"),
            track_id=7,
        )],
        hfov_deg=75.0, vfov_deg=60.0,
    )
    run_chain("thermal only", None, tf, None, expect_thermal=7)

    # ── Radar only ────────────────────────────────────────────
    rf = RadarFrame(
        timestamp=0.0, frame_id=1, connected=True,
        targets=[RadarTarget(
            tid=4, pos_x_m=0.0, pos_y_m=15.0, pos_z_m=0.0,
            vel_x_mps=0.0, vel_y_mps=0.0, vel_z_mps=0.0,
            size_x_m=1.0, size_y_m=1.0, size_z_m=1.0,
            confidence=0.9, num_points=10,
        )],
        max_range_m=50.0, fov_half_deg=60.0,
    )
    run_chain("radar only", None, None, rf, expect_radar=4)

    # ── Verify build_ws_message stamps fused_id on raw dets ──
    # This is the architectural simplification: backend resolves
    # per-sensor track_id → fused_id once, GUI just renders it.
    print()
    print("End-to-end build_ws_message fused_id stamping:")
    from gui.sensor_bridge import build_ws_message
    fm2 = FusionManager()
    # Push EO det through fusion so a fused track exists.
    eo_obs = fm2._observations_from_eo(ef)
    fm2._update_tracks([{
        "sensors": ["eo"], "primary": "eo",
        "class": eo_obs[0]["class"],
        "az": eo_obs[0]["az"], "el": eo_obs[0]["el"],
        "ang_w": eo_obs[0]["ang_w"], "ang_h": eo_obs[0]["ang_h"],
        "conf": eo_obs[0]["conf"],
        "eo_track_id": eo_obs[0]["eo_track_id"],
    }])
    trk = fm2._tracks[0]
    ft = FusedTrack(
        id=int(trk["id"]),
        target_class=TargetClass.VEHICLE,
        confidence=float(trk["conf"]),
        sensors=list(trk["sensor_misses"].keys()),
        primary=str(trk["primary"]),
        az_deg=float(trk["az"]), el_deg=float(trk["el"]),
        ang_w_deg=float(trk["ang_w"]), ang_h_deg=float(trk["ang_h"]),
        hits=int(trk["hits"]), misses=int(trk["misses"]),
        eo_track_id=trk.get("eo_track_id"),
    )
    payload = build_ws_message(tf=None, ef=ef, fused=[ft])
    eo_dets = payload["eo"]["detections"]
    if not eo_dets:
        fail("[ws] no EO detections in payload")
    if eo_dets[0].get("fused_id") != ft.id:
        fail(f"[ws] EO det.fused_id={eo_dets[0].get('fused_id')!r}, "
             f"expected {ft.id}")
    print(f"  EO det.fused_id={eo_dets[0]['fused_id']} "
          f"(matches fused #{ft.id})  OK")

    # ── EO + thermal pair (overlapping angular position) ──────
    # EO det at center → az=0, el=0. Thermal det at center too → fused.
    ef2 = EOFrame(
        timestamp=0.0, frame_id=2, connected=True, bgr=bgr,
        detections=[EODetection(
            bbox=BBox(900, 500, 120, 80),
            confidence=0.92, target_class=TargetClass.VEHICLE,
            track_id=11,
        )],
        hfov_deg=11.05, vfov_deg=9.23,
    )
    # Small thermal bbox at the centre so the angular IoU with the
    # narrow-FOV EO det is high enough to pair (cross-sensor pair gate
    # in fusion is ~0.05). EO at 11° HFOV / 1920 px → 5.7 px = 0.03°,
    # so a ~6×5 px thermal bbox at 75° HFOV is the matching scale.
    tf2 = ThermalFrame(
        timestamp=0.0, frame_id=2, connected=True, agc8=agc8,
        detections=[ThermalDetection(
            bbox=BBox(317, 253, 6, 5), area_px=30, contrast=10.0,
            classification=ClassificationResult(
                target_class=TargetClass.VEHICLE,
                confidence=0.80, classifier_used="yolo_hv"),
            track_id=22,
        )],
        hfov_deg=75.0, vfov_deg=60.0,
    )
    run_chain("EO+thermal pair", ef2, tf2, None,
              expect_eo=11, expect_thermal=22)

    print()
    print("PASS — per-sensor id link chain intact for all sensors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
