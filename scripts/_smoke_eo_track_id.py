"""End-to-end test of the EO track_id pass-through chain.

Verifies that an EO ByteTrack id placed on an EODetection survives:
    EOFrame
      → FusionManager._observations_from_eo
      → candidate dict
      → _update_tracks (birth path)
      → _update_tracks (update path)
      → FusedTrack.eo_track_id
      → fused_to_wire (GUI wire payload)
      → GUI matcher logic (mirrored in Python here)

Operator-reported 2026-04-27 the EO panel kept labeling raw dets
with E#N (the per-sensor ByteTrack id) even though a fused track
existed for the same target. Cause was a missing pass-through; this
test exists so a future blanket-revert can't silently break it again.

Run: python scripts/_smoke_eo_track_id.py
Exit 0 = chain intact; non-zero = something in the wire dropped the id.
"""
from __future__ import annotations
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.frames import (BBox, EODetection, EOFrame, FusedTrack,
                           TargetClass)
from fusion.fusion_manager import FusionManager
from gui.sensor_bridge import fused_to_wire


def fail(msg: str) -> None:
    sys.stderr.write(f"FAIL: {msg}\n")
    sys.exit(1)


def main() -> int:
    ef = EOFrame(
        timestamp=0.0, frame_id=1, connected=True,
        bgr=np.zeros((1080, 1920, 3), dtype=np.uint8),
        detections=[EODetection(
            bbox=BBox(x=900, y=500, w=120, h=80),
            confidence=0.92,
            target_class=TargetClass.VEHICLE,
            track_id=2,
        )],
        hfov_deg=11.05, vfov_deg=9.23, source_device=0,
    )

    fm = FusionManager()
    obs = fm._observations_from_eo(ef)
    if not obs:
        fail("_observations_from_eo returned empty")
    if obs[0].get("eo_track_id") != 2:
        fail(f"obs missing eo_track_id (got {obs[0].get('eo_track_id')!r})")

    cand = {
        "sensors": ["eo"], "primary": "eo", "class": obs[0]["class"],
        "az": obs[0]["az"], "el": obs[0]["el"],
        "ang_w": obs[0]["ang_w"], "ang_h": obs[0]["ang_h"],
        "conf": obs[0]["conf"],
        "eo_track_id": obs[0].get("eo_track_id"),
    }
    fm._update_tracks([cand])  # birth
    if not fm._tracks:
        fail("no fused track born")
    if fm._tracks[0].get("eo_track_id") != 2:
        fail("birth path didn't store eo_track_id")

    fm._update_tracks([cand])  # update
    if fm._tracks[0].get("eo_track_id") != 2:
        fail("update path lost eo_track_id")

    trk = fm._tracks[0]
    ft = FusedTrack(
        id=int(trk["id"]),
        target_class=TargetClass.VEHICLE,
        confidence=float(trk["conf"]),
        sensors=list(trk["sensor_misses"].keys()),
        primary=str(trk["primary"]),
        az_deg=float(trk["az"]),
        el_deg=float(trk["el"]),
        ang_w_deg=float(trk["ang_w"]),
        ang_h_deg=float(trk["ang_h"]),
        hits=int(trk["hits"]),
        misses=int(trk["misses"]),
        eo_track_id=trk.get("eo_track_id"),
    )
    if ft.eo_track_id != 2:
        fail(f"FusedTrack.eo_track_id missing ({ft.eo_track_id!r})")

    wire = fused_to_wire([ft], tf=None, ef=ef)
    if not wire:
        fail("fused_to_wire returned empty")
    if wire[0].get("eo_track_id") != 2:
        fail(f"wire payload missing eo_track_id ({wire[0].get('eo_track_id')!r})")

    # Mirror the JS matcher logic.
    det = {"track_id": 2}
    tid = int(det["track_id"])
    matched_id = None
    for t in wire:
        if t.get("eo_track_id") is not None and int(t["eo_track_id"]) == tid:
            matched_id = t["id"]
            break
    if matched_id != ft.id:
        fail(f"GUI matcher returned {matched_id}, expected {ft.id}")

    print(f"PASS — chain intact, GUI matcher returns fused id {matched_id} "
          f"for det.track_id={tid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
