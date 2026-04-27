"""
Self-contained smoke test for the recorder + replay pipeline.

Spins up the JSONLRecorder against a fresh FrameBus, fakes a few
seconds of thermal/eo/radar/gimbal/fused publishes plus user events,
stops the recorder, then runs replay_inspect.cmd_summary against the
written file. No hardware, no GUI.

Run:  python scripts/_smoke_recorder.py
Expects: 0 exit code, sensible summary on stdout.
"""
from __future__ import annotations

import os
import sys
import time
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from common.frame_bus import FrameBus
from common.frames import (BBox, EOFrame, FusedTrack, GimbalState,
                           RadarDetection, RadarFrame, RadarTarget,
                           TargetClass, ThermalDetection, ThermalFrame,
                           Topic)
from common.events import emit
from recording.jsonl_recorder import JSONLRecorder


def main() -> int:
    out_dir = os.path.join(_ROOT, "recordings")
    os.makedirs(out_dir, exist_ok=True)

    bus = FrameBus()
    rec = JSONLRecorder(bus, output_dir=out_dir, jpeg_quality=85)

    # NOTE: events module uses the singleton BUS, not our local one,
    # for its broadcast topic. The recorder uses a direct subscribe
    # callback that doesn't depend on the bus. So events flow fine
    # in this isolated test even though the bus instance differs.
    cfg = {"version": "smoke", "gimbal": {"track_lead_time_s": 0.3}}
    path = rec.start(config_snapshot=cfg)
    print(f"[smoke] recording -> {path}")

    emit("recording_started", {"path": path})

    # 30 ticks ≈ 1s at 30Hz of fake telemetry
    H, W = 60, 80
    img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    eo_img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    for i in range(30):
        ts = time.time()
        # Thermal
        tf = ThermalFrame(
            timestamp=ts, frame_id=i, connected=True,
            agc8=img,
            detections=[ThermalDetection(BBox(10, 10, 5, 5), 25, 1.5)],
            hfov_deg=75.0, vfov_deg=60.0, zoom_preset="full",
        )
        bus.publish(Topic.THERMAL, tf)
        # EO
        ef = EOFrame(
            timestamp=ts, frame_id=i, connected=True,
            bgr=eo_img,
            hfov_deg=11.05, vfov_deg=9.23, source_device=0,
        )
        bus.publish(Topic.EO, ef)
        # Radar
        rf = RadarFrame(
            timestamp=ts, frame_id=i, connected=True,
            detections=[RadarDetection(0.1*i, 5.0, 0.0, -1.2, 18.5,
                                       0.0, 5.0+0.1*i, 1.5, 0.0, 7)],
            targets=[RadarTarget(7, 0.1*i, 5.0, 0.0, 0.0, -1.2, 0.0)],
            num_points=1, num_targets=1, max_range_m=50.0,
        )
        bus.publish(Topic.RADAR, rf)
        # Gimbal
        gs = GimbalState(timestamp=ts, connected=True,
                         pan_deg=float(i*0.5), tilt_deg=0.0,
                         mode="auto", target_pan_deg=float(i*0.5),
                         target_tilt_deg=0.0, tracked_target_id=42)
        bus.publish(Topic.GIMBAL, gs)
        # Fused
        ft = FusedTrack(id=42, target_class=TargetClass.PERSON,
                        confidence=0.8, sensors=["eo", "radar"],
                        primary="eo", az_deg=2.0+0.1*i, el_deg=0.0,
                        ang_w_deg=1.0, ang_h_deg=1.0, hits=i+1)
        bus.publish(Topic.FUSED, [ft])

        # Sprinkle events across the timeline
        if i == 5:
            emit("track_engaged", {"target_id": 42})
        if i == 10:
            emit("track_predictor_step", {
                "tracked_id": 42, "now": ts, "cur_pan": 5.0,
                "cur_tilt": 0.0, "gimbal_dps": 0.5, "settled": True,
                "fresh_fused": True, "obs_world_az": 7.0,
                "obs_world_el": 0.0, "world_az": 7.0, "world_el": 0.0,
                "world_az_dot": 1.0, "world_el_dot": 0.0,
                "obs_count": 1, "age": 0.0, "confidence": 0.2,
                "lead": 0.3, "shift_az": 0.06, "shift_el": 0.0,
                "sp_pan": 7.06, "sp_tilt": 0.0,
            })
        if i == 20:
            emit("tilt_saturated_enter", {"cur_tilt": 0.0, "el_err": 0.5})
        if i == 25:
            emit("track_released", {"reason": "smoke_test_done"})

        time.sleep(0.03)

    emit("recording_stopped", {})
    closed = rec.stop()
    print(f"[smoke] stopped -> {closed}")

    # Run inspector summary against what we just wrote
    print()
    print("=== replay_inspect --summary ===")
    from scripts.replay_inspect import cmd_summary
    rc = cmd_summary(closed)
    if rc != 0:
        return rc

    # Quick sanity check on file shape
    import json
    n_lines = 0
    channels = set()
    with open(closed, "r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            d = json.loads(line)
            channels.add(d.get("channel"))
    print()
    print(f"file lines: {n_lines}")
    print(f"channels: {sorted(channels)}")
    assert "session/header" in channels, "missing header"
    assert "thermal/frame" in channels, "missing thermal"
    assert "events" in channels, "missing events"
    assert n_lines > 30, f"file too short: {n_lines}"
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
