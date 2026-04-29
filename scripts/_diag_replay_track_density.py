"""Quantify how many tracks are alive simultaneously across the
replay of a recording.

For the `multiple bbs.jsonl` debug we care less about total births
and more about: how many bboxes does the operator see at any one
tick? That's the user-visible 'multiple bbs' problem.

Walks through fusion_manager._tick offline (with the same matcher
the live system runs), tracks the count of alive tracks per tick,
and reports max + histogram.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Re-use the replay_fusion machinery
import scripts.replay_fusion as rf

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", "-r", required=True)
    ap.add_argument("--variant", default="world-frame-stamped",
                    choices=("camera-frame", "world-frame",
                             "world-frame-capture",
                             "world-frame-stamped"))
    ap.add_argument("--max-misses", type=int, default=300)
    args = ap.parse_args()

    # Reproduce replay's main loop but record alive-count per tick.
    fcfg = {}
    thermal_az = thermal_el = radar_az = radar_el = 0.0
    state = None
    rate_hz = 15.0
    next_fusion_t = None
    fusion_dt = 1.0 / rate_hz
    latest_thermal = None
    latest_eo = None
    latest_radar = None
    latest_pan = 0.0
    latest_tilt = 0.0
    latest_thermal_ts = 0.0
    latest_eo_ts = 0.0
    latest_radar_ts = 0.0
    pose_history = []

    alive_counts = []
    t0_ns = None
    for r in rf.iter_records(args.recording):
        ts = r.get("ts_ns")
        if ts is None: continue
        if t0_ns is None: t0_ns = int(ts)
        t = (int(ts) - t0_ns) / 1e9
        ch = r.get("channel", "")
        msg = r.get("msg") or {}
        if ch == "session/header":
            cfg = (msg.get("config_snapshot") or {})
            fcfg = cfg.get("fusion") or {}
            tcfg = (cfg.get("thermal") or {}).get("extrinsic") or {}
            rcfg = (cfg.get("radar") or {}).get("extrinsic") or {}
            thermal_az = float(tcfg.get("az_bias_deg", 0.0))
            thermal_el = float(tcfg.get("el_bias_deg", 0.0))
            radar_az = float(rcfg.get("az_bias_deg", 0.0))
            radar_el = float(rcfg.get("el_bias_deg", 0.0))
            rate_hz = float(fcfg.get("rate_hz", 15.0))
            cls = (rf.WorldFrameStampedFusionState if args.variant == "world-frame-stamped"
                    else rf.WorldFrameFusionState if args.variant == "world-frame"
                    else rf.WorldFrameCaptureFusionState if args.variant == "world-frame-capture"
                    else rf.FusionState)
            state = cls(max_misses=args.max_misses, sensor_grace_ticks=5)
            fusion_dt = 1.0 / rate_hz
            continue
        if state is None:
            cls = (rf.WorldFrameStampedFusionState if args.variant == "world-frame-stamped"
                    else rf.WorldFrameFusionState if args.variant == "world-frame"
                    else rf.WorldFrameCaptureFusionState if args.variant == "world-frame-capture"
                    else rf.FusionState)
            state = cls(max_misses=args.max_misses, sensor_grace_ticks=5)
        if ch == "events": continue
        if ch == "thermal/frame":
            latest_thermal = msg; latest_thermal_ts = t
        elif ch == "eo/frame":
            latest_eo = msg; latest_eo_ts = t
        elif ch == "radar/frame":
            latest_radar = msg; latest_radar_ts = t
        elif ch == "gimbal/state":
            latest_pan = float(msg.get("pan_deg") or 0.0)
            latest_tilt = float(msg.get("tilt_deg") or 0.0)
            pose_history.append((t, latest_pan, latest_tilt))
            if len(pose_history) > 60: pose_history.pop(0)

        if next_fusion_t is None:
            next_fusion_t = t + fusion_dt
        radar_iou_gate = float(fcfg.get("radar_iou_gate", 0.05))
        while next_fusion_t is not None and t >= next_fusion_t:
            thermal_obs = rf.thermal_obs_from_frame(latest_thermal or {}, thermal_az, thermal_el)
            eo_obs = rf.eo_obs_from_frame(latest_eo or {})
            radar_obs_list = rf.radar_obs_from_frame(latest_radar or {}, radar_az, radar_el)
            cands = rf.cross_sensor_associate(thermal_obs, eo_obs, radar_obs_list, radar_iou_gate)
            cands = rf.dedup_candidates(cands)
            state.step(cands, latest_pan, latest_tilt)
            state.tracks = rf.merge_overlapping_tracks(state.tracks)
            # Count visible tracks (active sensor sets non-empty)
            visible = sum(1 for tr in state.tracks
                           if tr.get("sensor_misses") and any(
                               m <= state.sensor_grace_ticks
                               for m in tr["sensor_misses"].values()))
            alive_counts.append(visible)
            next_fusion_t += fusion_dt

    print(f"variant: {args.variant}  max_misses: {args.max_misses}")
    print(f"ticks: {len(alive_counts)}")
    if not alive_counts:
        return 0
    print(f"alive max: {max(alive_counts)}")
    print(f"alive avg: {sum(alive_counts)/len(alive_counts):.1f}")
    hist = Counter(alive_counts)
    for k in sorted(hist):
        print(f"  {k} alive: {hist[k]} ticks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
