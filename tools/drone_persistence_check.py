"""Check if the drone has a 'persistent' signature distinct from operator/clutter.

A drone hovering at one location in the scene shows up as a STABLE
detection at the same (range, az) over many consecutive frames.
An operator walking around shows up as a MOVING detection across
many (range, az) cells.

For each of pre/drone/post windows, compute:
  - Mean detection LIFETIME (consecutive frames with hit at same
    (range, az) within tolerance).
  - Number of distinct stable clusters.
  - Variance of detection (range, az) within window.

If drone window shows a long-lifetime, low-variance stable cluster
that pre/post don't have, that's the drone.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REC = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\airborne1_v5+thermalv2_replay_RADAR.jsonl")


def load_dets():
    """Returns list of (t_rel, range_m, az_deg, vel, snr) for each frame."""
    out = []
    ts0 = None
    with open(REC, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ch = r.get("channel", "")
            if ts0 is None and ch != "session/header":
                ts0 = r.get("ts_ns")
            if ch != "radar/aa_frame":
                continue
            t_rel = (r.get("ts_ns", 0) - ts0) / 1e9 if ts0 else 0
            msg = r.get("msg", {})
            tgts = msg.get("targets", [])
            if tgts:
                t = tgts[0]
                out.append((t_rel, t.get("range_m", 0), t.get("az_deg", 0),
                            t.get("doppler_mps", 0), t.get("snr_db", 0)))
            else:
                out.append((t_rel, None, None, None, None))
    return out


def window_stats(dets, t_lo, t_hi, label):
    inside = [(t, r, a, v, s) for t, r, a, v, s in dets
              if t_lo <= t < t_hi and r is not None]
    n_total = sum(1 for t, *_ in dets if t_lo <= t < t_hi)
    if not inside:
        print(f"\n  {label} ({t_lo}-{t_hi}s): no detections ({n_total} frames)")
        return
    rs = np.array([r for t, r, a, v, s in inside])
    azs = np.array([a for t, r, a, v, s in inside])
    vs = np.array([v for t, r, a, v, s in inside])
    snrs = np.array([s for t, r, a, v, s in inside])
    print(f"\n  {label} ({t_lo}-{t_hi}s): {len(inside)}/{n_total} frames have det "
          f"({100*len(inside)/n_total:.0f}%)")
    print(f"    range : median {np.median(rs):.2f} m, "
          f"std {rs.std():.2f},  iqr [{np.percentile(rs,25):.1f}, {np.percentile(rs,75):.1f}]")
    print(f"    az    : median {np.median(azs):+.1f} deg, "
          f"std {azs.std():.1f},  iqr [{np.percentile(azs,25):+.1f}, {np.percentile(azs,75):+.1f}]")
    print(f"    vel   : median {np.median(vs):+.2f} m/s, "
          f"std {vs.std():.2f}")
    print(f"    snr   : median {np.median(snrs):.1f} dB, max {snrs.max():.1f}")

    # Persistence: count consecutive runs at the same (range, az) within tolerance
    R_TOL = 1.5  # m
    AZ_TOL = 5.0  # deg
    runs = []
    cur_len = 0
    cur_r = cur_a = None
    for t, r, a, v, s in inside:
        if cur_r is None or abs(r - cur_r) > R_TOL or abs(a - cur_a) > AZ_TOL:
            if cur_len > 0: runs.append(cur_len)
            cur_len = 1
            cur_r, cur_a = r, a
        else:
            cur_len += 1
            cur_r = 0.7 * cur_r + 0.3 * r
            cur_a = 0.7 * cur_a + 0.3 * a
    if cur_len > 0: runs.append(cur_len)
    if runs:
        runs_arr = np.array(runs)
        print(f"    runs  : count {len(runs_arr)}, longest {runs_arr.max()}, "
              f"mean {runs_arr.mean():.1f}")
        # Number of "long" runs (>=5 frames at same spot) — this is what
        # would identify a hovering drone.
        long_runs = runs_arr[runs_arr >= 5]
        print(f"    long-runs (>=5 consecutive at same spot): {len(long_runs)}")
        if len(long_runs) > 0:
            print(f"      durations: {long_runs.tolist()}")


def main():
    dets = load_dets()
    print(f"loaded {len(dets)} radar/aa_frame events from replay JSONL")

    window_stats(dets, 0,  30, "PRE  ")
    window_stats(dets, 30, 65, "DRONE")
    window_stats(dets, 65, 200, "POST ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
