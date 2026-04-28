"""Diagnostic: trace the timeline of frames + gimbal + births around
the phantom-birth window in `revert not helping ghosts.jsonl`.

What we know:
    t=973.972  #9, #10 born (eo+thermal at world_el ~4.0)
    t=974.051  #11, #12 born (eo+thermal at world_el ~2.5) — same world_az
                 as #9/#10, el shifted -1.5°.

The question this script answers: between 973.95 and 974.10, what
were the ThermalFrame/EOFrame/GimbalState timestamps and pan/tilt
values, and which fusion tick produced which birth? We need this
to understand whether the 1.5° drift is from:
    (a) fusion using fusion-tick pose for both ticks while frames
        captured at different actual times,
    (b) sensor frame timestamps being publish-time not capture-time,
    (c) gimbal/state arriving slower than expected, or
    (d) something else entirely.
"""
from __future__ import annotations

import json
import sys


def main(path: str, t_start: float, t_end: float) -> None:
    rel0 = None
    print(f"Window {t_start} .. {t_end} (rel s)")
    print(f"{'rel_t':>8} {'channel':>20} {'detail'}")
    print("-" * 80)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ts = r.get("ts_ns")
            if ts is None:
                continue
            if rel0 is None:
                rel0 = int(ts)
            t = (int(ts) - rel0) / 1e9
            if t < t_start or t > t_end:
                continue
            ch = r.get("channel", "")
            msg = r.get("msg") or {}
            if ch == "gimbal/state":
                print(f"{t:>8.3f} {ch:>20} pan={msg.get('pan_deg'):.3f} "
                      f"tilt={msg.get('tilt_deg'):.3f} ts={msg.get('timestamp')}")
            elif ch == "thermal/frame":
                dets = msg.get("detections") or []
                ts_msg = msg.get("timestamp")
                print(f"{t:>8.3f} {ch:>20} ts={ts_msg} "
                      f"n_det={len(dets)} fid={msg.get('frame_id')}")
                for i, d in enumerate(dets[:3]):
                    cl = (d.get("classification") or {}).get("target_class")
                    bb = d.get("bbox") or {}
                    print(f"           det{i} cls={cl} bbox=({bb.get('x')},{bb.get('y')},{bb.get('w')},{bb.get('h')})")
            elif ch == "eo/frame":
                dets = msg.get("detections") or []
                ts_msg = msg.get("timestamp")
                print(f"{t:>8.3f} {ch:>20} ts={ts_msg} "
                      f"n_det={len(dets)} fid={msg.get('frame_id')}")
                for i, d in enumerate(dets[:3]):
                    bb = d.get("bbox") or {}
                    print(f"           det{i} cls={d.get('target_class')} "
                          f"bbox=({bb.get('x')},{bb.get('y')},{bb.get('w')},{bb.get('h')})")
            elif ch == "fusion/tracks":
                tracks = msg.get("tracks") or []
                print(f"{t:>8.3f} {ch:>20} n_tracks={len(tracks)}")
                for tr in tracks:
                    print(f"           id={tr.get('id')} "
                          f"az={tr.get('az_deg')} el={tr.get('el_deg')} "
                          f"hits={tr.get('hits')} m={tr.get('misses')}")
            elif ch == "events":
                tp = msg.get("type")
                if tp in ("fused_track_born", "fused_track_dropped"):
                    pl = msg.get("payload") or {}
                    print(f"{t:>8.3f} {ch:>20} {tp} id={pl.get('id')} "
                          f"az={pl.get('az')} el={pl.get('el')}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nUsage: python _diag_phantom_birth.py RECORDING t_start t_end")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
