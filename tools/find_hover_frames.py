"""Find hover frames in airborne1 from chip TLV detections.

Reads `drone test airborne 1.jsonl`, walks all radar/frame messages,
collects frames where points exist with low |vel_y_mps| (<1 m/s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

JSONL = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\drone test airborne 1.jsonl")


def main() -> None:
    radar_idx = -1     # 0-based index into radar/frame stream (matches .bin frame index)
    hover_rows = []    # (radar_idx, frame_id, num_points, range_m, vel_y, vel_x)
    n_with_pts = 0
    first_frame_id = None
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("channel") != "radar/frame":
                continue
            radar_idx += 1
            msg = rec.get("msg", {})
            fid = msg.get("frame_id")
            if first_frame_id is None:
                first_frame_id = fid
            pts = msg.get("points") or []
            if not pts:
                continue
            n_with_pts += 1
            # vy near 0 across any point => hover candidate
            # Find any point with low |v| — chip's vradial is the
            # closest thing to "is the target hovering?".
            for p in pts:
                v = float(p.get("v") or 0.0)
                rng = float(p.get("r") or 0.0)
                az = float(p.get("az") or 0.0)
                if abs(v) <= 2.0 and 4.0 <= rng <= 80.0:
                    hover_rows.append((radar_idx, fid, len(pts), rng, v, az))
                    break
    print(f"first chip frame_id: {first_frame_id}")
    print(f"total radar/frame:   {radar_idx+1}")
    print(f"frames with points:  {n_with_pts}")
    print(f"hover candidates:    {len(hover_rows)}")
    print("\nradar_idx  fid     npts  rng_m   v_mps   az_deg")
    for r in hover_rows[:80]:
        print(f"  {r[0]:5d}  {r[1]:6d} {r[2]:4d}  {r[3]:6.1f}  {r[4]:+6.2f}  {r[5]:+6.1f}")


if __name__ == "__main__":
    main()
