"""Dump first non-empty radar point structure to inspect field names."""
import json
from pathlib import Path

JSONL = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\drone test airborne 1.jsonl")

n = 0
with open(JSONL, "r", encoding="utf-8") as f:
    for line in f:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("channel") != "radar/frame":
            continue
        msg = rec.get("msg", {})
        pts = msg.get("points") or []
        if pts:
            n += 1
            if n <= 3:
                print(f"frame_id={msg.get('frame_id')} num_points={msg.get('num_points')}")
                for p in pts[:3]:
                    print("   keys:", list(p.keys()))
                    print("   point:", json.dumps(p))
            else:
                break
