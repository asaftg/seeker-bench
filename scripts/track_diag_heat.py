"""Heat-track diagnostic — draw a synthetic_target bbox, then log
gimbal pan/tilt + heat-track centroid over time. Catches the
"gimbal circles the target instead of centering" bug.

Usage:  python scripts/track_diag_heat.py [--bbox X Y W H] [--duration 10]
        --bbox defaults to (200, 100, 80, 80) — top-leftish of a Boson 640x512.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
import time

import websockets


async def run(bbox: tuple[int, int, int, int],
              duration_s: float) -> int:
    print(f"connecting -> ws://127.0.0.1:8080/ws/sensors", flush=True)
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        # Eat first frame so we have current state
        first = await asyncio.wait_for(ws.recv(), timeout=5.0)
        m0 = json.loads(first)
        g = m0.get("gimbal") or {}
        print(f"connected. gimbal pan={g.get('pan'):+.2f} "
              f"tilt={g.get('tilt'):+.2f}  mode={g.get('mode')}",
              flush=True)

        # Send synthetic_target — this seeds a heat track at the
        # bbox center and auto-engages the gimbal lock on it.
        x, y, w, h = bbox
        print(f"\n-> synthetic_target bbox=[{x},{y},{w},{h}]  "
              f"center=({x + w//2}, {y + h//2})", flush=True)
        await ws.send(json.dumps({
            "command": "synthetic_target",
            "bbox": [x, y, w, h],
        }))

        # Log gimbal + heat-track centroid for `duration_s`.
        # Heat-track centroid lives under thermal.heat_tracks (debug
        # mode might be required; otherwise use the synthetic
        # detection that's tagged in thermal.detections).
        t0 = time.time()
        last_log = 0
        rows: list[dict] = []
        last_heat_id = None
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            elapsed = time.time() - t0
            if elapsed >= duration_s:
                break
            now = time.time()
            if now - last_log < 0.1: continue
            last_log = now

            m = json.loads(raw)
            g = m.get("gimbal") or {}
            tf = m.get("thermal") or {}
            tracked_heat_id = m.get("tracked_heat_id")
            if tracked_heat_id != last_heat_id and tracked_heat_id is not None:
                print(f"  heat lock -> H#{tracked_heat_id}", flush=True)
                last_heat_id = tracked_heat_id

            # Find the heat-track or detection matching the lock id
            cx = cy = None
            for d in (tf.get("detections") or []):
                if d.get("synthetic"):
                    bb = d.get("bbox") or {}
                    cx = bb.get("x", 0) + bb.get("w", 0) // 2
                    cy = bb.get("y", 0) + bb.get("h", 0) // 2
                    break
            for ht in (tf.get("heat_tracks") or []):
                if ht.get("id") == tracked_heat_id:
                    bb = ht.get("bbox") or {}
                    if cx is None:
                        cx = bb.get("x", 0) + bb.get("w", 0) // 2
                        cy = bb.get("y", 0) + bb.get("h", 0) // 2
                    break

            # Compute az/el of centroid (camera-relative) for
            # mathematical visibility — the gimbal sees the target
            # at this offset. tf.hfov_deg is the current zoom.
            tw = tf.get("width") or 640
            th_ = tf.get("height") or 512
            hfov = tf.get("hfov_deg") or 75.0
            vfov = tf.get("vfov_deg") or 60.0
            az = el = None
            if cx is not None and cy is not None:
                nx = (cx / tw) - 0.5
                ny = (cy / th_) - 0.5
                az = nx * hfov
                el = -ny * vfov

            row = {
                "t":     elapsed,
                "pan":   g.get("pan"),
                "tilt":  g.get("tilt"),
                "mode":  g.get("mode"),
                "lock":  tracked_heat_id,
                "cx":    cx,
                "cy":    cy,
                "az":    az,
                "el":    el,
            }
            rows.append(row)

        # Release the heat track.
        print("\n-> CLEAR synthetic", flush=True)
        try:
            await ws.send(json.dumps({"command": "clear_synthetic_target"}))
        except Exception:
            pass

        # Print the trajectory.
        print(f"\nlogged {len(rows)} samples over {duration_s}s:", flush=True)
        print(f"  {'t':>5}  {'pan':>7}  {'tilt':>6}  {'cx':>4} {'cy':>4}  "
              f"{'az':>7}  {'el':>6}  {'mode':>6}  {'lock':>4}",
              flush=True)
        for r in rows:
            pan_s = f"{r['pan']:+.2f}"  if r["pan"]  is not None else "  None"
            tilt_s= f"{r['tilt']:+.2f}" if r["tilt"] is not None else " None"
            cx_s  = f"{r['cx']}"        if r["cx"]   is not None else "  --"
            cy_s  = f"{r['cy']}"        if r["cy"]   is not None else "  --"
            az_s  = f"{r['az']:+.2f}"   if r["az"]   is not None else "  None"
            el_s  = f"{r['el']:+.2f}"   if r["el"]   is not None else "  None"
            print(f"  {r['t']:>5.2f}s {pan_s:>7} {tilt_s:>6} {cx_s:>4} {cy_s:>4} "
                  f"{az_s:>7} {el_s:>6} {(r['mode'] or '?'):>6} {str(r['lock']):>4}",
                  flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=int,
                    default=[200, 100, 80, 80],
                    help="Thermal bbox X Y W H in 640x512 raw pixels")
    ap.add_argument("--duration", type=float, default=10.0)
    args = ap.parse_args()
    try:
        return asyncio.run(run(tuple(args.bbox), args.duration))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
