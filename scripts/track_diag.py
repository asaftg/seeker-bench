"""Active diagnostic for the TRACK-overshoot bug.

Connects to a running seeker GUI's WebSocket, reads the live targets
list + gimbal state, picks the highest-confidence target, sends a
TRACK command, then logs the gimbal pan/tilt and the target's
reported az/el every 100 ms for ``--duration`` seconds.

Diagnostic value: side-by-side time series of (gimbal pan)
vs (target az_deg) tells us whether the closed-loop is converging
(both numbers approach each other / az approaches 0) or diverging
(one runs away). Direction and magnitude both visible at a glance.

Usage (from project root, with seeker already running):

    python scripts/track_diag.py [--duration 8] [--ws ws://127.0.0.1:8080/ws/sensors]

This script DOES NOT start or stop the seeker -- that's intentional
so we test the live system. Press Ctrl-C to bail mid-run; the script
sends a TRACK-release on its way out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# pip install websockets -- seeker already has this in requirements
try:
    import websockets
except ImportError:
    print("websockets not installed. pip install websockets", file=sys.stderr)
    sys.exit(2)


async def run(ws_url: str, duration_s: float, target_id: Optional[int]) -> int:
    print(f"connecting -> {ws_url}", flush=True)
    async with websockets.connect(ws_url, max_size=2**24) as ws:
        # Wait for the first frame so we have current gimbal + targets state
        first = await asyncio.wait_for(ws.recv(), timeout=5.0)
        msg = json.loads(first)
        gim = msg.get("gimbal") or {}
        targets = msg.get("top_targets") or msg.get("targets") or []
        print(f"connected. gimbal pan={gim.get('pan'):.1f} deg tilt={gim.get('tilt'):.1f} deg  "
              f"connected={gim.get('connected')}", flush=True)
        print(f"targets ({len(targets)}):", flush=True)
        for t in targets:
            print(f"  id={t.get('id')}  cls={t.get('target_class')}  "
                  f"az={t.get('az_deg'):+.2f} deg  el={t.get('el_deg'):+.2f} deg  "
                  f"conf={t.get('confidence'):.2f}  "
                  f"sensors={t.get('sensors')}", flush=True)

        # If no targets and no specific id requested, wait briefly --
        # the scene may just have nothing in view at this exact instant.
        if target_id is None and not targets:
            print("no targets yet -- waiting up to 10s for one...", flush=True)
            t_wait = time.time()
            while time.time() - t_wait < 10.0:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                m = json.loads(raw)
                tgts = m.get("top_targets") or m.get("targets") or []
                if tgts:
                    targets = tgts
                    gim = m.get("gimbal") or {}
                    print(f"target(s) appeared after {time.time()-t_wait:.1f}s. "
                          f"gimbal at pan={gim.get('pan'):.1f} tilt={gim.get('tilt'):.1f}",
                          flush=True)
                    for t in targets:
                        print(f"  id={t.get('id')} cls={t.get('target_class')} "
                              f"az={t.get('az_deg'):+.2f} el={t.get('el_deg'):+.2f} "
                              f"conf={t.get('confidence'):.2f}", flush=True)
                    break

        # Pick target
        if target_id is None:
            if not targets:
                print("no live targets after wait -- point camera at "
                      "something detectable, or pass --target-id N.",
                      flush=True)
                return 1
            # Highest confidence with non-trivial az (interesting trajectory)
            picked = max(targets,
                         key=lambda t: (abs(t.get("az_deg", 0)),
                                        t.get("confidence", 0)))
            target_id = int(picked["id"])
            print(f"\npicking target id={target_id} "
                  f"(az={picked.get('az_deg'):+.2f} deg, "
                  f"conf={picked.get('confidence'):.2f})", flush=True)
        else:
            print(f"\ntargeting id={target_id} as requested", flush=True)

        # Engage TRACK
        print("-> TRACK ON", flush=True)
        await ws.send(json.dumps({"command": "track", "track_id": target_id}))

        # Drain frames + log gimbal + target az every 100 ms
        t0 = time.time()
        next_log = t0
        rows: list[dict] = []
        while True:
            try:
                msg_raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                print("timeout waiting for frame", flush=True)
                continue
            elapsed = time.time() - t0
            if elapsed >= duration_s:
                break
            if time.time() < next_log:
                continue
            next_log = time.time() + 0.1
            m = json.loads(msg_raw)
            gim = m.get("gimbal") or {}
            tgts = m.get("top_targets") or m.get("targets") or []
            our = next((t for t in tgts if t.get("id") == target_id), None)
            row = {
                "t":     elapsed,
                "pan":   gim.get("pan"),
                "tilt":  gim.get("tilt"),
                "mode":  gim.get("mode"),
                "tracked_id": m.get("tracked_target_id"),
                "az":    (our or {}).get("az_deg"),
                "el":    (our or {}).get("el_deg"),
                "conf":  (our or {}).get("confidence"),
            }
            rows.append(row)

        # Release track on the way out
        print("-> TRACK OFF", flush=True)
        try:
            await ws.send(json.dumps({"command": "track", "track_id": None}))
        except Exception:
            pass

        # Print time series
        print(f"\nlogged {len(rows)} samples over {duration_s}s:", flush=True)
        print(f"  {'t':>5}  {'pan':>7}  {'tilt':>6}  {'az':>7}  {'el':>6}  "
              f"{'mode':>6}  {'lock':>4}  {'conf':>4}", flush=True)
        for r in rows:
            pan_s = f"{r['pan']:+.2f}"  if r["pan"]  is not None else "  None"
            tilt_s= f"{r['tilt']:+.2f}" if r["tilt"] is not None else " None"
            az_s  = f"{r['az']:+.2f}"   if r["az"]   is not None else "  None"
            el_s  = f"{r['el']:+.2f}"   if r["el"]   is not None else " None"
            conf_s= f"{r['conf']:.2f}"  if r["conf"] is not None else " None"
            print(f"  {r['t']:>5.2f}s {pan_s:>7} {tilt_s:>6} {az_s:>7} {el_s:>6} "
                  f"{(r['mode'] or '?'):>6} {str(r['tracked_id']):>4} {conf_s:>4}",
                  flush=True)

        # Verdict
        print("\n-- VERDICT --", flush=True)
        good = [r for r in rows if r["pan"] is not None and r["az"] is not None]
        if len(good) < 3:
            print("  not enough samples with both pan and az -- inconclusive.",
                  flush=True)
            return 0
        pan_total = good[-1]["pan"] - good[0]["pan"]
        az_first = good[0]["az"]
        # Camera-frame convention: target +az = right of boresight.
        # cameras_on_gimbal=true: gimbal pans +Δ to point at +az target,
        # then target's apparent az shrinks toward 0.
        # If pan changes opposite-sign of az_first, direction is INVERTED.
        # If pan changes same-sign as az_first AND |az| approaches 0,
        # tracking is correct.
        # If pan changes same-sign but |az| keeps growing, gimbal is
        # walking off (positive feedback / sign issue further upstream).
        az_final = good[-1]["az"]
        print(f"  initial: pan={good[0]['pan']:+.2f}  az={az_first:+.2f}",
              flush=True)
        print(f"  final:   pan={good[-1]['pan']:+.2f}  az={az_final:+.2f}",
              flush=True)
        print(f"  Δpan = {pan_total:+.2f} deg", flush=True)
        if pan_total * az_first < 0:
            print("  ** DIRECTION INVERTED -- pan moved AWAY from target.",
                  flush=True)
        elif abs(az_final) < abs(az_first) * 0.5:
            print("  OK converging -- target az shrunk by >50% during track.",
                  flush=True)
        elif abs(az_final) > abs(az_first) * 1.5:
            print("  ** DIVERGING -- target az grew while pan moved.",
                  flush=True)
        else:
            print("  ~ tracking but slow / not yet converged.",
                  flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="ws://127.0.0.1:8080/ws/sensors")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--target-id", type=int, default=None)
    args = ap.parse_args()
    try:
        return asyncio.run(run(args.ws, args.duration, args.target_id))
    except KeyboardInterrupt:
        print("\ninterrupted.", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
