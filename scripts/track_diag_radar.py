"""Park at (pan=-13, tilt=0), wait for a radar-only target, engage
TRACK, log gimbal motion + target az/el for 10 seconds. Diagnoses
the operator-reported "radar tracking doesn't move the gimbal" bug.
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
import websockets


async def goto(ws, pan: float, tilt: float, settle_s: float = 1.5) -> None:
    await ws.send(json.dumps({
        "command": "gimbal_absolute",
        "pan_deg":  float(pan),
        "tilt_deg": float(tilt),
    }))
    deadline = asyncio.get_event_loop().time() + settle_s + 2.0
    while asyncio.get_event_loop().time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            break
        m = json.loads(raw)
        g = m.get("gimbal") or {}
        if (g.get("pan") is not None and g.get("tilt") is not None
            and abs(g["pan"]  - pan)  < 0.5
            and abs(g["tilt"] - tilt) < 0.5):
            break
    await asyncio.sleep(0.4)
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.05)
        except asyncio.TimeoutError:
            break


async def run() -> int:
    print("connecting...", flush=True)
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5.0)
        print("parking at pan=-13 tilt=0...", flush=True)
        await goto(ws, -13.0, 0.0, settle_s=2.0)

        # Wait up to 20s for a radar-only target to appear, picking
        # the first one we see.
        print("waiting up to 20s for radar target...", flush=True)
        target_id = None
        target_initial_az = None
        deadline = time.time() + 20.0
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            m = json.loads(raw)
            for t in (m.get("top_targets") or []):
                sensors = t.get("sensors") or []
                primary = t.get("primary") or ""
                if primary == "radar" and "eo" not in sensors and "thermal" not in sensors:
                    target_id = int(t["id"])
                    target_initial_az = float(t.get("az_deg") or 0)
                    target_initial_el = float(t.get("el_deg") or 0)
                    print(f"  found radar target id={target_id} "
                          f"az={target_initial_az:+.2f} "
                          f"el={target_initial_el:+.2f}", flush=True)
                    break
            if target_id is not None:
                break
        if target_id is None:
            print("** no radar-only target appeared **", flush=True)
            return 1

        # Engage TRACK
        print(f"-> TRACK on id={target_id}", flush=True)
        await ws.send(json.dumps({"command": "track", "track_id": target_id}))

        # Log for 10s
        rows = []
        t0 = time.time()
        last_log = t0
        while time.time() - t0 < 10.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            now = time.time()
            if now - last_log < 0.1: continue
            last_log = now
            m = json.loads(raw)
            g = m.get("gimbal") or {}
            lock = m.get("tracked_target_id")
            our = next((t for t in (m.get("top_targets") or [])
                        if t.get("id") == target_id), None)
            row = {
                "t": now - t0,
                "pan":  g.get("pan"),
                "tilt": g.get("tilt"),
                "mode": g.get("mode"),
                "lock": lock,
                "az":   (our or {}).get("az_deg"),
                "el":   (our or {}).get("el_deg"),
                "hits": (our or {}).get("hits"),
                "primary": (our or {}).get("primary"),
            }
            rows.append(row)

        # Release
        await ws.send(json.dumps({"command": "track", "track_id": None}))

        # Print
        print(f"\nlogged {len(rows)} samples:", flush=True)
        print(f"  {'t':>5}  {'pan':>7}  {'tilt':>6}  "
              f"{'az':>7}  {'el':>6}  {'hits':>4}  {'mode':>6}  {'lock':>5}",
              flush=True)
        for r in rows:
            pan_s = f"{r['pan']:+.2f}"  if r['pan']  is not None else "  None"
            tilt_s= f"{r['tilt']:+.2f}" if r['tilt'] is not None else " None"
            az_s  = f"{r['az']:+.2f}"   if r['az']   is not None else "  None"
            el_s  = f"{r['el']:+.2f}"   if r['el']   is not None else " None"
            hits_s= f"{r['hits']}"      if r['hits'] is not None else " --"
            print(f"  {r['t']:>5.2f}s {pan_s:>7} {tilt_s:>6} {az_s:>7} {el_s:>6} "
                  f"{hits_s:>4}  {(r['mode'] or '?'):>6}  {str(r['lock']):>5}",
                  flush=True)

        # Verdict
        good = [r for r in rows if r['pan'] is not None]
        if len(good) >= 2:
            dpan = good[-1]['pan']  - good[0]['pan']
            dtilt= good[-1]['tilt'] - good[0]['tilt']
            print(f"\nNET: dpan={dpan:+.2f}  dtilt={dtilt:+.2f}", flush=True)
            print(f"Initial target az={target_initial_az:+.2f} "
                  f"el={target_initial_el:+.2f}", flush=True)
        return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        sys.exit(130)
