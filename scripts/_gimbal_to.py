"""Quick utility: WS-connect to running seeker, command gimbal to a
specific (pan, tilt), watch for 5s.

Usage:  python scripts/_gimbal_to.py PAN TILT
"""
import asyncio, json, sys, time
import websockets

async def main(pan: float, tilt: float):
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        # Eat first frame
        await asyncio.wait_for(ws.recv(), timeout=5.0)
        # Send absolute angle command
        await ws.send(json.dumps({
            "command": "gimbal_absolute",
            "pan_deg":  pan,
            "tilt_deg": tilt,
        }))
        print(f"sent gimbal_absolute pan={pan} tilt={tilt}", flush=True)
        t0 = time.time()
        last_log = 0
        while time.time() - t0 < 5.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            now = time.time()
            if now - last_log < 0.5: continue
            last_log = now
            m = json.loads(raw)
            g = m.get("gimbal") or {}
            tgts = m.get("targets") or []
            print(f"  t={now-t0:4.1f}s  pan={g.get('pan'):+.1f}  "
                  f"tilt={g.get('tilt'):+.1f}  mode={g.get('mode')}  "
                  f"targets={len(tgts)}", flush=True)

if __name__ == "__main__":
    pan  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    tilt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    asyncio.run(main(pan, tilt))
