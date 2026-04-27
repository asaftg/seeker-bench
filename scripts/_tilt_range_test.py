"""Quick: park at (0, 0), capture frame, then go to (0, 89), capture
frame. MAD between the two confirms tilt has full mechanical range."""
import asyncio, base64, io, json, sys
import numpy as np
import websockets
from PIL import Image


async def goto_and_grab(ws, pan: float, tilt: float, settle_s: float = 2.5) -> tuple[np.ndarray, dict]:
    await ws.send(json.dumps({
        "command": "gimbal_absolute",
        "pan_deg":  float(pan),
        "tilt_deg": float(tilt),
    }))
    deadline = asyncio.get_event_loop().time() + settle_s + 3.0
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
    await asyncio.sleep(0.5)
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.05)
        except asyncio.TimeoutError:
            break
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        m = json.loads(raw)
        eo = m.get("eo") or {}
        if eo.get("connected") and eo.get("jpeg_b64"):
            data = base64.b64decode(eo["jpeg_b64"])
            img = np.asarray(Image.open(io.BytesIO(data)).convert("L"),
                             dtype=np.uint8)
            return img, m


async def main():
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5.0)
        print("Step 1: pan=0 tilt=0", flush=True)
        a, m_a = await goto_and_grab(ws, 0.0, 0.0)
        ga = m_a["gimbal"]
        print(f"  reported pan={ga['pan']:+.2f} tilt={ga['tilt']:+.2f}", flush=True)
        Image.fromarray(a).save("logs/tilt_range_at_0.png")

        print("Step 2: pan=0 tilt=89", flush=True)
        b, m_b = await goto_and_grab(ws, 0.0, 89.0)
        gb = m_b["gimbal"]
        print(f"  reported pan={gb['pan']:+.2f} tilt={gb['tilt']:+.2f}", flush=True)
        Image.fromarray(b).save("logs/tilt_range_at_89.png")

        if a.shape != b.shape:
            b_pil = Image.fromarray(b).resize((a.shape[1], a.shape[0]))
            b = np.asarray(b_pil, dtype=np.uint8)
        mad = float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())
        # Compare also to a-vs-a (self): zero by definition.
        print(f"\nMAD(at_0, at_89) = {mad:.1f}", flush=True)
        print(f"  Reference: identical frames -> MAD ~ 0-3", flush=True)
        print(f"             totally different scene -> MAD ~ 30-80+", flush=True)
        if mad < 5.0:
            print("  ** TILT MECHANICALLY NOT MOVING **", flush=True)
        elif mad < 15.0:
            print("  ** TILT MOVING ONLY A LITTLE -- mechanical slop or partial servo **", flush=True)
        else:
            print("  TILT moves through full range OK", flush=True)
        # Park back at safe middle.
        await ws.send(json.dumps({"command": "gimbal_absolute",
                                  "pan_deg": 0.0, "tilt_deg": 30.0}))
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
