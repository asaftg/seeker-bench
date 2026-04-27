"""Peek at the seeker's WS payload — print what's actually in the
EO/thermal frames + targets list right now."""
import asyncio, json, sys
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        # Eat 5 frames to see detection variation
        for i in range(5):
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            m = json.loads(raw)
            eo = m.get("eo") or {}
            tf = m.get("thermal") or {}
            tgts = m.get("targets") or []
            eo_dets = eo.get("detections") or []
            tf_dets = tf.get("detections") or []
            g = m.get("gimbal") or {}
            print(f"frame {i}: pan={g.get('pan'):.1f} tilt={g.get('tilt'):.1f}  "
                  f"EO_dets={len(eo_dets)}  TH_dets={len(tf_dets)}  "
                  f"fused={len(tgts)}", flush=True)
            for d in eo_dets[:3]:
                print(f"  EO: cls={d.get('classification', {}).get('target_class')} "
                      f"conf={d.get('classification', {}).get('confidence'):.2f} "
                      f"bbox={d.get('bbox')}", flush=True)

asyncio.run(main())
