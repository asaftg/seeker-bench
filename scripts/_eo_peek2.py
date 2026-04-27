"""Deeper peek — see thermal/eo connected flag, all top-level keys."""
import asyncio, json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        for i in range(3):
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            m = json.loads(raw)
            keys = list(m.keys())
            eo = m.get("eo") or {}
            tf = m.get("thermal") or {}
            print(f"=== frame {i} ===", flush=True)
            print(f"  ws keys: {keys}", flush=True)
            print(f"  eo.connected={eo.get('connected')} "
                  f"eo.detections={len(eo.get('detections') or [])} "
                  f"eo.width={eo.get('width')}", flush=True)
            print(f"  th.connected={tf.get('connected')} "
                  f"th.detections={len(tf.get('detections') or [])} "
                  f"th.width={tf.get('width')}", flush=True)
            print(f"  top_targets={len(m.get('top_targets') or [])}", flush=True)
            print(f"  tracks={len(m.get('tracks') or [])}", flush=True)
            print(f"  fused={len(m.get('fused') or [])}", flush=True)
            for t in (m.get('fused') or [])[:3]:
                print(f"    fused: id={t.get('id')} "
                      f"cls={t.get('target_class')} "
                      f"sensors={t.get('sensors')} "
                      f"hits={t.get('hits')} "
                      f"az={t.get('az_deg'):+.2f}", flush=True)

asyncio.run(main())
