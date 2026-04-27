"""Gimbal small-step sensitivity test.

Uses the THERMAL panel (37 deg HFOV at the 'wide' zoom preset) instead
of EO because:
  * thermal at 37 deg = 17.3 px/deg, EO at 11 deg = 112 px/deg
  * for a 1 deg move the thermal frame shifts only ~17 px (well
    inside the FFT correlation's measurable range), while EO would
    shift ~112 px and large multi-degree moves blow past the image
    edges and become unmeasurable
  * smaller pixel scale = much better signal for detecting whether
    the gimbal really moved 0.5 / 1 / 2 deg or whether the motion
    was eaten by mechanical slop

Strategy:
  * park at (pan=0, tilt=mid) and capture a thermal baseline
  * for each amplitude in [0.5, 1, 2, 5] deg, command pan+/- and
    tilt+/-, capture, measure pixel shift via FFT phase correlation,
    convert back to degrees and compare to commanded
  * print a per-amplitude table so the operator can see the smallest
    amplitude that still produces motion (the deadband / slop floor)

Operator setup notes (2026-04-26):
  * gimbal mount lighter without the front case
  * scene is fully detailed for all pan and tilt up to ~25 deg
  * thermal preset switched to 'wide' (37 deg) automatically by
    this script via /api/config/thermal

Usage:  python scripts/gimbal_sensitivity.py [--mid-tilt 12]
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import io
import json
import sys
import time

import numpy as np
import websockets
from PIL import Image


def decode_jpeg(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def measure_shift_fft(prev: np.ndarray,
                      curr: np.ndarray) -> tuple[float, float, float]:
    """FFT phase correlation. Returns (dx_px, dy_px, peak_strength).

    peak_strength is the response value at the peak; useful as a
    quality indicator (>~0.05 = solid lock, <~0.01 = noise).
    """
    if prev.shape != curr.shape:
        cp = Image.fromarray(curr).resize((prev.shape[1], prev.shape[0]))
        curr = np.asarray(cp, dtype=np.uint8)
    h, w = prev.shape
    a = prev.astype(np.float32) - prev.mean()
    b = curr.astype(np.float32) - curr.mean()
    wy = np.hanning(h).astype(np.float32)[:, None]
    wx = np.hanning(w).astype(np.float32)[None, :]
    a *= wy * wx
    b *= wy * wx
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R /= (np.abs(R) + 1e-9)
    r = np.fft.ifft2(R).real
    py, px = np.unravel_index(np.argmax(r), r.shape)
    peak = float(r[py, px])
    if py > h // 2: py -= h
    if px > w // 2: px -= w
    # Sign convention: positive dx = content in `curr` is to the
    # RIGHT of where it was in `prev` (i.e. image content moved right).
    return float(-px), float(-py), peak


async def goto(ws, pan: float, tilt: float, settle_s: float = 1.5) -> dict:
    """Slew to absolute angles, wait for reported pose to match,
    drain backlog, return last gimbal-state dict."""
    await ws.send(json.dumps({
        "command": "gimbal_absolute",
        "pan_deg":  float(pan),
        "tilt_deg": float(tilt),
    }))
    deadline = asyncio.get_event_loop().time() + settle_s + 2.0
    last_g: dict = {}
    while asyncio.get_event_loop().time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            break
        m = json.loads(raw)
        g = m.get("gimbal") or {}
        if g: last_g = g
        if (g.get("pan") is not None and g.get("tilt") is not None
            and abs(g["pan"]  - pan)  < 0.3
            and abs(g["tilt"] - tilt) < 0.3):
            break
    # Mechanical settle then drain backlog so next read is fresh.
    await asyncio.sleep(0.5)
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.05)
        except asyncio.TimeoutError:
            break
    return last_g


async def grab_thermal(ws) -> tuple[np.ndarray, dict]:
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        m = json.loads(raw)
        tf = m.get("thermal") or {}
        if tf.get("connected") and tf.get("jpeg_b64"):
            return decode_jpeg(tf["jpeg_b64"]), m


async def set_thermal_zoom(ws, preset: str) -> None:
    """Switch thermal to a known zoom preset via the existing API.
    Done over HTTP, not the WS, so it's a separate call."""
    import urllib.request
    body = json.dumps({"zoom_preset": preset}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/config/thermal",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=2.0).read()
    except Exception as e:
        print(f"  ! thermal zoom set failed: {e}", flush=True)


async def run(mid_tilt: float) -> int:
    print("connecting -> ws://127.0.0.1:8080/ws/sensors", flush=True)
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        await asyncio.wait_for(ws.recv(), timeout=5.0)

        # Switch thermal to wide zoom (37 deg) for sub-degree
        # measurement headroom.
        print("switching thermal zoom -> 'wide' (37 deg HFOV)...", flush=True)
        await set_thermal_zoom(ws, "wide")
        await asyncio.sleep(1.0)

        # Park at midpoint and read frame metadata
        print(f"parking at pan=0 tilt={mid_tilt}...", flush=True)
        await goto(ws, 0.0, mid_tilt, settle_s=1.5)
        # Drain again after the zoom switch so we have current FOV
        # in the next thermal frame.
        while True:
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.05)
            except asyncio.TimeoutError:
                break
        baseline_img, baseline_msg = await grab_thermal(ws)
        tf = baseline_msg.get("thermal") or {}
        h, w = baseline_img.shape
        hfov = float(tf.get("hfov_deg") or 75.0)
        vfov = float(tf.get("vfov_deg") or 60.0)
        px_per_deg_x = w / hfov
        px_per_deg_y = h / vfov
        print(f"baseline: thermal frame {w}x{h}  "
              f"hfov={hfov:.2f} vfov={vfov:.2f}  "
              f"px/deg=({px_per_deg_x:.2f}, {px_per_deg_y:.2f})", flush=True)

        amplitudes = [0.5, 1.0, 2.0, 5.0]
        rows = []
        for amp in amplitudes:
            for axis, dpan, dtilt in [
                ("PAN +",  +amp, 0.0),
                ("PAN -",  -amp, 0.0),
                ("TILT +", 0.0, +amp),
                ("TILT -", 0.0, -amp),
            ]:
                tgt_pan  = 0.0 + dpan
                tgt_tilt = mid_tilt + dtilt
                if not (-45 <= tgt_pan <= 45):  continue
                if not (0 <= tgt_tilt <= 25):   continue
                # re-park
                await goto(ws, 0.0, mid_tilt, settle_s=1.0)
                pre_img, _ = await grab_thermal(ws)
                # commanded move
                g_post = await goto(ws, tgt_pan, tgt_tilt, settle_s=1.5)
                post_img, _ = await grab_thermal(ws)
                dx, dy, peak = measure_shift_fft(pre_img, post_img)
                # expected pixel shift (camera moves opposite of content)
                exp_dx = -dpan  * px_per_deg_x
                exp_dy = +dtilt * px_per_deg_y
                # measured deg
                meas_dpan_deg  = -dx / px_per_deg_x
                meas_dtilt_deg = +dy / px_per_deg_y
                mad = float(np.abs(pre_img.astype(np.int32)
                                   - post_img.astype(np.int32)).mean())
                row = {
                    "label":   f"{axis}{amp}",
                    "cmd_dpan": dpan, "cmd_dtilt": dtilt,
                    "meas_dpan": meas_dpan_deg, "meas_dtilt": meas_dtilt_deg,
                    "mad": mad, "peak": peak,
                }
                rows.append(row)
                tag = "  "
                if (axis.startswith("PAN") and abs(meas_dpan_deg - dpan) < amp*0.3
                    or axis.startswith("TILT") and abs(meas_dtilt_deg - dtilt) < amp*0.3):
                    tag = "OK"
                elif mad < 4.0:
                    tag = "** NO MOTION (MAD < 4) **"
                else:
                    tag = "?? PARTIAL"
                print(f"  {axis}{amp:<4}  cmd=({dpan:+.1f},{dtilt:+.1f})  "
                      f"meas=({meas_dpan_deg:+.2f},{meas_dtilt_deg:+.2f})  "
                      f"mad={mad:5.1f}  peak={peak:.3f}  {tag}",
                      flush=True)

        await goto(ws, 0.0, mid_tilt, settle_s=1.0)

        # Summary
        print("\n" + "=" * 68, flush=True)
        print(f"  {'case':>8}  {'cmd':>10}  {'measured':>14}  "
              f"{'err_amp':>8}  {'mad':>5}", flush=True)
        for r in rows:
            cmd_str  = f"({r['cmd_dpan']:+.1f},{r['cmd_dtilt']:+.1f})"
            meas_str = f"({r['meas_dpan']:+.2f},{r['meas_dtilt']:+.2f})"
            target_amp = (r['cmd_dpan']  if abs(r['cmd_dpan'])  > 0
                          else r['cmd_dtilt'])
            meas_amp   = (r['meas_dpan'] if abs(r['cmd_dpan'])  > 0
                          else r['meas_dtilt'])
            err_amp = meas_amp - target_amp
            print(f"  {r['label']:>8}  {cmd_str:>10}  {meas_str:>14}  "
                  f"{err_amp:+7.2f}  {r['mad']:5.1f}", flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid-tilt", type=float, default=12.0,
                    help="midpoint tilt deg (kept inside the visible band)")
    args = ap.parse_args()
    try:
        return asyncio.run(run(args.mid_tilt))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
