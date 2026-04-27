"""Optical closed-loop verification of gimbal mechanical motion.

Sends absolute (pan, tilt) commands, captures the EO frame before and
after each move, and uses 1D cross-correlation on the row/column
brightness profiles to MEASURE how many pixels the image shifted.
Compares that to the expected pixel shift from the commanded angle
+ EO FOV. Pass/fail per axis per amplitude.

Why this exists: operator reported (2026-04-25) that gimbal movement
"feels off" especially in tilt -- could be a sticking servo, wrong
calibration coefficient, or a mechanical slip. Internal Maestro
'Get Position' just echoes what we sent; only optical feedback can
confirm the camera (and therefore the mount) actually moved.

Test sequence: park at a midpoint, then nudge pan +5, -5, +10, -10
and tilt +5, -5, +10, -10. After each move, settle 1.0s, capture,
compute shift, log expected vs actual. Pan/tilt are tested
independently so cross-coupling shows up clearly.

Usage from project root, with seeker running:
    python scripts/gimbal_optical_verify.py [--mid-pan 0] [--mid-tilt 30]
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

try:
    from PIL import Image
except ImportError:
    print("Pillow needed: pip install pillow", file=sys.stderr)
    sys.exit(2)


def decode_jpeg(b64: str) -> np.ndarray:
    """base64-jpeg -> luma uint8 (H, W)."""
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def measure_shift(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """2D FFT phase correlation -- locks the global pixel shift even
    when scene is mostly-uniform sky with one small feature (the
    naive 1D xcorr on row/col profiles fails badly on those).

    Returns (dx_px, dy_px) -- positive dx means content in `curr` is
    to the RIGHT of where it was in `prev`. We use a Hann window to
    suppress border ringing, then take the IFFT of the cross power
    spectrum normalised to unit magnitude. The peak's position mod
    image-size gives the shift; we wrap to signed.
    """
    if prev.shape != curr.shape:
        curr_pil = Image.fromarray(curr)
        curr_pil = curr_pil.resize((prev.shape[1], prev.shape[0]))
        curr = np.asarray(curr_pil, dtype=np.uint8)
    h, w = prev.shape
    a = prev.astype(np.float32)
    b = curr.astype(np.float32)
    # Subtract DC (image mean) so the (0,0) FFT bin doesn't dominate.
    a -= a.mean()
    b -= b.mean()
    # Hann window kills the border-discontinuity peak the FFT would
    # otherwise plant at the corner.
    wy = np.hanning(h).astype(np.float32)[:, None]
    wx = np.hanning(w).astype(np.float32)[None, :]
    hann = wy * wx
    a *= hann
    b *= hann
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    R /= (np.abs(R) + 1e-9)
    r = np.fft.ifft2(R).real
    # Peak position in (row, col) of the response. Wrap to signed
    # offsets in [-h/2, h/2), [-w/2, w/2).
    py, px = np.unravel_index(np.argmax(r), r.shape)
    if py > h // 2: py -= h
    if px > w // 2: px -= w
    # Phase corr returns the shift TO ALIGN curr ONTO prev; flip sign
    # so positive dx means content moved right in curr (matches the
    # docstring + the earlier 1D xcorr convention).
    return float(-px), float(-py)


async def grab_frame(ws) -> tuple[np.ndarray, dict]:
    """Pull frames until we have a non-None EO jpeg + gimbal state."""
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
        m = json.loads(raw)
        eo = m.get("eo") or {}
        if not eo.get("connected") or not eo.get("jpeg_b64"):
            continue
        img = decode_jpeg(eo["jpeg_b64"])
        return img, m


async def goto(ws, pan: float, tilt: float,
               settle_s: float = 1.5,
               tol_deg: float = 0.5) -> None:
    """Slew to absolute angles and wait until reported gimbal pan/tilt
    matches commanded within tol_deg, then drain backlog so the next
    frame we read is FRESH (post-settle). Without the drain, the
    queued WS messages from before+during the slew get returned
    first and the test reads stale frames."""
    await ws.send(json.dumps({
        "command": "gimbal_absolute",
        "pan_deg":  float(pan),
        "tilt_deg": float(tilt),
    }))
    deadline = asyncio.get_event_loop().time() + settle_s + 1.5
    last_pan = last_tilt = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            break
        m = json.loads(raw)
        g = m.get("gimbal") or {}
        last_pan, last_tilt = g.get("pan"), g.get("tilt")
        if last_pan is not None and last_tilt is not None:
            if (abs(last_pan - pan) < tol_deg
                    and abs(last_tilt - tilt) < tol_deg):
                break
    # Tiny extra settle for mechanical lash, then DRAIN any backlog
    # so the next read is post-settle.
    await asyncio.sleep(0.3)
    drained = 0
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.05)
            drained += 1
        except asyncio.TimeoutError:
            break


async def run(mid_pan: float, mid_tilt: float) -> int:
    print(f"connecting -> ws://127.0.0.1:8080/ws/sensors", flush=True)
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors",
                                  max_size=2**24) as ws:
        # Eat first frame
        await asyncio.wait_for(ws.recv(), timeout=5.0)
        # Park at midpoint
        print(f"\nparking at pan={mid_pan} tilt={mid_tilt} for baseline...",
              flush=True)
        await goto(ws, mid_pan, mid_tilt, settle_s=1.5)
        baseline_img, baseline_msg = await grab_frame(ws)
        eo = baseline_msg.get("eo") or {}
        gim = baseline_msg.get("gimbal") or {}
        h, w = baseline_img.shape
        hfov = float(eo.get("hfov_deg") or 11.05)
        vfov = float(eo.get("vfov_deg") or 9.23)
        px_per_deg_x = w / hfov
        px_per_deg_y = h / vfov
        print(f"baseline: pan={gim.get('pan'):+.2f} tilt={gim.get('tilt'):+.2f} "
              f"frame={w}x{h} hfov={hfov:.2f} vfov={vfov:.2f} "
              f"px/deg=({px_per_deg_x:.2f}, {px_per_deg_y:.2f})", flush=True)

        # Test sequence: pan and tilt, +/- 5 and +/- 10 deg.
        # Each test: from midpoint, command +d, capture, command -d
        # back to mid (verify return), capture again. Per direction.
        cases = [
            ("PAN  +5",  +5, 0),
            ("PAN  -5",  -5, 0),
            ("PAN  +10", +10, 0),
            ("PAN  -10", -10, 0),
            ("TILT +5",  0, +5),
            ("TILT -5",  0, -5),
            ("TILT +10", 0, +10),
            ("TILT -10", 0, -10),
        ]
        rows = []
        for label, dpan, dtilt in cases:
            tgt_pan  = mid_pan  + dpan
            tgt_tilt = mid_tilt + dtilt
            # Skip if would hit limits
            if not (-45 <= tgt_pan <= 45):  continue
            if not (0   <= tgt_tilt <= 90): continue
            # Re-park at midpoint and grab pre-frame
            await goto(ws, mid_pan, mid_tilt, settle_s=1.5)
            pre_img, pre_msg = await grab_frame(ws)
            pre_g = pre_msg.get("gimbal") or {}
            # Command the move
            await goto(ws, tgt_pan, tgt_tilt, settle_s=1.5)
            post_img, post_msg = await grab_frame(ws)
            post_g = post_msg.get("gimbal") or {}
            # Measure pixel shift. dx = shift required so prev features
            # land at curr positions; equivalently, image content moved
            # by +dx pixels right. When the camera pans RIGHT (gimbal
            # +pan), real-world content appears to move LEFT in image
            # (negative dx). So expected dx = -dpan * px_per_deg_x.
            dx, dy = measure_shift(pre_img, post_img)
            # MAD between pre and post tells us if the FRAMES
            # ACTUALLY DIFFER -- regardless of whether xcorr can
            # localise the shift. If MAD ~0 the camera image is
            # identical (gimbal didn't physically move OR scene
            # is uniform). If MAD is high the camera moved but
            # xcorr might fail on a low-feature scene.
            mad = float(np.abs(pre_img.astype(np.int32) -
                               post_img.astype(np.int32)).mean())
            # Save the first PAN move's pre/post for visual inspect.
            if label == "PAN  +5":
                Image.fromarray(pre_img).save(
                    "logs/optical_pre_pan5.png")
                Image.fromarray(post_img).save(
                    "logs/optical_post_pan5.png")
            if label == "TILT +5":
                Image.fromarray(pre_img).save(
                    "logs/optical_pre_tilt5.png")
                Image.fromarray(post_img).save(
                    "logs/optical_post_tilt5.png")
            exp_dx = -dpan  * px_per_deg_x
            # Tilt: gimbal +tilt (UP) -> world content moves DOWN in
            # image (+dy). So expected dy = +dtilt * px_per_deg_y.
            exp_dy = +dtilt * px_per_deg_y
            # Convert measured pixel shift back to angle for comparison
            meas_dpan_deg  = -dx / px_per_deg_x
            meas_dtilt_deg = +dy / px_per_deg_y
            row = {
                "label": label,
                "cmd_dpan":  dpan,
                "cmd_dtilt": dtilt,
                "rep_pan_pre":  pre_g.get("pan"),
                "rep_pan_post": post_g.get("pan"),
                "rep_tilt_pre": pre_g.get("tilt"),
                "rep_tilt_post":post_g.get("tilt"),
                "exp_dx":   exp_dx,
                "exp_dy":   exp_dy,
                "meas_dx":  dx,
                "meas_dy":  dy,
                "meas_dpan_deg":  meas_dpan_deg,
                "meas_dtilt_deg": meas_dtilt_deg,
            }
            rows.append(row)
            print(f"\n[{label}] commanded d=({dpan:+.0f}, {dtilt:+.0f})", flush=True)
            print(f"  reported pan: {pre_g.get('pan'):+.2f} -> "
                  f"{post_g.get('pan'):+.2f}  "
                  f"(d_rep={post_g.get('pan')-pre_g.get('pan'):+.2f})", flush=True)
            print(f"  reported tilt:{pre_g.get('tilt'):+.2f} -> "
                  f"{post_g.get('tilt'):+.2f}  "
                  f"(d_rep={post_g.get('tilt')-pre_g.get('tilt'):+.2f})", flush=True)
            print(f"  optical shift: dx={dx:+5.0f}px dy={dy:+5.0f}px  "
                  f"(expected dx={exp_dx:+5.0f} dy={exp_dy:+5.0f})  "
                  f"frame-MAD={mad:.1f}", flush=True)
            print(f"  -> measured  dpan={meas_dpan_deg:+5.2f} deg  "
                  f"dtilt={meas_dtilt_deg:+5.2f} deg", flush=True)
            err_pan  = meas_dpan_deg  - dpan
            err_tilt = meas_dtilt_deg - dtilt
            tag = "OK"
            if abs(err_pan) > 1.5 or abs(err_tilt) > 1.5:
                tag = "WARN -- error > 1.5 deg"
            print(f"  ERR: dpan={err_pan:+.2f} dtilt={err_tilt:+.2f}  [{tag}]",
                  flush=True)

        # Park back at midpoint
        await goto(ws, mid_pan, mid_tilt, settle_s=1.0)

        # Summary
        print("\n" + "=" * 64, flush=True)
        print("SUMMARY:", flush=True)
        print(f"  {'case':>10}  {'cmd dpan':>8} {'cmd dtilt':>9}  "
              f"{'meas dpan':>9} {'meas dtilt':>10}  {'err pan':>7} {'err tilt':>8}",
              flush=True)
        for r in rows:
            print(f"  {r['label']:>10}  {r['cmd_dpan']:+8.1f} {r['cmd_dtilt']:+9.1f}  "
                  f"{r['meas_dpan_deg']:+9.2f} {r['meas_dtilt_deg']:+10.2f}  "
                  f"{r['meas_dpan_deg']-r['cmd_dpan']:+7.2f} "
                  f"{r['meas_dtilt_deg']-r['cmd_dtilt']:+8.2f}", flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mid-pan", type=float, default=0.0)
    ap.add_argument("--mid-tilt", type=float, default=30.0)
    args = ap.parse_args()
    try:
        return asyncio.run(run(args.mid_pan, args.mid_tilt))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
