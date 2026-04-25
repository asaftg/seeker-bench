"""Probe the LI-IMX568-GMSL2 over USB3 at native resolution.

Now that the kit is confirmed streaming in CameraTool (2472x2064 @
10.8 fps after switching to a true USB3 port), this probe asks the
same thing of OpenCV's DSHOW backend:

  * does cv2.VideoCapture open at 2472x2064?
  * what FOURCC do we actually get?
  * what is the real payload shape / dtype / value range?
  * is the data non-zero (was all-zero while stuck on USB2)?
  * what RAW12 packing is the FX3 bridge using?

The packing question matters for the real capture class. UVC bridges
carry RAW12 inside a YUY2-labeled stream two common ways:

  A. Packed 12-bit  -> 3 bytes per 2 pixels  (frame_bytes == W*H*3/2)
  B. MIPI-aligned   -> 4 bytes per 2 pixels  (frame_bytes == W*H*2)

Width * height = 5_102_208 px.
  Option A -> 7_653_312 bytes
  Option B -> 10_204_416 bytes

We compare what cv2 hands back to both and print which one matches.

Saves the first good frame as:
  scripts/eo_snapshots/imx568_raw_bytes.bin  (raw payload)
  scripts/eo_snapshots/imx568_naive_8bit.png (naive MSB-byte view)
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "eo_snapshots"
OUT_DIR.mkdir(exist_ok=True)

W, H = 2472, 2064
EXPECTED_PIXELS = W * H
PACK_A_BYTES = EXPECTED_PIXELS * 3 // 2   # packed 12-bit
PACK_B_BYTES = EXPECTED_PIXELS * 2        # MIPI-aligned 4:2


def fourcc_str(v: float) -> str:
    n = int(v)
    if n <= 0:
        return "----"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def identify_packing(nbytes: int) -> str:
    if nbytes == PACK_A_BYTES:
        return "A (packed 12-bit, 3 bytes per 2 pixels)"
    if nbytes == PACK_B_BYTES:
        return "B (MIPI-aligned, 4 bytes per 2 pixels)"
    # Some bridges pad to 16-bit per pixel (RAW16 carrier)
    if nbytes == EXPECTED_PIXELS * 2:
        return "B' (16-bit-per-pixel carrier — same byte count as B)"
    return f"UNKNOWN ({nbytes} bytes — not a recognized RAW12 packing)"


def probe(idx: int = 1) -> int:
    print(f"== IMX568 USB3 probe on index {idx} ==")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("  OPEN FAILED")
        return 1
    try:
        # Force native resolution. Bridge advertises YUY2 carrier.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        afps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
        print(f"  negotiated: {aw}x{ah} {fc} @ {afps:.1f} fps")

        # Warmup
        t0 = time.time()
        while time.time() - t0 < 0.5:
            cap.read()

        # Measure 3s
        frames = 0
        nonzero_frames = 0
        first_good: np.ndarray | None = None
        byte_counts: list[int] = []
        t0 = time.time()
        while time.time() - t0 < 3.0:
            ok, f = cap.read()
            if not ok or f is None:
                continue
            frames += 1
            byte_counts.append(int(f.nbytes))
            if float(f.mean()) > 1.0:
                nonzero_frames += 1
                if first_good is None:
                    first_good = f.copy()
        elapsed = time.time() - t0
        meas_fps = frames / elapsed if elapsed > 0 else 0.0
        print(f"  measured: {meas_fps:.1f} fps over {elapsed:.1f}s, "
              f"{nonzero_frames}/{frames} frames had payload")

        if not byte_counts:
            print("  NO FRAMES RECEIVED")
            return 2
        most_common = max(set(byte_counts), key=byte_counts.count)
        print(f"  frame byte-count: {most_common} "
              f"(pixels {EXPECTED_PIXELS}, A={PACK_A_BYTES}, B={PACK_B_BYTES})")
        print(f"  -> packing: {identify_packing(most_common)}")

        if first_good is None:
            print("  No non-zero frame captured; cannot analyze payload.")
            print("  If this is surprising, confirm the lens cap is off and")
            print("  the scene is not extremely dim.")
            return 3

        f = first_good
        print(f"  first-good frame: shape={f.shape} dtype={f.dtype} "
              f"min={f.min()} max={f.max()} mean={f.mean():.1f} std={f.std():.1f}")

        # Dump raw bytes for offline analysis
        raw_path = OUT_DIR / "imx568_raw_bytes.bin"
        raw_path.write_bytes(f.tobytes())
        print(f"  wrote raw bytes -> {raw_path}")

        # Naive 8-bit view: if frame is already uint8 HxWx3 (YUY2 decoded
        # to BGR by DirectShow), just save Y channel. If it's 2-channel
        # YUY2 uint8, take even bytes. This is NOT the correct RAW12
        # unpack — just a sanity image so we can eyeball whether the
        # scene content is there.
        try:
            if f.ndim == 3 and f.shape[2] == 3:
                y = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            elif f.ndim == 3 and f.shape[2] == 2:
                y = f[:, :, 0]  # Y of YUY2
            elif f.ndim == 2:
                y = f
            else:
                y = None
            if y is not None:
                png_path = OUT_DIR / "imx568_naive_8bit.png"
                cv2.imwrite(str(png_path), y)
                print(f"  wrote naive 8-bit view -> {png_path}")
        except Exception as e:
            print(f"  naive view failed: {e}")

    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    raise SystemExit(probe(idx))
