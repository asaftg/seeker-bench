"""Probe what's actually in the IMX568 frame, per-channel.

Answers three questions:
  1. Is the frame genuinely saturated (max == 255 across the board)?
  2. Do the B/G/R channels carry the same content, or does DirectShow's
     YUY2→BGR decode scramble them?
  3. What does a hand-covered frame look like vs an uncovered frame?

Run it twice:
    python scripts/eo_dynamic_range.py uncovered
    # then cover lens with your hand:
    python scripts/eo_dynamic_range.py covered

Saves per-channel stats + a luma PNG to scripts/eo_snapshots/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "eo_snapshots"
OUT_DIR.mkdir(exist_ok=True)


def main(label: str) -> int:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2472)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2064)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warmup so we're measuring steady-state
    t0 = time.time()
    while time.time() - t0 < 1.0:
        cap.read()

    ok, f = cap.read()
    if not ok or f is None:
        print("FAILED TO GRAB")
        return 1
    cap.release()

    print(f"\n== {label} ==")
    print(f"shape={f.shape} dtype={f.dtype}")
    for ch, name in enumerate(("B", "G", "R")):
        c = f[:, :, ch]
        pcts = np.percentile(c, [1, 10, 50, 90, 99])
        print(f"  {name}: min={c.min():3d} max={c.max():3d} "
              f"mean={c.mean():6.1f}  pct[1,10,50,90,99]={[int(x) for x in pcts]}")

    luma = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    pcts = np.percentile(luma, [1, 10, 50, 90, 99])
    print(f"  Y: min={luma.min():3d} max={luma.max():3d} "
          f"mean={luma.mean():6.1f}  pct[1,10,50,90,99]={[int(x) for x in pcts]}")

    # Saturation check: what fraction of luma pixels are at 255?
    frac_255 = float((luma == 255).sum()) / luma.size * 100
    frac_0 = float((luma == 0).sum()) / luma.size * 100
    print(f"  pixels at 255: {frac_255:5.1f}%   at 0: {frac_0:5.1f}%")

    # Save the luma channel (histogram-stretched so we always see SOMETHING)
    lo, hi = float(luma.min()), float(luma.max())
    if hi > lo:
        stretched = np.clip((luma.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                            0, 255).astype(np.uint8)
    else:
        stretched = luma
    p = OUT_DIR / f"imx568_{label}_luma_stretched.png"
    cv2.imwrite(str(p), stretched)
    print(f"  -> {p}")

    # Also save the unstretched luma so we can see absolute brightness
    p2 = OUT_DIR / f"imx568_{label}_luma_raw.png"
    cv2.imwrite(str(p2), luma)
    print(f"  -> {p2}")
    return 0


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "frame"
    raise SystemExit(main(label))
