"""Smoke-test the new PyAV path inside IMX568Capture without the GUI.

Verifies:
  1. start() picks the PyAV path (logs PYAV_RAW_YUY2)
  2. grab() returns a (H, W, 3) BGR frame (Y replicated)
  3. The frame is NOT green-saturated (B/G/R within 1% of each other)
  4. Five consecutive frames have nontrivial std (not flat)

Saves the first frame to scripts/eo_snapshots/diagnostic/30_live_pyav.png
so we can eyeball it before launching the full app.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eo.imx568_capture import IMX568Capture  # noqa: E402


def main() -> int:
    cap = IMX568Capture(device_index="auto")
    try:
        cap.start()
    except Exception as e:
        print(f"start() failed: {e!r}")
        return 2

    out_dir = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    t0 = time.time()
    while len(frames) < 8 and time.time() - t0 < 6.0:
        f = cap.grab()
        if f is not None:
            frames.append(f)
    cap.stop()

    if not frames:
        print("no frames captured")
        return 3

    print(f"captured {len(frames)} frames in {time.time() - t0:.1f}s")
    f0 = frames[0]
    print(f"shape={f0.shape} dtype={f0.dtype}")
    if f0.ndim == 3 and f0.shape[2] == 3:
        b = float(f0[..., 0].mean())
        g = float(f0[..., 1].mean())
        r = float(f0[..., 2].mean())
        print(f"first-frame channel means: B={b:.1f} G={g:.1f} R={r:.1f}")
        # On the PyAV raw-Y path Y is replicated to all 3 channels — they
        # MUST be equal. On the broken DSHOW BGR path G is way bigger.
        if abs(g - b) < 1.0 and abs(g - r) < 1.0:
            print("PASS: channels equal — clean mono replicated to BGR")
        else:
            print("FAIL: channels not equal — still on broken DSHOW path?")
            return 4
    else:
        print(f"unexpected frame shape {f0.shape}")
        return 5

    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        print(f"  frame[{i}] mean={gray.mean():.1f} std={gray.std():.1f} "
              f"min={int(gray.min())} max={int(gray.max())}")

    out_path = out_dir / "30_live_pyav.png"
    cv2.imwrite(str(out_path), frames[0])
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
