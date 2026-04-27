"""Smoke test: open IMX568Capture with SDK stream backend, grab N
frames, save the middle one + print stats. Confirms the
``eo.imx568_capture`` → ``eo.leopard_stream_capture`` → 32-bit helper
chain works end-to-end.

Two runs:
  1. AE auto (manual_exposure_ext=None) — what the live bench will
     boot into tomorrow. Should produce a non-saturated, well-exposed
     frame regardless of room lighting.
  2. AE manual @ 2000 — the calibration value from the reference match.
     Should match the previous reference comparison numerically.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eo.imx568_capture import IMX568Capture  # noqa: E402

OUT = REPO / "scripts" / "eo_snapshots" / "calibration" / "sdk_loop"


def run(label: str, manual_exposure_ext, n_frames: int = 25) -> None:
    print(f"\n=== {label} (manual_exposure_ext={manual_exposure_ext}) ===")
    cap = IMX568Capture(
        manual_exposure_ext=manual_exposure_ext,
    )
    t0 = time.time()
    cap.start()
    print(f"  start()  took {time.time()-t0:.1f}s, "
          f"sdk_stream_mode={cap._sdk_stream_mode}, "
          f"fourcc={cap._fourcc}")
    if not cap._sdk_stream_mode:
        print("  SDK stream did NOT engage — check helper / 32-bit Python")
        cap.stop()
        return
    means = []
    saved = None
    for i in range(n_frames):
        f = cap.grab()
        if f is None:
            print(f"  frame {i} = None (helper died?)")
            break
        means.append(float(f.mean()))
        if i == n_frames // 2:
            saved = f
    cap.stop()
    if saved is not None:
        out_path = OUT / f"smoke_{label}.bmp"
        cv2.imwrite(str(out_path), saved)
        b, g, r = (float(saved[..., 0].mean()),
                   float(saved[..., 1].mean()),
                   float(saved[..., 2].mean()))
        print(f"  middle frame B/G/R = {b:.2f} / {g:.2f} / {r:.2f}")
        print(f"  wrote: {out_path}")
    if means:
        print(f"  {len(means)} frames, mean span "
              f"{min(means):.2f}..{max(means):.2f}")


def main():
    # AE auto first (the new default)
    run("ae_auto", manual_exposure_ext=None, n_frames=25)
    # Then AE manual at the calibration value
    run("ae_manual_2000", manual_exposure_ext=2000, n_frames=25)
    return 0


if __name__ == "__main__":
    sys.exit(main())
