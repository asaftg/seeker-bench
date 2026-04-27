"""Dump live PyAV-raw-Y stats: 30 frames, per-frame histogram + AE drift.

Two questions to answer empirically:

  1. Range — does the bridge actually deliver Y across 0..255, or is it
     limited (e.g. 16..235) or compressed by an internal gamma so whites
     never reach the rails? If max stays around 200, the user's "white
     bottles look bad" is the bridge gamma'ing highlights down.

  2. Drift — does the per-frame mean change on a static scene? That's
     the AE breathing the user noticed. A 5..10 unit mean drift over a
     2-second capture is what we need to suppress.

Saves a 1-D histogram of frame[15] and writes scripts/eo_snapshots/
diagnostic/live_stats.txt with everything.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eo.imx568_capture import IMX568Capture  # noqa: E402


def main() -> int:
    cap = IMX568Capture(device_index="auto")
    cap.start()

    out_dir = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture one frame every ~0.5 s for 60 s so we can see bridge AE
    # converge from cold-open (blown white) into its steady state.
    frames: list[np.ndarray] = []
    sample_times: list[float] = []
    t0 = time.time()
    last_sample = -1e9
    while time.time() - t0 < 60.0 and len(frames) < 120:
        f = cap.grab()
        if f is None:
            continue
        now = time.time()
        if now - last_sample >= 0.5:
            frames.append(f.copy())
            sample_times.append(now - t0)
            last_sample = now
    cap.stop()
    print(f"captured {len(frames)} frames over {time.time() - t0:.1f}s")

    stats_lines = []
    means = []
    for i, f in enumerate(frames):
        y = f[..., 0] if f.ndim == 3 else f  # B=G=R on PyAV path
        mn, mx = int(y.min()), int(y.max())
        mean = float(y.mean())
        std = float(y.std())
        # full-frame coarse 8-bin histogram so we can read it visually
        hist, _ = np.histogram(y, bins=8, range=(0, 256))
        hist_pct = (hist / y.size) * 100
        means.append(mean)
        ts = sample_times[i] if i < len(sample_times) else 0.0
        line = (f"  t={ts:5.1f}s f[{i:02d}] min={mn:3d} max={mx:3d} "
                f"mean={mean:6.1f} std={std:5.1f}  hist%=" +
                "[" + " ".join(f"{p:4.1f}" for p in hist_pct) + "]")
        stats_lines.append(line)

    print("\n=== PER-FRAME STATS (8-bin histogram, 0-31, 32-63, ..., 224-255) ===")
    for line in stats_lines:
        print(line)

    if len(means) > 1:
        m = np.array(means)
        drift_pp = float(m.max() - m.min())
        print(f"\nAE breathing: mean span = {drift_pp:.1f} (over {len(m)} frames)")
        print(f"             mean(mean)  = {m.mean():.1f}")
        print(f"             std(mean)   = {m.std():.2f}")

    # Save
    out_path = out_dir / "live_stats.txt"
    out_path.write_text(
        "\n".join(stats_lines) +
        f"\n\nAE breathing span: {(np.max(means)-np.min(means)):.1f}\n",
        encoding="utf-8",
    )
    print(f"\nsaved {out_path}")

    # Also save a couple of raw frames for visual inspection.
    if frames:
        import cv2
        for idx in (0, len(frames) // 2, len(frames) - 1):
            f = frames[idx]
            cv2.imwrite(str(out_dir / f"live_frame_{idx:02d}.png"), f)
        print(f"saved live_frame_00/{len(frames)//2:02d}/{len(frames)-1:02d}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
