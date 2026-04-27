"""Sweep LPCamera quality knobs (Bits, SensorMode, ExposureExt) and save
captured frames + stats so we can visually pick the combo that best
matches/beats Leopard CameraTool quality on the user's reference scene.

Method per combo:
  1. Spawn 32-bit helper to apply the settings
  2. Wait 0.6s for FX3 to settle
  3. Open PyAV, capture ~20 frames, save the median frame as PNG
  4. Compute histogram + mean/std/saturated-pct
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eo.imx568_capture import IMX568Capture  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
PY32 = REPO / "tools" / "python311-x86" / "python.exe"
OUT = REPO / "scripts" / "eo_snapshots" / "diagnostic" / "quality_knobs"
OUT.mkdir(parents=True, exist_ok=True)


def apply(label: str, **kwargs) -> dict:
    cmd = [str(PY32), str(HELPER), "--ae", "off", "--json"]
    for k, v in kwargs.items():
        if v is None:
            continue
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return json.loads(out.stdout)
    except Exception:
        return {"raw": out.stdout, "err": out.stderr}


def capture_save(label: str, n_grab: int = 20) -> dict:
    cap = IMX568Capture(device_index="auto")
    cap.start()
    frames = []
    for _ in range(n_grab):
        f = cap.grab()
        if f is None:
            continue
        frames.append(f)
    cap.stop()
    if not frames:
        return {"err": "no frames"}
    # Use the LAST frame (settling) as the saved PNG, mean/std over last 5
    f = frames[-1]
    y = f[..., 0] if f.ndim == 3 else f
    last5 = [(fr[..., 0] if fr.ndim == 3 else fr) for fr in frames[-5:]]
    means = [float(yy.mean()) for yy in last5]
    stds = [float(yy.std()) for yy in last5]
    sats = [float((yy >= 254).mean()) for yy in last5]
    blacks = [float((yy <= 1).mean()) for yy in last5]
    out_path = OUT / f"{label}.png"
    cv2.imwrite(str(out_path), f)
    # also save a histogram-stretched version for visual
    p2, p98 = np.percentile(y, [2, 98])
    if p98 > p2:
        ystr = np.clip((y.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        cv2.imwrite(str(OUT / f"{label}_stretched.png"),
                    cv2.cvtColor(ystr, cv2.COLOR_GRAY2BGR))
    hist, _ = np.histogram(y, bins=8, range=(0, 256))
    return {
        "saved": str(out_path),
        "mean": round(float(np.mean(means)), 1),
        "std": round(float(np.mean(stds)), 1),
        "sat_pct": round(float(np.mean(sats)) * 100.0, 2),
        "black_pct": round(float(np.mean(blacks)) * 100.0, 2),
        "hist8": [int(h) for h in hist],
    }


COMBOS = [
    ("baseline_ext1000",        dict(exposure_ext=1000)),
    ("bits10_ext1000",          dict(exposure_ext=1000, bits=10)),
    ("bits12_ext1000",          dict(exposure_ext=1000, bits=12)),
    ("bits10_mode1_ext1000",    dict(exposure_ext=1000, bits=10, sensor_mode=1)),
    ("bits10_mode2_ext1000",    dict(exposure_ext=1000, bits=10, sensor_mode=2)),
    ("ext500",                  dict(exposure_ext=500)),
    ("ext2000",                 dict(exposure_ext=2000)),
    ("ext1000_native",          dict(exposure_ext=1000)),  # rerun for repeatability
]


def main() -> int:
    rows = []
    for label, params in COMBOS:
        print(f"\n=== {label}: {params} ===")
        helper_rep = apply(label, **params)
        time.sleep(0.6)
        cap_stats = capture_save(label)
        print(f"  helper: stage={helper_rep.get('stage')!r}")
        print(f"  capture: {cap_stats}")
        rows.append((label, cap_stats))
    print("\n=== SUMMARY ===")
    print(f"{'label':<25s}  {'mean':>5s}  {'std':>5s}  {'sat%':>5s}  {'blk%':>5s}  hist8")
    for label, st in rows:
        if "err" in st:
            print(f"{label:<25s}  {st['err']}")
            continue
        print(f"{label:<25s}  {st['mean']:>5.1f}  {st['std']:>5.1f}  "
              f"{st['sat_pct']:>5.2f}  {st['black_pct']:>5.2f}  {st['hist8']}")
    print(f"\nframes saved in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
