"""Test mono-Y recovery from a YUY2-with-zero-chroma BGR frame.

Reads the raw bridge dump produced by ``eo_diagnostic.py`` and writes
out alternative luma extractions so we can see which actually preserves
sensor dynamic range. Run after ``python -m scripts.eo_diagnostic``.

The IMX568 is a mono sensor; the FX3 bridge ships YUY2 with U=V=0.
DirectShow's YUY2->BGR with U=V=0 produces:
    B = Y - 227   (clips to 0 for Y < 227)
    G = Y + 135   (clips to 255 for Y > 120)
    R = Y - 179   (clips to 0 for Y < 179)
so BGR2GRAY = 0.114B + 0.587G + 0.299R loses the entire Y in [120, 179]
plateau. G - 135 (clipped) recovers the low half exactly, and R + 179
(when G is saturated) recovers the upper half.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _stats(name: str, y: np.ndarray) -> dict:
    flat = y.reshape(-1)
    p1, p50, p99 = np.percentile(flat, [1, 50, 99]).tolist()
    return {
        "name": name,
        "min": int(flat.min()),
        "max": int(flat.max()),
        "mean": round(float(flat.mean()), 2),
        "std": round(float(flat.std()), 2),
        "p1": round(p1, 2),
        "p50": round(p50, 2),
        "p99": round(p99, 2),
    }


def recover_y_from_yuy2_bridge(bgr: np.ndarray) -> np.ndarray:
    """Reconstruct mono Y from a YUY2-with-zero-chroma BGR decode.

    Stitches G - 135 (low half, Y in [0, 120]) with R + 179 (upper half,
    Y in [179, 255]) and linearly interpolates the saturated [120, 179]
    plateau where both color channels are pinned.
    """
    bgr_i = bgr.astype(np.int16)
    g = bgr_i[..., 1]
    r = bgr_i[..., 2]

    # Low-half estimate: G - 135. Valid where G isn't saturated.
    y_low = np.clip(g - 135, 0, 255).astype(np.uint8)
    # High-half estimate: R + 179. Valid where R isn't pinned at 0.
    y_high = np.clip(r + 179, 0, 255).astype(np.uint8)

    # Choose y_high where R has lifted off zero (Y > 179), else y_low.
    use_high = r > 0
    y = np.where(use_high, y_high, y_low).astype(np.uint8)
    return y


def main() -> int:
    diag_dir = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
    raw_path = diag_dir / "01_raw_bridge.png"
    if not raw_path.exists():
        print(f"missing {raw_path} — run eo_diagnostic first")
        return 2

    bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3:
        print(f"could not read {raw_path}")
        return 3

    out = diag_dir
    variants = {
        "11_recover_g_minus_135.png":
            np.clip(bgr.astype(np.int16)[..., 1] - 135, 0, 255).astype(np.uint8),
        "12_recover_y_stitched.png":
            recover_y_from_yuy2_bridge(bgr),
        "13_recover_y_stretched.png": None,  # filled below
        "14_recover_y_gamma.png": None,
    }
    y = recover_y_from_yuy2_bridge(bgr)
    # Mild stretch on recovered Y: use 1..99 percentile of the recovered
    # signal, not 0.5..99.5, so we don't blow up sensor noise.
    lo, hi = np.percentile(y, [1, 99])
    if hi > lo:
        y_str = np.clip((y.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                        0, 255).astype(np.uint8)
    else:
        y_str = y
    variants["13_recover_y_stretched.png"] = y_str
    # Stretched + mild gamma 0.85 (midtone lift)
    inv = 1.0 / 0.85
    lut = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)],
                   dtype=np.uint8)
    variants["14_recover_y_gamma.png"] = cv2.LUT(y_str, lut)

    report = []
    for name, gray in variants.items():
        bgr_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(out / name), bgr_out)
        report.append(_stats(name, gray))

    (out / "recover_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {len(variants)} variants to {out}")
    print(f"{'name':40s} mean   std   p1   p99")
    for st in report:
        print(f"{st['name']:40s} {st['mean']:5.1f} {st['std']:5.1f}"
              f" {st['p1']:5.1f} {st['p99']:5.1f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
