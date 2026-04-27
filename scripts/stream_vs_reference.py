"""Render explicit frame-to-frame comparison: a streamed SDK frame
vs the Leopard CameraTool reference BMP. Produces a thumbnail-size
PNG so the Read tool can render it visually.

Also prints per-channel B/G/R means so the numerical comparison is
adjacent to the visual one in the log.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CAL = REPO / "scripts" / "eo_snapshots" / "calibration"
SDK = CAL / "sdk_loop"

REF_BMP = CAL / "leopard_reference.bmp"
# Pick a settled streamed frame mid-run
STREAM_BMP = SDK / "stream_settled_014.bmp"
OUT_PNG = SDK / "stream_vs_reference.png"
OUT_PNG_AGC = SDK / "stream_vs_reference_agc.png"

# Allow CLI override so we can compare e.g. the AE-auto smoke output
# against the same reference: `python scripts/stream_vs_reference.py
# scripts/eo_snapshots/calibration/sdk_loop/smoke_ae_auto.bmp`
if len(sys.argv) > 1:
    STREAM_BMP = Path(sys.argv[1])
    OUT_PNG = SDK / f"compare_{STREAM_BMP.stem}_vs_ref.png"
    OUT_PNG_AGC = SDK / f"compare_{STREAM_BMP.stem}_vs_ref_agc.png"


def channel_means(img: np.ndarray) -> tuple[float, float, float]:
    return (float(img[..., 0].mean()),
            float(img[..., 1].mean()),
            float(img[..., 2].mean()))


def agc(img: np.ndarray, lo_pct=0.5, hi_pct=99.5) -> np.ndarray:
    p_lo, p_hi = np.percentile(img, [lo_pct, hi_pct])
    p_hi = max(p_hi, p_lo + 1)
    return np.clip((img.astype(np.float32) - p_lo) /
                   (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)


def main():
    if not STREAM_BMP.exists():
        print(f"missing: {STREAM_BMP}")
        return 1
    if not REF_BMP.exists():
        print(f"missing: {REF_BMP}")
        return 1

    ours = cv2.imread(str(STREAM_BMP))   # BGR
    ref = cv2.imread(str(REF_BMP))       # BGR

    print(f"ours shape: {ours.shape}  ref shape: {ref.shape}")
    om = channel_means(ours)
    rm = channel_means(ref)
    print(f"OURS  B/G/R = {om[0]:6.2f} / {om[1]:6.2f} / {om[2]:6.2f}")
    print(f"REF   B/G/R = {rm[0]:6.2f} / {rm[1]:6.2f} / {rm[2]:6.2f}")
    print(f"DIFF  B/G/R = {om[0]-rm[0]:+6.2f} / {om[1]-rm[1]:+6.2f} / "
          f"{om[2]-rm[2]:+6.2f}")

    # Resize to thumbnail. Both images: 2064 x 2472 (HxW)
    h2 = 540
    w2 = int(ours.shape[1] * h2 / ours.shape[0])
    o_th = cv2.resize(ours, (w2, h2))
    r_th = cv2.resize(ref, (w2, h2))
    sep = np.full((h2, 6, 3), 255, dtype=np.uint8)
    panel = np.hstack([o_th, sep, r_th])
    cv2.putText(panel, f"OURS (SDK stream frame, BG debayer)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(panel, f"B={om[0]:.1f} G={om[1]:.1f} R={om[2]:.1f}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(panel, "LEOPARD REFERENCE BMP", (w2 + 16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(panel, f"B={rm[0]:.1f} G={rm[1]:.1f} R={rm[2]:.1f}",
                (w2 + 16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    cv2.imwrite(str(OUT_PNG), panel)
    print(f"wrote: {OUT_PNG}")

    # Also AGC version for visibility
    o_agc = agc(ours)
    r_agc = agc(ref)
    o_agc_th = cv2.resize(o_agc, (w2, h2))
    r_agc_th = cv2.resize(r_agc, (w2, h2))
    panel2 = np.hstack([o_agc_th, sep, r_agc_th])
    cv2.putText(panel2, "OURS AGC (visibility)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(panel2, "LEOPARD AGC", (w2 + 16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imwrite(str(OUT_PNG_AGC), panel2)
    print(f"wrote: {OUT_PNG_AGC}")

    # Numerical match summary
    ratio_o = np.array(om) / max(sum(om), 1)
    ratio_r = np.array(rm) / max(sum(rm), 1)
    ratio_diff = float(np.abs(ratio_o - ratio_r).sum())
    abs_mean_diff = float(np.abs(np.array(om) - np.array(rm)).mean())
    print(f"\nratio_diff (B/G/R proportions) = {ratio_diff:.5f} "
          f"(0 = identical color balance)")
    print(f"abs mean diff (raw value) = {abs_mean_diff:.3f}")
    if ratio_diff < 0.01 and abs_mean_diff < 2.0:
        print("\n*** FRAME-TO-FRAME MATCH CONFIRMED ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
