"""Reproduce Leopard's exact color path: take raw12, clip to uint8
(no AGC), debayer, save BMP. Compare side-by-side with the
leopard_reference.bmp.

Leopard reference BMP per-channel means: B=22.4, G=38.8, R=31.1
Our raw12 has the same distribution (mean=32.6, max=255). So if we
debayer the raw12 directly as uint8, we should match Leopard's BMP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "scripts" / "eo_snapshots" / "calibration" / "sdk_loop"
W, H = 2472, 2064


def decode_raw12(bin_path: Path) -> np.ndarray:
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    n2 = W * H * 2
    img16 = np.frombuffer(raw[:n2].tobytes(), dtype="<u2").reshape(H, W)
    return (img16 >> 4).astype(np.uint16)


def main():
    bin_path = OUT / "frame_e2000_w12.bin"
    if not bin_path.exists():
        # Pick any captured frame
        bins = sorted(OUT.glob("frame_*.bin"))
        if not bins:
            print("no captured frames yet")
            return 1
        bin_path = bins[-1]
    print("using:", bin_path.name)

    raw12 = decode_raw12(bin_path)
    print(f"raw12 stats: mean={raw12.mean():.2f} std={raw12.std():.2f} "
          f"min={raw12.min()} max={raw12.max()}")

    # Clip raw12 to uint8 directly (no AGC) — same as Leopard saves
    raw8 = np.clip(raw12, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "match_raw8_mono.bmp"), raw8)

    ref_bgr = cv2.imread(str(REPO /
        "scripts/eo_snapshots/calibration/leopard_reference.bmp"))
    ref_means = [ref_bgr[..., c].mean() for c in range(3)]
    print(f"reference BMP B/G/R: {ref_means[0]:.2f} {ref_means[1]:.2f} "
          f"{ref_means[2]:.2f}")

    best = None
    for code, name in [(cv2.COLOR_BAYER_BG2BGR, "BG"),
                       (cv2.COLOR_BAYER_GB2BGR, "GB"),
                       (cv2.COLOR_BAYER_RG2BGR, "RG"),
                       (cv2.COLOR_BAYER_GR2BGR, "GR")]:
        bgr = cv2.cvtColor(raw8, code)
        m = [bgr[..., c].mean() for c in range(3)]
        ratio_diff = sum(abs(m[i]/sum(m) - ref_means[i]/sum(ref_means))
                         for i in range(3))
        print(f"  Bayer {name}: B={m[0]:.2f} G={m[1]:.2f} R={m[2]:.2f} "
              f"ratio_diff={ratio_diff:.4f}")
        cv2.imwrite(str(OUT / f"match_color_{name}.bmp"), bgr)
        if best is None or ratio_diff < best["ratio_diff"]:
            best = {"name": name, "ratio_diff": ratio_diff,
                    "bgr": bgr, "means": m}

    print(f"\nBest pattern: {best['name']} (ratio_diff={best['ratio_diff']:.4f})")
    print(f"  ours B/G/R: {best['means'][0]:.2f} {best['means'][1]:.2f} "
          f"{best['means'][2]:.2f}")
    print(f"  ref  B/G/R: {ref_means[0]:.2f} {ref_means[1]:.2f} "
          f"{ref_means[2]:.2f}")

    # Save chosen result + comparison panel
    cv2.imwrite(str(OUT / "match_best_color.bmp"), best["bgr"])
    h2 = 768
    w2 = int(W * h2 / H)
    o = cv2.resize(best["bgr"], (w2, h2))
    r = cv2.resize(ref_bgr, (w2, h2))
    sep = np.full((h2, 6, 3), 255, dtype=np.uint8)
    panel = np.hstack([o, sep, r])
    cv2.putText(panel, f"OURS (Bayer {best['name']}, raw8)", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(panel, "LEOPARD BMP", (w2 + 18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / "match_compare_color.png"), panel)

    # Also AGC version of best for visibility
    p_lo, p_hi = np.percentile(best["bgr"], [0.5, 99.5])
    agc = np.clip((best["bgr"].astype(np.float32) - p_lo) /
                  max(p_hi - p_lo, 1) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "match_best_color_agc.bmp"), agc)
    # AGC ref similarly
    rp_lo, rp_hi = np.percentile(ref_bgr, [0.5, 99.5])
    ragc = np.clip((ref_bgr.astype(np.float32) - rp_lo) /
                   max(rp_hi - rp_lo, 1) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "match_ref_color_agc.bmp"), ragc)
    o2 = cv2.resize(agc, (w2, h2))
    r2 = cv2.resize(ragc, (w2, h2))
    panel2 = np.hstack([o2, sep, r2])
    cv2.putText(panel2, f"OURS AGC (Bayer {best['name']})", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(panel2, "LEOPARD AGC", (w2 + 18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / "match_compare_color_agc.png"), panel2)
    print("wrote:", OUT / "match_compare_color.png")
    print("wrote:", OUT / "match_compare_color_agc.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
