"""SDK frame format identified:

  buffer = W*H*3 bytes (SDK reports bpp=24 but lies about layout)
  first  W*H*2 bytes = packed RAW12 little-endian, value = uint16 >> 4
  last   W*H*1 bytes = zero padding (uninitialized)

This script:
  1. Loads the .bin
  2. Treats first W*H*2 as <u2 → unpacks raw12 = u16 >> 4
  3. Stretches and debayers to BGR
  4. Saves both an AGC-stretched version (visible) and a raw-scaled version
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 2472, 2064


def auto_levels(img: np.ndarray, lo_pct: float = 0.5,
                hi_pct: float = 99.5) -> np.ndarray:
    p_lo, p_hi = np.percentile(img, [lo_pct, hi_pct])
    p_hi = max(p_hi, p_lo + 1)
    out = np.clip((img.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255.0,
                  0, 255).astype(np.uint8)
    return out


def decode(bin_path: Path) -> None:
    print(f"\n=== {bin_path.name} ===")
    raw_bytes = np.fromfile(str(bin_path), dtype=np.uint8)
    n2 = W * H * 2
    if raw_bytes.size < n2:
        print(f"file too small: {raw_bytes.size} < {n2}")
        return

    # Unpack as uint16 LE
    raw16 = np.frombuffer(raw_bytes[:n2].tobytes(),
                          dtype="<u2").reshape(H, W)
    # The SDK delivers (raw12 << 4) — recover the 12-bit value
    raw12 = (raw16 >> 4).astype(np.uint16)
    print(f"  raw12 stats: mean={raw12.mean():.2f} "
          f"min={raw12.min()} max={raw12.max()} "
          f"std={raw12.std():.2f}  range=[0..4095]")

    # Quick mono visualization at native scale
    mono_native = (raw12 / 16).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(bin_path.parent / f"v3_{bin_path.stem}_mono_native.bmp"),
                mono_native)

    # AGC stretched
    mono_agc = auto_levels(raw12)
    print(f"  mono_agc stats: mean={mono_agc.mean():.1f} "
          f"min={mono_agc.min()} max={mono_agc.max()}")
    cv2.imwrite(str(bin_path.parent / f"v3_{bin_path.stem}_mono_agc.bmp"),
                mono_agc)

    # Debayer at 8-bit (after stretch). Try all 4 patterns.
    for code, name in [(cv2.COLOR_BAYER_BG2BGR, "BG"),
                       (cv2.COLOR_BAYER_GB2BGR, "GB"),
                       (cv2.COLOR_BAYER_RG2BGR, "RG"),
                       (cv2.COLOR_BAYER_GR2BGR, "GR")]:
        bgr = cv2.cvtColor(mono_agc, code)
        b, g, r = [bgr[..., i].mean() for i in range(3)]
        print(f"  debayer {name}: B={b:.1f} G={g:.1f} R={r:.1f}")
        cv2.imwrite(str(bin_path.parent / f"v3_{bin_path.stem}_color_{name}.bmp"),
                    bgr)

    # Also try 16-bit debayer (preserves more detail) on raw12<<4
    raw_for_debayer16 = (raw12.astype(np.uint16) << 4)
    for code, name in [(cv2.COLOR_BAYER_BG2BGR, "BG"),
                       (cv2.COLOR_BAYER_GB2BGR, "GB"),
                       (cv2.COLOR_BAYER_RG2BGR, "RG"),
                       (cv2.COLOR_BAYER_GR2BGR, "GR")]:
        bgr16 = cv2.cvtColor(raw_for_debayer16, code)
        # AGC per channel-aware: stretch the joint percentiles
        p_lo, p_hi = np.percentile(bgr16, [0.5, 99.5])
        bgr8 = np.clip((bgr16.astype(np.float32) - p_lo) /
                       max(p_hi - p_lo, 1) * 255.0, 0, 255).astype(np.uint8)
        b, g, r = [bgr8[..., i].mean() for i in range(3)]
        print(f"  debayer16 {name}+AGC: B={b:.1f} G={g:.1f} R={r:.1f}")
        cv2.imwrite(str(bin_path.parent / f"v3_{bin_path.stem}_color16_{name}.bmp"),
                    bgr8)


def main():
    if len(sys.argv) > 1:
        bins = [Path(x) for x in sys.argv[1:]]
    else:
        base = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\scripts\eo_snapshots\calibration\sdk_capture_test")
        bins = [base / "frame.bin", base / "frame_rgb888.bin"]
    for b in bins:
        if b.exists():
            decode(b)
        else:
            print(f"missing: {b}")


if __name__ == "__main__":
    main()
