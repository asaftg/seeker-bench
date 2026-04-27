"""Decode the SDK-captured .bin file every which way to figure out
what format the SDK actually delivered.

Saves:
    decode_yuyv_y.bmp        — first W*H*2 bytes as YUYV, take Y plane
    decode_yuyv_color.bmp    — first W*H*2 bytes as YUYV, full color
    decode_bgr.bmp           — full W*H*3 bytes as BGR
    decode_rgb.bmp           — full W*H*3 bytes as RGB
    decode_raw12_bayer.bmp   — full bytes as RAW12 packed → debayered
    decode_full_y_only.bmp   — full bytes as YUYV (3 bytes per Y2-pair?)

Prints mean/min/max for each interpretation so we can rank them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 2472, 2064


def stats(name: str, img: np.ndarray) -> None:
    if img.ndim == 3:
        means = [float(img[..., c].mean()) for c in range(img.shape[2])]
        print(f"  {name:30s} shape={img.shape} dtype={img.dtype} "
              f"mean(B,G,R)={means[0]:.2f},{means[1]:.2f},{means[2]:.2f} "
              f"min={img.min()} max={img.max()}")
    else:
        print(f"  {name:30s} shape={img.shape} dtype={img.dtype} "
              f"mean={float(img.mean()):.2f} min={img.min()} max={img.max()}")


def try_yuyv(data: np.ndarray, out_dir: Path, tag: str) -> None:
    n_yuyv = W * H * 2
    if data.size < n_yuyv:
        print(f"  YUYV: not enough bytes ({data.size} < {n_yuyv})")
        return
    yuyv = data[:n_yuyv].reshape(H, W, 2)
    y = yuyv[..., 0].copy()
    stats(f"YUYV Y plane ({tag})", y)
    cv2.imwrite(str(out_dir / f"decode_{tag}_yuyv_y.bmp"), y)
    # Full color via OpenCV — needs (H, W, 2) two-channel input
    yuyv2 = data[:n_yuyv].reshape(H, W, 2)
    try:
        bgr = cv2.cvtColor(yuyv2, cv2.COLOR_YUV2BGR_YUYV)
        stats(f"YUYV color ({tag})", bgr)
        cv2.imwrite(str(out_dir / f"decode_{tag}_yuyv_color.bmp"), bgr)
    except Exception as e:
        print(f"    YUYV color failed: {e}")
    try:
        bgr2 = cv2.cvtColor(yuyv2, cv2.COLOR_YUV2BGR_UYVY)
        stats(f"UYVY color ({tag})", bgr2)
        cv2.imwrite(str(out_dir / f"decode_{tag}_uyvy_color.bmp"), bgr2)
    except Exception as e:
        print(f"    UYVY color failed: {e}")
    # Also try interpreting odd bytes as Y (UYVY style)
    y_alt = data[1:n_yuyv:2].copy()
    if y_alt.size == W * H:
        y_alt = y_alt.reshape(H, W)
        stats(f"odd-byte Y ({tag})", y_alt)
        cv2.imwrite(str(out_dir / f"decode_{tag}_uyvy_y.bmp"), y_alt)


def try_packed3(data: np.ndarray, out_dir: Path, tag: str) -> None:
    n3 = W * H * 3
    if data.size < n3:
        print(f"  packed3: not enough bytes ({data.size} < {n3})")
        return
    arr = data[:n3].reshape(H, W, 3)
    stats(f"as BGR ({tag})", arr)
    cv2.imwrite(str(out_dir / f"decode_{tag}_bgr.bmp"), arr)
    rgb_swapped = arr[..., ::-1].copy()
    stats(f"as RGB->BGR ({tag})", rgb_swapped)
    cv2.imwrite(str(out_dir / f"decode_{tag}_rgb_as_bgr.bmp"), rgb_swapped)


def try_raw12(data: np.ndarray, out_dir: Path, tag: str) -> None:
    """RAW12 packed: every 3 bytes encode 2 pixels (12+12 = 24 bits)."""
    n12 = (W * H * 3) // 2
    if data.size < n12:
        print(f"  RAW12: not enough bytes ({data.size} < {n12})")
        return
    chunk = data[:n12].reshape(-1, 3).astype(np.uint16)
    # pixel0 = (chunk[:,0] << 4) | (chunk[:,1] & 0x0F)
    # pixel1 = (chunk[:,2] << 4) | ((chunk[:,1] >> 4) & 0x0F)
    p0 = (chunk[:, 0] << 4) | (chunk[:, 1] & 0x0F)
    p1 = (chunk[:, 2] << 4) | ((chunk[:, 1] >> 4) & 0x0F)
    pixels = np.empty(p0.size + p1.size, dtype=np.uint16)
    pixels[0::2] = p0
    pixels[1::2] = p1
    img12 = pixels.reshape(H, W)
    img8 = (img12 >> 4).astype(np.uint8)
    stats(f"RAW12 packed mono ({tag})", img8)
    cv2.imwrite(str(out_dir / f"decode_{tag}_raw12_mono.bmp"), img8)
    # Try debayering
    for code, name in [
        (cv2.COLOR_BAYER_BG2BGR, "BG"),
        (cv2.COLOR_BAYER_GB2BGR, "GB"),
        (cv2.COLOR_BAYER_RG2BGR, "RG"),
        (cv2.COLOR_BAYER_GR2BGR, "GR"),
    ]:
        try:
            bgr = cv2.cvtColor(img8, code)
            stats(f"RAW12 deb-{name} ({tag})", bgr)
            cv2.imwrite(str(out_dir / f"decode_{tag}_raw12_deb_{name}.bmp"), bgr)
        except Exception as e:
            print(f"    debayer {name} failed: {e}")


def main(argv):
    if len(argv) < 2:
        bins = [
            r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\scripts\eo_snapshots\calibration\sdk_capture_test\frame.bin",
            r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\scripts\eo_snapshots\calibration\sdk_capture_test\frame_rgb888.bin",
        ]
    else:
        bins = argv[1:]

    for b in bins:
        p = Path(b)
        if not p.exists():
            print(f"missing: {p}")
            continue
        data = np.fromfile(str(p), dtype=np.uint8)
        print(f"\n=== {p.name}  size={data.size} ===")
        out_dir = p.parent
        tag = p.stem
        try_yuyv(data, out_dir, tag)
        try_packed3(data, out_dir, tag)
        try_raw12(data, out_dir, tag)


if __name__ == "__main__":
    main(sys.argv)
