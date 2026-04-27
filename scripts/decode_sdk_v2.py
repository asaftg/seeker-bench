"""Forensic decoder. The captured .bin has very specific stats:

  byte size  = 15,306,624 = W*H*3
  even bytes = mean 240, range 0..255
  odd bytes  = mean 0.0,  max = 1     <-- almost all zero

That means the first ~10.2MB looks like 16-bit LE mono with all values < 256
(LO byte rich, HI byte zero). The trailing 5.1MB is something else.

Try every interpretation and report.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H = 2472, 2064


def stats(name, img):
    if img.ndim == 3:
        means = [float(img[..., c].mean()) for c in range(img.shape[2])]
        print(f"  {name:42s} shape={img.shape} dtype={img.dtype} "
              f"mean(B,G,R)=({means[0]:.1f},{means[1]:.1f},{means[2]:.1f}) "
              f"min={img.min()} max={img.max()} std={float(img.std()):.1f}")
    else:
        print(f"  {name:42s} shape={img.shape} dtype={img.dtype} "
              f"mean={float(img.mean()):.2f} min={img.min()} max={img.max()} "
              f"std={float(img.std()):.2f}")


def go(p: Path):
    data = np.fromfile(str(p), dtype=np.uint8)
    print(f"\n=== {p.name}  size={data.size} ===")

    # Look at the byte distribution in 4 quartiles to find structure
    n = data.size
    q = n // 4
    for i in range(4):
        chunk = data[i * q:(i + 1) * q]
        print(f"  bytes[{i*q:>9d}..{(i+1)*q:>9d}] mean={chunk.mean():.1f} "
              f"min={chunk.min()} max={chunk.max()} std={chunk.std():.1f} "
              f"nonzero={int((chunk!=0).sum())}/{chunk.size}")

    # Treat first W*H*2 bytes as mono16 LE
    n2 = W * H * 2
    if data.size >= n2:
        m16 = np.frombuffer(data[:n2].tobytes(), dtype="<u2").reshape(H, W)
        m8 = np.clip(m16, 0, 255).astype(np.uint8)
        stats(f"mono16<256 -> u8", m8)
        cv2.imwrite(str(p.parent / f"v2_{p.stem}_mono16_clip.bmp"), m8)
        # Also right-shift 4 (RAW12)
        m8_rshift4 = (m16 >> 4).astype(np.uint8)
        stats(f"mono16 >>4", m8_rshift4)
        cv2.imwrite(str(p.parent / f"v2_{p.stem}_mono16_rshift4.bmp"), m8_rshift4)
        # And right-shift 6 (RAW10)
        m8_rshift6 = (m16 >> 6).astype(np.uint8)
        stats(f"mono16 >>6", m8_rshift6)
        cv2.imwrite(str(p.parent / f"v2_{p.stem}_mono16_rshift6.bmp"), m8_rshift6)
        # Treat as little-endian uint16 with values in upper 8 bits (BE-ish)
        # i.e. LO byte = real, HI byte zero
        # So just take LO byte
        lo = data[0:n2:2].reshape(H, W)
        stats(f"LO bytes only (mono8)", lo)
        cv2.imwrite(str(p.parent / f"v2_{p.stem}_lo_byte_mono.bmp"), lo)
        # Try debayering this lo-byte mono
        for code, name in [(cv2.COLOR_BAYER_BG2BGR, "BG"),
                           (cv2.COLOR_BAYER_GB2BGR, "GB"),
                           (cv2.COLOR_BAYER_RG2BGR, "RG"),
                           (cv2.COLOR_BAYER_GR2BGR, "GR")]:
            try:
                bgr = cv2.cvtColor(lo, code)
                stats(f"  debayer {name}", bgr)
                cv2.imwrite(str(p.parent / f"v2_{p.stem}_debayer_{name}.bmp"), bgr)
            except Exception as e:
                print(f"    debayer {name} failed: {e}")

    # Look at the "tail" 5.1MB
    tail = data[n2:]
    print(f"\n  TAIL ({tail.size} bytes): mean={tail.mean():.1f} "
          f"min={tail.min()} max={tail.max()} std={tail.std():.1f} "
          f"nonzero={int((tail!=0).sum())}/{tail.size}")
    # If tail is also W*H*1 = 5,099,328 -> close to W*H/2 sized chroma plane?
    nv = W * H
    if tail.size >= nv:
        chroma = tail[:nv].reshape(H, W)
        stats(f"  tail as full-size chroma", chroma)
        cv2.imwrite(str(p.parent / f"v2_{p.stem}_tail_full.bmp"), chroma)
    nv_half = (W * H) // 2
    if tail.size >= nv_half:
        chrm = tail[:nv_half]
        stats(f"  tail half ({chrm.size} bytes raw)",
              chrm.reshape(H // 2, W))


def main():
    base = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\scripts\eo_snapshots\calibration\sdk_capture_test")
    for nm in ["frame.bin", "frame_rgb888.bin"]:
        p = base / nm
        if p.exists():
            go(p)


if __name__ == "__main__":
    main()
