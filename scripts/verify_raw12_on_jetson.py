"""Capture a live frame on Jetson, treat as RAW12 u16, save stretched PNG.

This is the proof-of-concept on hardware. If the live frame produces the
same kind of clean stretched mono image that the Windows .raw analysis
showed, the entire EO quality gap on Linux is closed.
"""
from __future__ import annotations
import os, sys, subprocess
import numpy as np

DEV = "/dev/video0"
W, H = 2472, 2064
RAW = "/tmp/raw12_proof.bin"

# Set format and capture a few frames
subprocess.run(
    ["v4l2-ctl", f"--device={DEV}",
     f"--set-fmt-video=width={W},height={H},pixelformat=YUYV"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
if os.path.exists(RAW):
    os.remove(RAW)
subprocess.run(
    ["timeout", "5", "v4l2-ctl", f"--device={DEV}",
     "--stream-mmap=4", "--stream-count=4",
     f"--stream-to={RAW}"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

if not os.path.exists(RAW) or os.path.getsize(RAW) < 1000:
    print("FAIL: no frame captured")
    sys.exit(1)

b = np.fromfile(RAW, dtype=np.uint8)
print(f"Captured: {len(b)} bytes ({len(b) // (W*H*2)} frames)")
fb = W * H * 2
n_frames = len(b) // fb
last = b[(n_frames-1) * fb : n_frames * fb]

# Interpretation A: YUYV (current Linux behaviour)
even = last[::2]; odd = last[1::2]
print(f"\n=== Current Linux interpretation (YUYV mono) ===")
print(f"  Y bytes:  mean={even.mean():.1f}  std={even.std():.1f}  max={even.max()}")
print(f"  UV bytes: mean={odd.mean():.1f}  std={odd.std():.1f}  max={odd.max()}")

# Interpretation B: RAW12 u16 LE
u16 = np.frombuffer(last.tobytes(), dtype="<u2").reshape(H, W)
print(f"\n=== RAW12 reinterpretation (u16 LE) ===")
print(f"  u16 values: mean={u16.mean():.1f}  std={u16.std():.1f}  max={u16.max()}  min={u16.min()}")
print(f"  >> {'RAW12 confirmed (max <= 4095)' if u16.max() <= 4095 else 'Values > 4095 - NOT RAW12'}")
print(f"  histogram (deciles): {np.percentile(u16, [10, 25, 50, 75, 90, 99]).astype(int).tolist()}")

# Bayer phase check
p_ee = u16[::2, ::2]; p_eo = u16[::2, 1::2]
p_oe = u16[1::2, ::2]; p_oo = u16[1::2, 1::2]
print(f"\n=== Bayer phases (RGGB if eo == oe and they exceed ee, oo) ===")
print(f"  ee (R?): mean={p_ee.mean():.1f}")
print(f"  eo (G):  mean={p_eo.mean():.1f}")
print(f"  oe (G):  mean={p_oe.mean():.1f}")
print(f"  oo (B?): mean={p_oo.mean():.1f}")

# Save 8-bit visualization with p1/p99 stretch
p1 = float(np.percentile(u16, 1))
p99 = float(np.percentile(u16, 99))
print(f"\n=== AGC stretch (matching Windows leopard_stream_capture) ===")
print(f"  p1={p1:.0f}  p99={p99:.0f}  span={p99-p1:.0f}")
scaled = np.clip((u16.astype(np.float32) - p1) * 255.0 / max(p99 - p1, 1.0), 0, 255).astype(np.uint8)

try:
    from PIL import Image
    Image.fromarray(scaled).save("/tmp/raw12_jetson_stretched.png")
    print("  saved /tmp/raw12_jetson_stretched.png")
except Exception as e:
    print(f"  PIL save failed: {e}")

# Optional: try debayer to color (RGGB -> BGR)
try:
    import cv2
    # cv2 wants 8 or 16 bit data. Use full u16 and downcast result.
    bgr16 = cv2.cvtColor(u16, cv2.COLOR_BAYER_RG2BGR)
    # Stretch each channel
    p1c = float(np.percentile(bgr16, 1))
    p99c = float(np.percentile(bgr16, 99))
    bgr8 = np.clip((bgr16.astype(np.float32) - p1c) * 255.0 / max(p99c - p1c, 1.0), 0, 255).astype(np.uint8)
    cv2.imwrite("/tmp/raw12_jetson_color.jpg", bgr8, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  saved /tmp/raw12_jetson_color.jpg (RGGB debayer)")
except Exception as e:
    print(f"  cv2 debayer failed: {e}")
