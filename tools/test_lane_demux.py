"""Test the 2-lane CBUFF demux hypothesis."""
import numpy as np
from scipy import fft as scipy_fft

with open("recordings/seeker_2026-05-05_21-14-35_radar.bin", "rb") as f:
    buf = f.read(1179648)
raw = np.frombuffer(buf, dtype=np.int16)

# Per-chirp: 1536 BYTES = 768 int16. With 4 lane-positions per cycle (2 active +
# 2 zero), 768 / 4 = 192 cycles per chirp.
# Lane 0 at idx%4==0, lane 1 at idx%4==1, zero pad at idx%4==2,3
raw_chirp = raw.reshape(768, 768)   # (n_chirps, int16_per_chirp)
print(f"chirp 0 first 32 int16: {raw_chirp[0,:32].tolist()}")
print()

lane0 = raw_chirp[0, 0::4]  # 384 samples
lane1 = raw_chirp[0, 1::4]
ext1  = raw_chirp[0, 2::4]
ext2  = raw_chirp[0, 3::4]
print(f"lane0 (idx%4==0): first 16 = {lane0[:16].tolist()}, std={lane0.std():.1f}")
print(f"lane1 (idx%4==1): first 16 = {lane1[:16].tolist()}, std={lane1.std():.1f}")
print(f"ext1  (idx%4==2): nonzero count = {(ext1!=0).sum()}, std={ext1.std():.2f}")
print(f"ext2  (idx%4==3): nonzero count = {(ext2!=0).sum()}, std={ext2.std():.2f}")
print()

# Per the 2-lane real CBUFF format 0x75316420 / 0x64207531:
# Lane 0 emits sample-indices [0, 2, 4, 6, ...] (even positions in the 4-RX × 2-time burst)
# Lane 1 emits sample-indices [1, 3, 5, 7, ...] (odd positions)
# 4-RX interleaved ADCBuf burst[0..7] = [RX0_s0, RX1_s0, RX2_s0, RX3_s0, RX0_s1, RX1_s1, RX2_s1, RX3_s1]
# So lane 0 (even sample idx): [RX0_s0, RX2_s0, RX0_s1, RX2_s1, ...] → alternates RX0/RX2
# Lane 1 (odd sample idx):  [RX1_s0, RX3_s0, RX1_s1, RX3_s1, ...] → alternates RX1/RX3

lane0_rx0 = lane0[0::2]  # RX0 (192 samples — but that's wrong, total RX0 samples should be 192)
lane0_rx2 = lane0[1::2]  # RX2
lane1_rx1 = lane1[0::2]  # RX1
lane1_rx3 = lane1[1::2]  # RX3

print(f"Hypothesized RX0 (lane0[0::2]): first 12 = {lane0_rx0[:12].tolist()}")
print(f"Hypothesized RX1 (lane1[0::2]): first 12 = {lane1_rx1[:12].tolist()}")
print(f"Hypothesized RX2 (lane0[1::2]): first 12 = {lane0_rx2[:12].tolist()}")
print(f"Hypothesized RX3 (lane1[1::2]): first 12 = {lane1_rx3[:12].tolist()}")
print()
print(f"std: RX0={lane0_rx0.std():.1f} RX1={lane1_rx1.std():.1f} RX2={lane0_rx2.std():.1f} RX3={lane1_rx3.std():.1f}")
print()

def rfft_mag(s, n=192):
    if len(s) < n:
        s = np.pad(s, (0, n - len(s)))
    s = s[:n].astype(np.float32)
    s = s - s.mean()
    win = np.hanning(n).astype(np.float32)
    return np.abs(scipy_fft.rfft(s * win))

r0 = rfft_mag(lane0_rx0)
r1 = rfft_mag(lane1_rx1)
r2 = rfft_mag(lane0_rx2)
r3 = rfft_mag(lane1_rx3)

print("Range FFT correlations (high = both real radar channels):")
print(f"  corr(RX0, RX2) = {np.corrcoef(r0, r2)[0,1]:.3f}")
print(f"  corr(RX1, RX3) = {np.corrcoef(r1, r3)[0,1]:.3f}")
print(f"  corr(RX0, RX1) = {np.corrcoef(r0, r1)[0,1]:.3f}")
print(f"  corr(RX2, RX3) = {np.corrcoef(r2, r3)[0,1]:.3f}")
print(f"  corr(RX0, RX3) = {np.corrcoef(r0, r3)[0,1]:.3f}")
print()
print("Top-3 range bins per channel:")
for lbl, r in [("RX0", r0), ("RX1", r1), ("RX2", r2), ("RX3", r3)]:
    top3 = np.argsort(-r)[:3]
    print(f"  {lbl}: top3={top3.tolist()}, max={r.max():.0f}")
