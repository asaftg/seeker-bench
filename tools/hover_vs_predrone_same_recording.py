"""Compare hover frames vs pre-drone frames WITHIN the same airborne1
recording. Same scene, same operator, same chip state. The ONLY
difference is drone presence/absence.

If the "consistent excess at rb=17 (44m)" we saw vs the separate
background recording is REALLY the drone, it'll show up here too.
If it disappears, the signature was environmental drift between
two separate recordings, not the drone.

Run:
    py -3.11 tools/hover_vs_predrone_same_recording.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_FFT_SLOW = 4096
BIN_HZ = PRF_HZ / N_FFT_SLOW
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

AIRBORNE_BIN = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin"
)


def load_frame(path, idx):
    with open(path, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME: raise EOFError
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
              .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def per_frame_pwr(path, frame_idx):
    cube = load_frame(path, frame_idx)
    windowed = cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2).astype(np.complex64)
    rfft_out[:, 1:-1, :] *= 2.0
    rfft_out -= rfft_out.mean(axis=0, keepdims=True)
    rxsum = rfft_out.sum(axis=2)
    win = np.hanning(N_CHIRPS).astype(np.float32)
    spec = np.fft.fft(rxsum * win[:, None], n=N_FFT_SLOW, axis=0)
    pwr = (spec.real ** 2 + spec.imag ** 2)
    return pwr[: N_FFT_SLOW // 2, :].astype(np.float64)


def integrate_n(path, start, n):
    acc = None
    for i in range(n):
        try: spec = per_frame_pwr(path, start + i)
        except EOFError: break
        if acc is None: acc = spec.copy()
        else: acc += spec
    return acc


def main():
    print(f"PRF={PRF_HZ:.0f}, bin_hz={BIN_HZ:.2f}, FFT={N_FFT_SLOW}")
    print()
    print("Comparing HOVER vs PRE-DRONE within airborne1 recording.")
    print("Pre-drone: t=0-24s = frames 0-330 (drone hadn't entered scene)")
    print("Hover: t=33-100s = frames 463-1387 (drone hovering per visual)")
    print()

    # Build pre-drone reference: integrate 60 frames pre-drone
    print("Building pre-drone reference (60 frames from frames 50-309)...")
    predrone = integrate_n(AIRBORNE_BIN, 50, 60)
    print(f"Pre-drone shape: {predrone.shape}")
    print(f"Pre-drone total power: {predrone.sum():.2e}")

    # Specific (range, freq) cells we previously flagged as "drone signature"
    # via vs-background subtraction:
    SUSPECT_CELLS = [
        # (range_bin, freq_hz, label)
        (17, 2091, "rb=17 @2091Hz (top suspect)"),
        (17, 2083, "rb=17 @2083Hz"),
        (17, 1525, "rb=17 @1525Hz"),
        (18, 2500, "rb=18 @2500Hz"),
        (19, 283, "rb=19 @283Hz"),
        (19, 1205, "rb=19 @1205Hz"),
        (16, 1325, "rb=16 @1325Hz"),
        (29, 1890, "rb=29 @1890Hz"),
        (7, 2619, "rb=7 @2619Hz"),
        # Also check the chip-detected drone range bins
        (2, 800, "rb=2 (5.3m) @800Hz - DJI hover BPF"),
        (2, 1500, "rb=2 (5.3m) @1500Hz - DJI max BPF"),
        (5, 800, "rb=5 (13.2m) @800Hz"),
        (10, 800, "rb=10 (26.4m) @800Hz"),
    ]

    bin_50 = int(50 / BIN_HZ)
    bin_3000 = int(3000 / BIN_HZ)

    HOVER_SAMPLES = [(470, 33.9), (600, 43.3), (750, 54.1),
                      (900, 64.9), (1050, 75.8), (1200, 86.6), (1350, 97.4)]

    # Per hover window, compute hover/predrone ratio at each suspect cell
    print()
    print("=" * 100)
    print("HOVER vs PRE-DRONE (same recording, same scene, same chip state)")
    print("=" * 100)
    cell_results = {label: [] for _, _, label in SUSPECT_CELLS}
    for start, t_rel in HOVER_SAMPLES:
        hov = integrate_n(AIRBORNE_BIN, start, 6)
        if hov is None: continue
        # Normalize for global gain (different time = different chip warm-up)
        scale = predrone.sum() / hov.sum()
        hov_norm = hov * scale
        ratio_db = 10.0 * np.log10((hov_norm + 1e-30) / (predrone + 1e-30))

        print(f"\n--- t={t_rel:.1f}s, frames {start}-{start+5} (gain scale={scale:.3f}) ---")
        print(f"  {'cell':<40} {'excess_dB':>9}")
        for rb, freq, label in SUSPECT_CELLS:
            freq_bin = int(round(freq / BIN_HZ))
            if freq_bin >= ratio_db.shape[0] or rb >= ratio_db.shape[1]:
                continue
            v = float(ratio_db[freq_bin, rb])
            cell_results[label].append(v)
            mark = " <-- positive" if v > 3.0 else ""
            print(f"  {label:<40} {v:>9.1f}{mark}")

    # Summary
    print()
    print("=" * 100)
    print("CONSISTENCY: cells that show positive excess in N of 7 hover windows")
    print("=" * 100)
    print(f"{'cell':<45} {'count_>3dB':>10} {'mean_excess':>12}")
    for label, vals in cell_results.items():
        n_pos = sum(1 for v in vals if v > 3.0)
        mean = sum(vals) / len(vals) if vals else 0
        marker = ""
        if n_pos == 7: marker = " <<-- DRONE-like (consistent)"
        elif n_pos >= 5: marker = " <- mostly consistent"
        print(f"{label:<45} {n_pos:>10}/7 {mean:>12.1f} dB{marker}")

    # Also: a generic scan to find any (range, freq) cell with consistent excess
    print()
    print("=" * 100)
    print("SCAN: ANY (range, freq) cell with consistent >3dB excess in all 7 hover windows")
    print("=" * 100)
    excess_count = np.zeros((bin_3000 - bin_50, 30), dtype=int)
    excess_sum = np.zeros((bin_3000 - bin_50, 30), dtype=float)
    for start, _ in HOVER_SAMPLES:
        hov = integrate_n(AIRBORNE_BIN, start, 6)
        if hov is None: continue
        scale = predrone.sum() / hov.sum()
        ratio_db = 10.0 * np.log10((hov * scale + 1e-30) / (predrone + 1e-30))
        for rb in range(1, 30):
            band = ratio_db[bin_50:bin_3000, rb]
            for j in range(band.shape[0]):
                if band[j] > 3.0:
                    excess_count[j, rb] += 1
                    excess_sum[j, rb] += float(band[j])

    # Top consistent cells
    flat_idx = np.argsort(-(excess_count.flatten() * 1000 + excess_sum.flatten() / 7))
    print(f"{'count':>5} {'rb':>3} {'range_m':>7} {'avg_dB':>7} {'freq_Hz':>8}")
    n_shown = 0
    for idx in flat_idx[:30]:
        j, rb = np.unravel_index(idx, excess_count.shape)
        if excess_count[j, rb] < 4: continue
        freq = (bin_50 + j) * BIN_HZ
        avg_db = excess_sum[j, rb] / max(excess_count[j, rb], 1)
        print(f"{excess_count[j, rb]:>5} {rb:>3} {rb*RANGE_RES_M:>7.1f} {avg_db:>7.1f} {freq:>8.0f}")
        n_shown += 1
        if n_shown >= 25: break
    if n_shown == 0:
        print("  ZERO cells consistently exceed pre-drone reference!")
        print("  -> The 'drone signature' from vs-background was environmental drift.")


if __name__ == "__main__":
    sys.exit(main())
