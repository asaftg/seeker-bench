"""Background subtraction approach: hover_spectrum - background_spectrum.

Logic: chip artifacts (peaks at 2540 Hz / multiples thereof) appear at
the SAME range bins and SAME frequencies in both background and hover
recordings. So subtraction cancels them. Only the drone-induced energy
should survive.

Per range bin per freq bin:
   delta = hover_integrated_pwr / background_integrated_pwr (in dB)
A real drone-only signal shows positive delta at the drone's range bin
at non-chip-artifact frequencies. Chip artifacts show ~0 dB delta everywhere.

Run:
    py -3.11 tools/bg_subtract_drone.py
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
BG_BIN = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-54-23_radar.bin"
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
    rfft_out -= rfft_out.mean(axis=0, keepdims=True)  # MTI
    rxsum = rfft_out.sum(axis=2)  # (768, 97)
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
    print(f"PRF={PRF_HZ:.0f}, Nyquist={PRF_HZ/2:.0f}, bin_hz={BIN_HZ:.2f}")
    print(f"FFT={N_FFT_SLOW}, RX-coherent sum, 6-frame integrated power, NO notch, NO DDMA")
    print()

    # Build a stable background spectrum by integrating MANY background frames
    print("Building background reference (40 frames integrated)...")
    bg = integrate_n(BG_BIN, 100, 40)  # 40 frames -> ~3 sec, robust
    bg_per_freq_med = np.median(bg, axis=1, keepdims=True)  # (n_freq, 1)
    print(f"Background ref shape: {bg.shape}")

    # Per (range, freq) compute log10 ratio of hover spec vs background
    HOVER_SAMPLES = [(470, 33.9), (600, 43.3), (750, 54.1),
                      (900, 64.9), (1050, 75.8), (1200, 86.6), (1350, 97.4)]

    # Range bins of interest: 1-30 (drone-candidate territory)
    DRONE_RB_MAX = 30
    # Frequencies to inspect: 50-3000 Hz (drone band)
    bin_50 = int(50 / BIN_HZ)
    bin_3000 = int(3000 / BIN_HZ)

    # Tally: which (range, freq) consistently show large excess across all hover windows?
    excess_count = np.zeros((bin_3000 - bin_50, DRONE_RB_MAX), dtype=int)
    excess_sum = np.zeros((bin_3000 - bin_50, DRONE_RB_MAX), dtype=float)

    print()
    print("=" * 90)
    print("HOVER vs BACKGROUND, per range bin: top in-band peaks of hover/bg ratio")
    print("=" * 90)
    for start, t_rel in HOVER_SAMPLES:
        hov = integrate_n(AIRBORNE_BIN, start, 6)
        if hov is None: continue
        # Normalize for global gain mismatch: scale so total power matches bg total power
        scale = bg.sum() / hov.sum()
        hov_norm = hov * scale
        # Ratio in dB
        ratio_db = 10.0 * np.log10((hov_norm + 1e-30) / (bg + 1e-30))

        print(f"\n--- t={t_rel:.1f}s, frames {start}-{start+5} (scale={scale:.3f}) ---")
        # For each range bin in 1-30, find the strongest in-band excess
        rows = []
        for rb in range(1, DRONE_RB_MAX):
            band = ratio_db[bin_50:bin_3000, rb]
            peak_idx = int(np.argmax(band))
            peak_db = float(band[peak_idx])
            peak_freq = (bin_50 + peak_idx) * BIN_HZ
            rows.append((rb, peak_db, peak_freq))
            # Update tallies
            for j in range(band.shape[0]):
                if band[j] > 3.0:  # 3 dB excess
                    excess_count[j, rb] += 1
                    excess_sum[j, rb] += band[j]
        rows.sort(key=lambda x: -x[1])
        print(f"  {'rb':>3} {'range_m':>7} {'excess_dB':>9} {'peak_Hz':>8}")
        for rb, db, f in rows[:10]:
            mark = " <-- positive" if db > 3.0 else ""
            print(f"  {rb:>3} {rb*RANGE_RES_M:>7.1f} {db:>9.1f} {f:>8.0f}{mark}")

    # Final tally: which (range, freq) cells consistently showed excess across multiple windows?
    print()
    print("=" * 90)
    print("CONSISTENT EXCESS across all 7 hover windows (>3 dB excess in N of 7)")
    print("=" * 90)
    print(f"{'count':>5} {'rb':>3} {'range_m':>7} {'avg_dB':>7} {'freq_Hz':>8}")
    flat_idx = np.argsort(-(excess_count.flatten() * 1000 + excess_sum.flatten() / 7))
    n_shown = 0
    for idx in flat_idx[:60]:
        j, rb = np.unravel_index(idx, excess_count.shape)
        if excess_count[j, rb] < 3: continue
        freq = (bin_50 + j) * BIN_HZ
        avg_db = excess_sum[j, rb] / max(excess_count[j, rb], 1)
        print(f"{excess_count[j, rb]:>5} {rb:>3} {rb*RANGE_RES_M:>7.1f} {avg_db:>7.1f} {freq:>8.0f}")
        n_shown += 1
        if n_shown >= 30: break
    if n_shown == 0:
        print("  None! No (range, freq) cells consistently exceed background.")


if __name__ == "__main__":
    sys.exit(main())
