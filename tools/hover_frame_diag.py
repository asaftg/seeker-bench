"""Comprehensive per-timestamp diagnostic.

For each of N selected hover timestamps:
  - Load the corresponding radar bin frame
  - Run MIMO detector with proper N=6 warm-up
  - Dump the slow-time POWER SPECTRUM at the predicted drone range bin
    (computed from EO pixel position + gimbal pose)
  - Report what MIMO returned at this frame
  - Output a CSV summary

Predicted drone position per visual EO inspection (2026-05-06 PM):
  t=45.6s: drone at az=-5.9 deg, el=19.7 deg, range~50m -> rb~19
  t=60.8s: drone at smaller pixel size (further) -> need re-measure
  t=91.2s: drone hovering, position barely changed
"""
from __future__ import annotations

import sys
import json
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

from radar_dca.ddma import ddma_unfold

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_TX = 4
EFF_PRF = PRF_HZ / N_TX
N_FFT_VA = 1024
BIN_HZ = EFF_PRF / N_FFT_VA
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

# Full-PRF analysis as well (no DDMA)
N_FFT_FULL = 4096
BIN_HZ_FULL = PRF_HZ / N_FFT_FULL  # 7.44 Hz
HANN_SLOW = np.hanning(N_CHIRPS).astype(np.float32)

BIN_PATH = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin"
)
JSONL_PATH = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\drone test airborne 1.jsonl"
)


def load_frame(idx):
    with open(BIN_PATH, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME: raise EOFError
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
              .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def stage1(real_cube):
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2).astype(np.complex64)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out


def per_frame_full_prf_pwr(idx):
    """Returns (n_fft//2, n_range) power spectrum at full PRF, RX-coherent sum."""
    cube = load_frame(idx)
    rc = stage1(cube)
    rc -= rc.mean(axis=0, keepdims=True)
    rxsum = rc.sum(axis=2)  # (768, 97) coherent
    spec = np.fft.fft(rxsum * HANN_SLOW[:, None], n=N_FFT_FULL, axis=0)
    pwr = (spec.real**2 + spec.imag**2)
    return pwr[: N_FFT_FULL // 2, :].astype(np.float64)


def find_aa_frame_by_ts(aa_records, target_ts_ns):
    best_i, best_d = 0, abs(aa_records[0][0] - target_ts_ns)
    for i, (ts, _) in enumerate(aa_records):
        d = abs(ts - target_ts_ns)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def analyze_frame(radar_idx, t_label):
    """Dump spectrum at multiple range bins for ONE radar frame."""
    print(f"\n=== t={t_label}, radar_idx={radar_idx} ===")

    # Single-frame full-PRF spectrum
    pwr = per_frame_full_prf_pwr(radar_idx)

    # Mask out DC + low-freq leakage + chip artifact at PRF/3
    keep = np.ones(pwr.shape[0], dtype=bool)
    keep[:8] = False
    artifact_freqs = [PRF_HZ/3, PRF_HZ/2 - 100, PRF_HZ/4]
    for af in artifact_freqs:
        ab = int(round(af / BIN_HZ_FULL))
        keep[max(0, ab-3):min(pwr.shape[0], ab+4)] = False

    # Per range bin: find peak in 50-3000 Hz, peak in 3000-15000 Hz
    bin_50 = int(50 / BIN_HZ_FULL)
    bin_3k = int(3000 / BIN_HZ_FULL)
    bin_15k = pwr.shape[0]

    # Focus on candidate drone range bins (rb=15-30 = 40-79m)
    candidate_bins = list(range(15, 30))
    print(f"{'rb':>3} {'range_m':>7} {'floor':>10} {'low_peak_Hz':>12} {'low_dB':>7} "
          f"{'high_peak_Hz':>12} {'high_dB':>7}")
    for rb in candidate_bins:
        spec = pwr[:, rb].copy()
        spec[~keep] = 0
        floor = float(np.median(pwr[keep, rb]))
        if floor <= 0: continue
        low_band = spec[bin_50:bin_3k]
        high_band = spec[bin_3k:bin_15k]
        if low_band.max() <= 0 or high_band.max() <= 0: continue
        low_idx = int(np.argmax(low_band))
        high_idx = int(np.argmax(high_band))
        low_freq = (bin_50 + low_idx) * BIN_HZ_FULL
        high_freq = (bin_3k + high_idx) * BIN_HZ_FULL
        low_db = 10*np.log10(low_band[low_idx] / floor)
        high_db = 10*np.log10(high_band[high_idx] / floor)
        print(f"{rb:>3} {rb*RANGE_RES_M:>7.1f} {floor:>10.2e} "
              f"{low_freq:>12.0f} {low_db:>7.1f} "
              f"{high_freq:>12.0f} {high_db:>7.1f}")


def main():
    # Build aa_frame index
    print("Indexing JSONL...")
    aa = []
    base_ts = None
    base_aa_fid = None
    with io.open(JSONL_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            try: r = json.loads(line)
            except: continue
            if r.get("channel") != "radar/aa_frame": continue
            ts = r.get("ts_ns")
            msg = r.get("msg") or {}
            fid = msg.get("frame_id")
            if base_ts is None: base_ts = ts
            if base_aa_fid is None: base_aa_fid = fid
            aa.append((ts, msg))
    print(f"Indexed {len(aa)} aa_frames, base_ts={base_ts}, base_fid={base_aa_fid}")

    # Sample timestamps spanning the recording (every 10s in the hover window)
    samples_t = [33.4, 45.6, 60.8, 76.0, 91.2, 100.3]
    for t_rel in samples_t:
        target_ts = base_ts + int(t_rel * 1e9)
        ai = find_aa_frame_by_ts(aa, target_ts)
        ts_actual, msg = aa[ai]
        radar_idx = msg.get("frame_id", 0) - base_aa_fid
        analyze_frame(radar_idx, f"{t_rel:.1f}s (ts diff={(ts_actual-target_ts)/1e6:.0f}ms)")


if __name__ == "__main__":
    sys.exit(main())
