"""Offline replay: run herm_detector_v2 (slot-0 blade-pass detector) across
each frame of a recording. Per the 2026-05-07 corrected layout findings.

Usage:
    python -m radar_dca.herm_replay_v2 drone_fly --threshold 6
    python -m radar_dca.herm_replay_v2 airborne1
    python -m radar_dca.herm_replay_v2 background

Output:
    runs/herm_v2/<rec>_<threshold>db/
        per_frame.csv       — every frame, every detection
        track_summary.csv   — per-bin statistics (detection rate, peak freq stability)
        track_plot.png      — visualization
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy import fft as scipy_fft

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[0]))

from tools.diagnose_pmm._common_fixed import (  # noqa: E402
    iter_real_frames, resolve_recording, stage1_range_fft,
)
from radar_dca.herm_detector_v2 import detect_blade_pass  # noqa: E402


def replay(rec_arg: str, snr_threshold_db: float, max_frames: int | None = None,
           band_low_hz: float = 300.0, band_high_hz: float = 2200.0,
           out_dir: Path | None = None):
    rec = resolve_recording(rec_arg)
    if out_dir is None:
        out_dir = Path('runs/herm_v2') / f"{rec.name.replace(' ', '_')}_{int(snr_threshold_db)}db"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recording: {rec.name}")
    print(f"  bin: {rec.bin_path}")
    print(f"  SNR threshold: {snr_threshold_db:.1f} dB")
    print(f"  Search band: [{band_low_hz:.0f}, {band_high_hz:.0f}] Hz")
    print(f"  per-VA PRF: {rec.dims.per_va_prf_hz:.0f} Hz")
    print(f"  Output: {out_dir}")
    print()

    n_range = rec.dims.n_range_bins
    fft_n = 1024
    freqs = np.fft.fftfreq(fft_n, d=1.0 / rec.dims.per_va_prf_hz)
    pos_mask = freqs >= 0
    n_va_chirps = rec.dims.n_chirps // 6  # 128
    win = np.hanning(n_va_chirps).astype(np.float32)

    per_frame_csv = open(out_dir / 'per_frame.csv', 'w')
    per_frame_csv.write('frame_idx,t_s,range_bin,range_m,peak_freq_hz,peak_power_db,'
                        'noise_floor_db,snr_db,secondary_freq_hz,secondary_power_db,confidence\n')

    bin_stats = {r: {'detections': 0, 'peak_freqs': [], 'snr_db': []}
                 for r in range(n_range)}
    n_processed = 0

    for idx, cube in iter_real_frames(rec.bin_path, rec.dims, max_frames=max_frames):
        # Range FFT
        rfft = stage1_range_fft(cube)         # (768, 97, n_rx_active=4)
        # Slot-0 decimation: every 6th chirp starting at 0
        slot0 = rfft[0::6, :, :]              # (128, 97, 4)
        slot0 = slot0.sum(axis=-1)            # broadside RX sum: (128, 97) complex

        # FFT each range bin's slow-time
        slot0_w = slot0 * win[:, None]
        spec = np.abs(scipy_fft.fft(slot0_w, n=fft_n, axis=0))   # (fft_n, 97)

        # Per-bin detection
        t_s = idx * rec.dims.framePeriodicity_s
        for r in range(n_range):
            result = detect_blade_pass(
                spec[:, r],
                prf_per_va_hz=rec.dims.per_va_prf_hz,
                band_low_hz=band_low_hz,
                band_high_hz=band_high_hz,
                snr_threshold_db=snr_threshold_db,
                fft_already_done=True,
            )
            if result.detected:
                rng_m = r * rec.dims.range_resolution_m
                per_frame_csv.write(
                    f'{idx},{t_s:.3f},{r},{rng_m:.2f},{result.peak_freq_hz:.1f},'
                    f'{result.peak_power_db:.1f},{result.noise_floor_db:.1f},'
                    f'{result.snr_db:.2f},{result.secondary_freq_hz:.1f},'
                    f'{result.secondary_power_db:.1f},{result.confidence:.3f}\n'
                )
                bin_stats[r]['detections'] += 1
                bin_stats[r]['peak_freqs'].append(result.peak_freq_hz)
                bin_stats[r]['snr_db'].append(result.snr_db)

        n_processed += 1
        if n_processed % 50 == 0:
            print(f'  ... frame {n_processed} done')

    per_frame_csv.close()

    # Per-bin summary
    summary_path = out_dir / 'track_summary.csv'
    with open(summary_path, 'w') as f:
        f.write('range_bin,range_m,detections,detection_rate,peak_freq_median_hz,'
                'peak_freq_mad_hz,snr_median_db\n')
        for r in range(n_range):
            d = bin_stats[r]
            rate = 100 * d['detections'] / max(n_processed, 1)
            if d['peak_freqs']:
                pf_med = float(np.median(d['peak_freqs']))
                pf_mad = float(np.median(np.abs(np.array(d['peak_freqs']) - pf_med)))
                snr_med = float(np.median(d['snr_db']))
            else:
                pf_med = float('nan'); pf_mad = float('nan'); snr_med = float('nan')
            f.write(f'{r},{r*rec.dims.range_resolution_m:.2f},{d["detections"]},'
                    f'{rate:.1f},{pf_med:.1f},{pf_mad:.1f},{snr_med:.2f}\n')

    # Print summary to console — top bins by detection rate
    print()
    print(f'Processed {n_processed} frames.')
    print(f'\nTop range bins by detection rate (out of {n_processed} frames):')
    print(f'  {"bin":<5} {"range_m":<8} {"hits":<6} {"rate%":<7} {"peak_Hz_med":<13} {"peak_mad":<10} {"snr_db_med":<10}')
    rows = []
    for r in range(n_range):
        d = bin_stats[r]
        if d['detections'] > 0:
            pf_med = float(np.median(d['peak_freqs']))
            pf_mad = float(np.median(np.abs(np.array(d['peak_freqs']) - pf_med)))
            snr_med = float(np.median(d['snr_db']))
            rate = 100 * d['detections'] / max(n_processed, 1)
            rows.append((r, rate, pf_med, pf_mad, snr_med, d['detections']))
    rows.sort(key=lambda x: -x[1])
    for r, rate, pf, pf_mad, snr, hits in rows[:20]:
        print(f'  {r:<5} {r*rec.dims.range_resolution_m:<8.1f} {hits:<6} {rate:<7.1f} {pf:<13.1f} {pf_mad:<10.1f} {snr:<10.2f}')

    print(f'\nResults written to:')
    print(f'  {out_dir/"per_frame.csv"}')
    print(f'  {summary_path}')

    return bin_stats, n_processed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('recording')
    ap.add_argument('--threshold', type=float, default=6.0)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--band-low', type=float, default=300.0)
    ap.add_argument('--band-high', type=float, default=2200.0)
    args = ap.parse_args()
    replay(args.recording, args.threshold, args.max_frames,
           args.band_low, args.band_high)
    return 0


if __name__ == '__main__':
    sys.exit(main())
