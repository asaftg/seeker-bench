"""Offline replay: run `herm_detector.detect_comb` (Phase 1
comb-of-harmonics detector) on each frame of a recording.

Mirrors `herm_replay_v2.py` but calls the comb detector instead of the
single-peak-in-band rule. Reuses the (now-fixed) `_common_fixed.py`
helpers — `iter_real_frames`, `stage1_range_fft`.

Output:
    runs/herm_comb/<rec>_<threshold>db/
        per_frame.csv       — every frame, every detection
        track_summary.csv   — per-bin statistics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import fft as scipy_fft

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[0]))

from tools.diagnose_pmm._common_fixed import (  # noqa: E402
    iter_real_frames, resolve_recording, stage1_range_fft,
)
from radar_dca.herm_detector import detect_comb  # noqa: E402


def replay(rec_arg: str, comb_threshold_db: float, max_frames: int | None = None,
           band_low_hz: float = 200.0, band_high_hz: float = 1500.0,
           min_harmonics: int = 3, per_harmonic_floor_db: float = 3.0,
           out_dir: Path | None = None):
    rec = resolve_recording(rec_arg)
    if out_dir is None:
        out_dir = Path('runs/herm_comb') / f"{rec.name.replace(' ', '_')}_{int(comb_threshold_db)}db"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Recording: {rec.name}")
    print(f"  bin: {rec.bin_path}")
    print(f"  comb threshold: {comb_threshold_db:.1f} dB total")
    print(f"  rotor search band: [{band_low_hz:.0f}, {band_high_hz:.0f}] Hz")
    print(f"  min active harmonics: {min_harmonics} (per-harmonic floor "
          f"{per_harmonic_floor_db:.1f} dB)")
    print(f"  per-VA PRF: {rec.dims.per_va_prf_hz:.0f} Hz")
    print(f"  Output: {out_dir}")
    print()

    n_range = rec.dims.n_range_bins
    fft_n = 1024
    n_va_chirps = rec.dims.n_chirps // 6  # 128
    win = np.hanning(n_va_chirps).astype(np.float32)

    per_frame_csv = open(out_dir / 'per_frame.csv', 'w')
    per_frame_csv.write('frame_idx,t_s,range_bin,range_m,rotor_freq_hz,'
                        'comb_score_db,n_active_harmonics,snr_per_harm_db,'
                        'noise_floor_db,h1_hz,h2_hz,h3_hz,h4_hz\n')

    bin_stats = {r: {'detections': 0, 'rotor_freqs': [], 'scores_db': []}
                 for r in range(n_range)}
    n_processed = 0

    for idx, cube in iter_real_frames(rec.bin_path, rec.dims, max_frames=max_frames):
        rfft = stage1_range_fft(cube)         # (768, 97, 4)
        slot0 = rfft[0::6, :, :]              # (128, 97, 4)
        slot0 = slot0.sum(axis=-1)            # broadside RX sum: (128, 97)

        slot0_w = slot0 * win[:, None]
        spec = np.abs(scipy_fft.fft(slot0_w, n=fft_n, axis=0))   # (fft_n, 97)

        t_s = idx * rec.dims.framePeriodicity_s
        for r in range(n_range):
            res = detect_comb(
                spec[:, r],
                prf_per_va_hz=rec.dims.per_va_prf_hz,
                band_low_hz=band_low_hz,
                band_high_hz=band_high_hz,
                min_harmonics=min_harmonics,
                per_harmonic_floor_db=per_harmonic_floor_db,
                comb_threshold_db=comb_threshold_db,
            )
            if res.detected:
                rng_m = r * rec.dims.range_resolution_m
                pf = res.peak_freqs_hz
                per_frame_csv.write(
                    f'{idx},{t_s:.3f},{r},{rng_m:.2f},{res.rotor_freq_hz:.1f},'
                    f'{res.comb_score_db:.2f},{res.n_active_harmonics},'
                    f'{res.snr_db:.2f},{res.noise_floor_db:.1f},'
                    f'{pf[0]:.1f},{pf[1]:.1f},{pf[2]:.1f},{pf[3]:.1f}\n'
                )
                bin_stats[r]['detections'] += 1
                bin_stats[r]['rotor_freqs'].append(res.rotor_freq_hz)
                bin_stats[r]['scores_db'].append(res.comb_score_db)

        n_processed += 1
        if n_processed % 100 == 0:
            print(f'  ... frame {n_processed} done')

    per_frame_csv.close()

    summary_path = out_dir / 'track_summary.csv'
    with open(summary_path, 'w') as f:
        f.write('range_bin,range_m,detections,detection_rate,rotor_freq_med_hz,'
                'rotor_freq_mad_hz,comb_score_med_db\n')
        for r in range(n_range):
            d = bin_stats[r]
            rate = 100 * d['detections'] / max(n_processed, 1)
            if d['rotor_freqs']:
                pf_med = float(np.median(d['rotor_freqs']))
                pf_mad = float(np.median(np.abs(np.array(d['rotor_freqs']) - pf_med)))
                snr_med = float(np.median(d['scores_db']))
            else:
                pf_med = float('nan'); pf_mad = float('nan'); snr_med = float('nan')
            f.write(f'{r},{r*rec.dims.range_resolution_m:.2f},{d["detections"]},'
                    f'{rate:.1f},{pf_med:.1f},{pf_mad:.1f},{snr_med:.2f}\n')

    print()
    print(f'Processed {n_processed} frames.')
    overall_dets = sum(d['detections'] for d in bin_stats.values())
    overall_rate = 100 * overall_dets / max(n_processed * n_range, 1)
    print(f'Overall detection rate (any range bin): '
          f'{sum(1 for r in range(n_range) if bin_stats[r]["detections"]>0):d} '
          f'bins fired at least once; '
          f'{overall_dets} total detections '
          f'({overall_rate:.2f}% per (frame,bin) cell)')
    print()
    print(f'Top range bins by detection rate (out of {n_processed} frames):')
    print(f'  {"bin":<5} {"range_m":<8} {"hits":<6} {"rate%":<7} '
          f'{"rotor_Hz_med":<13} {"rotor_mad":<10} {"comb_db_med":<11}')
    rows = []
    for r in range(n_range):
        d = bin_stats[r]
        if d['detections'] > 0:
            pf_med = float(np.median(d['rotor_freqs']))
            pf_mad = float(np.median(np.abs(np.array(d['rotor_freqs']) - pf_med)))
            snr_med = float(np.median(d['scores_db']))
            rate = 100 * d['detections'] / max(n_processed, 1)
            rows.append((r, rate, pf_med, pf_mad, snr_med, d['detections']))
    rows.sort(key=lambda x: -x[1])
    for r, rate, pf, pf_mad, snr, hits in rows[:20]:
        print(f'  {r:<5} {r*rec.dims.range_resolution_m:<8.1f} {hits:<6} '
              f'{rate:<7.1f} {pf:<13.1f} {pf_mad:<10.1f} {snr:<11.2f}')

    print(f'\nResults written to:')
    print(f'  {out_dir/"per_frame.csv"}')
    print(f'  {summary_path}')

    return bin_stats, n_processed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('recording', help='recording name (drone_fly|airborne1|background) or .meta.yaml path')
    ap.add_argument('--threshold', type=float, default=12.0,
                    help='total comb score threshold in dB (sum across active harmonics)')
    ap.add_argument('--band-low', type=float, default=200.0,
                    help='rotor fundamental search band low (Hz)')
    ap.add_argument('--band-high', type=float, default=1500.0,
                    help='rotor fundamental search band high (Hz)')
    ap.add_argument('--min-harm', type=int, default=3,
                    help='minimum number of harmonics that must individually exceed the per-harmonic floor')
    ap.add_argument('--per-harm-floor', type=float, default=3.0,
                    help='per-harmonic SNR floor in dB above local noise')
    ap.add_argument('--max-frames', type=int, default=None)
    args = ap.parse_args()

    replay(args.recording, args.threshold, max_frames=args.max_frames,
           band_low_hz=args.band_low, band_high_hz=args.band_high,
           min_harmonics=args.min_harm, per_harmonic_floor_db=args.per_harm_floor)
    return 0


if __name__ == '__main__':
    sys.exit(main())
