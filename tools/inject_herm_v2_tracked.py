"""Inject TRACK-FILTERED herm_detector_v2 detections into a radar replay JSONL.

Improvements over inject_herm_v2_detections.py:
1. Per-frame computes the FULL spectrum (no thresholding yet).
2. Across all frames in the window, find the dominant range-bin band
   (the bin where in-band power is most consistently elevated).
3. Restrict per-frame detection to that range-bin band ±2.
4. Require consistency: a detection counts only if a near-bin
   detection also fired in ≥3 of the last 5 frames.
5. Inject only confirmed detections.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import deque

import numpy as np
from scipy import fft as scipy_fft

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from tools.diagnose_pmm._common_fixed import (    # noqa: E402
    iter_real_frames, resolve_recording, stage1_range_fft,
)


def find_dominant_bin(rec, t_start: float, t_end: float,
                       band_low_hz: float, band_high_hz: float,
                       bins_to_search):
    """Average per-frame magnitude spectra across the window. Return the
    bin (within `bins_to_search`) with the highest median in-band power
    above the per-bin out-of-band pedestal.
    """
    sf = int(t_start * 20)
    n_frames = int((t_end - t_start) * 20)
    fft_n = 1024
    win = np.hanning(128).astype(np.float32)
    prf_va = rec.dims.per_va_prf_hz
    freqs = np.fft.fftfreq(fft_n, d=1.0 / prf_va)
    pos = freqs >= 0
    freqs_pos = freqs[pos]
    band_mask = (freqs_pos >= band_low_hz) & (freqs_pos <= band_high_hz)
    oob_mask = (freqs_pos >= band_high_hz + 100) & (freqs_pos < freqs_pos[-1] - 50)

    accum = np.zeros((np.sum(pos), 97), dtype=np.float64)
    n = 0
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims,
                                                     start_frame=sf,
                                                     max_frames=n_frames)):
        rfft = stage1_range_fft(cube)
        slot0 = rfft[0::6, :, :].sum(axis=-1)
        spec = np.abs(scipy_fft.fft(slot0 * win[:, None], n=fft_n, axis=0))
        accum += spec[pos]
        n += 1
    spec = accum / max(n, 1)

    band_pwr = spec[band_mask].mean(axis=0)
    oob_pwr = spec[oob_mask].mean(axis=0) if oob_mask.any() else np.full(97, 1.0)
    excess_db = 20 * np.log10(np.maximum(band_pwr / np.maximum(oob_pwr, 1e-6), 1e-6))

    # Restrict to the bins of interest, find best
    candidate_bins = list(bins_to_search)
    candidate_excess = [(r, excess_db[r]) for r in candidate_bins]
    candidate_excess.sort(key=lambda x: -x[1])
    print(f'  Dominant-bin scan, top 5 candidates by in-band excess:')
    for r, ex in candidate_excess[:5]:
        print(f'    bin {r:3d} ({r*rec.dims.range_resolution_m:6.1f} m): excess {ex:+5.2f} dB')
    return candidate_excess[0][0]


def detect_with_tracking(rec, t_start: float, t_end: float,
                          dominant_bin: int, bin_window: int,
                          band_low_hz: float, band_high_hz: float,
                          snr_threshold_db: float,
                          consistency_n: int = 3, consistency_window: int = 5):
    """Per-frame detection restricted to ±bin_window of dominant_bin.
    Apply consistency: emit a detection only if the last `consistency_window`
    frames have at least `consistency_n` detections within ±2 bins.
    """
    sf = int(t_start * 20)
    n_frames = int((t_end - t_start) * 20)
    fft_n = 1024
    win = np.hanning(128).astype(np.float32)
    prf_va = rec.dims.per_va_prf_hz
    freqs = np.fft.fftfreq(fft_n, d=1.0 / prf_va)
    pos = freqs >= 0
    freqs_pos = freqs[pos]
    band_mask = (freqs_pos >= band_low_hz) & (freqs_pos <= band_high_hz)

    bins_to_search = list(range(max(0, dominant_bin - bin_window),
                                min(97, dominant_bin + bin_window + 1)))

    raw = []  # per-frame (bin, snr, freq) for the frame's best detection
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims,
                                                     start_frame=sf,
                                                     max_frames=n_frames)):
        rfft = stage1_range_fft(cube)
        slot0 = rfft[0::6, :, :].sum(axis=-1)
        spec = np.abs(scipy_fft.fft(slot0 * win[:, None], n=fft_n, axis=0))
        spec_pos = spec[pos]

        # For each candidate bin, compute SNR
        best = None
        for r in bins_to_search:
            sb = np.where(band_mask, spec_pos[:, r], 0)
            peak_idx = int(np.argmax(sb))
            peak_pwr = sb[peak_idx]
            sb2 = sb.copy()
            sb2[max(0, peak_idx - 5):peak_idx + 6] = 0
            noise = float(np.median(sb2[band_mask][sb2[band_mask] > 0])) if (sb2[band_mask] > 0).any() else 1.0
            snr_db = 20 * np.log10(max(peak_pwr / max(noise, 1e-6), 1e-6))
            if snr_db >= snr_threshold_db:
                if best is None or snr_db > best[1]:
                    best = (r, snr_db, freqs_pos[peak_idx])
        raw.append(best)  # None if no detection passed threshold this frame

    # Apply consistency filter — emit only frames whose last `consistency_window`
    # frames have ≥`consistency_n` detections within ±2 bins of the current best.
    confirmed = []
    history = deque(maxlen=consistency_window)
    for k, det in enumerate(raw):
        history.append(det)
        if det is None:
            continue
        cur_bin = det[0]
        n_consistent = sum(1 for h in history if h is not None and abs(h[0] - cur_bin) <= 2)
        if n_consistent >= consistency_n:
            t_s = (sf + k) / 20.0
            confirmed.append({
                't_s_in_bin': t_s,
                'bin': det[0],
                'rng_m': det[0] * rec.dims.range_resolution_m,
                'peak_hz': det[2],
                'snr_db': det[1],
            })
    return confirmed


def find_session_start_ns(jsonl_path: Path) -> int:
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get('channel') == 'session/header':
                return int(msg['msg']['started_at_ns'])
    raise RuntimeError(f'No session/header in {jsonl_path}')


def inject(in_jsonl: Path, out_jsonl: Path, detections: list,
            t_start: float, t_end: float):
    session_start_ns = find_session_start_ns(in_jsonl)
    det_t = np.array([d['t_s_in_bin'] for d in detections]) if detections else np.array([])
    n_in = n_out = n_injected = n_radar = n_parse = 0
    with open(in_jsonl, 'r') as fin, open(out_jsonl, 'w') as fout:
        for line in fin:
            n_in += 1
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line if line.endswith('\n') else line + '\n')
                n_out += 1; n_parse += 1; continue

            if msg.get('channel') != 'radar/frame':
                fout.write(json.dumps(msg) + '\n'); n_out += 1; continue

            n_radar += 1
            ts_ns = msg.get('ts_ns')
            t_s = (ts_ns - session_start_ns) / 1e9 if ts_ns else None
            if t_s is None or not (t_start <= t_s <= t_end) or len(det_t) == 0:
                fout.write(json.dumps(msg) + '\n'); n_out += 1; continue

            i = int(np.argmin(np.abs(det_t - t_s)))
            if abs(det_t[i] - t_s) > 0.05:
                fout.write(json.dumps(msg) + '\n'); n_out += 1; continue

            d = detections[i]
            r_m = float(d['rng_m'])
            snr = float(d['snr_db'])
            blade_hz = float(d['peak_hz'])
            point = {
                'x': r_m, 'y': 0.0, 'z': 0.0,
                'v': 0.0, 'snr': snr,
                'r': r_m, 'az': 0.0, 'el': 0.0,
                'tid': 1, 'src': 'herm_v2',
                'blade_freq_hz': blade_hz,
            }
            target = {
                'tid': 1, 'x': r_m, 'y': 0.0, 'z': 0.0,
                'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
                'conf': 0.9, 'src': 'herm_v2',
                'np': 1, 'coasting': False, 'hits': 1, 'misses': 0,
            }
            msg['msg']['num_points'] = 1
            msg['msg']['num_targets'] = 1
            msg['msg']['points'] = [point]
            msg['msg']['targets'] = [target]
            fout.write(json.dumps(msg) + '\n')
            n_out += 1; n_injected += 1
    return dict(n_input_lines=n_in, n_output_lines=n_out, n_radar_frames=n_radar,
                n_injected=n_injected, n_parse_errors=n_parse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-jsonl', required=True)
    ap.add_argument('--out-jsonl', required=True)
    ap.add_argument('--recording', default='airborne1')
    ap.add_argument('--t-start', type=float, default=30.0)
    ap.add_argument('--t-end', type=float, default=40.0)
    ap.add_argument('--threshold', type=float, default=8.0)
    ap.add_argument('--band-low', type=float, default=300.0)
    ap.add_argument('--band-high', type=float, default=2050.0)
    ap.add_argument('--bin-min', type=int, default=0)
    ap.add_argument('--bin-max', type=int, default=30)
    ap.add_argument('--bin-window', type=int, default=2)
    ap.add_argument('--consistency-n', type=int, default=3)
    ap.add_argument('--consistency-window', type=int, default=5)
    args = ap.parse_args()

    rec = resolve_recording(args.recording)
    print(f'Recording: {rec.name}')
    print(f'  bin: {rec.bin_path}')
    print(f'  Window: t={args.t_start}-{args.t_end} s')
    print()
    print(f'Step 1: find dominant range bin in [{args.bin_min},{args.bin_max-1}]...')
    dom_bin = find_dominant_bin(rec, args.t_start, args.t_end,
                                  args.band_low, args.band_high,
                                  range(args.bin_min, args.bin_max))
    print(f'  Dominant bin: {dom_bin} ({dom_bin*rec.dims.range_resolution_m:.1f} m)')
    print()
    print(f'Step 2: per-frame detection within ±{args.bin_window} of bin {dom_bin}, threshold {args.threshold} dB')
    print(f'  with consistency filter: ≥{args.consistency_n}-of-{args.consistency_window} frames within ±2 bins')
    confirmed = detect_with_tracking(rec, args.t_start, args.t_end,
                                       dom_bin, args.bin_window,
                                       args.band_low, args.band_high,
                                       args.threshold,
                                       args.consistency_n, args.consistency_window)
    print(f'  {len(confirmed)} confirmed detections')
    if confirmed:
        bins = sorted(set(d['bin'] for d in confirmed))
        print(f'  bins observed: {bins}')
        freqs = [d['peak_hz'] for d in confirmed]
        print(f'  peak freq median {np.median(freqs):.0f} Hz, range [{min(freqs):.0f}, {max(freqs):.0f}]')
        print(f'  SNR median {np.median([d["snr_db"] for d in confirmed]):.1f} dB')
        print(f'  Sample first 5: ', [(f't={d["t_s_in_bin"]:.2f}', f'bin{d["bin"]}', f'r={d["rng_m"]:.1f}m',
                                       f'peak={d["peak_hz"]:.0f}Hz', f'snr={d["snr_db"]:.1f}dB') for d in confirmed[:5]])

    print()
    print(f'Injecting into {args.out_jsonl}')
    stats = inject(Path(args.in_jsonl), Path(args.out_jsonl), confirmed,
                    args.t_start, args.t_end)
    for k, v in stats.items():
        print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
