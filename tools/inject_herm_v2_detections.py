"""Inject herm_detector_v2 detections into a radar replay JSONL.

Takes the existing radar replay JSONL (which has empty detections in the
30-40s window because the live pipeline was buggy) and adds my real
detections at the right frame timestamps. Output is a new JSONL the
user can play through the replay viewer to see the radar boxes appear
on the EO/thermal stream at the drone's position.

Usage:
  python tools/inject_herm_v2_detections.py \
    --in-jsonl  recordings/airborne1_v5+thermalv2_replay_RADAR.jsonl \
    --out-jsonl recordings/airborne1_v5+thermalv2_replay_RADAR_with_herm.jsonl \
    --bin       recordings/seeker_2026-05-06_12-58-59_radar.bin \
    --t-start   30.0 \
    --t-end     40.0 \
    --threshold 5.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import fft as scipy_fft

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from tools.diagnose_pmm._common_fixed import (    # noqa: E402
    iter_real_frames, resolve_recording, stage1_range_fft,
)
from radar_dca.herm_detector_v2 import detect_blade_pass    # noqa: E402


def detect_window(rec, t_start: float, t_end: float, threshold_db: float = 5.0,
                  band_low_hz: float = 300.0, band_high_hz: float = 2050.0,
                  bins_to_search=range(0, 30)):
    """Per-frame detection across the time window, returns list of dicts.

    Output keys: t_s_in_bin (float), bin (int), rng_m (float),
    peak_hz (float), snr_db (float).
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

    detections = []
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims,
                                                     start_frame=sf,
                                                     max_frames=n_frames)):
        rfft = stage1_range_fft(cube)
        slot0 = rfft[0::6, :, :].sum(axis=-1)
        spec = np.abs(scipy_fft.fft(slot0 * win[:, None], n=fft_n, axis=0))
        spec_pos = spec[pos]

        # Pick the bin with strongest in-band peak
        best = None
        for r in bins_to_search:
            if r >= 97:
                continue
            res = detect_blade_pass(
                spec_pos[:, r], prf_per_va_hz=prf_va,
                band_low_hz=band_low_hz, band_high_hz=band_high_hz,
                snr_threshold_db=threshold_db, fft_already_done=True,
            )
            if res.detected:
                if best is None or res.snr_db > best['snr_db']:
                    best = {'bin': r, 'rng_m': r * rec.dims.range_resolution_m,
                            'peak_hz': res.peak_freq_hz, 'snr_db': res.snr_db}
        if best is not None:
            t_s_in_bin = (sf + k) / 20.0
            best['t_s_in_bin'] = t_s_in_bin
            detections.append(best)
    return detections


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


def inject(in_jsonl: Path, out_jsonl: Path, detections: list, t_start: float,
           t_end: float, gimbal_pan_deg: float = 0.0, gimbal_tilt_deg: float = 0.0):
    """For each radar/frame line whose timestamp falls in [t_start, t_end],
    inject the closest detection (by t).
    """
    session_start_ns = find_session_start_ns(in_jsonl)
    print(f'Session start ns: {session_start_ns}')
    # Sort detections by t_s for binary search
    det_t = np.array([d['t_s_in_bin'] for d in detections])

    n_in = 0
    n_out = 0
    n_injected = 0
    n_parse_err = 0
    n_radar = 0

    with open(in_jsonl, 'r') as fin, open(out_jsonl, 'w') as fout:
        for line in fin:
            n_in += 1
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Pass through bad lines unchanged
                fout.write(line if line.endswith('\n') else line + '\n')
                n_out += 1
                n_parse_err += 1
                continue

            if msg.get('channel') != 'radar/frame':
                fout.write(json.dumps(msg) + '\n')
                n_out += 1
                continue

            n_radar += 1
            ts_ns = msg.get('ts_ns')
            t_s = (ts_ns - session_start_ns) / 1e9 if ts_ns else None
            if t_s is None or not (t_start <= t_s <= t_end):
                fout.write(json.dumps(msg) + '\n')
                n_out += 1
                continue

            # Find nearest detection in time
            i = int(np.argmin(np.abs(det_t - t_s)))
            dt = abs(det_t[i] - t_s)
            if dt > 0.05:  # outside frame period — no detection at this frame
                fout.write(json.dumps(msg) + '\n')
                n_out += 1
                continue

            d = detections[i]
            r_m = d['rng_m']
            # Cartesian: assume broadside (az=0, el=0); user can refine if gimbal
            # data lets us project differently
            az_deg = 0.0
            el_deg = 0.0
            x = r_m * math.cos(math.radians(el_deg)) * math.cos(math.radians(az_deg))
            y = r_m * math.cos(math.radians(el_deg)) * math.sin(math.radians(az_deg))
            z = r_m * math.sin(math.radians(el_deg))

            point = {
                'x': x, 'y': y, 'z': z,
                'v': 0.0,
                'snr': d['snr_db'],
                'r': r_m,
                'az': az_deg,
                'el': el_deg,
                'tid': 1,
                'src': 'herm_v2',
                'blade_freq_hz': d['peak_hz'],
            }
            target = {
                'tid': 1,
                'x': x, 'y': y, 'z': z,
                'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
                'conf': 0.9,
                'src': 'herm_v2',
                'np': 1, 'coasting': False, 'hits': 1, 'misses': 0,
            }
            # Replace empty arrays with our detection
            msg['msg']['num_points'] = 1
            msg['msg']['num_targets'] = 1
            msg['msg']['points'] = [point]
            msg['msg']['targets'] = [target]
            fout.write(json.dumps(msg) + '\n')
            n_out += 1
            n_injected += 1

    return {
        'n_input_lines': n_in,
        'n_output_lines': n_out,
        'n_radar_frames': n_radar,
        'n_injected': n_injected,
        'n_parse_errors': n_parse_err,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-jsonl', required=True)
    ap.add_argument('--out-jsonl', required=True)
    ap.add_argument('--recording', default='airborne1',
                    help='Friendly name (resolves to .bin) — used for detection')
    ap.add_argument('--t-start', type=float, default=30.0)
    ap.add_argument('--t-end', type=float, default=40.0)
    ap.add_argument('--threshold', type=float, default=5.0)
    ap.add_argument('--band-low', type=float, default=300.0)
    ap.add_argument('--band-high', type=float, default=2050.0)
    ap.add_argument('--bin-min', type=int, default=0)
    ap.add_argument('--bin-max', type=int, default=30)
    args = ap.parse_args()

    rec = resolve_recording(args.recording)
    print(f'Recording: {rec.name}')
    print(f'  bin: {rec.bin_path}')
    print(f'  Window: t={args.t_start}-{args.t_end} s')
    print(f'  Detection: SNR>={args.threshold} dB, band [{args.band_low:.0f}, {args.band_high:.0f}] Hz, bins {args.bin_min}-{args.bin_max-1}')
    print()

    print('Running per-frame detection...')
    detections = detect_window(
        rec, args.t_start, args.t_end, args.threshold,
        args.band_low, args.band_high,
        bins_to_search=range(args.bin_min, args.bin_max),
    )
    print(f'  {len(detections)} detections in {int((args.t_end - args.t_start) * 20)} frames')
    if detections:
        print(f'  Sample: t={detections[0]["t_s_in_bin"]:.2f}s bin{detections[0]["bin"]} r={detections[0]["rng_m"]:.1f}m peak={detections[0]["peak_hz"]:.0f}Hz snr={detections[0]["snr_db"]:.1f}dB')
        print(f'          ... (range bins observed: {sorted(set(d["bin"] for d in detections))})')
        print(f'          ... (peak freq min/max: {min(d["peak_hz"] for d in detections):.0f}-{max(d["peak_hz"] for d in detections):.0f} Hz)')

    print()
    print(f'Injecting into {args.in_jsonl} -> {args.out_jsonl}')
    stats = inject(Path(args.in_jsonl), Path(args.out_jsonl), detections,
                   args.t_start, args.t_end)
    for k, v in stats.items():
        print(f'  {k}: {v}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
