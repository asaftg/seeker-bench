"""Range-Doppler movie v2 — with background subtraction.

For airborne1 30-40s: per frame, compute RD map, subtract a slow-moving
window-average RD map (which captures static clutter and the persistent
+/-Nyquist artifact), and render the residual. Any moving target should
appear as a bright spot in the residual.

Also: tighter dB scale, a "track view" panel showing range vs time at
the expected Doppler.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import fft as scipy_fft

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from tools.diagnose_pmm._common_fixed import (    # noqa: E402
    iter_real_frames, resolve_recording, stage1_range_fft,
)


def compute_rd(cube, dims):
    rfft = stage1_range_fft(cube)
    slot0 = rfft[0::6, :, :].sum(axis=-1)
    win = np.hanning(128).astype(np.float32)
    slot0_w = slot0 * win[:, None]
    rd = scipy_fft.fftshift(scipy_fft.fft(slot0_w, n=128, axis=0), axes=0)
    return np.abs(rd).astype(np.float32)   # magnitude (not dB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--recording', default='airborne1')
    ap.add_argument('--t-start', type=float, default=30.0)
    ap.add_argument('--t-end', type=float, default=40.0)
    ap.add_argument('--out-dir', default='runs/rd_movie_airborne1_30_40_v2')
    ap.add_argument('--frame-step', type=int, default=4)
    ap.add_argument('--bg-window-frames', type=int, default=80,
                    help='Number of frames before/after to use for background subtract (sliding mean)')
    ap.add_argument('--expected-range-m', type=float, default=44.0)
    ap.add_argument('--expected-velocity-mps', type=float, default=2.8)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rec = resolve_recording(args.recording)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Recording: {rec.name}, window {args.t_start}-{args.t_end}s')

    sf = int(args.t_start * 20)
    n_frames = int((args.t_end - args.t_start) * 20)

    # Pass 1: compute every RD map and stash
    print(f'  Computing {n_frames} RD maps...')
    rds = np.zeros((n_frames, 128, 97), dtype=np.float32)
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims,
                                                     start_frame=sf,
                                                     max_frames=n_frames)):
        rds[k] = compute_rd(cube, rec.dims)

    # Background = sliding mean across the whole window. Subtraction in
    # magnitude space (not dB) — moving targets show as residuals.
    rd_bg = rds.mean(axis=0)
    print(f'  Mag range: rd values [{rds.min():.0f}, {rds.max():.0f}], bg [{rd_bg.min():.0f}, {rd_bg.max():.0f}]')

    # Residuals (mag - bg). Clip to >0 (static clutter that's BIGGER on average
    # than this frame produces negative residuals; we don't care about those.)
    residuals = np.maximum(rds - rd_bg[None, :, :], 0)
    res_db = 20 * np.log10(residuals + 1e-3)
    print(f'  Residual dB range: [{res_db.min():.1f}, {res_db.max():.1f}]')
    # Use percentile-based vmin/vmax for clearer visualization
    vmin_res = float(np.percentile(res_db, 80))
    vmax_res = float(np.percentile(res_db, 99.9))
    print(f'  Color scale: [{vmin_res:.1f}, {vmax_res:.1f}] dB')

    prf_va = rec.dims.per_va_prf_hz
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(128, d=1.0 / prf_va))
    range_m = np.arange(97) * rec.dims.range_resolution_m

    expected_doppler_pos = 2 * args.expected_velocity_mps / (3e8 / 77e9)
    expected_doppler_neg = -expected_doppler_pos

    # Render per-frame residual maps + a tracking panel
    print('  Rendering residual frames...')
    rendered = 0
    for k in range(n_frames):
        if k % args.frame_step != 0:
            continue
        t_s = (sf + k) / 20.0

        fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                                  gridspec_kw={'width_ratios': [3, 1.3]})

        # Residual RD map
        ax = axes[0]
        extent = [range_m[0], range_m[-1], doppler_freqs[0], doppler_freqs[-1]]
        im = ax.imshow(res_db[k], aspect='auto', origin='lower',
                       extent=extent, cmap='inferno',
                       vmin=vmin_res, vmax=vmax_res)
        ax.axvline(args.expected_range_m, color='cyan', lw=0.7, ls='--', alpha=0.7)
        ax.axhline(expected_doppler_pos, color='cyan', lw=0.6, ls='--', alpha=0.4)
        ax.axhline(expected_doppler_neg, color='cyan', lw=0.6, ls='--', alpha=0.4)
        ax.scatter([args.expected_range_m], [expected_doppler_pos],
                   s=140, marker='o', edgecolor='cyan', facecolor='none', lw=1.5)
        ax.scatter([args.expected_range_m], [expected_doppler_neg],
                   s=140, marker='o', edgecolor='cyan', facecolor='none', lw=1.5, alpha=0.6)
        ax.axhline(0, color='white', lw=0.3, alpha=0.3)
        ax.set_xlabel('range (m)')
        ax.set_ylabel('Doppler freq (Hz)')
        ax.set_title(f'airborne1 t={t_s:.2f}s — RD map BG-subtracted '
                     f'(moving targets only). cyan = expected drone '
                     f'(GUI: 44 m, ±2.8 m/s)')
        cb = plt.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label('residual dB')

        # Right panel: track view = range vs time, taking max over Doppler
        # excluding zero-Doppler ±100 Hz. Shows moving targets only.
        ax2 = axes[1]
        # exclude zero-Doppler
        exclude_mask = np.abs(doppler_freqs) > 200
        # max over Doppler in non-zero band per (frame, range)
        track_data = res_db[:, exclude_mask, :].max(axis=1)   # (n_frames, n_range)
        time_axis = np.arange(n_frames) * 0.05 + args.t_start
        ax2.imshow(track_data.T, aspect='auto', origin='lower',
                   extent=[time_axis[0], time_axis[-1], range_m[0], range_m[-1]],
                   cmap='inferno', vmin=vmin_res, vmax=vmax_res)
        ax2.axvline(t_s, color='lime', lw=0.7, alpha=0.7)
        ax2.axhline(args.expected_range_m, color='cyan', lw=0.6, ls='--', alpha=0.7)
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('range (m)')
        ax2.set_title(f'Track view: max-over-Doppler (excl. ±200Hz) per range bin\n'
                      f'Bright streaks = moving targets vs time')

        out_path = out_dir / f'rd_t{t_s:06.2f}.png'
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        rendered += 1

    print(f'  Rendered {rendered} frames')

    # Summary: window-averaged residual
    res_db_avg = 20 * np.log10(residuals.mean(axis=0) + 1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                              gridspec_kw={'width_ratios': [3, 1.3]})
    ax = axes[0]
    extent = [range_m[0], range_m[-1], doppler_freqs[0], doppler_freqs[-1]]
    im = ax.imshow(res_db_avg, aspect='auto', origin='lower',
                   extent=extent, cmap='inferno',
                   vmin=vmin_res, vmax=vmax_res)
    ax.axvline(args.expected_range_m, color='cyan', lw=0.7, ls='--')
    ax.axhline(expected_doppler_pos, color='cyan', lw=0.6, ls='--', alpha=0.4)
    ax.axhline(expected_doppler_neg, color='cyan', lw=0.6, ls='--', alpha=0.4)
    ax.scatter([args.expected_range_m], [expected_doppler_pos],
               s=160, marker='o', edgecolor='cyan', facecolor='none', lw=1.5,
               label='expected: 44m, +2.8m/s')
    ax.scatter([args.expected_range_m], [expected_doppler_neg],
               s=160, marker='o', edgecolor='cyan', facecolor='none', lw=1.5,
               alpha=0.6, label='expected: 44m, -2.8m/s')
    ax.set_xlabel('range (m)')
    ax.set_ylabel('Doppler freq (Hz)')
    ax.set_title(f'WINDOW-AVG residual RD map (background-subtracted, MAGNITUDE-mean)')
    ax.legend(loc='upper right', fontsize=9)
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label('residual dB')

    # Range vs time tracking
    ax2 = axes[1]
    exclude_mask = np.abs(doppler_freqs) > 200
    track_data = res_db[:, exclude_mask, :].max(axis=1)
    time_axis = np.arange(n_frames) * 0.05 + args.t_start
    ax2.imshow(track_data.T, aspect='auto', origin='lower',
               extent=[time_axis[0], time_axis[-1], range_m[0], range_m[-1]],
               cmap='inferno', vmin=vmin_res, vmax=vmax_res)
    ax2.axhline(args.expected_range_m, color='cyan', lw=0.6, ls='--', alpha=0.7,
                label='expected drone @ 44m')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('range (m)')
    ax2.set_title('Range-vs-time of moving targets (max over Doppler excl ±200Hz)')
    ax2.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    summary = out_dir / 'WINDOW_SUMMARY.png'
    fig.savefig(summary, dpi=140)
    plt.close(fig)
    print(f'  Wrote summary: {summary}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
