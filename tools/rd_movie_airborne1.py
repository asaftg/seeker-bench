"""Range-Doppler 'movie' for airborne1 30-40s.

For each frame in the window, compute a slot-0-decimated range-Doppler
map (128 Doppler bins x 97 range bins) and render it as a PNG. The
sequence of PNGs is the movie.

Annotates the expected drone position based on the GUI screenshot at
t=0:38: 44 m range, 2.8 m/s radial velocity (sign uncertain, so we
mark both +1436 Hz and -1436 Hz Doppler).

A real drone with a body return at the right range and Doppler should
show up as a bright spot AT the marker. Propeller modulation sidebands
should appear as additional symmetric peaks.
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


def compute_rd_map(cube: np.ndarray, dims) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slot-0 decimated range-Doppler map.

    Inputs: cube (n_chirps=768, n_samples=192, n_rx=2)
    Returns:
      rd_db: (n_doppler=128, n_range=97) magnitude in dB
      doppler_freqs: (128,) Doppler freqs in Hz, fftshifted
      range_m: (97,) range in m
    """
    rfft = stage1_range_fft(cube)              # (768, 97, 2)
    slot0 = rfft[0::6, :, :].sum(axis=-1)      # (128, 97), RX-summed
    win = np.hanning(128).astype(np.float32)
    slot0_w = slot0 * win[:, None]
    rd = scipy_fft.fftshift(scipy_fft.fft(slot0_w, n=128, axis=0), axes=0)
    rd_db = 20 * np.log10(np.abs(rd) + 1e-3)

    prf_va = dims.per_va_prf_hz
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(128, d=1.0 / prf_va))
    range_m = np.arange(rfft.shape[1]) * dims.range_resolution_m
    return rd_db, doppler_freqs, range_m


def doppler_hz_for_velocity(v_mps: float, freq_ghz: float = 77.0) -> float:
    """Doppler frequency for a radial velocity. Positive Doppler = target
    moving toward radar in TI mmwave convention.
    """
    c = 3e8
    wavelength = c / (freq_ghz * 1e9)
    return 2 * v_mps / wavelength


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--recording', default='airborne1')
    ap.add_argument('--t-start', type=float, default=30.0)
    ap.add_argument('--t-end', type=float, default=40.0)
    ap.add_argument('--out-dir', default='runs/rd_movie_airborne1_30_40')
    ap.add_argument('--frame-step', type=int, default=2,
                    help='Render every Nth frame (default: 2 -> 100 frames over 10s)')
    ap.add_argument('--expected-range-m', type=float, default=44.0,
                    help='Expected drone range from GUI/EO (m)')
    ap.add_argument('--expected-velocity-mps', type=float, default=2.8,
                    help='Expected drone radial velocity from GUI (m/s, sign uncertain)')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rec = resolve_recording(args.recording)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Recording: {rec.name}')
    print(f'  Window: t={args.t_start}-{args.t_end}s')
    print(f'  Output: {out_dir}')

    sf = int(args.t_start * 20)
    n_frames_total = int((args.t_end - args.t_start) * 20)

    # Expected drone Doppler
    expected_doppler_pos = doppler_hz_for_velocity(args.expected_velocity_mps)
    expected_doppler_neg = -expected_doppler_pos
    print(f'  Expected drone: range={args.expected_range_m:.1f} m, '
          f'Doppler=±{abs(expected_doppler_pos):.0f} Hz (v=±{args.expected_velocity_mps:.1f} m/s)')
    expected_range_bin = int(round(args.expected_range_m / rec.dims.range_resolution_m))
    print(f'    → range bin {expected_range_bin}')

    # Two-pass: first pass to compute global vmin/vmax, second to render
    print('  Pre-scanning to find dB scale...')
    rd_min = np.inf
    rd_max = -np.inf
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims,
                                                     start_frame=sf,
                                                     max_frames=n_frames_total)):
        if k % args.frame_step != 0:
            continue
        rd_db, _, _ = compute_rd_map(cube, rec.dims)
        rd_min = min(rd_min, np.percentile(rd_db, 5))
        rd_max = max(rd_max, np.percentile(rd_db, 99.9))
    vmin = rd_min
    vmax = rd_max
    print(f'  dB scale: [{vmin:.1f}, {vmax:.1f}]')

    # Render pass
    print('  Rendering frames...')
    n_rendered = 0
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims,
                                                     start_frame=sf,
                                                     max_frames=n_frames_total)):
        if k % args.frame_step != 0:
            continue
        t_s = (sf + k) / 20.0
        rd_db, doppler_freqs, range_m = compute_rd_map(cube, rec.dims)

        fig, ax = plt.subplots(figsize=(11, 7))
        # Image: y=Doppler (rows), x=range (cols). Transpose so range on x.
        # rd_db shape is (128, 97) = (doppler, range). imshow takes (rows, cols)
        # so we want (doppler, range) → display with extent.
        extent = [range_m[0], range_m[-1], doppler_freqs[0], doppler_freqs[-1]]
        im = ax.imshow(rd_db, aspect='auto', origin='lower',
                       extent=extent, cmap='inferno',
                       vmin=vmin, vmax=vmax)
        # Annotate expected drone location
        ax.axvline(args.expected_range_m, color='cyan', lw=0.6, ls='--', alpha=0.6)
        ax.axhline(expected_doppler_pos, color='cyan', lw=0.6, ls='--', alpha=0.4)
        ax.axhline(expected_doppler_neg, color='cyan', lw=0.6, ls='--', alpha=0.4)
        ax.scatter([args.expected_range_m], [expected_doppler_pos],
                   s=80, marker='o', edgecolor='cyan', facecolor='none', lw=1.2,
                   label=f'expected: r={args.expected_range_m:.0f}m, +v={args.expected_velocity_mps:.1f}m/s')
        ax.scatter([args.expected_range_m], [expected_doppler_neg],
                   s=80, marker='o', edgecolor='cyan', facecolor='none', lw=1.2,
                   alpha=0.5,
                   label=f'expected: r={args.expected_range_m:.0f}m, -v={args.expected_velocity_mps:.1f}m/s')
        # Zero-Doppler line
        ax.axhline(0, color='white', lw=0.3, alpha=0.3)

        ax.set_xlabel('range (m)')
        ax.set_ylabel('Doppler freq (Hz) — positive = approaching radar')
        ax.set_title(f'airborne1  t={t_s:.2f}s  frame={sf+k}  '
                     f'(slot-0 decimated, RX0+RX1, per-VA PRF=5080 Hz)')
        ax.legend(loc='upper right', fontsize=7)
        cb = plt.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label('dB')

        # Add a velocity axis on the right
        c = 3e8
        wavelength = c / (77e9)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim()[0] * wavelength / 2,
                     ax.get_ylim()[1] * wavelength / 2)
        ax2.set_ylabel('radial velocity (m/s)')

        out_path = out_dir / f'rd_t{t_s:06.2f}.png'
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        n_rendered += 1
        if n_rendered % 20 == 0:
            print(f'    rendered {n_rendered} frames')

    print(f'  Total rendered: {n_rendered}')
    print()

    # Also produce a 3D summary: range-Doppler map AVERAGED across the
    # entire window, with the expected drone annotation. Easier to spot
    # a faint persistent target.
    print('  Computing window-averaged RD map...')
    rd_accum = np.zeros((128, 97), dtype=np.float64)
    n_acc = 0
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims,
                                                     start_frame=sf,
                                                     max_frames=n_frames_total)):
        rd_db, _, _ = compute_rd_map(cube, rec.dims)
        rd_accum += 10**(rd_db / 20)
        n_acc += 1
    rd_avg = 20 * np.log10(rd_accum / n_acc + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 8))
    extent = [range_m[0], range_m[-1], doppler_freqs[0], doppler_freqs[-1]]
    im = ax.imshow(rd_avg, aspect='auto', origin='lower',
                   extent=extent, cmap='inferno')
    ax.axvline(args.expected_range_m, color='cyan', lw=0.6, ls='--')
    ax.axhline(expected_doppler_pos, color='cyan', lw=0.6, ls='--')
    ax.axhline(expected_doppler_neg, color='cyan', lw=0.6, ls='--', alpha=0.5)
    ax.scatter([args.expected_range_m], [expected_doppler_pos],
               s=120, marker='o', edgecolor='cyan', facecolor='none', lw=1.5,
               label=f'expected drone, +{args.expected_velocity_mps}m/s')
    ax.scatter([args.expected_range_m], [expected_doppler_neg],
               s=120, marker='o', edgecolor='cyan', facecolor='none', lw=1.5,
               alpha=0.5, label=f'expected drone, -{args.expected_velocity_mps}m/s')
    ax.set_xlabel('range (m)')
    ax.set_ylabel('Doppler freq (Hz) — positive = approaching')
    ax.set_title(f'airborne1 t={args.t_start}-{args.t_end}s — INCOHERENT-AVG range-Doppler map\n'
                 f'(should show persistent drone return at the cyan marker)')
    ax.legend(loc='upper right', fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label('dB')
    fig.tight_layout()
    summary_path = out_dir / 'WINDOW_AVERAGE.png'
    fig.savefig(summary_path, dpi=140)
    plt.close(fig)
    print(f'  Wrote window-average: {summary_path}')

    print()
    print(f'Done. {n_rendered} per-frame PNGs in {out_dir}')
    print(f'Window-average summary at {summary_path}')
    print()
    print('To inspect:')
    print(f'  - Open {summary_path} first. If you see a bright peak at the cyan')
    print(f'    marker (44 m, ±1436 Hz), the drone IS visible in our data.')
    print(f'  - Otherwise, it isn\'t — and we have an honest negative result for')
    print(f'    that hardware/range/algorithm combination.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
