"""Phase-0 diagnostic 0.1: per-range-bin slow-time spectrogram.

For a chosen recording and time window, compute the per-frame slow-time
spectrum at every range bin and render heatmaps with the propeller
blade-pass band [rpm_low, rpm_high] Hz overlaid.

Per-frame (NOT cross-frame) FFT is used because there is a ~25 ms quiet
interval between every 50 ms frame; concatenating slow-time across that
gap smears any periodic signal. Each frame's 768 chirps at the chip
PRF (30478 Hz) zero-padded to 4096 gives ~7.4 Hz spectral resolution,
enough to resolve a 300-1100 Hz blade-pass comb.

DDMA is NOT un-mixed in this diagnostic (per the user's "quick-and-dirty
first" decision). A real propeller comb at f_bp will appear at f_bp
itself plus three DDMA-fold copies at f_bp ± PRF/4, ± PRF/2. We plot
up to 2.5 kHz so the fundamental band and the first negative-Doppler
mirror are visible without wading through the fold copies.

The script ranks range bins by their in-band (300-1100 Hz) power
excluding zero-Doppler, so the operator only has to look at the top-K
candidate bins per recording.

Usage:
    python -m tools.diagnose_pmm.04_spectrogram_grid drone_fly --t-start 0 --t-end 25
    python -m tools.diagnose_pmm.04_spectrogram_grid airborne1 --t-start 30 --t-end 35
    python -m tools.diagnose_pmm.04_spectrogram_grid airborne1 --t-start 60 --t-end 65
    python -m tools.diagnose_pmm.04_spectrogram_grid background --t-start 5 --t-end 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import fft as scipy_fft

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))

from tools.diagnose_pmm._common import (  # noqa: E402
    in_notch_zone,
    iter_real_frames,
    notch_zones,
    resolve_recording,
    stage1_range_fft,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", help="drone_fly|airborne1|background or path to .meta.yaml")
    ap.add_argument("--t-start", type=float, required=True, help="window start (s)")
    ap.add_argument("--t-end", type=float, required=True, help="window end (s)")
    ap.add_argument("--rpm-low", type=float, default=300.0)
    ap.add_argument("--rpm-high", type=float, default=1100.0)
    ap.add_argument("--top-k", type=int, default=12,
                    help="Render top-K bins by in-band power (excluding notch)")
    ap.add_argument("--zero-pad-mult", type=int, default=4,
                    help="Slow-time FFT zero-pad multiplier (resolution boost)")
    ap.add_argument("--apply-notch", action="store_true",
                    help="Apply the existing 24-bin chip-artifact notch (off by "
                         "default so we see what the bins look like before notching)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default: runs/spectrogram_grid/<rec>_<tstart>-<tend>)")
    args = ap.parse_args()

    repo_root = _THIS.parents[1]
    rec = resolve_recording(args.recording)
    suffix = "_notched" if args.apply_notch else ""
    out_dir = (
        Path(args.out) if args.out else
        repo_root / "runs" / "spectrogram_grid" /
        f"{rec.name.replace(' ', '_')}_{args.t_start:.0f}-{args.t_end:.0f}{suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_s = rec.dims.framePeriodicity_s
    f_start = int(round(args.t_start / fp_s))
    f_end = int(round(args.t_end / fp_s))
    n_frames = f_end - f_start
    if n_frames <= 0:
        print("FATAL: t-end must be > t-start")
        return 1

    n_chirps = rec.dims.n_chirps
    n_range = rec.dims.n_range_bins
    prf = rec.dims.prf_hz
    n_fft = max(1024, args.zero_pad_mult * 2 ** int(np.ceil(np.log2(n_chirps))))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / prf)
    pos = freqs >= 0
    freqs_pos = freqs[pos]                              # (n_fft/2 + 1,)
    bin_low_idx = int(np.searchsorted(freqs_pos, args.rpm_low))
    bin_high_idx = int(np.searchsorted(freqs_pos, args.rpm_high))
    plot_freq_max = max(2500.0, args.rpm_high * 2.5)
    plot_high_idx = int(np.searchsorted(freqs_pos, plot_freq_max))
    notch = notch_zones(rec.dims.n_samples, radius=6)
    nyq_dop_idx = int(np.searchsorted(freqs_pos, 50.0))  # zero-Doppler exclude band

    print(f"Recording: {rec.name}")
    print(f"  window:        {args.t_start:.1f}-{args.t_end:.1f} s "
          f"(frames {f_start}-{f_end}, {n_frames} frames)")
    print(f"  PRF (total):   {prf:.0f} Hz  (per-VA {rec.dims.per_va_prf_hz:.0f} Hz after un-fold)")
    print(f"  slow-FFT N:    {n_fft}  ({prf/n_fft:.2f} Hz/bin)")
    print(f"  band:          [{args.rpm_low:.0f}, {args.rpm_high:.0f}] Hz "
          f"(FFT bins {bin_low_idx}-{bin_high_idx})")
    print(f"  notch zones (radius=6): {notch}  apply={args.apply_notch}")
    print(f"  output:        {out_dir}")
    print()

    # ------------------------------------------------------------------ load
    # Per-frame slow-time spectrum at each range bin, RX-summed.
    print(f"  parsing & FFTing {n_frames} frames...", end=" ", flush=True)
    spec = np.zeros((n_frames, n_range, plot_high_idx), dtype=np.float32)
    win_slow = np.hanning(n_chirps).astype(np.float32)
    for k, (idx, cube) in enumerate(
        iter_real_frames(rec.bin_path, rec.dims, start_frame=f_start,
                         max_frames=n_frames)
    ):
        rfft = stage1_range_fft(cube)              # (n_chirps, n_range, n_rx)
        if args.apply_notch:
            for lo, hi in notch:
                rfft[:, lo:hi, :] = 0
        slow = rfft.sum(axis=2)                    # (n_chirps, n_range), RX-summed
        slow_w = slow * win_slow[:, None]
        sp = np.abs(scipy_fft.fft(slow_w, n=n_fft, axis=0, workers=2))
        spec[k] = sp[:plot_high_idx, :].T.astype(np.float32)  # (n_range, plot_high_idx)
    print("done")

    # ----------------------------------------------------- rank by in-band power
    # Mean magnitude in [rpm_low, rpm_high] excluding bins that fall in the
    # ±50 Hz zero-Doppler exclusion. Subtract a per-bin out-of-band noise
    # floor so we measure ABOVE-floor energy.
    band = spec[:, :, bin_low_idx:bin_high_idx]            # (n_frames, n_range, n_band)
    band_pwr = np.mean(band, axis=(0, 2))                  # (n_range,)
    # Local out-of-band: 1500-2500 Hz (well above the prop band and above
    # the first DDMA-fold replica of the prop band would only show with
    # un-mix; this slice is "noise / sidelobe pedestal").
    oob_lo = int(np.searchsorted(freqs_pos, 1500.0))
    oob_hi = int(np.searchsorted(freqs_pos, 2500.0))
    oob_pwr = np.mean(spec[:, :, oob_lo:oob_hi], axis=(0, 2))  # (n_range,)
    excess_db = 20.0 * np.log10(np.maximum(band_pwr / np.maximum(oob_pwr, 1e-6), 1e-6))

    # CSV: per range bin, in-band power, OOB power, excess dB
    rng_csv = out_dir / "ranking.csv"
    with open(rng_csv, "w") as f:
        f.write("range_bin,range_m,in_notch_zone,band_pwr,oob_pwr,excess_db\n")
        for r in range(n_range):
            f.write(f"{r},{r*rec.dims.range_resolution_m:.2f},"
                    f"{int(in_notch_zone(r, rec.dims.n_samples))},"
                    f"{band_pwr[r]:.3f},{oob_pwr[r]:.3f},{excess_db[r]:.2f}\n")
    print(f"  ranking written to {rng_csv}")

    # Top-K bins, preferring those NOT in the notch zone (so we don't waste
    # plots on bins that were going to be killed anyway).
    order = np.argsort(-excess_db)
    plot_bins = []
    for r in order:
        if len(plot_bins) >= args.top_k:
            break
        if not in_notch_zone(int(r), rec.dims.n_samples):
            plot_bins.append(int(r))
    # Always include a few in-notch bins for comparison if we have headroom
    notch_extras = [int(r) for r in order if in_notch_zone(int(r), rec.dims.n_samples)][:3]
    plot_bins += notch_extras

    print(f"  top-{args.top_k} clean bins by excess dB:")
    for r in plot_bins[: args.top_k]:
        rng_m = r * rec.dims.range_resolution_m
        print(f"    bin {r:3d}  range {rng_m:6.1f} m  in-band={band_pwr[r]:.1f} "
              f"oob={oob_pwr[r]:.1f}  excess={excess_db[r]:+.1f} dB")
    if notch_extras:
        print(f"  + {len(notch_extras)} in-notch bins (for comparison):")
        for r in notch_extras:
            rng_m = r * rec.dims.range_resolution_m
            print(f"    bin {r:3d}  range {rng_m:6.1f} m  excess={excess_db[r]:+.1f} dB (NOTCHED)")

    # --------------------------------------------------------------- plotting
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("FATAL: matplotlib not available; cannot render PDF")
        return 1

    pdf_path = out_dir / "spectrograms.pdf"
    times_s = (np.arange(n_frames) + f_start) * fp_s
    band_lines = [args.rpm_low, args.rpm_high]
    extra_lines = [args.rpm_low * 2, args.rpm_high]  # 2× harmonic markers

    with PdfPages(pdf_path) as pdf:
        # Cover sheet
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(n_range), excess_db, width=0.8,
               color=["crimson" if in_notch_zone(int(r), rec.dims.n_samples)
                      else "steelblue" for r in range(n_range)])
        ax.set_xlabel("range bin")
        ax.set_ylabel("in-band excess (dB above OOB pedestal)")
        ax.set_title(f"{rec.name}  t={args.t_start:.0f}-{args.t_end:.0f}s "
                     f"band {args.rpm_low:.0f}-{args.rpm_high:.0f} Hz "
                     f"(red = chip-artifact notch zone)")
        ax.grid(alpha=0.3)
        # mark notch zones with grey background
        for lo, hi in notch:
            ax.axvspan(lo - 0.5, hi - 0.5, alpha=0.1, color="black")
        fig.tight_layout()
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

        # One page per top-K bin
        for r in plot_bins:
            rng_m = r * rec.dims.range_resolution_m
            in_zone = in_notch_zone(int(r), rec.dims.n_samples)
            S = spec[:, r, :]                              # (n_frames, plot_high_idx)
            S_db = 20.0 * np.log10(S + 1e-6)
            vmax = float(np.percentile(S_db, 99.5))
            vmin = vmax - 40.0

            fig, ax = plt.subplots(figsize=(11, 5))
            extent = [times_s[0], times_s[-1], 0, freqs_pos[plot_high_idx - 1]]
            im = ax.imshow(
                S_db.T, aspect="auto", origin="lower", extent=extent,
                cmap="inferno", vmin=vmin, vmax=vmax,
            )
            for fl in band_lines:
                ax.axhline(fl, color="lime", lw=0.8, alpha=0.7, ls="--")
            ax.axhline(args.rpm_low * 2, color="cyan", lw=0.6, alpha=0.5, ls=":")
            ax.axhline(args.rpm_high * 2, color="cyan", lw=0.6, alpha=0.5, ls=":")
            ax.axhline(args.rpm_low * 3, color="cyan", lw=0.6, alpha=0.4, ls=":")
            zone_tag = "  [IN CHIP-ARTIFACT NOTCH ZONE]" if in_zone else ""
            ax.set_title(
                f"{rec.name}  bin {r}  range {rng_m:.1f} m  "
                f"excess {excess_db[r]:+.1f} dB{zone_tag}\n"
                f"green = blade-pass band [{args.rpm_low:.0f}, {args.rpm_high:.0f}]; "
                f"cyan = 2nd & 3rd harmonics"
            )
            ax.set_xlabel("time (s)")
            ax.set_ylabel("Doppler / slow-time freq (Hz)")
            cb = plt.colorbar(im, ax=ax, fraction=0.04)
            cb.set_label("dB (rel. arb.)")
            fig.tight_layout()
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

    print(f"  wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
