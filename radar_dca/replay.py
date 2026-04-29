"""Offline replay of a DCA1000 raw-ADC `.bin` capture through the PMM
detector pipeline.

Lets us validate the radar_dca pipeline against a recorded capture
without needing the chip live. This is the bench-test path that
unblocks PMM detector tuning before driveway tests.

Usage:
    python -m radar_dca.replay <bin_path> --cfg studio_capture/fpv_long_range.mmwave.json

Output:
    Per-frame summary line with: frame_idx, drone_hits, best_range_m,
    best_blade_freq_hz, best_band_snr_db.

    A final summary at the end with: total frames, total hits, best
    overall SNR, % of frames with at least one hit.

This is also a smoke test for ``radar_dca.bin_parser`` — if your
.bin doesn't have the layout we expect, you'll see it here as garbage
range bins.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Make the repo importable when running as a script
import os
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..")))

from radar_dca.bin_parser import (  # noqa: E402
    CaptureDims, dims_from_mmwave_json,
    parse_bin_streaming, range_fft, integrate_rx,
)
from radar_dca.pmm_detector import scan_range_bins, PMMResult  # noqa: E402


def replay(
    bin_path: Path,
    dims: CaptureDims,
    *,
    pmm_band_low_hz: float = 50.0,
    pmm_band_high_hz: float = 500.0,
    pmm_threshold_db: float = 6.0,
    max_frames: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run PMM scan over every frame in ``bin_path``. Return summary stats."""
    n_frames_processed = 0
    n_total_hits = 0
    best_snr_db = float("-inf")
    best_range_m = float("nan")
    best_blade_freq_hz = float("nan")
    frames_with_hits = 0
    t_start = time.monotonic()

    for frame_idx, cube in parse_bin_streaming(bin_path, dims):
        if max_frames is not None and frame_idx >= max_frames:
            break

        # cube shape: (n_chirps, n_samples, n_rx) complex
        # 1. Range FFT across fast-time axis
        rng_fft = range_fft(cube)
        # 2. Coherent sum across RX (broadside beam)
        slow_grid = integrate_rx(rng_fft)  # (n_chirps, n_range_bins)
        # 3. Reorder to (n_range_bins, n_chirps) for scan_range_bins
        slow_grid = slow_grid.T  # (n_range_bins, n_chirps)

        # 4. PMM scan
        hits: List[tuple] = scan_range_bins(
            slow_grid,
            prf_hz=dims.prf_hz,
            band_low_hz=pmm_band_low_hz,
            band_high_hz=pmm_band_high_hz,
            threshold_db=pmm_threshold_db,
        )

        if hits:
            frames_with_hits += 1
            n_total_hits += len(hits)
            for range_bin, result in hits:
                if result.band_snr_db > best_snr_db:
                    best_snr_db = result.band_snr_db
                    best_range_m = range_bin * dims.range_resolution_m
                    best_blade_freq_hz = result.blade_freq_hz

        if verbose:
            if hits:
                top = max(hits, key=lambda h: h[1].band_snr_db)
                rng_m = top[0] * dims.range_resolution_m
                print(f"  frame {frame_idx:4d}: {len(hits):2d} hits | "
                      f"top: {rng_m:6.2f} m, blade={top[1].blade_freq_hz:5.1f} Hz, "
                      f"SNR={top[1].band_snr_db:5.1f} dB")
            else:
                print(f"  frame {frame_idx:4d}: 0 hits")

        n_frames_processed += 1

    wall_s = time.monotonic() - t_start

    summary = {
        "n_frames_processed": n_frames_processed,
        "n_total_hits": n_total_hits,
        "frames_with_hits": frames_with_hits,
        "best_snr_db": best_snr_db,
        "best_range_m": best_range_m,
        "best_blade_freq_hz": best_blade_freq_hz,
        "wall_s": wall_s,
        "frames_per_s": n_frames_processed / wall_s if wall_s > 0 else 0.0,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_path", help="Path to adc_data_Raw_0.bin")
    ap.add_argument("--cfg", required=True,
                    help="Path to the .mmwave.json that produced the capture")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Process at most N frames (default: all)")
    ap.add_argument("--band-low", type=float, default=50.0,
                    help="PMM blade-rate search band lower bound (Hz)")
    ap.add_argument("--band-high", type=float, default=500.0,
                    help="PMM blade-rate search band upper bound (Hz)")
    ap.add_argument("--threshold", type=float, default=6.0,
                    help="PMM detection threshold (dB above noise floor)")
    ap.add_argument("--quiet", action="store_true",
                    help="Skip per-frame log; print summary only")
    args = ap.parse_args()

    dims = dims_from_mmwave_json(args.cfg)
    bin_path = Path(args.bin_path)
    if not bin_path.exists():
        print(f"FATAL: {bin_path} not found")
        return 1

    print(f"Replaying {bin_path}")
    print(f"  cfg:                 {args.cfg}")
    print(f"  n_rx:                {dims.n_rx}")
    print(f"  n_tx:                {dims.n_tx}")
    print(f"  n_samples:           {dims.n_samples}")
    print(f"  n_chirps_per_frame:  {dims.n_chirps_per_frame}")
    print(f"  PRF:                 {dims.prf_hz:.0f} Hz "
          f"(chirp period {dims.chirp_period_s*1e6:.1f} us)")
    print(f"  range res:           {dims.range_resolution_m*100:.1f} cm")
    print(f"  max range:           {dims.max_range_m:.1f} m")
    print(f"  bytes/frame:         {dims.bytes_per_frame}")
    print(f"  PMM band:            [{args.band_low:.0f}, {args.band_high:.0f}] Hz "
          f"@ {args.threshold:.1f} dB")
    print()

    summary = replay(
        bin_path, dims,
        pmm_band_low_hz=args.band_low,
        pmm_band_high_hz=args.band_high,
        pmm_threshold_db=args.threshold,
        max_frames=args.max_frames,
        verbose=not args.quiet,
    )

    print()
    print("=" * 60)
    print(f"Summary:")
    print(f"  frames processed:   {summary['n_frames_processed']}")
    print(f"  frames with a hit:  {summary['frames_with_hits']} "
          f"({100*summary['frames_with_hits']/max(summary['n_frames_processed'],1):.0f}%)")
    print(f"  total PMM hits:     {summary['n_total_hits']}")
    if summary['best_snr_db'] > float("-inf"):
        print(f"  best hit:           range={summary['best_range_m']:.2f} m, "
              f"blade={summary['best_blade_freq_hz']:.1f} Hz, "
              f"SNR={summary['best_snr_db']:.1f} dB")
    else:
        print(f"  best hit:           NO HITS")
    print(f"  wall:               {summary['wall_s']:.2f} s "
          f"({summary['frames_per_s']:.1f} frames/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
