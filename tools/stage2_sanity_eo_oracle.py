"""Stage 2 single sanity pass on airborne1 30-40s.

Applies:
  - corrected layout (RX0+RX1 only)
  - complex BG subtraction using background.bin's mean RD
  - 1-second coherent multi-frame integration
  - EO oracle: known drone cell at 44 m / 2.8 m/s (from GUI screenshot)

If the integrated magnitude at the EO-oracle cell is significantly
above neighboring range bins: WIN.
Else: write the negative result; the recording's airborne1 drone is
beyond the 2-RX SNR floor with current algorithms.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as scipy_fft

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from tools.diagnose_pmm._common_fixed import (   # noqa: E402
    iter_real_frames, resolve_recording, stage1_range_fft,
)


def main() -> int:
    print("=== Stage 2 sanity pass — airborne1 30-40s ===\n")

    EXPECTED_RANGE_M = 44.0
    EXPECTED_VELOCITY_MPS = 2.8
    EXPECTED_BIN = int(round(EXPECTED_RANGE_M / 2.638))
    print(f"EO oracle: drone at {EXPECTED_RANGE_M} m / {EXPECTED_VELOCITY_MPS} m/s (range bin {EXPECTED_BIN})")

    # 1. complex BG reference from background.bin (10s)
    print("\n1. Computing COMPLEX BG reference from background.bin...")
    bg = resolve_recording("background")
    N_BG = 200
    bg_rd_accum = np.zeros((128, 97), dtype=np.complex128)
    n = 0
    win = np.hanning(128).astype(np.float32)
    for k, (idx, cube) in enumerate(iter_real_frames(bg.bin_path, bg.dims, max_frames=N_BG)):
        rfft = stage1_range_fft(cube)
        slot0 = rfft[0::6, :, :].sum(axis=-1)
        rd = scipy_fft.fftshift(scipy_fft.fft(slot0 * win[:, None], n=128, axis=0), axes=0)
        bg_rd_accum += rd
        n += 1
    bg_rd_complex = bg_rd_accum / n
    print(f"  Averaged {n} background frames")

    # 2. airborne1 30-40s residuals (complex - BG)
    print("\n2. airborne1 30-40s with complex BG subtraction...")
    ab = resolve_recording("airborne1")
    sf = int(30.0 * 20)
    n_frames = 200
    ab_rd_residuals = np.zeros((n_frames, 128, 97), dtype=np.complex128)
    for k, (idx, cube) in enumerate(iter_real_frames(ab.bin_path, ab.dims, start_frame=sf, max_frames=n_frames)):
        rfft = stage1_range_fft(cube)
        slot0 = rfft[0::6, :, :].sum(axis=-1)
        rd = scipy_fft.fftshift(scipy_fft.fft(slot0 * win[:, None], n=128, axis=0), axes=0)
        ab_rd_residuals[k] = rd - bg_rd_complex
    print(f"  Computed {n_frames} BG-subtracted RD frames")

    # 3. 1-second coherent integration (20 frames)
    print("\n3. 1-second coherent integration (20 frames)...")
    N_INT = 20
    n_starts = n_frames - N_INT
    integrated = np.zeros((n_starts, 128, 97), dtype=np.complex128)
    for k0 in range(n_starts):
        integrated[k0] = ab_rd_residuals[k0:k0 + N_INT].sum(axis=0)
    print(f"  {n_starts} integration windows")

    # 4. Track view at EO oracle cell
    prf_va = ab.dims.per_va_prf_hz
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(128, d=1.0 / prf_va))
    range_m = np.arange(97) * ab.dims.range_resolution_m
    expected_dop_pos = 2 * EXPECTED_VELOCITY_MPS / (3e8 / 77e9)
    expected_dop_neg = -expected_dop_pos
    dop_bin_pos = int(np.argmin(np.abs(doppler_freqs - expected_dop_pos)))
    dop_bin_neg = int(np.argmin(np.abs(doppler_freqs - expected_dop_neg)))
    print(f"\n4. EO oracle cell: bin {EXPECTED_BIN}, expected Doppler ±{abs(expected_dop_pos):.0f} Hz")
    print(f"   Doppler bins: pos={dop_bin_pos} ({doppler_freqs[dop_bin_pos]:.0f} Hz), "
          f"neg={dop_bin_neg} ({doppler_freqs[dop_bin_neg]:.0f} Hz)")

    dop_tol = 5
    track_pos = np.abs(integrated[:, max(0, dop_bin_pos - dop_tol):dop_bin_pos + dop_tol + 1, :]).max(axis=1)
    track_neg = np.abs(integrated[:, max(0, dop_bin_neg - dop_tol):dop_bin_neg + dop_tol + 1, :]).max(axis=1)
    track_either = np.maximum(track_pos, track_neg)
    track_either_db = 20 * np.log10(track_either + 1e-3)

    target_track = track_either_db[:, EXPECTED_BIN]
    control_track = track_either_db[:, EXPECTED_BIN + 10:EXPECTED_BIN + 30].max(axis=1)
    median_at_target = float(np.median(target_track))
    median_at_control = float(np.median(control_track))
    diff = median_at_target - median_at_control

    print(f"\n5. Result:")
    print(f"   Median magnitude at bin {EXPECTED_BIN} ({EXPECTED_RANGE_M} m): {median_at_target:.1f} dB")
    print(f"   Median magnitude at neighboring bins (60-100 m): {median_at_control:.1f} dB")
    print(f"   Difference: {diff:+.1f} dB")

    # Plot summary
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    k0_show = n_starts // 2
    extent = [range_m[0], range_m[-1], doppler_freqs[0], doppler_freqs[-1]]

    ax = axes[0, 0]
    im = ax.imshow(20 * np.log10(np.abs(integrated[k0_show]) + 1e-3),
                   aspect="auto", origin="lower", extent=extent, cmap="inferno")
    ax.axvline(EXPECTED_RANGE_M, color="cyan", lw=0.8, ls="--", alpha=0.8)
    ax.axhline(expected_dop_pos, color="cyan", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(expected_dop_neg, color="cyan", lw=0.6, ls="--", alpha=0.5)
    ax.scatter([EXPECTED_RANGE_M], [expected_dop_pos], s=180, marker="o",
               edgecolor="cyan", facecolor="none", lw=1.6)
    ax.scatter([EXPECTED_RANGE_M], [expected_dop_neg], s=180, marker="o",
               edgecolor="cyan", facecolor="none", lw=1.6, alpha=0.6)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(f"BG-subtracted, 1-sec coherent RD map (k0={k0_show})")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[0, 1]
    time_axis = (sf + np.arange(n_starts) + N_INT / 2) / 20.0
    ax.imshow(track_either_db.T, aspect="auto", origin="lower",
              extent=[time_axis[0], time_axis[-1], range_m[0], range_m[-1]],
              cmap="inferno")
    ax.axhline(EXPECTED_RANGE_M, color="cyan", lw=0.6, ls="--", alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    ax.set_title(f"Range-vs-time at expected Doppler ±{dop_tol}-bin window")

    ax = axes[1, 0]
    ax.plot(time_axis, target_track, "r-", label=f"bin {EXPECTED_BIN} ({EXPECTED_RANGE_M}m, EO oracle)")
    ax.plot(time_axis, control_track, "k-", alpha=0.5, label="max bins 60-100m (control)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title("EO oracle vs control over time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    spec_at_oracle = 20 * np.log10(np.abs(integrated[k0_show, :, EXPECTED_BIN]) + 1e-3)
    ax.plot(doppler_freqs, spec_at_oracle, "b-")
    ax.axvline(expected_dop_pos, color="cyan", lw=0.6, ls="--")
    ax.axvline(expected_dop_neg, color="cyan", lw=0.6, ls="--")
    ax.set_xlabel("Doppler (Hz)")
    ax.set_ylabel("mag (dB)")
    ax.set_title(f"Slow-time spectrum at EO cell (bin {EXPECTED_BIN}, k0={k0_show})")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Stage 2 sanity pass — BG-subtract + 1s coherent integration\n"
                 f"EO oracle: drone at {EXPECTED_RANGE_M} m / {EXPECTED_VELOCITY_MPS} m/s — diff={diff:+.1f} dB",
                 fontsize=12)
    fig.tight_layout()
    out = Path("runs/stage2_sanity/airborne1_30_40_eo_oracle.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nWrote summary: {out}")

    if diff > 3:
        print(f"\nVERDICT: WIN. Drone signature DETECTED at EO oracle cell.")
    else:
        print(f"\nVERDICT: NEGATIVE. Drone NOT detected at EO oracle cell.")
        print(f"  BG-subtract + 1s coherent integration insufficient with 2-RX coherent SNR.")
        print(f"  Recommend re-flying with 4-RX firmware fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
