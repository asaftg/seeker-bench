"""Stage 2 SECOND PASS with per-frame phase correction.

The cross-frame phase coherence test found a consistent ~0.83 rad
systematic phase jump at every frame boundary (PLL retune during the
~25 ms quiet interval between frames). Without correction, our Stage 2
1-sec coherent integration was somewhere between coherent (intra-frame)
and incoherent (inter-frame), giving ~13 dB gain instead of 26 dB.

This pass corrects the per-frame phase using a strong static reference
range bin. After correction, all 20 frames in the 1-sec window share a
common phase reference, and the coherent sum should give the full ~26 dB.

If the airborne1 30-40s drone becomes visible after this correction,
that's the win. If not, we have stronger evidence the 2-RX SNR limit
is the floor.
"""
import sys
import numpy as np
from scipy import fft as scipy_fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from tools.diagnose_pmm._common_fixed import iter_real_frames, resolve_recording, stage1_range_fft

EXPECTED_RANGE_M = 44.0
EXPECTED_VELOCITY_MPS = 2.8
EXPECTED_BIN = int(round(EXPECTED_RANGE_M / 2.638))
EXPECTED_DOPPLER = 2 * EXPECTED_VELOCITY_MPS / (3e8 / 77e9)


def compute_rd_with_phase_correction(rec, t_start, n_frames, phase_ref_bin=None):
    """Compute slot-0 RD maps with per-frame phase correction.

    phase_ref_bin: if given, use this range bin as the per-frame phase
    reference. If None, auto-pick the brightest static bin.
    """
    sf = int(t_start * 20)
    win = np.hanning(128).astype(np.float32)

    # First pass: collect all frames, find phase reference if not provided
    rfft_buf = []
    for k, (idx, cube) in enumerate(iter_real_frames(rec.bin_path, rec.dims, start_frame=sf, max_frames=n_frames)):
        rfft = stage1_range_fft(cube)
        slot0 = rfft[0::6, :, :].sum(axis=-1)  # (128, 97) RX-summed
        rfft_buf.append(slot0)
    rfft_buf = np.array(rfft_buf)  # (n_frames, 128, 97)

    if phase_ref_bin is None:
        # Brightest static (excl DC)
        mean_mag = np.abs(rfft_buf).mean(axis=(0, 1))
        mean_mag[0] = 0
        phase_ref_bin = int(np.argmax(mean_mag))
    print(f"  Phase reference bin: {phase_ref_bin} ({phase_ref_bin*2.638:.1f} m)")

    # Per-frame phase correction: each frame's phase_ref bin should have
    # constant phase across the slot-0 chirps (it's static). Take the
    # mean angle across the 128 chirps of that frame at phase_ref_bin and
    # use it as the per-frame phase to remove.
    n_f, n_chirps, n_range = rfft_buf.shape
    rfft_corrected = np.zeros_like(rfft_buf)
    per_frame_offset = np.zeros(n_f, dtype=np.complex128)
    for fi in range(n_f):
        ref = rfft_buf[fi, :, phase_ref_bin]
        # Use median direction (more robust)
        z = ref.sum()
        offset_phase = np.angle(z)
        per_frame_offset[fi] = np.exp(-1j * offset_phase)
        rfft_corrected[fi] = rfft_buf[fi] * per_frame_offset[fi]

    # Compute Doppler FFT for each frame's corrected slot-0 cube
    rd_buf = np.zeros((n_f, 128, n_range), dtype=np.complex128)
    for fi in range(n_f):
        rd = scipy_fft.fftshift(
            scipy_fft.fft(rfft_corrected[fi] * win[:, None], n=128, axis=0),
            axes=0,
        )
        rd_buf[fi] = rd

    return rd_buf, phase_ref_bin


def main():
    print("=" * 70)
    print("Stage 2 with per-frame phase correction — airborne1 30-40s")
    print("=" * 70)

    # 1. Compute COMPLEX BG reference from background.bin (with phase correction)
    print("\n1. BG reference (phase-corrected) from background.bin...")
    bg = resolve_recording("background")
    bg_rd, bg_ref_bin = compute_rd_with_phase_correction(bg, 0, 200)
    bg_rd_mean = bg_rd.mean(axis=0)  # (128, n_range)
    print(f"  BG RD mean computed, used phase ref bin {bg_ref_bin}")

    # 2. airborne1 30-40s with phase correction (use SAME ref bin so refs match)
    print("\n2. airborne1 30-40s (phase-corrected)...")
    ab = resolve_recording("airborne1")
    ab_rd, ab_ref_bin = compute_rd_with_phase_correction(ab, 30.0, 200, phase_ref_bin=bg_ref_bin)
    print(f"  airborne1 RD computed (ref bin {ab_ref_bin})")

    # 3. Subtract complex BG and 1-second coherent integration
    print("\n3. BG-subtract and 1-second coherent integration...")
    residuals = ab_rd - bg_rd_mean[None, :, :]
    N_INT = 20
    n_frames = ab_rd.shape[0]
    n_starts = n_frames - N_INT
    integrated = np.zeros((n_starts, 128, 97), dtype=np.complex128)
    for k0 in range(n_starts):
        integrated[k0] = residuals[k0:k0 + N_INT].sum(axis=0)

    # 4. Look at EO oracle cell
    prf_va = ab.dims.per_va_prf_hz
    doppler_freqs = np.fft.fftshift(np.fft.fftfreq(128, d=1.0 / prf_va))
    range_m = np.arange(97) * ab.dims.range_resolution_m
    dop_bin_pos = int(np.argmin(np.abs(doppler_freqs - EXPECTED_DOPPLER)))
    dop_bin_neg = int(np.argmin(np.abs(doppler_freqs - (-EXPECTED_DOPPLER))))
    print(f"\n4. EO oracle cell: bin {EXPECTED_BIN} ({EXPECTED_RANGE_M} m), "
          f"Doppler ±{EXPECTED_DOPPLER:.0f} Hz")

    dop_tol = 5
    track_pos = np.abs(integrated[:, max(0, dop_bin_pos - dop_tol):dop_bin_pos + dop_tol + 1, :]).max(axis=1)
    track_neg = np.abs(integrated[:, max(0, dop_bin_neg - dop_tol):dop_bin_neg + dop_tol + 1, :]).max(axis=1)
    track_either = np.maximum(track_pos, track_neg)
    track_either_db = 20 * np.log10(track_either + 1e-3)

    target_track = track_either_db[:, EXPECTED_BIN]
    control_track = track_either_db[:, EXPECTED_BIN + 10:EXPECTED_BIN + 30].max(axis=1)
    median_target = float(np.median(target_track))
    median_control = float(np.median(control_track))
    diff = median_target - median_control

    print(f"\n5. Result with phase correction:")
    print(f"   Median magnitude at bin {EXPECTED_BIN} ({EXPECTED_RANGE_M} m): {median_target:.1f} dB")
    print(f"   Median magnitude at neighbors (60-100 m): {median_control:.1f} dB")
    print(f"   Difference: {diff:+.1f} dB  (previous, no phase correction: -2.2 dB)")

    # Also report best single-window
    best_k = int(np.argmax(target_track))
    print(f"   Best window: t={(int(30*20)+best_k+N_INT/2)/20:.2f}s  "
          f"target={target_track[best_k]:.1f} dB  "
          f"control={control_track[best_k]:.1f} dB  "
          f"diff={target_track[best_k]-control_track[best_k]:+.1f} dB")

    # Plot
    out = Path("runs/stage2_phase_corrected")
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    k0_show = n_starts // 2
    extent = [range_m[0], range_m[-1], doppler_freqs[0], doppler_freqs[-1]]

    ax = axes[0, 0]
    im = ax.imshow(20 * np.log10(np.abs(integrated[k0_show]) + 1e-3),
                   aspect="auto", origin="lower", extent=extent, cmap="inferno")
    ax.axvline(EXPECTED_RANGE_M, color="cyan", lw=0.8, ls="--", alpha=0.8)
    ax.axhline(EXPECTED_DOPPLER, color="cyan", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(-EXPECTED_DOPPLER, color="cyan", lw=0.6, ls="--", alpha=0.5)
    ax.scatter([EXPECTED_RANGE_M], [EXPECTED_DOPPLER], s=180, marker="o",
               edgecolor="cyan", facecolor="none", lw=1.6)
    ax.scatter([EXPECTED_RANGE_M], [-EXPECTED_DOPPLER], s=180, marker="o",
               edgecolor="cyan", facecolor="none", lw=1.6, alpha=0.6)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(f"Phase-corrected, 1-sec coherent RD map (k0={k0_show})")
    plt.colorbar(im, ax=ax, fraction=0.04)

    ax = axes[0, 1]
    time_axis = (int(30 * 20) + np.arange(n_starts) + N_INT / 2) / 20.0
    ax.imshow(track_either_db.T, aspect="auto", origin="lower",
              extent=[time_axis[0], time_axis[-1], range_m[0], range_m[-1]],
              cmap="inferno")
    ax.axhline(EXPECTED_RANGE_M, color="cyan", lw=0.6, ls="--", alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    ax.set_title(f"Range-vs-time at expected Doppler ±{dop_tol}-bin")

    ax = axes[1, 0]
    ax.plot(time_axis, target_track, "r-", label=f"bin {EXPECTED_BIN} ({EXPECTED_RANGE_M}m, EO oracle)")
    ax.plot(time_axis, control_track, "k-", alpha=0.5, label="control 60-100m")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("magnitude (dB)")
    ax.set_title(f"EO oracle vs control. diff={diff:+.1f} dB (was -2.2 without phase correction)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    spec = 20 * np.log10(np.abs(integrated[best_k, :, EXPECTED_BIN]) + 1e-3)
    ax.plot(doppler_freqs, spec, "b-")
    ax.axvline(EXPECTED_DOPPLER, color="cyan", lw=0.6, ls="--")
    ax.axvline(-EXPECTED_DOPPLER, color="cyan", lw=0.6, ls="--")
    ax.set_xlabel("Doppler (Hz)")
    ax.set_ylabel("mag (dB)")
    ax.set_title(f"Slow-time spectrum at EO cell (bin {EXPECTED_BIN}), best window k0={best_k}")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Stage 2 PHASE-CORRECTED — diff at EO oracle = {diff:+.1f} dB",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "phase_corrected_summary.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote: {out/'phase_corrected_summary.png'}")

    if diff > 3:
        print(f"\nVERDICT: WIN. Drone signature DETECTED with phase correction (+{diff:.1f} dB).")
    elif diff > 0:
        print(f"\nVERDICT: MARGINAL. Phase correction improves but not decisive.")
    else:
        print(f"\nVERDICT: STILL NEGATIVE. Phase correction did not help. 2-RX SNR is genuinely too low.")


if __name__ == "__main__":
    main()
