"""DDMA slot-0 verification + cross-frame phase coherence test."""
import sys
import numpy as np
from scipy import fft as scipy_fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from tools.diagnose_pmm._common_fixed import iter_real_frames, resolve_recording, stage1_range_fft

print("=" * 70)
print("DDMA SLOT-0 VERIFICATION (synthetic targets)")
print("=" * 70)

# Cfg: 6 chirp slots per loop, 128 loops, all 4 TX active each slot.
# ddmPhaseShiftAntOrder 0 2 3 1 — interpretation: ant_order[i] is the
# DDMA position of physical TX[i]. So phase per TX[i] per slot k =
# ant_order[i] * (2pi/N_slots) * k.
n_slots = 6
n_loops = 128
n_chirps = n_slots * n_loops  # 768
ant_order = [0, 2, 3, 1]
n_tx = 4

per_tx_phase = np.zeros((n_tx, n_chirps))
for chirp_idx in range(n_chirps):
    slot = chirp_idx % n_slots
    for tx_idx in range(n_tx):
        per_tx_phase[tx_idx, chirp_idx] = ant_order[tx_idx] * (2 * np.pi / n_slots) * slot

slot0_chirps = np.arange(0, n_chirps, n_slots)
print("Slot-0 phase per TX (first 5 slot-0 chirps, deg):")
for tx_idx in range(n_tx):
    phases_deg = np.rad2deg(per_tx_phase[tx_idx, slot0_chirps[:5]])
    print(f"  TX{tx_idx}: {phases_deg}")

prf_total = 30478.51
f_target = 1429.0
slow_time_t = np.arange(n_chirps) / prf_total
target_signal = np.exp(1j * 2 * np.pi * f_target * slow_time_t)

# 4-TX coherent test (target reflected by all TX simultaneously)
received_all = np.zeros(n_chirps, dtype=np.complex128)
for tx in range(n_tx):
    received_all += target_signal * np.exp(1j * per_tx_phase[tx, :])
slow_slot0_all = received_all[slot0_chirps]
win = np.hanning(len(slow_slot0_all))
spec_all = np.abs(scipy_fft.fftshift(scipy_fft.fft(slow_slot0_all * win, n=1024)))
freqs = np.fft.fftshift(np.fft.fftfreq(1024, d=n_slots / prf_total))
peak_idx_all = int(np.argmax(spec_all))
print(f"\n4-TX coherent target test (target Doppler {f_target} Hz):")
print(f"  Slot-0 FFT peak at: {freqs[peak_idx_all]:.1f} Hz")
print(f"  Peak magnitude: {spec_all[peak_idx_all]:.0f}")

# Single-TX test
received_tx0 = target_signal * np.exp(1j * per_tx_phase[0, :])
slow_slot0_tx0 = received_tx0[slot0_chirps]
spec_tx0 = np.abs(scipy_fft.fftshift(scipy_fft.fft(slow_slot0_tx0 * win, n=1024)))
peak_idx_tx0 = int(np.argmax(spec_tx0))
print(f"\nTX0-only target test:")
print(f"  Slot-0 FFT peak at: {freqs[peak_idx_tx0]:.1f} Hz")
print(f"  Peak magnitude: {spec_tx0[peak_idx_tx0]:.0f}")
print(f"  4-TX/1-TX ratio: {spec_all[peak_idx_all] / spec_tx0[peak_idx_tx0]:.2f}")
print(f"  Expected ratio = 4.0 if all 4 TX add coherently at slot 0.")

# Test: TX1-only (the highest phase increment, k*pi/3 per slot)
received_tx1 = target_signal * np.exp(1j * per_tx_phase[3, :])  # ant_order[3]=1
slow_slot0_tx1 = received_tx1[slot0_chirps]
spec_tx1 = np.abs(scipy_fft.fftshift(scipy_fft.fft(slow_slot0_tx1 * win, n=1024)))
peak_idx_tx1 = int(np.argmax(spec_tx1))
print(f"\nTX1-only target test:")
print(f"  Slot-0 FFT peak at: {freqs[peak_idx_tx1]:.1f} Hz")
print(f"  Peak magnitude: {spec_tx1[peak_idx_tx1]:.0f}")
print(f"  Should match TX0-only magnitude (slot-0 has all TX at phase 0).")

print()
print("=" * 70)
print("CROSS-FRAME PHASE COHERENCE (background.bin, stationary range bin)")
print("=" * 70)

bg = resolve_recording("background")
N = 20
print(f"Loading {N} background frames...")
all_rfft = []
for k, (idx, cube) in enumerate(iter_real_frames(bg.bin_path, bg.dims, max_frames=N)):
    rfft = stage1_range_fft(cube)
    summed = rfft.sum(axis=-1)
    all_rfft.append(summed)
all_rfft = np.array(all_rfft)
print(f"  cube shape: {all_rfft.shape}")

mean_mag = np.abs(all_rfft).mean(axis=(0, 1))
mean_mag_no_dc = mean_mag.copy()
mean_mag_no_dc[0] = 0
brightest = int(np.argmax(mean_mag_no_dc))
print(f"Brightest static range bin (excl DC): bin {brightest} ({brightest * 2.638:.1f} m)")

slot0_at_bin = all_rfft[:, ::6, brightest]
long_signal = slot0_at_bin.flatten()
phase = np.unwrap(np.angle(long_signal))
delta_phase = np.diff(phase)

median_delta = float(np.median(delta_phase))
mad_delta = float(np.median(np.abs(delta_phase - median_delta)))
print(f"\nSlot-0 chirp Δφ stats:")
print(f"  median: {median_delta:.4f} rad ({np.rad2deg(median_delta):.2f} deg)")
print(f"  MAD:    {mad_delta:.4f} rad ({np.rad2deg(mad_delta):.2f} deg)")

# 128 slot-0 chirps per frame. Frame boundary = end of slot-0 set N
# (chirp idx 127 in long signal) → start of next set (chirp 128).
chirps_per_frame = 128
frame_boundary_idx = np.arange(chirps_per_frame - 1, len(long_signal) - 1, chirps_per_frame)
boundary_jumps = delta_phase[frame_boundary_idx]
print(f"\nFrame-boundary phase jumps (last slot-0 chirp of frame N → first of frame N+1):")
print(f"  count: {len(boundary_jumps)}")
print(f"  median: {np.median(boundary_jumps):.4f} rad")
print(f"  range:  [{boundary_jumps.min():.3f}, {boundary_jumps.max():.3f}] rad")

threshold = max(0.1, 6 * mad_delta)
n_outliers = int(np.sum(np.abs(boundary_jumps - median_delta) > threshold))
print(f"\n  Frame boundaries with |Δφ - median| > {threshold:.3f} rad: {n_outliers}/{len(boundary_jumps)}")

if n_outliers > len(boundary_jumps) // 4:
    print("  → COHERENCE BROKEN at frame boundaries.")
    print("  → Cross-frame coherent integration is NOT valid; only incoherent works.")
else:
    print("  → COHERENCE PRESERVED at frame boundaries.")
    print("  → Cross-frame coherent integration is valid.")

# Plot
out = Path("runs/phase_coherence_test")
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(13, 9))

ax = axes[0]
ax.plot(np.arange(len(long_signal)), np.angle(long_signal), "b-", lw=0.4, alpha=0.6, label="raw angle")
ax.plot(np.arange(len(long_signal)), phase, "r-", lw=0.4, label="unwrapped")
for fb in range(chirps_per_frame - 1, len(long_signal), chirps_per_frame):
    ax.axvline(fb, color="k", lw=0.3, alpha=0.3)
ax.set_xlabel("slot-0 chirp index across 20 frames")
ax.set_ylabel("phase (rad)")
ax.set_title(f"Phase across {N} frames at static bin {brightest} ({brightest*2.638:.1f}m)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(np.arange(len(delta_phase)), delta_phase, "b-", lw=0.4)
for fb in frame_boundary_idx:
    ax.axvline(fb, color="r", lw=0.4, alpha=0.4)
ax.axhline(median_delta, color="g", ls="--", lw=0.6, label=f"median={median_delta:.3f}")
ax.set_xlabel("chirp index")
ax.set_ylabel("Δφ (rad)")
ax.set_title("Per-chirp Δφ. Red verticals = frame boundaries.")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(np.arange(len(boundary_jumps)), boundary_jumps, "r-o", label="boundary Δφ")
ax.axhline(median_delta, color="g", ls="--", lw=0.6, label="median Δφ")
ax.set_xlabel("frame boundary index")
ax.set_ylabel("Δφ at boundary (rad)")
ax.set_title("Frame-boundary phase jumps")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out / "phase_coherence.png", dpi=140)
plt.close(fig)
print(f"\nWrote plot: {out/'phase_coherence.png'}")
