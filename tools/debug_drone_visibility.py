"""Debug script: where is the drone in the RD map?

Skip CFAR. Just show the magnitude of every range bin (0..96) summed
across non-DC Doppler bins, for a few sample frames in each recording.
Looking for: a clear peak at the drone's expected range bin.

drone-fly: drone hovers ~5 m -> bin 2 (5/2.638 = 1.9)
airborne1: drone fly-away, "slow + close" per user — try bins 0-30
"""
from __future__ import annotations
import os, sys
import numpy as np
import scipy.fft as sfft

N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
N_RANGE = N_SAMPLES // 2 + 1
N_TX = 4
PRF_HZ = 30_478.51264858275
RANGE_RES_M = 2.638
INTEGRATE_CHIRPS = 16
N_GROUPS = N_CHIRPS // INTEGRATE_CHIRPS
LAM_M = 3e8 / 77e9
BPF = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)
HANN_SLOW = np.hanning(N_GROUPS).astype(np.float32)

REC_AIRBORNE = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-58-59_radar.bin"
REC_DRONEFLY = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin"


def load_frame(path, idx):
    with open(path, "rb") as f:
        f.seek(idx * BPF)
        buf = f.read(BPF)
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
            .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def rd_no_notches(cube):
    """Range FFT -> RD map. NO notches (we want to see EVERYTHING)."""
    rc = sfft.rfft(cube * HANN_FAST[None, :, None], axis=1, workers=2).astype(np.complex64)
    rc[:, 1:-1, :] *= 2.0
    # No artifact notch.
    trimmed = rc[: N_GROUPS * INTEGRATE_CHIRPS]
    integrated = trimmed.reshape(N_GROUPS, INTEGRATE_CHIRPS, N_RANGE, N_RX).mean(axis=1)
    rd = np.fft.fftshift(
        sfft.fft(integrated * HANN_SLOW[:, None, None], axis=0, workers=2),
        axes=0,
    )
    # No DC notch.
    rd_mag = np.abs(rd.sum(axis=2))
    return rd, rd_mag


def rd_with_notches(cube):
    """Range FFT -> RD map. With notches (matches patched dca_pipeline)."""
    rc = sfft.rfft(cube * HANN_FAST[None, :, None], axis=1, workers=2).astype(np.complex64)
    rc[:, 1:-1, :] *= 2.0
    # Artifact notch: every 24 bins +/-3
    for b in range(24, N_RANGE, 24):
        lo, hi = max(b - 3, 0), min(b + 4, N_RANGE)
        rc[:, lo:hi, :] = 0
    trimmed = rc[: N_GROUPS * INTEGRATE_CHIRPS]
    integrated = trimmed.reshape(N_GROUPS, INTEGRATE_CHIRPS, N_RANGE, N_RX).mean(axis=1)
    rd = np.fft.fftshift(
        sfft.fft(integrated * HANN_SLOW[:, None, None], axis=0, workers=2),
        axes=0,
    )
    # DC notch +/-1
    dc = rd.shape[0] // 2
    rd[max(dc - 1, 0): dc + 2, :, :] = 0
    rd_mag = np.abs(rd.sum(axis=2))
    return rd, rd_mag


def show_per_range(label, rd_mag):
    """Per-range-bin: max across Doppler. Tells us at which range a target sits."""
    n_dop, n_rng = rd_mag.shape
    per_rng_max = rd_mag.max(axis=0)
    per_rng_db = 20 * np.log10(np.maximum(per_rng_max, 1e-9))
    floor = np.median(per_rng_db)
    print(f"\n  {label}: per-range-bin max-across-Doppler (dB), floor={floor:.1f}")
    print("    rb  range_m  dB    excess_over_floor")
    for rb in range(0, min(40, n_rng)):
        m = per_rng_db[rb]
        ex = m - floor
        bar = "#" * max(0, int(ex))
        marker = ""
        if rb % 24 == 0 and rb > 0:
            marker = "  <- notch grid"
        print(f"    {rb:2d}  {rb*RANGE_RES_M:5.1f}  {m:6.1f}  +{ex:5.1f}  {bar}{marker}")
    # find top 5 bins
    top5 = np.argsort(-per_rng_max)[:8]
    print("  TOP 8 bins by max magnitude:")
    for rb in top5:
        rng = rb * RANGE_RES_M
        # Which Doppler bin had the max?
        dop_idx = int(np.argmax(rd_mag[:, rb]))
        dop_off_dc = dop_idx - rd_mag.shape[0] // 2
        fd_scale = PRF_HZ / INTEGRATE_CHIRPS / N_GROUPS
        vel = -LAM_M / 2.0 * (dop_off_dc) * fd_scale
        print(f"    rb={rb:2d} ({rng:5.1f} m) dB={per_rng_db[rb]:5.1f}  dop_off_dc={dop_off_dc:+3d}  vel={vel:+5.2f} m/s")


def main():
    # drone-fly: drone hovers at 5 m. Look at frames spread across recording.
    print("=" * 70)
    print("DRONE-FLY (drone hovering at 5m)")
    print("=" * 70)
    for fid in [50, 150, 250, 350, 450]:
        cube = load_frame(REC_DRONEFLY, fid)
        # No notches first — see the raw signal
        _, rd_mag_raw = rd_no_notches(cube)
        _, rd_mag_patched = rd_with_notches(cube)
        print(f"\n--- drone-fly frame {fid} ---")
        show_per_range("RAW (no notches)", rd_mag_raw)
        show_per_range("PATCHED (artifact notch + DC notch)", rd_mag_patched)

    # airborne1 fly-away: drone "slow and close" per user.
    print("\n" + "=" * 70)
    print("AIRBORNE1 FLY-AWAY (drone slow + close per user)")
    print("=" * 70)
    for fid in [395, 410, 425, 440, 460]:
        cube = load_frame(REC_AIRBORNE, fid)
        _, rd_mag_raw = rd_no_notches(cube)
        _, rd_mag_patched = rd_with_notches(cube)
        print(f"\n--- airborne1 frame {fid} ---")
        show_per_range("RAW (no notches)", rd_mag_raw)
        show_per_range("PATCHED", rd_mag_patched)

    # airborne1 control: pre-flyaway (no drone airborne)
    print("\n" + "=" * 70)
    print("AIRBORNE1 PRE-FLY (frame 100; control)")
    print("=" * 70)
    cube = load_frame(REC_AIRBORNE, 100)
    _, rd_mag_raw = rd_no_notches(cube)
    show_per_range("RAW", rd_mag_raw)


if __name__ == "__main__":
    main()
