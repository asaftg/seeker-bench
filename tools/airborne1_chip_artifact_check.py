"""Is the 2575 Hz peak we see at every range bin a chip artifact or real?

Hypothesis: 2537/2552/2575 Hz is the SAME chip-internal interference at
ALL range bins regardless of whether a drone is present. To verify, we
compare:
  - Mean DDMA per-VA spectrum at the chip-detected drone bin (rb=2)
  - Mean DDMA per-VA spectrum at a baseline bin (e.g. rb=50, no drone)
  - Mean DDMA per-VA spectrum AVERAGED across all non-notched range bins
  - All on the SAME hover frame (535)

If 2552 Hz is the same magnitude in all three, it's a chip artifact and
the median floor of the per-VA spectrum is dominated by it.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_dca.ddma import ddma_unfold, DEFAULT_ANTENNA_ORDER  # noqa: E402

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF = 30478.51264858275
EFF_PRF = PRF / 4
BYTES = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)
BIN_PATH = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-58-59_radar.bin")


def load_range_cube(frame_idx: int, notch: bool) -> np.ndarray:
    off = frame_idx * BYTES
    with open(BIN_PATH, "rb") as f:
        f.seek(off); buf = f.read(BYTES)
    raw = np.frombuffer(buf, dtype=np.int16)
    rc = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
            .transpose(0, 2, 1).astype(np.float32))
    rc -= rc.mean(axis=1, keepdims=True)
    rc = rc * HANN_FAST[None, :, None]
    rfft = sfft.rfft(rc, axis=1, workers=2)
    rfft[:, 1:-1, :] *= 2.0
    rfft = rfft.astype(np.complex64)
    rfft -= rfft.mean(axis=0, keepdims=True)  # MTI
    if notch:
        period = N_SAMPLES // 8
        step = max(period // 2, 1)
        radius = 5
        n_range = rfft.shape[1]
        for b in range(step, n_range, step):
            lo = max(b - radius, 0); hi = min(b + radius + 1, n_range)
            rfft[:, lo:hi, :] = 0
    return rfft


def per_va_pwr_spectrum(range_cube: np.ndarray, rb: int, n_fft: int = 1024) -> np.ndarray:
    virtual = ddma_unfold(range_cube, antenna_order=DEFAULT_ANTENNA_ORDER)
    n_per = virtual.shape[0]
    slow = virtual[:, rb, :, :].reshape(n_per, N_RX * 4)
    win = np.hanning(n_per).astype(np.float32)
    spec = np.fft.fft(slow * win[:, None], n=n_fft, axis=0)
    pwr = (spec.real * spec.real + spec.imag * spec.imag).mean(axis=1)
    return pwr[: n_fft // 2].astype(np.float64)


def main() -> None:
    n_fft = 1024
    bin_hz = EFF_PRF / n_fft
    print(f"bin_hz={bin_hz:.3f}, eff_prf={EFF_PRF:.0f} Hz, Nyq={EFF_PRF/2:.0f} Hz")

    # Hover frame 535: drone at rb=2. Take un-notched cube to inspect.
    rc_535 = load_range_cube(535, notch=False)

    # Spectra at: drone bin (rb=2), and a few "no drone" range bins
    test_bins = [2, 5, 7, 18, 20, 22, 50, 70]   # avoid notched 12-17, 19-29 etc
    print("\nfront-frame 535: per-VA-avg power (linear) at key freqs vs range bin")
    print(f"{'rb':>4} {'med':>10} {'p@60':>10} {'p@1250':>10} {'p@2500':>10} {'p@2575':>10} {'p@3750':>10}")
    for rb in test_bins:
        pwr = per_va_pwr_spectrum(rc_535, rb, n_fft)
        med = float(np.median(pwr))
        def at(hz):
            c = int(round(hz / bin_hz))
            lo = max(c - 3, 0); hi = min(c + 4, len(pwr))
            return float(pwr[lo:hi].max())
        print(f"{rb:>4} {med:>10.2e} {at(60):>10.2e} {at(1250):>10.2e} {at(2500):>10.2e} {at(2575):>10.2e} {at(3750):>10.2e}")

    # Average per-VA spectrum across many non-notched range bins (clutter background)
    notched = set()
    period = N_SAMPLES // 8; step = max(period // 2, 1); radius = 5
    for b in range(step, 97, step):
        for k in range(b - radius, b + radius + 1):
            if 0 <= k < 97:
                notched.add(k)
    safe = [r for r in range(4, 90) if r not in notched]
    avg = np.zeros(n_fft // 2)
    for rb in safe:
        avg += per_va_pwr_spectrum(rc_535, rb, n_fft)
    avg /= len(safe)
    print(f"\naverage per-VA spectrum over {len(safe)} non-notched range bins of frame 535:")
    med = float(np.median(avg))
    print(f"   median floor: {med:.3e}")
    for hz in (60, 250, 500, 1000, 1250, 1500, 1900, 2200, 2500, 2575, 3000, 3500, 3750):
        c = int(round(hz / bin_hz))
        lo = max(c - 3, 0); hi = min(c + 4, len(avg))
        pk = float(avg[lo:hi].max())
        db = 10 * np.log10(pk / max(med, 1e-30))
        print(f"   {hz:>5d} Hz: pwr={pk:.3e}  ({db:+5.1f} dB above background median)")

    # Compare a CLEAN BACKGROUND frame (no chip detection at this index) at rb=2
    print("\n--- FRAME 100 (no chip detection at this frame) ---")
    rc_clean = load_range_cube(100, notch=False)
    for rb in [2, 5, 7, 50]:
        pwr = per_va_pwr_spectrum(rc_clean, rb, n_fft)
        med = float(np.median(pwr))
        def at(hz):
            c = int(round(hz / bin_hz))
            lo = max(c - 3, 0); hi = min(c + 4, len(pwr))
            return float(pwr[lo:hi].max())
        print(f" rb={rb}: med={med:.2e}  60={at(60):.2e}  2575={at(2575):.2e}  ratio2575={at(2575)/max(med,1e-30):.1f}x")


if __name__ == "__main__":
    main()
