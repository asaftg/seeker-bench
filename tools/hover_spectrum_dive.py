"""Hover spectrum dive: drone-vs-background spectral feature contrast.

Builds the per-range-bin "fingerprint" of the airborne1 hover window
(t_rel_s 33..100) vs a no-drone background recording, using ONLY:
  - Stage 1 range FFT
  - Stage 2 MTI (slow-time mean subtract)
  - Coherent RX sum at full PRF
  - 4096-point slow-time FFT per range bin

NO notch (kills 86% of bins at n_samples=192).
NO DDMA unfold (loses ~3 dB SNR vs RX-coherent-sum at full PRF).
NO CFAR (cannot scale to 200m for small drones per spec).

For every hover frame and every non-DC range bin (0..96), we compute six
candidate features designed to detect a small UAV signature in the
PRESENCE of clutter, NOT against an absolute energy threshold:

  total_power_db  : in-band power 50..15000 Hz (sanity check; this is
                    what energy-style detectors look at — expected to be
                    similar for clutter and drone-rim alike)
  peak_db         : peak power in 100..3000 Hz (HERM-band) and its freq
  spec_kurtosis   : excess-kurtosis of slow-time |X|^2 across 50..15000 Hz
                    (drone rim → broadband + spiky → high kurtosis;
                     diffuse clutter → near-Gaussian → ~0 excess)
  cepstrum_peak   : peak in real cepstrum at quefrency 5..300 samples
                    (corresponds to periods 0.16..9.8 ms = blade-pass and
                     prop-rotation for DJI FPV ~25 kRPM, 3 blades →
                     fblade ~1250 Hz → quefrency ~24 samples at PRF=30478)
  papr            : peak-to-average power ratio in band
  n_supra_median  : count of 100..3000 Hz bins with power > 6x median in
                    that band (a "comb count" without forcing harmonics)

For each feature we then dump per-range-bin (drone-mean / background-mean)
ratio over the same range bins. Top candidates surface as the
"strongest hover-vs-background contrast" feature/range-bin pairs.

Usage
-----
    python hover_spectrum_dive.py
    # writes diagnostic stdout + a CSV summary at:
    #   tools/hover_spectrum_dive_output.csv
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ────────────────────────── constants ───────────────────────────────────

N_CHIRPS    = 768
N_RX        = 4
N_SAMPLES   = 192
PRF_HZ      = 30478.51264858275
RANGE_RES_M = 2.638466734211415
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2

# Recording is 2148 frames over ~155 s → 13.86 fps actual rate.
# t_rel_s → bin_idx: bin_idx = round(t_rel_s / 0.072) but the user gave
# 1/0.072 ≈ 13.89 fps. We use the recording's empirical fps for the
# actual recording length we have (2148 / 155 ≈ 13.86).
RECORDING_DURATION_S = 155.0
ACTUAL_FPS = 2148.0 / RECORDING_DURATION_S  # 13.858

# Slow-time FFT params
N_FFT       = 4096
HANN_FAST   = np.hanning(N_SAMPLES).astype(np.float32)
HANN_SLOW   = np.hanning(N_CHIRPS).astype(np.float32)

# Bands of interest
BAND_TOTAL_LO_HZ  = 50.0       # drop DC
BAND_TOTAL_HI_HZ  = 15000.0    # near Nyquist (15239 Hz)
BAND_HERM_LO_HZ   = 100.0      # HERM-band (where blade harmonics live)
BAND_HERM_HI_HZ   = 3000.0
QUE_LO            = 5          # cepstrum quefrency window (samples)
QUE_HI            = 300

# Range bins to scan: skip bin 0 (DC range = self-leakage), keep 1..96.
# n_range = N_SAMPLES//2 + 1 = 97. Beyond bin 96 is mirrored.
RB_LO = 1
RB_HI = 97  # exclusive

DRONE_BIN_PATH = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-58-59_radar.bin")
BG_BIN_PATH    = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-54-23_radar.bin")
TIMELINE_CSV   = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\airborne1_hover_timeline.csv")

OUT_CSV = Path(__file__).parent / "hover_spectrum_dive_output.csv"

# ────────────────────────── pipeline (Stage 1 + MTI) ────────────────────

def load_frame_real(path: Path, frame_idx: int) -> np.ndarray:
    """Read one frame, return real cube (n_chirps, n_samples, n_rx) float32.

    Wire layout per radar_dca/dca_pipeline.py:_stage1_range_fft —
    int16 reshape (chirps, rx, samples), transpose, DC subtract per
    (chirp, rx) across samples.
    """
    off = frame_idx * BYTES_PER_FRAME
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME:
        raise IOError(f"short read frame {frame_idx} from {path.name}")
    raw = np.frombuffer(buf, dtype=np.int16)
    real_cube = (
        raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
           .transpose(0, 2, 1)
           .astype(np.float32)
    )
    real_cube -= real_cube.mean(axis=1, keepdims=True)
    return real_cube


def stage1_range_fft(real_cube: np.ndarray) -> np.ndarray:
    """Range FFT, analytic scaling. Output (n_chirps, n_range, n_rx) cplx."""
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def stage2_mti(range_cube: np.ndarray) -> np.ndarray:
    """Slow-time MTI: mean-subtract along chirp axis."""
    return range_cube - range_cube.mean(axis=0, keepdims=True)


def coherent_rx_sum(range_cube_mti: np.ndarray) -> np.ndarray:
    """Coherent RX sum → (n_chirps, n_range) complex.

    NOT per-VA averaging. NOT DDMA. Full PRF preserved → Nyquist 15239 Hz.
    """
    return range_cube_mti.sum(axis=2)


# ────────────────────────── feature extraction ──────────────────────────

@dataclass
class BinFeatures:
    rb: int
    range_m: float
    total_power_db: float
    peak_db: float
    peak_freq_hz: float
    spec_kurtosis: float
    cepstrum_peak_db: float
    cepstrum_peak_que: int
    cepstrum_peak_freq_hz: float
    papr_db: float
    n_supra_median: int


# Pre-compute frequency axis
_FREQS_FULL = np.fft.fftfreq(N_FFT, d=1.0/PRF_HZ)
# Use only positive freqs from idx 1..N_FFT//2 (drop DC, drop neg freqs)
_POS_IDX     = np.arange(1, N_FFT // 2)
_FREQS_POS   = _FREQS_FULL[_POS_IDX]
_TOTAL_MASK  = (_FREQS_POS >= BAND_TOTAL_LO_HZ) & (_FREQS_POS <= BAND_TOTAL_HI_HZ)
_HERM_MASK   = (_FREQS_POS >= BAND_HERM_LO_HZ)  & (_FREQS_POS <= BAND_HERM_HI_HZ)


def features_at_range_bin(slow_time: np.ndarray, rb: int) -> BinFeatures:
    """Compute six spectral features for one range bin's slow-time signal.

    `slow_time` is shape (n_chirps,) complex64. Mean-subtracted (MTI was
    already applied by caller), Hann-windowed, zero-padded to N_FFT,
    full FFT (positive AND negative freqs available — we use positive).
    """
    s = slow_time - slow_time.mean()
    s = s * HANN_SLOW
    X = sfft.fft(s, n=N_FFT)
    P = (X.real * X.real + X.imag * X.imag).astype(np.float64)
    # Positive-frequency power (drop DC and Nyquist guard)
    P_pos = P[_POS_IDX]

    # Total power 50..15000 Hz (linear sum then dB)
    P_total_band = P_pos[_TOTAL_MASK]
    total_power = float(P_total_band.sum())
    total_power_db = 10.0 * np.log10(total_power + 1e-30)

    # Peak in HERM band 100..3000 Hz
    P_herm = P_pos[_HERM_MASK]
    f_herm = _FREQS_POS[_HERM_MASK]
    if P_herm.size == 0:
        peak_db = -300.0
        peak_freq_hz = 0.0
    else:
        ki = int(np.argmax(P_herm))
        peak_db = float(10.0 * np.log10(P_herm[ki] + 1e-30))
        peak_freq_hz = float(f_herm[ki])

    # Spectral kurtosis (excess) of P_pos in 50..15000 Hz band.
    # Captures "drone-rim broadband + spiky" vs Gaussian clutter (≈0).
    if P_total_band.size > 4:
        m = P_total_band.mean()
        sd = P_total_band.std() + 1e-30
        z = (P_total_band - m) / sd
        spec_kurtosis = float(np.mean(z ** 4) - 3.0)
    else:
        spec_kurtosis = 0.0

    # Real cepstrum (log-power → IFFT → magnitude). Use the FULL spectrum
    # (positive + negative) for proper cepstrum support.
    log_P = np.log(P + 1e-30)
    cepstrum = np.fft.ifft(log_P).real
    que_window = cepstrum[QUE_LO:QUE_HI]
    if que_window.size:
        qi = int(np.argmax(np.abs(que_window)))
        que_idx = QUE_LO + qi
        cep_peak_db = float(10.0 * np.log10(abs(que_window[qi]) + 1e-30))
        # Quefrency q (samples) at sample rate PRF → period = q / PRF (s) →
        # fundamental = PRF / q. q=24 → 1270 Hz.
        cep_peak_freq = float(PRF_HZ / max(que_idx, 1))
    else:
        que_idx = 0
        cep_peak_db = -300.0
        cep_peak_freq = 0.0

    # PAPR: peak/average in 50..15000 band (dB)
    if P_total_band.size > 1:
        papr_db = float(
            10.0 * np.log10(P_total_band.max() / (P_total_band.mean() + 1e-30))
        )
    else:
        papr_db = 0.0

    # n_supra_median: count of HERM-band bins above 6x band median
    if P_herm.size > 0:
        med = np.median(P_herm) + 1e-30
        n_supra = int(np.sum(P_herm > 6.0 * med))
    else:
        n_supra = 0

    return BinFeatures(
        rb=rb,
        range_m=rb * RANGE_RES_M,
        total_power_db=total_power_db,
        peak_db=peak_db,
        peak_freq_hz=peak_freq_hz,
        spec_kurtosis=spec_kurtosis,
        cepstrum_peak_db=cep_peak_db,
        cepstrum_peak_que=que_idx,
        cepstrum_peak_freq_hz=cep_peak_freq,
        papr_db=papr_db,
        n_supra_median=n_supra,
    )


def features_for_frame(path: Path, frame_idx: int) -> List[BinFeatures]:
    """Stage 1 + MTI + RX-coherent-sum, then features at each range bin."""
    real_cube  = load_frame_real(path, frame_idx)
    range_cube = stage1_range_fft(real_cube)        # (n_chirps, n_range, n_rx)
    range_cube = stage2_mti(range_cube)
    chirp_x_range = coherent_rx_sum(range_cube)     # (n_chirps, n_range)
    out: List[BinFeatures] = []
    for rb in range(RB_LO, RB_HI):
        out.append(features_at_range_bin(chirp_x_range[:, rb], rb))
    return out


# ────────────────────────── frame selection ────────────────────────────

def hover_frame_indices(timeline_csv: Path, n_target: int = 20) -> List[int]:
    """Pick ~n_target evenly-spaced frame indices in the t=33..100 hover window."""
    hover_t: List[float] = []
    with open(timeline_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["t_rel_s"])
            except (ValueError, TypeError):
                continue
            if row["phase"].strip() == "hover" and 33.0 <= t <= 100.0:
                hover_t.append(t)
    if not hover_t:
        raise RuntimeError("no hover rows in 33..100 s window")
    # Sample evenly
    if len(hover_t) <= n_target:
        chosen_t = hover_t
    else:
        idx = np.linspace(0, len(hover_t) - 1, n_target).round().astype(int)
        chosen_t = [hover_t[i] for i in idx]
    # Convert to frame indices
    return [int(round(t * ACTUAL_FPS)) for t in chosen_t]


def background_frame_indices(bg_path: Path, n_target: int = 20) -> List[int]:
    """Pick n_target evenly-spaced frames in the background recording, skipping
    the first/last 5% to avoid startup/shutdown transients."""
    n_frames = bg_path.stat().st_size // BYTES_PER_FRAME
    lo = int(n_frames * 0.05)
    hi = int(n_frames * 0.95)
    return list(np.linspace(lo, hi, n_target).round().astype(int))


# ────────────────────────── aggregate + report ─────────────────────────

def aggregate_per_bin(
    frames: List[List[BinFeatures]],
) -> Dict[str, np.ndarray]:
    """Stack per-frame features into per-range-bin mean/median arrays."""
    n_frames = len(frames)
    n_bins   = len(frames[0])
    keys = (
        "total_power_db", "peak_db", "spec_kurtosis",
        "cepstrum_peak_db", "papr_db", "n_supra_median",
    )
    M: Dict[str, np.ndarray] = {k: np.zeros((n_frames, n_bins)) for k in keys}
    for fi, fbins in enumerate(frames):
        for bi, b in enumerate(fbins):
            M["total_power_db"][fi, bi]  = b.total_power_db
            M["peak_db"][fi, bi]         = b.peak_db
            M["spec_kurtosis"][fi, bi]   = b.spec_kurtosis
            M["cepstrum_peak_db"][fi, bi]= b.cepstrum_peak_db
            M["papr_db"][fi, bi]         = b.papr_db
            M["n_supra_median"][fi, bi]  = b.n_supra_median
    out: Dict[str, np.ndarray] = {}
    for k, v in M.items():
        out[k + "_mean"]   = v.mean(axis=0)
        out[k + "_median"] = np.median(v, axis=0)
    return out


def main() -> int:
    if not DRONE_BIN_PATH.exists():
        print(f"ERROR: drone bin not found at {DRONE_BIN_PATH}", file=sys.stderr)
        return 2
    if not BG_BIN_PATH.exists():
        print(f"ERROR: background bin not found at {BG_BIN_PATH}", file=sys.stderr)
        return 2
    if not TIMELINE_CSV.exists():
        print(f"ERROR: timeline csv not found at {TIMELINE_CSV}", file=sys.stderr)
        return 2

    n_drone_frames = DRONE_BIN_PATH.stat().st_size // BYTES_PER_FRAME
    n_bg_frames    = BG_BIN_PATH.stat().st_size    // BYTES_PER_FRAME
    print(f"drone bin: {DRONE_BIN_PATH.name}  ({n_drone_frames} frames)")
    print(f"bg bin   : {BG_BIN_PATH.name}     ({n_bg_frames} frames)")
    print(f"actual fps: {ACTUAL_FPS:.3f}  (PRF={PRF_HZ:.1f} Hz, range_res={RANGE_RES_M:.3f} m/bin)")

    drone_frame_idxs = hover_frame_indices(TIMELINE_CSV, n_target=20)
    bg_frame_idxs    = background_frame_indices(BG_BIN_PATH, n_target=20)
    print(f"\nhover frames ({len(drone_frame_idxs)}): {drone_frame_idxs}")
    print(f"bg frames    ({len(bg_frame_idxs)})   : {bg_frame_idxs}")

    print("\n[scanning hover frames]")
    drone_frames: List[List[BinFeatures]] = []
    for fi in drone_frame_idxs:
        if fi >= n_drone_frames:
            print(f"  frame {fi}: out of range, skipping")
            continue
        try:
            drone_frames.append(features_for_frame(DRONE_BIN_PATH, fi))
        except Exception as e:
            print(f"  frame {fi}: {e}")
    print(f"  {len(drone_frames)} drone frames OK")

    print("[scanning background frames]")
    bg_frames: List[List[BinFeatures]] = []
    for fi in bg_frame_idxs:
        if fi >= n_bg_frames:
            print(f"  frame {fi}: out of range, skipping")
            continue
        try:
            bg_frames.append(features_for_frame(BG_BIN_PATH, fi))
        except Exception as e:
            print(f"  frame {fi}: {e}")
    print(f"  {len(bg_frames)} bg frames OK")

    if not drone_frames or not bg_frames:
        print("ERROR: insufficient frames", file=sys.stderr)
        return 3

    drone_agg = aggregate_per_bin(drone_frames)
    bg_agg    = aggregate_per_bin(bg_frames)

    rb_axis = np.arange(RB_LO, RB_HI)
    range_m_axis = rb_axis * RANGE_RES_M

    # Per-bin contrast (linear ratio for power-style features, raw delta
    # for kurtosis since it can be ~0 or negative).
    feature_keys = (
        ("total_power_db",    "ratio_db_to_lin"),
        ("peak_db",           "ratio_db_to_lin"),
        ("spec_kurtosis",     "delta"),
        ("cepstrum_peak_db",  "ratio_db_to_lin"),
        ("papr_db",           "delta"),       # PAPR already a ratio
        ("n_supra_median",    "delta"),
    )

    print("\n" + "=" * 78)
    print("PER-RANGE-BIN CONTRAST TABLE  (drone hover vs no-drone background)")
    print("=" * 78)
    header = (
        f"{'rb':>3} {'range_m':>7}"
        f" {'tot_d':>6} {'tot_b':>6} {'tot_x':>6}"
        f" {'pk_d':>6} {'pk_b':>6} {'pk_x':>6}"
        f" {'kurt_d':>7} {'kurt_b':>7} {'kurt_dx':>8}"
        f" {'cep_d':>6} {'cep_b':>6} {'cep_x':>6}"
        f" {'papr_d':>7} {'papr_b':>7}"
        f" {'sup_d':>6} {'sup_b':>6}"
    )
    print(header)
    rows: List[Dict[str, float]] = []
    for i, rb in enumerate(rb_axis):
        d_tot = drone_agg["total_power_db_mean"][i]
        b_tot = bg_agg["total_power_db_mean"][i]
        d_pk  = drone_agg["peak_db_mean"][i]
        b_pk  = bg_agg["peak_db_mean"][i]
        d_kr  = drone_agg["spec_kurtosis_mean"][i]
        b_kr  = bg_agg["spec_kurtosis_mean"][i]
        d_ce  = drone_agg["cepstrum_peak_db_mean"][i]
        b_ce  = bg_agg["cepstrum_peak_db_mean"][i]
        d_pa  = drone_agg["papr_db_mean"][i]
        b_pa  = bg_agg["papr_db_mean"][i]
        d_su  = drone_agg["n_supra_median_mean"][i]
        b_su  = bg_agg["n_supra_median_mean"][i]
        # x_lin = 10^((d - b)/10)
        tot_x = 10.0 ** ((d_tot - b_tot) / 10.0)
        pk_x  = 10.0 ** ((d_pk  - b_pk)  / 10.0)
        cep_x = 10.0 ** ((d_ce  - b_ce)  / 10.0)
        kurt_dx = d_kr - b_kr
        rows.append(dict(
            rb=int(rb), range_m=float(range_m_axis[i]),
            d_tot=d_tot, b_tot=b_tot, tot_x=tot_x,
            d_pk=d_pk, b_pk=b_pk, pk_x=pk_x,
            d_kurt=d_kr, b_kurt=b_kr, kurt_dx=kurt_dx,
            d_cep=d_ce, b_cep=b_ce, cep_x=cep_x,
            d_papr=d_pa, b_papr=b_pa,
            d_supra=d_su, b_supra=b_su,
        ))
        print(
            f"{int(rb):>3} {float(range_m_axis[i]):>7.1f}"
            f" {d_tot:>6.1f} {b_tot:>6.1f} {tot_x:>6.2f}"
            f" {d_pk:>6.1f} {b_pk:>6.1f} {pk_x:>6.2f}"
            f" {d_kr:>7.2f} {b_kr:>7.2f} {kurt_dx:>8.2f}"
            f" {d_ce:>6.1f} {b_ce:>6.1f} {cep_x:>6.2f}"
            f" {d_pa:>7.1f} {b_pa:>7.1f}"
            f" {d_su:>6.1f} {b_su:>6.1f}"
        )

    # ── ranking summaries ──
    print("\n" + "=" * 78)
    print("WHICH RANGE BINS HAVE THE STRONGEST CONTRAST PER FEATURE?")
    print("=" * 78)
    for key, mode in feature_keys:
        d = drone_agg[key + "_mean"]
        b = bg_agg[key + "_mean"]
        if mode == "ratio_db_to_lin":
            score = 10.0 ** ((d - b) / 10.0)   # contrast (1 = no diff)
            unit = "x"
        else:
            score = d - b                       # additive contrast
            unit = "delta"
        order = np.argsort(-score)[:5]
        top = [(int(rb_axis[k]), float(range_m_axis[k]), float(score[k])) for k in order]
        print(f"\n  {key:>20s}  top-5 (drone {unit} bg):")
        for rb, rng_m, sc in top:
            print(f"      rb={rb:3d} ({rng_m:5.1f} m)  score={sc:>8.3f}")

    # ── frame-wise consistency: for the top-kurtosis bin, how stable is it? ──
    print("\n" + "=" * 78)
    print("PER-FRAME PEAK FREQUENCY AND CEPSTRUM AT THE TOP-KURTOSIS RANGE BIN")
    print("=" * 78)
    kurt_score = drone_agg["spec_kurtosis_mean"] - bg_agg["spec_kurtosis_mean"]
    top_rb_i = int(np.argmax(kurt_score))
    top_rb = int(rb_axis[top_rb_i])
    top_rb_range = float(range_m_axis[top_rb_i])
    print(f"\n  best kurtosis-contrast bin: rb={top_rb} ({top_rb_range:.1f} m)")
    print(f"  per-frame features at this bin (drone hover):")
    print(f"    {'frame':>6} {'peak_Hz':>10} {'peak_dB':>9} {'kurt':>7} "
          f"{'cep_q':>6} {'cep_dB':>8} {'cep_freq':>10}")
    for fi_real, fbins in zip(drone_frame_idxs[:len(drone_frames)], drone_frames):
        b = fbins[top_rb_i]
        print(f"    {fi_real:>6d} {b.peak_freq_hz:>10.1f} {b.peak_db:>9.1f} "
              f"{b.spec_kurtosis:>7.2f} {b.cepstrum_peak_que:>6d} "
              f"{b.cepstrum_peak_db:>8.2f} {b.cepstrum_peak_freq_hz:>10.1f}")

    # Per-frame peak-frequency consistency: print a histogram of
    # peak_freq_hz across all frames + bins, drone vs bg. If the drone
    # has a real spectral signature, hover frames should show a tight
    # cluster of peak-freqs at one value (e.g. 1250 Hz blade-pass).
    print("\n" + "=" * 78)
    print("PEAK-FREQUENCY HISTOGRAM (per-frame per-bin) — drone hover vs bg")
    print("=" * 78)
    edges_hz = list(range(0, 3001, 100)) + [4000, 5000, 7500, 10000, 15000]
    drone_pks: List[float] = []
    bg_pks: List[float] = []
    for fbins in drone_frames:
        for b in fbins:
            drone_pks.append(b.peak_freq_hz)
    for fbins in bg_frames:
        for b in fbins:
            bg_pks.append(b.peak_freq_hz)
    drone_h, _ = np.histogram(drone_pks, bins=edges_hz)
    bg_h, _    = np.histogram(bg_pks, bins=edges_hz)
    print(f"  {'lo_Hz':>6} {'hi_Hz':>6} {'drone_n':>8} {'bg_n':>6}")
    for i in range(len(edges_hz) - 1):
        print(f"  {edges_hz[i]:>6d} {edges_hz[i+1]:>6d} "
              f"{drone_h[i]:>8d} {bg_h[i]:>6d}")

    # Per-frame trace: at suspected drone-range bins (18..28 = 47..74m),
    # does any one bin show consistent above-background features
    # frame-after-frame? This is the real test of "drone localized at
    # one range".
    print("\n" + "=" * 78)
    print("PER-FRAME TRACE AT SUSPECTED-DRONE RANGE BINS (rb=18..30)")
    print("=" * 78)
    trace_rbs = list(range(18, 31))
    print(f"\n  Drone hover frames — peak_dB at each rb:")
    print(f"    {'frame':>6} | " + " ".join(f"rb{rb:02d}" for rb in trace_rbs))
    for fi_real, fbins in zip(drone_frame_idxs[:len(drone_frames)], drone_frames):
        vals = " ".join(f"{fbins[rb-RB_LO].peak_db:5.1f}" for rb in trace_rbs)
        print(f"    {fi_real:>6d} | {vals}")
    print(f"\n  Background frames — peak_dB at each rb:")
    print(f"    {'frame':>6} | " + " ".join(f"rb{rb:02d}" for rb in trace_rbs))
    for fi_real, fbins in zip(bg_frame_idxs[:len(bg_frames)], bg_frames):
        vals = " ".join(f"{fbins[rb-RB_LO].peak_db:5.1f}" for rb in trace_rbs)
        print(f"    {int(fi_real):>6d} | {vals}")
    print(f"\n  Drone hover frames — kurtosis at each rb:")
    print(f"    {'frame':>6} | " + " ".join(f"rb{rb:02d}" for rb in trace_rbs))
    for fi_real, fbins in zip(drone_frame_idxs[:len(drone_frames)], drone_frames):
        vals = " ".join(f"{fbins[rb-RB_LO].spec_kurtosis:5.1f}" for rb in trace_rbs)
        print(f"    {fi_real:>6d} | {vals}")
    print(f"\n  Background frames — kurtosis at each rb:")
    print(f"    {'frame':>6} | " + " ".join(f"rb{rb:02d}" for rb in trace_rbs))
    for fi_real, fbins in zip(bg_frame_idxs[:len(bg_frames)], bg_frames):
        vals = " ".join(f"{fbins[rb-RB_LO].spec_kurtosis:5.1f}" for rb in trace_rbs)
        print(f"    {int(fi_real):>6d} | {vals}")

    # ── cepstrum fingerprint check ─────────────────────────────────────
    # Is there a stable quefrency in the drone hover that's absent in bg?
    print("\n" + "=" * 78)
    print("CEPSTRUM QUEFRENCY HISTOGRAM (where do the peaks land?)")
    print("=" * 78)
    drone_qs = []
    bg_qs    = []
    for fbins in drone_frames:
        for b in fbins:
            drone_qs.append(b.cepstrum_peak_que)
    for fbins in bg_frames:
        for b in fbins:
            bg_qs.append(b.cepstrum_peak_que)
    bins = list(range(QUE_LO, QUE_HI + 1, 10))
    drone_h, _ = np.histogram(drone_qs, bins=bins)
    bg_h, _    = np.histogram(bg_qs, bins=bins)
    print(f"  {'q_lo':>5} {'q_hi':>5} {'fblade_Hz':>10} {'drone_n':>8} {'bg_n':>6}")
    for i in range(len(bins) - 1):
        f_blade = PRF_HZ / max((bins[i] + bins[i+1]) / 2.0, 1.0)
        print(f"  {bins[i]:>5d} {bins[i+1]:>5d} {f_blade:>10.1f} "
              f"{drone_h[i]:>8d} {bg_h[i]:>6d}")

    # ── median-vs-median per-bin contrast (more robust than mean) ─────
    # Global noise-floor differences between recordings inflate every
    # mean-based contrast metric. Use per-bin median and subtract a
    # global per-recording shift to isolate range-bin-localized features.
    print("\n" + "=" * 78)
    print("NORMALIZED PER-BIN CONTRAST (median, global shift removed)")
    print("=" * 78)
    # For each feature, subtract the global median of background and
    # global median of drone (so a uniform recording-level offset
    # disappears) before reporting per-bin delta.
    for key in ("peak_db", "spec_kurtosis", "papr_db", "n_supra_median"):
        d = drone_agg[key + "_median"]
        b = bg_agg[key + "_median"]
        shift_d = float(np.median(d))
        shift_b = float(np.median(b))
        d_n = d - shift_d
        b_n = b - shift_b
        local_delta = d_n - b_n  # +ve = drone has more local excess
        order = np.argsort(-local_delta)[:10]
        print(f"\n  {key} (shift_d={shift_d:+.2f}, shift_b={shift_b:+.2f})")
        print(f"    top-10 local-excess range bins:")
        print(f"      {'rb':>3} {'range_m':>7} {'d_local':>9} {'b_local':>9} "
              f"{'delta':>7}")
        for k in order:
            print(f"      {int(rb_axis[k]):>3d} {float(range_m_axis[k]):>7.1f} "
                  f"{d_n[k]:>9.2f} {b_n[k]:>9.2f} {local_delta[k]:>7.2f}")

    # ── write CSV ─────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[OK] wrote per-bin contrast table to: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
