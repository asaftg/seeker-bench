"""Diagnose the every-N-range-bins artifact in the AWR2944P range axis.

NOTE on running output: T1 showed strong spurs at bins 24, 48, 72
(every 24 bins, NOT every 12). 192/24 = 8 -> likely 8-way time-
interleaved ADC. Spur freq = 24 * 30 MHz / 192 = 3.75 MHz IF.
The existing notch in dca_pipeline.py was zeroing bins 12, 24, 36, 48 ...
(every 12) which is twice as aggressive as needed.

Six read-only tests on the existing recordings, no notch filter,
no MTI in T5. Outputs a ranked verdict on the root cause:
- ADC interleave mismatch (chip-internal, deterministic at fs/16)
- USB/cable EMI (per-RX variance)
- PMIC switching (time-varying, broadband)
- Notch self-signature (artifact only present after notch is applied)

Cfg: 768 chirps x 4 RX x 192 samples x 2 bytes int16 (real ADC,
adcCfg 2 0). Sample rate 30 Msps. Bin 12 = 12 * 30e6 / 192 =
1.875 MHz IF. PRF 30478.51 Hz.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import scipy.fft as sfft

# ─────────────── constants from awr2944P_unified.cfg ───────────────
N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
N_RANGE = N_SAMPLES // 2 + 1            # 97 (rfft on real input)
FS_HZ = 30_000_000.0
PRF_HZ = 30_478.51264858275
N_TX = 4
BPF = N_CHIRPS * N_RX * N_SAMPLES * 2   # bytes per frame, real int16
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

SUSPECT_BINS = [24, 48, 72, 96]                 # multiples of 24 (8-way interleave)
CONTROL_BINS = [7, 19, 31, 43, 55, 67]           # NOT multiples of 24
PSEUDO_BINS  = [12, 36, 60, 84]                 # claimed by old notch but T1 shows clean

REC_AIRBORNE = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-58-59_radar.bin"
REC_DRONEFLY = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin"
REC_BACKGROUND = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-54-23_radar.bin"


# ─────────────────── frame loader (no notch, no MTI) ───────────────
def n_frames(path: str) -> int:
    return os.path.getsize(path) // BPF


def load_frame_raw(path: str, idx: int) -> np.ndarray:
    """Returns float32 cube (n_chirps, n_samples, n_rx) — DC-subtracted
    per chirp/RX exactly like _stage1_range_fft. NO range FFT yet."""
    with open(path, "rb") as f:
        f.seek(idx * BPF)
        buf = f.read(BPF)
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (
        raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
            .transpose(0, 2, 1)
            .astype(np.float32)
    )
    cube -= cube.mean(axis=1, keepdims=True)   # per-chirp/RX DC subtract
    return cube                                 # (chirps, samples, rx)


def range_fft_no_notch(cube: np.ndarray) -> np.ndarray:
    """Hann + rfft along samples axis. No notch. Returns (chirps, range, rx) complex64."""
    rc = sfft.rfft(cube * HANN_FAST[None, :, None], axis=1, workers=2).astype(np.complex64)
    rc[:, 1:-1, :] *= 2.0
    return rc


def stack_frames(path: str, n: int = 8, stride: int = 64) -> np.ndarray:
    """Load n frames spread across the recording. Returns (n, chirps, range, rx) complex."""
    total = n_frames(path)
    if total == 0:
        raise RuntimeError(f"no frames in {path}")
    # Pick frames evenly spaced across the recording
    indices = np.linspace(0, max(total - 1, 0), num=min(n, total), dtype=int)
    out = np.empty((len(indices), N_CHIRPS, N_RANGE, N_RX), dtype=np.complex64)
    for i, idx in enumerate(indices):
        out[i] = range_fft_no_notch(load_frame_raw(path, idx))
    return out


# ─────────────────────── test helpers ──────────────────────────────
def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-30))


def banner(title: str, char: str = "=") -> None:
    print()
    print(char * 70)
    print(title)
    print(char * 70)


# ─────────────────────────── T1 ────────────────────────────────────
def t1_bypass_notch(rec_label: str, rc_stack: np.ndarray) -> dict:
    """Bypass notch; check pattern persists across many frames.

    rc_stack shape: (n_frames, chirps, range, rx).
    Decision: peak excess at suspect bins vs control bins, averaged over frames.
    """
    banner(f"T1 — Bypass notch [{rec_label}]")
    print("Hypothesis : every-12-range-bins pattern persists when notch is OFF.")
    print("Measurement: mean |R(k)| in dB over chirps/RX/frames at suspect vs control bins.")

    # Average power over chirps and RX, then average across frames
    pwr = (np.abs(rc_stack) ** 2).mean(axis=(1, 3))     # (n_frames, range)
    spec_db = 10.0 * np.log10(np.maximum(pwr.mean(axis=0), 1e-30))   # (range,)

    sus_dB = np.array([spec_db[b] for b in SUSPECT_BINS])
    ctl_dB = np.array([spec_db[b] for b in CONTROL_BINS])
    excess = sus_dB.mean() - ctl_dB.mean()

    print(f"  suspect bins: {SUSPECT_BINS}")
    for b in SUSPECT_BINS:
        print(f"    bin {b:2d}: {spec_db[b]:+6.1f} dB")
    print(f"  control bins: {CONTROL_BINS}")
    for b in CONTROL_BINS:
        print(f"    bin {b:2d}: {spec_db[b]:+6.1f} dB")
    print(f"  excess (suspect - control mean): {excess:+.1f} dB")

    if excess > 6.0:
        verdict = "REAL_ARTIFACT"
        print("  Decision: >6 dB excess -> artifact is REAL (not notch self-signature).")
    elif excess > 1.0:
        verdict = "BORDERLINE"
        print("  Decision: 1-6 dB excess -> artifact present but weak.")
    else:
        verdict = "NOTCH_SELF_SIGNATURE"
        print("  Decision: <1 dB excess -> notch was creating phantom (Wave 1 hypothesis).")
    return {"excess_db": float(excess), "spec_db": spec_db, "verdict": verdict}


# ─────────────────────────── T2 ────────────────────────────────────
def t2_cepstrum(rec_label: str, t1_result: dict) -> dict:
    """Cepstrum (FFT of log magnitude spectrum). Peak at normalized freq 1/12."""
    banner(f"T2 — Cepstrum (spacing test) [{rec_label}]")
    print("Hypothesis : artifact has exact 24-bin period (corrected after T1).")
    print("Measurement: FFT of log magnitude spectrum; peak at freq 1/24 = 0.0417.")

    spec_db = t1_result["spec_db"]
    # Normalize: subtract mean, taper to avoid edge effects
    s = (spec_db - spec_db.mean()) * np.hanning(len(spec_db))
    cep = np.abs(sfft.rfft(s))
    # Skip the very-low freq part (broadband shape), look in the periodic regime
    # Bin k of cep corresponds to period N_RANGE/k bins
    # For 12-bin period: k = N_RANGE / 12 = 97/12 ≈ 8
    target_k = round(N_RANGE / 24)   # period = 24 bins (corrected after T1)
    # Window: ±2 around target
    region = cep[max(target_k - 2, 0): target_k + 3]
    rest = np.concatenate([cep[3: max(target_k - 2, 3)], cep[target_k + 3: -2]])
    if region.size == 0 or rest.size == 0:
        print("  Cepstrum bins out of range; skipping.")
        return {"peak_at_target": False}
    peak_target = region.max()
    peak_rest = rest.max()
    margin = 20 * np.log10(peak_target / max(peak_rest, 1e-9))

    print(f"  cepstrum peak at k~{target_k} (24-bin period): {peak_target:.2f}")
    print(f"  max elsewhere (excluding low-k baseline): {peak_rest:.2f}")
    print(f"  margin: {margin:+.1f} dB")
    if margin > 3:
        print("  Decision: peak strongly at expected period -> deterministic 12-bin spacing.")
        verdict = "EXACT_12_PERIOD"
    elif margin > 0:
        print("  Decision: weak peak -> spacing approximate.")
        verdict = "WEAK_12_PERIOD"
    else:
        print("  Decision: no peak at 12 -> not an exact 12-bin period.")
        verdict = "NO_12_PERIOD"
    return {"peak_target": float(peak_target), "margin_db": float(margin), "verdict": verdict}


# ─────────────────────────── T3 ────────────────────────────────────
def t3_chirp_coherence(rec_label: str, rc_stack: np.ndarray) -> dict:
    """Chirp-to-chirp coherence at lags 1, 16, 64, 256. Average over RX, suspect bins, frames.

    Coherence = |sum_n R[n+lag,bin] * conj(R[n,bin])| / (|...|*|...|).
    """
    banner(f"T3 — Chirp-to-chirp coherence [{rec_label}]")
    print("Hypothesis : deterministic spur -> high lag-1 coherence; random noise -> low.")
    print("Measurement: coherence at lags 1, 16, 64, 256, averaged over RX/suspect-bins/frames.")

    lags = [1, 16, 64, 256]
    # rc_stack: (n_frames, chirps, range, rx)
    n_frm = rc_stack.shape[0]
    coh_susp = {lag: [] for lag in lags}
    coh_ctl = {lag: [] for lag in lags}
    for fi in range(n_frm):
        for rx in range(N_RX):
            for b_set, bag in ((SUSPECT_BINS, coh_susp), (CONTROL_BINS, coh_ctl)):
                for b in b_set:
                    s = rc_stack[fi, :, b, rx]
                    for lag in lags:
                        s1 = s[: N_CHIRPS - lag]
                        s2 = s[lag:]
                        num = np.abs(np.vdot(s1, s2))
                        den = np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-12
                        bag[lag].append(num / den)

    print("  Suspect bins (mean ± std across all RX/bin/frame samples):")
    print(f"    lag       coh_suspect          coh_control")
    susp_means = {}
    ctl_means = {}
    for lag in lags:
        a = np.array(coh_susp[lag])
        c = np.array(coh_ctl[lag])
        susp_means[lag] = a.mean()
        ctl_means[lag] = c.mean()
        print(f"    {lag:>4d}    {a.mean():.3f} ± {a.std():.3f}     {c.mean():.3f} ± {c.std():.3f}")

    # Decision based on lag=1 suspect
    s1 = susp_means[1]
    c1 = ctl_means[1]
    if s1 > c1 + 0.2 and s1 > 0.5:
        verdict = "DETERMINISTIC"
        print(f"  Decision: suspect lag-1 coherence ({s1:.2f}) >> control ({c1:.2f}) -> deterministic spur.")
    elif s1 > c1 + 0.05:
        verdict = "PARTIALLY_DETERMINISTIC"
        print(f"  Decision: suspect lag-1 marginally above control -> partly deterministic.")
    else:
        verdict = "INCOHERENT"
        print(f"  Decision: suspect ≈ control -> incoherent / broadband noise.")

    # Drift check: lag-1 vs lag-256
    drift = susp_means[1] - susp_means[256]
    print(f"  Drift (lag-1 - lag-256 coherence): {drift:+.2f}")
    if drift > 0.3:
        print("    -> deterministic but DRIFTING (PLL/clock with phase walk).")
    return {"susp_means": susp_means, "ctl_means": ctl_means, "drift": float(drift), "verdict": verdict}


# ─────────────────────────── T4 ────────────────────────────────────
def t4_rx_signature(rec_label: str, rc_stack: np.ndarray) -> dict:
    """Per-RX magnitude std + inter-RX phase difference at suspect bins."""
    banner(f"T4 — Per-RX signature [{rec_label}]")
    print("Hypothesis : pre-RX shared (clock/supply) -> low std + zero phase diff.")
    print("             RX-specific (cable) -> high std + nonzero phase diff.")
    print("Measurement: per-RX magnitude std (dB) and inter-RX phase diff (deg) at suspect bins.")

    # Average over frames and chirps to get one complex value per (range, rx)
    # rc_stack shape: (n_frames, chirps, range, rx)
    avg = rc_stack.mean(axis=(0, 1))   # (range, rx) complex
    mag_db = db(avg)                    # (range, rx)
    phase = np.angle(avg, deg=True)     # (range, rx)

    print(f"  bin   |  RX0    RX1    RX2    RX3   | mag_std_dB | rel_phase[1-0,2-0,3-0] deg")
    for b in SUSPECT_BINS:
        m = mag_db[b]
        ph = phase[b]
        std = m.std()
        rel_ph = (ph[1:] - ph[0] + 180) % 360 - 180
        print(f"  {b:3d}  | {m[0]:+5.1f} {m[1]:+5.1f} {m[2]:+5.1f} {m[3]:+5.1f}  | {std:6.2f}    | {rel_ph[0]:+6.1f} {rel_ph[1]:+6.1f} {rel_ph[2]:+6.1f}")

    # Aggregate: average mag std and average abs phase diff at suspect bins
    avg_std = float(np.mean([mag_db[b].std() for b in SUSPECT_BINS]))
    avg_phase_diff = float(np.mean([np.mean(np.abs(((phase[b][1:] - phase[b][0] + 180) % 360 - 180))) for b in SUSPECT_BINS]))
    print(f"\n  Aggregate at suspect bins: mag_std={avg_std:.2f} dB, mean |Δphase|={avg_phase_diff:.1f} deg")

    if avg_std < 1.0 and avg_phase_diff < 30:
        verdict = "PRE_RX_SHARED"
        print("  Decision: low std + ~zero phase -> pre-RX shared origin (sample clock or supply).")
    elif avg_std > 3.0 or avg_phase_diff > 60:
        verdict = "RX_SPECIFIC"
        print("  Decision: high std or large phase diff -> RX-specific (cable/connector).")
    else:
        verdict = "MIXED"
        print("  Decision: mixed -> partially shared.")
    return {"avg_std_db": avg_std, "avg_phase_diff_deg": avg_phase_diff, "verdict": verdict}


# ─────────────────────────── T5 ────────────────────────────────────
def t5_doppler_localization(rec_label: str, rc_stack: np.ndarray) -> dict:
    """Slow-time FFT WITHOUT MTI; check if artifact at suspect bins is at zero Doppler."""
    banner(f"T5 — Doppler-axis localization (no MTI) [{rec_label}]")
    print("Hypothesis : DC-confined -> static internal source (clock/supply).")
    print("             Spread -> time-varying (PMIC switching with jitter).")
    print("Measurement: ratio of suspect-bin energy at DC vs spread across Doppler.")

    # Use first frame, no MTI. Slow-time FFT on the chirps axis.
    rc_first = rc_stack[0]                                       # (chirps, range, rx)
    rc_summed = rc_first.sum(axis=2)                             # (chirps, range)
    win = np.hanning(N_CHIRPS).astype(np.float32)
    spec = sfft.fft(rc_summed * win[:, None], axis=0)            # (chirps, range)
    spec = np.fft.fftshift(spec, axes=0)
    pwr = (np.abs(spec) ** 2)
    n_dop = N_CHIRPS
    dc = n_dop // 2
    dc_window = slice(dc - 1, dc + 2)                            # ±1 bin around DC

    print(f"  bin  | DC±1 power(dB)  rest_max(dB)  diff(dB)")
    dc_minus_rest = []
    for b in SUSPECT_BINS:
        col = pwr[:, b]
        dc_pwr = col[dc_window].max()
        # Rest = exclude DC ±5 to be safe (Hann sidelobes leak)
        rest_mask = np.ones(n_dop, bool)
        rest_mask[dc - 5: dc + 6] = False
        rest_pwr = col[rest_mask].max()
        d = 10 * np.log10(dc_pwr / max(rest_pwr, 1e-30))
        dc_minus_rest.append(d)
        print(f"  {b:3d}  | {10*np.log10(dc_pwr):+7.1f}        {10*np.log10(rest_pwr):+7.1f}      {d:+5.1f}")

    avg_excess = float(np.mean(dc_minus_rest))
    print(f"\n  Aggregate DC excess at suspect bins: {avg_excess:+.1f} dB")

    if avg_excess > 10:
        verdict = "DC_CONFINED"
        print("  Decision: DC-confined -> static internal source.")
    elif avg_excess > 0:
        verdict = "DC_BIASED"
        print("  Decision: weakly DC-biased -> mostly static with some drift.")
    else:
        verdict = "SPREAD"
        print("  Decision: spread across Doppler -> time-varying (PMIC switching).")
    return {"dc_excess_db": avg_excess, "verdict": verdict}


# ─────────────────────────── T6 ────────────────────────────────────
def t6_strength_vs_floor(rec_label: str, t1_result: dict) -> dict:
    """How strong is the artifact in dB above the local floor?"""
    banner(f"T6 -- Strength vs noise floor [{rec_label}]")
    print("Measurement: per-suspect-bin dB above the median of nearby non-suspect bins.")

    spec_db = t1_result["spec_db"]
    excess = []
    print(f"  bin  | level(dB)  local_floor(dB)  dB_above_floor")
    for b in SUSPECT_BINS:
        # Local floor: median of bins within +/-5 of b, EXCLUDING multiples-of-24 in that range
        lo, hi = max(b - 5, 0), min(b + 6, N_RANGE)
        nearby = [k for k in range(lo, hi) if k % 24 != 0]
        if not nearby:
            continue
        floor = np.median([spec_db[k] for k in nearby])
        ex = spec_db[b] - floor
        excess.append(ex)
        print(f"  {b:3d}  | {spec_db[b]:+7.1f}    {floor:+7.1f}        {ex:+5.1f}")

    avg_excess = float(np.mean(excess))
    max_excess = float(np.max(excess))
    print(f"\n  Suspect bins: mean +{avg_excess:.1f} dB, max +{max_excess:.1f} dB above local floor")

    if max_excess > 20:
        verdict = "DOMINANT"
        print("  Severity: DOMINANT (>20 dB) — artifact must be removed for any detection.")
    elif max_excess > 6:
        verdict = "SIGNIFICANT"
        print("  Severity: SIGNIFICANT (6-20 dB) — affects CFAR but detection still possible.")
    elif max_excess > 0:
        verdict = "MARGINAL"
        print("  Severity: MARGINAL (<6 dB) — barely above floor.")
    else:
        verdict = "NEGLIGIBLE"
        print("  Severity: NEGLIGIBLE — at or below floor; nothing to remove.")
    return {"avg_excess_db": avg_excess, "max_excess_db": max_excess, "verdict": verdict}


# ─────────────────── per-recording driver ──────────────────────────
def run_recording(path: str, label: str) -> dict:
    print()
    print(("#" * 70))
    print(f"# RECORDING: {label}")
    print(f"# path: {path}")
    nf = n_frames(path)
    print(f"# total frames: {nf}")
    print(("#" * 70))

    rc_stack = stack_frames(path, n=8, stride=64)
    t1 = t1_bypass_notch(label, rc_stack)
    t2 = t2_cepstrum(label, t1)
    t3 = t3_chirp_coherence(label, rc_stack)
    t4 = t4_rx_signature(label, rc_stack)
    t5 = t5_doppler_localization(label, rc_stack)
    t6 = t6_strength_vs_floor(label, t1)

    return {"label": label, "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6}


# ─────────────────────── final ranked verdict ──────────────────────
def synthesize_verdict(results: list[dict]) -> None:
    banner("FINAL RANKED VERDICT", char="*")
    print()

    # Score 4 hypotheses based on the test outcomes
    scores = {
        "ADC_INTERLEAVE": 0,    # 16-way time-interleaved ADC mismatch
        "PMIC_SWITCHING": 0,    # PMIC switching regulator at 1.875 MHz
        "USB_CABLE_EMI":  0,    # Per-channel cable / USB-coupled EMI
        "NOTCH_SELF":     0,    # Notch filter is creating its own signature
    }

    for r in results:
        # T1 verdict gates everything
        if r["T1"]["verdict"] == "NOTCH_SELF_SIGNATURE":
            scores["NOTCH_SELF"] += 5
            continue
        if r["T1"]["verdict"] == "REAL_ARTIFACT":
            # Real artifact — narrow down by other tests
            scores["ADC_INTERLEAVE"] += 1   # baseline (real & periodic)
            scores["PMIC_SWITCHING"] += 1
            scores["USB_CABLE_EMI"]  += 1

        # T2 — exact 12-period strongly favors ADC interleave (deterministic chip clock divider)
        if r["T2"].get("verdict") == "EXACT_12_PERIOD":
            scores["ADC_INTERLEAVE"] += 2
            scores["PMIC_SWITCHING"] += 1   # PMIC at exactly fs/16 less likely

        # T3 — high deterministic coherence -> ADC interleave, low -> PMIC
        if r["T3"]["verdict"] == "DETERMINISTIC":
            scores["ADC_INTERLEAVE"] += 3
            scores["PMIC_SWITCHING"] -= 1
        elif r["T3"]["verdict"] == "INCOHERENT":
            scores["PMIC_SWITCHING"] += 2
            scores["ADC_INTERLEAVE"] -= 2

        # T4 — pre-RX shared favors ADC/clock; RX-specific favors USB
        if r["T4"]["verdict"] == "PRE_RX_SHARED":
            scores["ADC_INTERLEAVE"] += 2
            scores["USB_CABLE_EMI"]  -= 2
        elif r["T4"]["verdict"] == "RX_SPECIFIC":
            scores["USB_CABLE_EMI"]  += 3
            scores["ADC_INTERLEAVE"] -= 1

        # T5 — DC-confined favors static (ADC/clock); spread favors time-varying (PMIC)
        if r["T5"]["verdict"] == "DC_CONFINED":
            scores["ADC_INTERLEAVE"] += 2
            scores["PMIC_SWITCHING"] -= 2
        elif r["T5"]["verdict"] == "SPREAD":
            scores["PMIC_SWITCHING"] += 3
            scores["ADC_INTERLEAVE"] -= 1

    print("HYPOTHESIS SCORES (across all recordings):")
    for h, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"   {h:>20s}: {s:+3d}")

    winner = max(scores, key=scores.get)
    print()
    print(f"  -> WINNER: {winner}")
    print()
    print("Recommended Step 2 fix:")
    if winner == "ADC_INTERLEAVE":
        print("  Implement sample-domain per-sub-ADC offset calibration in")
        print("  _stage1_range_fft BEFORE the per-chirp DC subtract on line 539.")
        print("  Then DELETE _notch_harmonic_artifact.")
    elif winner == "PMIC_SWITCHING":
        print("  Keep notch but tighten radius from ±5 to ±2 (recovers ~70% of range).")
        print("  Add a hardware ferrite-bead recommendation to bench setup runbook.")
    elif winner == "USB_CABLE_EMI":
        print("  Disconnect J8 XDS110 USB at runtime; re-record and verify.")
        print("  Tighten notch to ±2 if residual remains.")
    elif winner == "NOTCH_SELF":
        print("  Comment out _notch_harmonic_artifact call site. Re-validate.")
    print()


# ─────────────────────────── main ──────────────────────────────────
def main(argv: list[str]) -> int:
    print(f"diagnose_artifact.py — every-12-bin artifact diagnostic")
    print(f"  N_CHIRPS={N_CHIRPS}, N_SAMPLES={N_SAMPLES}, FS={FS_HZ/1e6} MHz")
    print(f"  bin 12 IF freq = 12 * FS / N_SAMPLES = {12 * FS_HZ / N_SAMPLES / 1e6} MHz")
    print(f"  suspect bins: {SUSPECT_BINS}")
    print(f"  control bins: {CONTROL_BINS}")

    results = []
    for path, label in [
        (REC_AIRBORNE, "airborne1 (drone fly-away)"),
        (REC_DRONEFLY, "drone-fly (5m hover)"),
        (REC_BACKGROUND, "background (no drone)"),
    ]:
        if not os.path.exists(path):
            print(f"\n[skip] {label}: file not found")
            continue
        results.append(run_recording(path, label))

    if results:
        synthesize_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
