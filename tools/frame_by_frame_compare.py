"""Frame-by-frame correlation: EO + thermal + radar at matched timestamps.

For each EO frame (with drone visible per user), find:
  1. The matching thermal frame at same wall-clock
  2. The matching radar bin frame at same wall-clock
  3. The drone's pixel position in EO (manual inspection from saved JPGs)
  4. The expected drone azimuth from pixel position (using HFOV)
  5. The radar spectrum at expected drone range
  6. What MIMO detector returns at this frame

Outputs a per-timestamp CSV + saves a side-by-side preview for each
sample so we can visually align "drone in EO" vs "what radar saw".

Run:
    py -3.11 tools/frame_by_frame_compare.py
"""
from __future__ import annotations

import sys
import json
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

from radar_dca.ddma import ddma_unfold
from radar_dca.mimo_coherence_detector import MIMOCoherenceDetector

# Constants
N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_TX = 4
EFF_PRF = PRF_HZ / N_TX
N_FFT = 1024
BIN_HZ = EFF_PRF / N_FFT
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

# EO camera
EO_WIDTH = 1236
EO_HEIGHT = 1032
EO_HFOV_DEG = 11.05  # per gui code default
EO_VFOV_DEG = 9.23

BIN_PATH = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin"
)
JSONL_PATH = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\drone test airborne 1.jsonl"
)


def load_radar_frame(idx):
    with open(BIN_PATH, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME: raise EOFError
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
              .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def stage1(real_cube):
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2).astype(np.complex64)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out


def per_frame_va_spec(idx):
    cube = load_radar_frame(idx)
    rc = stage1(cube)
    rc -= rc.mean(axis=0, keepdims=True)  # MTI
    virtual = ddma_unfold(rc)  # (192, 97, 4, 4)
    n_per_tx, n_range, _, _ = virtual.shape
    slow = virtual.reshape(n_per_tx, n_range, N_RX * N_TX)
    win = np.hanning(n_per_tx).astype(np.float32)
    spec = np.fft.fft(slow * win[:, None, None], n=N_FFT, axis=0)
    return spec[: N_FFT // 2, :, :]  # (512, 97, 16) complex


def coherence_per_freq(spec_va_int, freq_lo, freq_hi):
    """For accumulated covariance, compute spread per (freq, range)."""
    pass  # not used here — we'll do per-frame snapshot covariance


def aoa_eigenvec_to_az(eigvec):
    """Cheap az estimate from dominant eigenvector across 16 VAs."""
    from radar_dca.pmm_detector import _VA_COLS, _COL_SPACING_LAMBDA
    cols = np.array(_VA_COLS, dtype=np.float64)
    cols = cols - cols.min()
    az_grid = np.arange(-60.0, 60.5, 0.5)
    az_rad = np.deg2rad(az_grid)
    steering = np.exp(1j * 2.0 * np.pi * _COL_SPACING_LAMBDA *
                      np.sin(az_rad)[:, None] * cols[None, :])
    beam = np.abs((steering.conj() * eigvec[None, :]).sum(axis=1))
    return float(az_grid[int(np.argmax(beam))])


def build_jsonl_index():
    """Index the JSONL by channel: (ts_ns, msg) lists for eo, thermal, aa_frame."""
    eo = []
    thermal = []
    aa = []
    with io.open(JSONL_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            try: r = json.loads(line)
            except: continue
            ch = r.get("channel")
            ts = r.get("ts_ns")
            msg = r.get("msg") or {}
            if ch == "eo/frame":
                eo.append((ts, msg))
            elif ch == "thermal/frame":
                thermal.append((ts, msg))
            elif ch == "radar/aa_frame":
                aa.append((ts, msg))
    return eo, thermal, aa


def find_closest(records, target_ts):
    """Return index of record with timestamp closest to target_ts."""
    if not records: return -1
    best_i, best_d = 0, abs(records[0][0] - target_ts)
    for i, (ts, _) in enumerate(records):
        d = abs(ts - target_ts)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def detect_drone_in_eo(eo_msg):
    """Hard-coded crude detector: look for compact dark blob against sky.
    For now, just report any YOLO bbox if present; otherwise None.
    """
    dets = eo_msg.get("detections") or []
    # YOLO often misclassifies; collect all
    bboxes = []
    for d in dets:
        bb = d.get("bbox") or {}
        bboxes.append((bb.get("x", 0), bb.get("y", 0), bb.get("w", 0), bb.get("h", 0),
                       d.get("target_class", "?"), d.get("confidence", 0)))
    return bboxes


def main():
    print("Building JSONL index...")
    eo_idx, thermal_idx, aa_idx = build_jsonl_index()
    print(f"EO frames: {len(eo_idx)}, thermal: {len(thermal_idx)}, aa: {len(aa_idx)}")
    if not aa_idx:
        print("No aa_frames!"); return 1

    base_ts = aa_idx[0][0]
    base_aa_fid = aa_idx[0][1].get("frame_id", 0)
    print(f"base ts_ns = {base_ts}, base aa frame_id = {base_aa_fid}")
    print()

    # Sample timestamps every 6 seconds across the recording
    duration_s = (aa_idx[-1][0] - base_ts) / 1e9
    n_samples = int(duration_s / 6.0)
    print(f"Recording duration: {duration_s:.1f}s, sampling every 6s -> {n_samples} samples")
    print()

    # Build a pre-warmed MIMO detector — feed N=6 frames before each sample
    detector = MIMOCoherenceDetector(
        prf_hz=PRF_HZ, n_chirps_per_tx=N_SAMPLES,
        n_integration_frames=6, threshold_db=3.0,  # lower thr to see what's there
        scan_freq_lo_hz=50.0, scan_freq_hi_hz=3000.0,
        range_bin_min=4,
    )
    # Wait — n_chirps_per_tx is 192 not N_SAMPLES. Let me fix.
    detector = MIMOCoherenceDetector(
        prf_hz=PRF_HZ, n_chirps_per_tx=192,
        n_integration_frames=6, threshold_db=3.0,
        scan_freq_lo_hz=50.0, scan_freq_hi_hz=3000.0,
        range_bin_min=4,
    )

    print(f"{'t_rel':>6} {'EO fid':>8} {'EO bbox class/conf':<35} "
          f"{'rad_idx':>8} {'#hits':>5} {'top hits (rb@m / Hz / dB / az)':<60}")
    print("-" * 140)

    for s in range(n_samples):
        target_ts = base_ts + int(s * 6e9)
        # Find matching aa_frame
        aa_i = find_closest(aa_idx, target_ts)
        aa_ts, aa_msg = aa_idx[aa_i]
        radar_idx = aa_msg.get("frame_id", 0) - base_aa_fid
        if radar_idx < 0 or radar_idx > 2148: continue

        t_rel = (aa_ts - base_ts) / 1e9

        # Find matching EO
        eo_i = find_closest(eo_idx, aa_ts)
        eo_ts, eo_msg = eo_idx[eo_i]
        eo_fid = eo_msg.get("frame_id", 0)
        eo_bbs = detect_drone_in_eo(eo_msg)
        eo_bb_str = "no det"
        if eo_bbs:
            x, y, w, h, cls, conf = eo_bbs[0]
            eo_bb_str = f"{cls}@{conf:.2f} bb=({x},{y},{w}x{h})"

        # Reset detector state and feed 6 prior frames + current
        detector.reset()
        try:
            for k in range(6):
                idx = max(0, radar_idx - 5 + k)
                cube = load_radar_frame(idx)
                rc = stage1(cube)
                rc -= rc.mean(axis=0, keepdims=True)
                hits = detector.process_frame(rc)
        except EOFError:
            print(f"{t_rel:>6.1f} radar load failed at idx {radar_idx}")
            continue

        # Last frame was current — collect hits
        # Sort by spread descending
        hits = sorted(hits, key=lambda h: -h[1].band_snr_db)
        hits_str = "; ".join(
            f"rb{rb}@{rb*RANGE_RES_M:.0f}m/{r.blade_freq_hz:.0f}Hz/{r.band_snr_db:.1f}dB/az{r.az_deg:+.0f}"
            for rb, r in hits[:3]
        )
        print(f"{t_rel:>6.1f} {eo_fid:>8} {eo_bb_str:<35} "
              f"{radar_idx:>8} {len(hits):>5} {hits_str:<60}")


if __name__ == "__main__":
    sys.exit(main())
