"""Run patched detector on airborne1 binary + inject radar bbs into replay JSONL.

Output: a new JSONL `airborne1_v5+thermalv2_replay_RADAR.jsonl` that
mirrors the original replay file but with `radar/aa_frame` events
populated with the drone target detected by the patched pipeline.

Usage:
    py -3.11 tools/inject_radar_bbs.py
    py -3.11 scripts/replay_server.py recordings/airborne1_v5+thermalv2_replay_RADAR.jsonl

The browser at localhost:8081 will show the EO + thermal + radar
panels side by side, with the radar panel rendering my bbs.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.fft as sfft

# ─────────────── cfg constants (awr2944P_unified.cfg) ──────────────
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

NOTCH_STEP = 24
NOTCH_RADIUS = 3

CFAR_GR, CFAR_TR = 2, 8
CFAR_THRESHOLD_DB = 6.0

# Drone gate: close-range, off-DC Doppler, |az| <= 30°
DRONE_RANGE_MIN_M = 1.5
DRONE_RANGE_MAX_M = 12.0
DRONE_AZ_LIMIT_DEG = 30.0
DRONE_VEL_MIN_MPS = 0.10

REC_DIR = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings")
AIR_BIN = REC_DIR / "seeker_2026-05-06_12-58-59_radar.bin"
REPLAY_IN = REC_DIR / "airborne1_v5+thermalv2_replay.jsonl"
REPLAY_OUT = REC_DIR / "airborne1_v5+thermalv2_replay_RADAR.jsonl"


# ─────────────── pipeline stages (mirror dca_pipeline.py) ──────────
def stage1_range_fft(buf: bytes) -> np.ndarray:
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
            .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    rfft = sfft.rfft(cube * HANN_FAST[None, :, None], axis=1, workers=2).astype(np.complex64)
    rfft[:, 1:-1, :] *= 2.0
    return rfft


def notch(rc: np.ndarray) -> None:
    for b in range(NOTCH_STEP, rc.shape[1], NOTCH_STEP):
        lo, hi = max(b - NOTCH_RADIUS, 0), min(b + NOTCH_RADIUS + 1, rc.shape[1])
        rc[:, lo:hi, :] = 0


def stage3_rd(rc: np.ndarray):
    trimmed = rc[: N_GROUPS * INTEGRATE_CHIRPS]
    integrated = trimmed.reshape(N_GROUPS, INTEGRATE_CHIRPS, N_RANGE, N_RX).mean(axis=1)
    rd = np.fft.fftshift(
        sfft.fft(integrated * HANN_SLOW[:, None, None], axis=0, workers=2),
        axes=0,
    )
    dc = rd.shape[0] // 2
    rd[max(dc - 1, 0): dc + 2, :, :] = 0
    return rd, np.abs(rd.sum(axis=2)).astype(np.float32)


def detect_drone(buf: bytes) -> Optional[dict]:
    """Run detector, return best close-range drone-like cell or None.

    Strategy: scan range bins 1-4 (the bins where the drone lives in
    these recordings) for the strongest off-DC cell that beats the
    local noise by CFAR_THRESHOLD_DB. Return the strongest such cell
    if it passes the drone gate.
    """
    rc = stage1_range_fft(buf)
    notch(rc)
    rd, rd_mag = stage3_rd(rc)
    n_dop, n_rng = rd_mag.shape
    pwr = rd_mag.astype(np.float32) ** 2
    threshold_lin = 10 ** (CFAR_THRESHOLD_DB / 10.0)
    fd_scale = PRF_HZ / INTEGRATE_CHIRPS / N_GROUPS

    best = None
    rb_min = max(1, int(DRONE_RANGE_MIN_M / RANGE_RES_M))
    rb_max = int(DRONE_RANGE_MAX_M / RANGE_RES_M) + 1
    for r in range(rb_min, min(rb_max, n_rng)):
        for d in range(2, n_dop - 2):
            cell = pwr[d, r]
            if cell <= 0:
                continue
            # Right-side training; left-side if available.
            right = pwr[d, r + CFAR_GR + 1: r + CFAR_GR + CFAR_TR + 1]
            left = pwr[d, max(r - CFAR_GR - CFAR_TR, 0): max(r - CFAR_GR, 0)]
            ring = np.concatenate([left, right]) if left.size else right
            if ring.size == 0:
                continue
            noise = ring.mean() + 1e-9
            if cell <= threshold_lin * noise:
                continue
            snr_db = 10.0 * np.log10(cell / noise)
            # Velocity gate
            fd = (d - n_dop / 2.0) * fd_scale
            vel = -LAM_M / 2.0 * fd
            if abs(vel) < DRONE_VEL_MIN_MPS:
                continue
            # AoA via 64-pt FFT across RX
            rx_vec = rd[d, r, :]
            az_spec = np.fft.fftshift(sfft.fft(rx_vec, n=64))
            az_bin = int(np.argmax(np.abs(az_spec)))
            sin_theta = float(np.clip((az_bin - 32) / 32.0, -1.0, 1.0))
            az_deg = float(np.degrees(np.arcsin(sin_theta)))
            if abs(az_deg) > DRONE_AZ_LIMIT_DEG:
                continue
            range_m = float(r) * RANGE_RES_M
            if best is None or snr_db > best["snr_db"]:
                best = {
                    "range_m": range_m,
                    "az_deg": az_deg,
                    "vel_mps": float(vel),
                    "snr_db": float(snr_db),
                    "rb": int(r),
                    "dop_idx": int(d),
                }
    return best


# ─────────────── replay JSONL injection ────────────────────────────
def make_target_dict(det: dict, tid: int = 1) -> dict:
    """Build a RadarTarget-shaped dict consumed by gui/static/js/radar_view.js.

    Sensor frame: +x right, +y forward (boresight), +z up. Convert
    polar (range_m, az_deg) -> (x, y) in the radar's local frame.
    """
    az_rad = float(np.radians(det["az_deg"]))
    x = det["range_m"] * float(np.sin(az_rad))
    y = det["range_m"] * float(np.cos(az_rad))
    # Velocity vector — assume velocity is radial (along the line of sight).
    vx = det["vel_mps"] * float(np.sin(az_rad))
    vy = det["vel_mps"] * float(np.cos(az_rad))
    return {
        "tid": tid,
        "pos_x_m": float(x),
        "pos_y_m": float(y),
        "pos_z_m": 0.0,
        "vel_x_mps": float(vx),
        "vel_y_mps": float(vy),
        "vel_z_mps": 0.0,
        "size_x_m": 0.5,
        "size_y_m": 0.5,
        "size_z_m": 0.5,
        "confidence": min(1.0, max(0.4, (det["snr_db"] - 4) / 20)),
        "source": "rd_drone",
        "num_points": 1,
        "coasting": False,
        "hits": 1,
        "misses": 0,
        # GUI-friendly fields radar_view.js looks for
        "x": float(x),
        "y": float(y),
        "z": 0.0,
        "range_m": float(det["range_m"]),
        "az_deg": float(det["az_deg"]),
        "doppler_mps": float(det["vel_mps"]),
        "snr_db": float(det["snr_db"]),
        "target_class": "drone",
    }


def main() -> int:
    if not AIR_BIN.exists():
        print(f"missing: {AIR_BIN}", file=sys.stderr); return 1
    if not REPLAY_IN.exists():
        print(f"missing: {REPLAY_IN}", file=sys.stderr); return 1

    n_bin_frames = os.path.getsize(AIR_BIN) // BPF
    print(f"binary file: {n_bin_frames} frames")

    # Build mapping: aa_frame_event_index -> binary_frame_index.
    # Simplest: assume sequential 1:1.
    # Run the detector for every aa_frame in the JSONL.
    print("collecting radar/aa_frame timestamps from replay JSONL...")
    aa_indices = []   # line indices of radar/aa_frame events
    with open(REPLAY_IN, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
                if r.get("channel") == "radar/aa_frame":
                    aa_indices.append(i)
            except Exception:
                pass
    print(f"  {len(aa_indices)} radar/aa_frame events found")

    n_to_run = min(len(aa_indices), n_bin_frames)
    print(f"running detector on {n_to_run} binary frames...")

    detections: dict[int, Optional[dict]] = {}   # binary frame idx -> det
    fbin = open(AIR_BIN, "rb")
    try:
        for i in range(n_to_run):
            fbin.seek(i * BPF)
            buf = fbin.read(BPF)
            if len(buf) != BPF:
                break
            det = detect_drone(buf)
            detections[i] = det
            if i % 100 == 0:
                print(f"  frame {i}/{n_to_run}  det={det}")
    finally:
        fbin.close()

    n_hits = sum(1 for v in detections.values() if v is not None)
    print(f"detected drone in {n_hits}/{n_to_run} frames "
          f"({100*n_hits/max(n_to_run,1):.1f}%)")

    # Walk the replay JSONL, injecting detections at radar/aa_frame events.
    print(f"writing patched replay JSONL to {REPLAY_OUT.name}...")
    aa_seen = 0
    with open(REPLAY_IN, "r", encoding="utf-8") as fin, \
         open(REPLAY_OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except Exception:
                fout.write(line)
                continue
            if r.get("channel") == "radar/aa_frame":
                bin_idx = aa_seen
                aa_seen += 1
                msg = r.get("msg", {}) or {}
                # Force connected on so the GUI renders something.
                msg["connected"] = True
                if msg.get("max_range_m") in (None, 0, 0.0):
                    msg["max_range_m"] = 250.0
                if msg.get("fov_half_deg") in (None, 0, 0.0):
                    msg["fov_half_deg"] = 60.0
                det = detections.get(bin_idx)
                if det is not None:
                    target = make_target_dict(det, tid=1)
                    msg["targets"] = [target]
                    msg["num_targets"] = 1
                    msg["points"] = [target]
                    msg["num_points"] = 1
                else:
                    msg["targets"] = []
                    msg["num_targets"] = 0
                    msg["points"] = []
                    msg["num_points"] = 0
                r["msg"] = msg
                fout.write(json.dumps(r) + "\n")
            else:
                fout.write(line)

    print(f"done. wrote {aa_seen} radar/aa_frame events into {REPLAY_OUT}")
    print()
    print("To watch the result:")
    print(f"  py -3.11 scripts/replay_server.py {REPLAY_OUT.name}")
    print("  open http://localhost:8081/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
