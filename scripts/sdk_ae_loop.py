"""Software auto-exposure loop driven directly through the Leopard SDK
helper.

Why this script exists
----------------------
The FX3 bridge's "auto exposure" (cam.AE = True) is broken on this
sensor in daylight: it leaves the IMX568 fully analog-saturated through
the 35 mm NIR-pass lens and never recovers, regardless of how many AE
ticks pass. This was verified on 2026-04-25: "Auto" mode in the GUI
produced a pure-white image; CameraTool was equally white.

So bridge AE is unusable. We need to do AE ourselves: capture a frame,
look at the raw u16 statistics, write a new ExposureExt, repeat.

This script proves the algorithm before it gets baked into eo_manager
as a live loop. Two reasons to keep it as a standalone script too:

  1. It's the only camera-attached test we can run without launching
     the full GUI / pipeline. Useful any time AE looks wrong in the
     wild — point this at the scene and let the iterations show what
     it converges on.
  2. The numbers it prints per iteration are the ground truth — if
     the live AE inside eo_manager misbehaves later, run this to
     verify the camera+SDK path itself is fine.

Algorithm — bracketing AE
-------------------------
Targets ``raw u16 p99`` (99th percentile of the 16-bit-padded raw12
payload). p99 instead of mean: it tracks "is the sensor about to clip?"
much better than mean, which drifts with scene composition. We accept
any p99 inside [target_lo, target_hi] as converged; we don't try to hit
a single number, because empirically the IMX568 + 35 mm NIR-pass lens
has a regime (cloudy daylight) where 4× exposure produces 12× p99 —
proportional control just oscillates across the cliff.

Instead, the controller maintains two brackets while iterating:
    low_floor  = largest known ExposureExt where p99 was too dim
    high_brake = smallest known ExposureExt where the sensor saturated

Each step picks the next ExposureExt by:
    - if both brackets known → midpoint of [low_floor, high_brake]
    - if only low known      → 2× (catch up fast)
    - if only high known     → /2 (back off fast)
    - if no brackets yet     → proportional p99-ratio move, capped at 2×
The brackets close in until they're 1 apart; if neither sat nor noise
inside that bracket, we accept the lower one.

This converges on bizarre lens+light combos without overshoot. Tested
2026-04-25 on cloudy daylight + 35mm NIR-pass: cliff between exp=4
and exp=16 (no integer value gives "ideal" p99 ≈ 2200, but exp=4 with
p99=341 is perfectly usable for downstream AGC stretch).

Run
---
    Close the seeker GUI first (the SDK is single-consumer).
    python scripts/sdk_ae_loop.py
        [--target-p99 2200] [--start-exp 1264] [--max-iters 12]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PY32 = REPO / "tools" / "python311-x86" / "python.exe"
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
OUT_DIR = (REPO / "scripts" / "eo_snapshots" / "diagnostic"
           / time.strftime("ae_loop_%Y%m%d_%H%M%S"))

W, H = 2472, 2064

# Firmware-accepted ExposureExt range. We empirically know:
#   - 1 works (tested in sdk_daylight_diagnostic.py at exp=1, u16 mean=17)
#   - 50000 works (no upper-side test failures yet in nighttime)
# We don't go below 1; the SDK's wiggle/decoy code already handles the
# bridge-firmware-reuse quirk down to small values.
EXP_MIN = 1
EXP_MAX = 50000

# Saturation thresholds. "Fully clipped" = practically every pixel in
# the brightest 1% is jammed against the analog ceiling — common in
# bright daylight before the AE has had a chance to step down.
SAT_P99 = 4090
SAT_MEAN = 4000
NOISE_P99 = 50


def helper_capture(exposure_ext: int, frame_path: Path,
                   warmup: int = 8, timeout: float = 60.0) -> dict:
    """One-shot RAW12 capture at fixed ExposureExt via the SDK helper."""
    cmd = [
        str(PY32), str(HELPER),
        "--ae", "off",
        "--exposure-ext", str(int(exposure_ext)),
        "--data-mode", "RAW12",
        "--capture-width", str(W),
        "--capture-height", str(H),
        "--capture-warmup", str(int(warmup)),
        "--capture-frame", str(frame_path),
        "--json",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    info: dict
    try:
        info = json.loads(cp.stdout) if cp.stdout else {}
    except Exception:
        info = {"raw_stdout": cp.stdout[-500:]}
    info["_returncode"] = cp.returncode
    if cp.stderr:
        info["_stderr_tail"] = cp.stderr[-800:]
    return info


def decode_u16(bin_path: Path) -> np.ndarray:
    """SDK packs RAW12 as uint16 LE in the low 12 bits → range [0, 4095]."""
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    n2 = W * H * 2
    if raw.size < n2:
        raise RuntimeError(f"frame too small: {raw.size} < {n2}")
    return np.frombuffer(raw[:n2].tobytes(), dtype="<u2").reshape(H, W)


def stats_u16(u16: np.ndarray) -> dict:
    p = np.percentile(u16, [1, 50, 99]).tolist()
    return {
        "mean": float(u16.mean()),
        "std": float(u16.std()),
        "min": int(u16.min()),
        "max": int(u16.max()),
        "p1": float(p[0]), "p50": float(p[1]), "p99": float(p[2]),
        "frac_clip": float((u16 >= SAT_P99).mean()),
    }


class AEController:
    """Bracketing AE controller. See module docstring for the algorithm."""

    def __init__(self, target_lo: float, target_hi: float) -> None:
        self.target_lo = float(target_lo)
        self.target_hi = float(target_hi)
        # Largest exp known to give p99 < target_lo (too dim).
        self.low_floor: int | None = None
        # Smallest exp known to saturate or be above target_hi (too bright).
        self.high_brake: int | None = None

    def converged(self, p99: float, frac_clip: float) -> bool:
        # Accept any frame in the target zone with negligible clipping.
        return (self.target_lo <= p99 <= self.target_hi
                and frac_clip < 0.01)

    def step(self, exp: int, p99: float, mean: float,
             frac_clip: float) -> int:
        # ── classify the current frame ──
        is_sat = (frac_clip > 0.01) or (p99 >= SAT_P99)
        is_dim = (p99 < self.target_lo) and not is_sat

        if is_sat:
            # exp is too high. Tighten the high bracket.
            if self.high_brake is None or exp < self.high_brake:
                self.high_brake = exp
        elif is_dim:
            # exp is too low. Tighten the low bracket.
            if self.low_floor is None or exp > self.low_floor:
                self.low_floor = exp
        elif p99 > self.target_hi:
            # In-bounds-but-too-bright. Acts like "high but not yet
            # clipped" — store as a soft brake so we narrow toward
            # darker side.
            if self.high_brake is None or exp < self.high_brake:
                self.high_brake = exp

        # ── propose a next exposure ──
        if self.low_floor is not None and self.high_brake is not None:
            # We have both brackets. If they're already adjacent and
            # the lower one isn't dim, accept it.
            if self.high_brake - self.low_floor <= 1:
                # No integer between brackets; fall back to low_floor
                # (better dim than saturated).
                return self.low_floor
            # Midpoint of the bracket. Round down to bias toward
            # darker, because saturation is the worse failure mode.
            new = (self.low_floor + self.high_brake) // 2
            if new == exp:
                # Force a step in whichever direction makes sense.
                new = exp - 1 if is_sat else exp + 1
            return max(EXP_MIN, min(EXP_MAX, new))

        # Only one or no bracket yet — fall back to coarse moves.
        if is_sat:
            return max(EXP_MIN, exp // 2)
        if p99 < NOISE_P99:
            return min(EXP_MAX, max(EXP_MIN, exp * 4))
        if is_dim:
            return min(EXP_MAX, max(EXP_MIN, exp * 2))
        if p99 > self.target_hi:
            # Cap to 2× cut so we don't overshoot the cliff.
            return max(EXP_MIN, exp // 2)
        # In-zone but not converged (shouldn't happen — convergence
        # check fires first). Hold.
        return exp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-lo", type=float, default=300.0,
                    help="Lower bound of acceptable raw u16 p99")
    ap.add_argument("--target-hi", type=float, default=3500.0,
                    help="Upper bound of acceptable raw u16 p99")
    ap.add_argument("--start-exp", type=int, default=1264,
                    help="Initial ExposureExt to seed the loop")
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ae] output dir: {OUT_DIR}")
    print(f"[ae] target p99 zone: [{args.target_lo:.0f}, {args.target_hi:.0f}]")
    print(f"[ae] firmware exposure range: [{EXP_MIN}, {EXP_MAX}]")
    print(f"[ae] starting at ExposureExt={args.start_exp}")

    ae = AEController(args.target_lo, args.target_hi)
    history = []
    exp = int(args.start_exp)
    for i in range(args.max_iters):
        bin_path = OUT_DIR / f"iter{i:02d}_e{exp}.bin"
        t0 = time.time()
        info = helper_capture(exp, bin_path, warmup=args.warmup)
        dt = time.time() - t0
        if not bin_path.exists():
            err = info.get("set_ExposureExt_err") or info.get("err") \
                or info.get("_stderr_tail", "")[-200:]
            print(f"  iter {i:02d}  exp={exp:>5d}  HELPER FAILED  {err}")
            return 2
        u16 = decode_u16(bin_path)
        st = stats_u16(u16)
        history.append({"iter": i, "exp": exp, "stats": st, "dt": dt,
                         "low_floor": ae.low_floor,
                         "high_brake": ae.high_brake})
        print(f"  iter {i:02d}  exp={exp:>5d}  "
              f"p99={st['p99']:7.1f}  mean={st['mean']:7.1f}  "
              f"max={st['max']:>4d}  clip={st['frac_clip']*100:5.1f}%  "
              f"bracket=[{ae.low_floor},{ae.high_brake}]  ({dt:4.1f}s)")
        if ae.converged(st["p99"], st["frac_clip"]):
            print(f"\n*** CONVERGED at ExposureExt={exp} after "
                  f"{i+1} iterations ***")
            print(f"  final p99={st['p99']:.0f}  mean={st['mean']:.0f}  "
                  f"max={st['max']}  frac_clip={st['frac_clip']*100:.2f}%")
            (OUT_DIR / "summary.json").write_text(
                json.dumps({"converged": True, "exposure_ext": exp,
                            "iterations": i + 1, "history": history,
                            "target_lo": args.target_lo,
                            "target_hi": args.target_hi},
                           indent=2, default=str),
                encoding="utf-8")
            return 0
        new_exp = ae.step(exp, st["p99"], st["mean"], st["frac_clip"])
        # Convergence shortcut: if the bracket has closed and we keep
        # bouncing, accept the low_floor (better dim than saturated).
        if (ae.low_floor is not None and ae.high_brake is not None
                and ae.high_brake - ae.low_floor <= 1
                and new_exp == exp):
            print(f"\n*** BRACKET COLLAPSED, accepting "
                  f"ExposureExt={ae.low_floor} (dim but unsaturated) ***")
            (OUT_DIR / "summary.json").write_text(
                json.dumps({"converged": True,
                            "exposure_ext": ae.low_floor,
                            "iterations": i + 1, "history": history,
                            "target_lo": args.target_lo,
                            "target_hi": args.target_hi,
                            "note": "bracket collapsed"},
                           indent=2, default=str),
                encoding="utf-8")
            return 0
        exp = new_exp

    print(f"\n*** DID NOT CONVERGE in {args.max_iters} iterations ***")
    print(f"  last exp={exp}, history saved to {OUT_DIR}/summary.json")
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"converged": False, "iterations": args.max_iters,
                    "history": history,
                    "target_lo": args.target_lo,
                    "target_hi": args.target_hi},
                   indent=2, default=str),
        encoding="utf-8")
    return 1


if __name__ == "__main__":
    sys.exit(main())
