"""End-to-end driver: spawn the 32-bit Leopard SDK helper, capture
a frame, decode RAW12 → mono+debayer → AGC, and compare to the
leopard reference image.

Loops over (exposure_ext × warmup_count) until our capture's stats
approach the reference.

Reference (leopard_reference.raw):
    raw12 mean=32.7  std=19.2  range=[15..255]
    BMP B/G/R means: 22.4 / 38.8 / 31.1

Stops successfully when:
    abs(mean_diff) < 5   AND   our_std > 8   AND   our_max > 60
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
# Force UTF-8 stdout/stderr to avoid Windows cp1252 issues
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PY32 = REPO / "tools" / "python311-x86" / "python.exe"
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
OUT_DIR = REPO / "scripts" / "eo_snapshots" / "calibration" / "sdk_loop"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REF_RAW = REPO / "scripts" / "eo_snapshots" / "calibration" / "leopard_reference.raw"
REF_BMP = REPO / "scripts" / "eo_snapshots" / "calibration" / "leopard_reference.bmp"

W, H = 2472, 2064


def load_ref_raw() -> np.ndarray:
    data = np.fromfile(str(REF_RAW), dtype="<u2")
    img16 = data.reshape(H, W)
    raw12 = (img16 >> 4).astype(np.uint16)
    return raw12


def load_ref_bmp() -> np.ndarray:
    return cv2.imread(str(REF_BMP))  # BGR


def call_helper(exposure_ext: int, warmup: int, frame_path: Path,
                data_mode: str = "RAW12") -> dict:
    cmd = [
        str(PY32), str(HELPER),
        "--ae", "off",
        "--exposure-ext", str(int(exposure_ext)),
        "--data-mode", data_mode,
        "--capture-width", str(W),
        "--capture-height", str(H),
        "--capture-warmup", str(int(warmup)),
        "--capture-frame", str(frame_path),
        "--json",
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    info = {}
    try:
        info = json.loads(cp.stdout) if cp.stdout else {}
    except Exception:
        info = {"raw_stdout": cp.stdout, "raw_stderr": cp.stderr}
    info["_returncode"] = cp.returncode
    return info


def decode_raw12(bin_path: Path) -> np.ndarray:
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    n2 = W * H * 2
    if raw.size < n2:
        raise RuntimeError(f"frame too small: {raw.size} < {n2}")
    img16 = np.frombuffer(raw[:n2].tobytes(), dtype="<u2").reshape(H, W)
    raw12 = (img16 >> 4).astype(np.uint16)
    return raw12


def agc(raw12: np.ndarray, lo_pct=0.5, hi_pct=99.5) -> np.ndarray:
    p_lo, p_hi = np.percentile(raw12, [lo_pct, hi_pct])
    p_hi = max(p_hi, p_lo + 1)
    return np.clip((raw12.astype(np.float32) - p_lo) /
                   (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)


def debayer_best_match(raw12: np.ndarray, ref_bgr: np.ndarray) -> tuple:
    """Try all 4 bayer patterns, pick the one whose B/G/R ratio is
    closest to the reference BMP's. Returns (bgr_image, pattern_name,
    score)."""
    # First make an 8-bit AGC version (debayer needs uint8 or uint16)
    raw8 = agc(raw12)
    ref_means = np.array([ref_bgr[..., c].mean() for c in range(3)])
    ref_ratio = ref_means / max(ref_means.sum(), 1)
    best = None
    for code, name in [
        (cv2.COLOR_BAYER_BG2BGR, "BG"),
        (cv2.COLOR_BAYER_GB2BGR, "GB"),
        (cv2.COLOR_BAYER_RG2BGR, "RG"),
        (cv2.COLOR_BAYER_GR2BGR, "GR"),
    ]:
        bgr = cv2.cvtColor(raw8, code)
        m = np.array([bgr[..., c].mean() for c in range(3)])
        rat = m / max(m.sum(), 1)
        score = float(np.abs(rat - ref_ratio).sum())
        if best is None or score < best[2]:
            best = (bgr, name, score)
    return best


def stats(raw12: np.ndarray) -> dict:
    return {
        "mean": float(raw12.mean()),
        "std": float(raw12.std()),
        "min": int(raw12.min()),
        "max": int(raw12.max()),
        "p1": float(np.percentile(raw12, 1)),
        "p99": float(np.percentile(raw12, 99)),
    }


def render_comparison(ours_raw12: np.ndarray, ref_raw12: np.ndarray,
                      out_path: Path) -> None:
    """Side-by-side AGC versions of ours and the reference."""
    ours8 = agc(ours_raw12)
    ref8 = agc(ref_raw12)
    # Resize each to a manageable thumb
    h2 = 768
    w2 = int(W * h2 / H)
    ours_th = cv2.resize(ours8, (w2, h2))
    ref_th = cv2.resize(ref8, (w2, h2))
    # Stack
    sep = np.full((h2, 6), 255, dtype=np.uint8)
    panel = np.hstack([ours_th, sep, ref_th])
    # Add labels
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    cv2.putText(panel_bgr, "OURS", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(panel_bgr, "LEOPARD REF", (w2 + 18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), panel_bgr)


def render_color_comparison(ours_raw12: np.ndarray, ref_bgr: np.ndarray,
                            out_path: Path) -> tuple:
    """Debayer ours, side-by-side vs reference BMP. Returns chosen
    pattern + per-channel means."""
    bgr_ours, pat, score = debayer_best_match(ours_raw12, ref_bgr)
    h2 = 768
    w2 = int(W * h2 / H)
    o_th = cv2.resize(bgr_ours, (w2, h2))
    r_th = cv2.resize(ref_bgr, (w2, h2))
    sep = np.full((h2, 6, 3), (255, 255, 255), dtype=np.uint8)
    panel = np.hstack([o_th, sep, r_th])
    cv2.putText(panel, f"OURS color (Bayer {pat})", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(panel, "LEOPARD BMP", (w2 + 18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), panel)
    return pat, score, bgr_ours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-attempts", type=int, default=15)
    ap.add_argument("--start-exposure", type=int, default=1264)
    ap.add_argument("--start-warmup", type=int, default=8)
    args = ap.parse_args()

    ref_raw12 = load_ref_raw()
    ref_bgr = load_ref_bmp()
    ref_stats = stats(ref_raw12)
    print(f"REFERENCE raw12: {ref_stats}")
    print(f"REFERENCE BMP   B/G/R means: "
          f"{ref_bgr[...,0].mean():.1f} {ref_bgr[...,1].mean():.1f} "
          f"{ref_bgr[...,2].mean():.1f}")

    # Try a sweep of (exposure_ext, warmup)
    schedule = [
        (1264, 12),  # default with longer warmup
        (1264, 25),
        (2000, 12),
        (3000, 15),
        (5000, 15),
        (1500, 30),
        (1000, 40),
        (8000, 15),
        (12000, 15),
        (1264, 60),  # really long warmup
    ]
    best = None
    for attempt, (exp, wu) in enumerate(schedule[:args.max_attempts]):
        t0 = time.time()
        frame_path = OUT_DIR / f"frame_e{exp}_w{wu}.bin"
        info = call_helper(exp, wu, frame_path)
        elapsed = time.time() - t0
        if not frame_path.exists():
            print(f"[#{attempt+1}] exp={exp} wu={wu}: helper FAILED "
                  f"({elapsed:.1f}s) info={json.dumps(info)[:300]}")
            continue
        raw12 = decode_raw12(frame_path)
        st = stats(raw12)
        diff_mean = abs(st["mean"] - ref_stats["mean"])
        diff_std = abs(st["std"] - ref_stats["std"])
        score = diff_mean + diff_std
        ok = (diff_mean < 5.0 and st["std"] > 8 and st["max"] > 60)
        print(f"[#{attempt+1}] exp={exp:>5d} wu={wu:>3d} "
              f"mean={st['mean']:6.2f} std={st['std']:6.2f} "
              f"min={st['min']} max={st['max']} "
              f"dmean={diff_mean:5.2f} dstd={diff_std:5.2f} "
              f"score={score:6.2f}{'  *** MATCH' if ok else ''} "
              f"({elapsed:.1f}s)")
        if best is None or score < best["score"]:
            # Save AGC + comparison
            best = {
                "exp": exp,
                "wu": wu,
                "stats": st,
                "score": score,
                "frame": frame_path,
            }
            cv2.imwrite(str(OUT_DIR / "best_mono_agc.bmp"), agc(raw12))
            render_comparison(raw12, ref_raw12,
                              OUT_DIR / "best_compare_mono.png")
            pat, cs, bgr_ours = render_color_comparison(
                raw12, ref_bgr, OUT_DIR / "best_compare_color.png")
            cv2.imwrite(str(OUT_DIR / "best_color.bmp"), bgr_ours)
            print(f"   → new best (color pattern={pat}, color_score={cs:.4f})")
        if ok:
            print("\n*** MATCH FOUND ***")
            print(f"  exp_ext={exp}  warmup={wu}")
            print(f"  ours: {st}")
            print(f"  ref:  {ref_stats}")
            return 0

    print("\nNo full match. Best:")
    print(json.dumps(best, indent=2, default=str))
    return 1


if __name__ == "__main__":
    sys.exit(main())
