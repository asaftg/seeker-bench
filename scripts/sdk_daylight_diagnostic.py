"""Daylight-saturation diagnostic for the IMX568 + 35 mm NIR-pass lens.

Why this script
---------------
Morning of 2026-04-25 the user reported the EO sensor is "completely
white" on a cloudy garage scene, even at our minimum manual exposure
of ExposureExt=50. Last night with an NIR flashlight the image was
clean. Critically: Leopard CameraTool — the *vendor's* official
viewer, which we know reads the sensor through their full SDK pipeline
— is also saturated white on the same scene. So the problem is NOT
something downstream of the sensor (debayer, AGC, classifier); it's
either:

  (A) Sensor analog clip: raw12 hitting 4095 across most of the
      frame, no software fix possible. Need physical ND filter,
      lower analog gain, or shorter integration time than the
      firmware allows.
  (B) Decode/AGC bug in our pipeline (and Leopard's) where raw12 is
      actually in some normal range like [200, 1500] but our hard
      `np.clip(raw12, 0, 255)` decode in leopard_stream_capture.py
      forces everything to white. (This is a real possibility — the
      live-stream decode is calibrated for very dark scenes where
      raw12 max ≈ 255.)

This script produces the data needed to tell A from B and to find
the (ExposureExt, Gain) combo that escapes saturation, if one exists
within the firmware's accepted range.

What it does
------------
1. Probe firmware-accepted ranges for Exposure (UVC IAMCameraControl
   property) and Gain (IAMVideoProcAmp). This gives the hard min/max
   the FX3 will actually accept — which may differ from our UI's 50.

2. Sweep a small grid of (exposure_ext, gain) combinations. For each:
     - Capture one RAW12 frame via the SDK (same call CameraTool uses
       to write its .raw / .bmp output).
     - Decode raw12 (uint16 LE >> 4 → [0, 4095]).
     - Report the full distribution: mean, std, p1/p50/p99, fraction
       of pixels at the analog ceiling (raw12 >= 4090), fraction at
       the floor (raw12 <= 5).
     - Also report how that frame would look through the seeker's
       current decode (clip to 255 then debayer): % of stream pixels
       that would saturate to 255.
     - Save AGC-stretched mono PNG and the seeker's actual stream
       view side by side.

3. Print a verdict block at the end:
     - "SCENARIO A (analog clip)" if any combo has > 50% pixels at
       raw12 >= 4090. Recommend ND filter or firmware mod.
     - "SCENARIO B (our decode is the issue)" if some combo has p99
       well under 4095 BUT seeker's stream-decode would still saturate
       (because raw12 > 255 maps to 255). Recommend updating
       leopard_stream_capture.py to AGC-stretch instead of hard clip.
     - "SCENARIO C (mixed)" if both effects are present at different
       gains.

Run
---
    Close the Seeker GUI and any running CameraTool — the SDK can
    only have one consumer of the FX3 device at a time.

    python scripts/sdk_daylight_diagnostic.py
        [--exposures 50,200,1000,5000]
        [--gains 0,8,16,32]
        [--out scripts/eo_snapshots/diagnostic/daylight_2026_04_25/]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PY32 = REPO / "tools" / "python311-x86" / "python.exe"
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
DEFAULT_OUT = (REPO / "scripts" / "eo_snapshots" / "diagnostic"
               / time.strftime("daylight_%Y%m%d_%H%M%S"))

# IMX568 native RAW12 mode dimensions
W, H = 2472, 2064
RAW12_CEIL = 4090   # pixels at/above this count as analog-clipped
RAW12_FLOOR = 5     # pixels at/below this count as noise-floor


# ───────────────────────── helper invocations ─────────────────────────

def helper_probe_ranges(timeout: float = 60.0) -> dict:
    """Run the SDK helper in --probe-ranges mode to pull firmware
    min/max/step for Exposure and Gain. Returns the parsed JSON dict
    or an error dict on failure."""
    cmd = [str(PY32), str(HELPER), "--probe-ranges", "--json"]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"err": "helper timeout in --probe-ranges"}
    try:
        return json.loads(cp.stdout) if cp.stdout else {
            "err": "no stdout", "stderr": cp.stderr[-2000:]}
    except Exception as e:
        return {"err": f"json parse: {e}",
                "stdout": cp.stdout[-1000:],
                "stderr": cp.stderr[-1000:]}


def helper_capture(exposure_ext: int, gain: int | None, frame_path: Path,
                   warmup: int = 12, timeout: float = 90.0) -> dict:
    """One-shot RAW12 capture at fixed exposure (and optionally gain)
    via the SDK helper. Returns the parsed result dict and writes the
    raw payload to frame_path."""
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
    if gain is not None:
        cmd[2:2] = ["--gain", str(int(gain))]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"err": "helper timeout in --capture-frame"}
    info: dict
    try:
        info = json.loads(cp.stdout) if cp.stdout else {}
    except Exception:
        info = {"raw_stdout": cp.stdout[-1000:]}
    info["_returncode"] = cp.returncode
    if cp.stderr:
        info["_stderr_tail"] = cp.stderr[-1500:]
    return info


# ───────────────────────── decode + stats ─────────────────────────

def decode_raw12(bin_path: Path) -> np.ndarray:
    """Read the helper's RAW12 payload and return a uint16 array of
    raw12 values in [0, 4095]."""
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    n2 = W * H * 2
    if raw.size < n2:
        raise RuntimeError(f"frame too small: {raw.size} < {n2}")
    img16 = np.frombuffer(raw[:n2].tobytes(), dtype="<u2").reshape(H, W)
    return (img16 >> 4).astype(np.uint16)


def raw12_stats(raw12: np.ndarray) -> dict:
    """Full distributional summary, including the two saturation
    fractions that decide A vs B."""
    n = raw12.size
    p = np.percentile(raw12, [1, 50, 99]).tolist()
    sat_top = float((raw12 >= RAW12_CEIL).sum() / n)
    sat_bot = float((raw12 <= RAW12_FLOOR).sum() / n)
    # How would the seeker's current stream decode see this frame?
    # leopard_stream_capture.py does: raw8 = np.clip(raw12, 0, 255).
    # So any raw12 > 255 becomes white. That fraction is the answer
    # to "would the live stream go white even if the sensor isn't
    # actually clipping?"
    stream_white = float((raw12 > 255).sum() / n)
    return {
        "mean": float(raw12.mean()),
        "std": float(raw12.std()),
        "min": int(raw12.min()),
        "max": int(raw12.max()),
        "p1": float(p[0]), "p50": float(p[1]), "p99": float(p[2]),
        "frac_analog_clip": sat_top,    # raw12 >= 4090 → real sensor saturation
        "frac_noise_floor": sat_bot,    # raw12 <= 5    → underexposed
        "frac_stream_white": stream_white,  # raw12 > 255 → seeker's white
    }


def agc_stretch(raw12: np.ndarray, lo_pct=0.5, hi_pct=99.5) -> np.ndarray:
    p_lo, p_hi = np.percentile(raw12, [lo_pct, hi_pct])
    p_hi = max(p_hi, p_lo + 1)
    return np.clip((raw12.astype(np.float32) - p_lo) /
                   (p_hi - p_lo) * 255.0, 0, 255).astype(np.uint8)


def stream_decode(raw12: np.ndarray) -> np.ndarray:
    """The decode the live seeker is using right now (clip-to-255).
    Reproduce it so the diagnostic image shows what the GUI sees."""
    return np.clip(raw12, 0, 255).astype(np.uint8)


def render_panel(raw12: np.ndarray, st: dict, out_path: Path,
                 label: str) -> None:
    """Write a side-by-side: AGC-stretched view (what's actually in
    the data) and stream-clip view (what the seeker's GUI sees right
    now), with stats overlaid."""
    agc = agc_stretch(raw12)
    stream = stream_decode(raw12)
    h2 = 540
    w2 = int(W * h2 / H)
    a_th = cv2.resize(agc, (w2, h2))
    s_th = cv2.resize(stream, (w2, h2))
    sep = np.full((h2, 8), 255, dtype=np.uint8)
    panel = np.hstack([a_th, sep, s_th])
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

    # Header strip
    header_h = 90
    header = np.zeros((header_h, panel_bgr.shape[1], 3), dtype=np.uint8)
    lines = [
        f"{label}",
        f"raw12  mean={st['mean']:7.1f}  std={st['std']:6.1f}  "
        f"max={st['max']:>4d}  p99={st['p99']:7.1f}",
        f"frac analog-clip(>={RAW12_CEIL})={st['frac_analog_clip']*100:5.1f}%   "
        f"stream-white(>255)={st['frac_stream_white']*100:5.1f}%   "
        f"noise-floor(<={RAW12_FLOOR})={st['frac_noise_floor']*100:5.1f}%",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(header, txt, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1,
                    cv2.LINE_AA)
    # Column titles on the panel
    cv2.putText(panel_bgr, "AGC stretch (true data)", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(panel_bgr, "Seeker stream decode (clip 0..255)",
                (w2 + 18, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    full = np.vstack([header, panel_bgr])
    cv2.imwrite(str(out_path), full)


# ───────────────────────── main ─────────────────────────

def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposures", default="50,200,1000,5000",
                    help="Comma-separated ExposureExt values to try")
    ap.add_argument("--gains", default="none,0,4,16",
                    help="Comma-separated Gain values; use 'none' to "
                         "leave gain untouched (firmware default)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output directory for snapshots + report.json")
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--skip-ranges", action="store_true",
                    help="Skip the --probe-ranges call (faster reruns)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diag] output dir: {out_dir}")

    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exposures": args.exposures,
        "gains": args.gains,
        "frames": [],
    }

    # ── Step 1: probe firmware-accepted ranges ──
    if not args.skip_ranges:
        print("[diag] probing firmware ranges (Exposure / Gain)...")
        ranges = helper_probe_ranges()
        report["ranges"] = ranges
        # Pretty print the few keys we care most about
        if "ranges" in ranges:
            r = ranges["ranges"]
            for key in ("camctrl.Exposure", "vproc.Gain",
                        "camctrl.Exposure_err", "vproc.Gain_err"):
                if key in r:
                    print(f"   {key}: {r[key]}")
        else:
            print(f"   (no ranges payload — got: {list(ranges)[:6]})")

    # ── Step 2: sweep (exposure, gain) ──
    exposures = parse_int_list(args.exposures)
    gain_tokens = [t.strip() for t in args.gains.split(",") if t.strip()]
    gains: list[int | None] = [
        None if t.lower() == "none" else int(t) for t in gain_tokens
    ]

    print(f"[diag] sweeping {len(exposures)} exposures × {len(gains)} gains "
          f"= {len(exposures) * len(gains)} captures")
    any_clip = False
    any_stream_white_no_clip = False

    for exp in exposures:
        for g in gains:
            tag_g = "auto" if g is None else f"g{g}"
            tag = f"e{exp}_{tag_g}"
            bin_path = out_dir / f"frame_{tag}.bin"
            png_path = out_dir / f"frame_{tag}.png"
            t0 = time.time()
            info = helper_capture(exp, g, bin_path, warmup=args.warmup)
            dt = time.time() - t0
            if not bin_path.exists():
                err = info.get("set_ExposureExt_err") or info.get("err") \
                    or info.get("_stderr_tail", "")[-300:]
                print(f"  {tag:>14s}  HELPER FAILED ({dt:4.1f}s)  {err}")
                report["frames"].append({"tag": tag, "exp": exp, "gain": g,
                                          "err": str(err)[:400]})
                continue
            try:
                raw12 = decode_raw12(bin_path)
                st = raw12_stats(raw12)
                render_panel(raw12, st, png_path, label=tag)
            except Exception as e:
                print(f"  {tag:>14s}  DECODE FAILED: {e}")
                report["frames"].append({"tag": tag, "exp": exp, "gain": g,
                                          "decode_err": str(e)})
                continue
            report["frames"].append({
                "tag": tag, "exp": exp, "gain": g,
                "stats": st, "png": str(png_path),
            })
            print(f"  {tag:>14s}  mean={st['mean']:7.1f}  "
                  f"max={st['max']:>4d}  "
                  f"clip={st['frac_analog_clip']*100:5.1f}%  "
                  f"streamwhite={st['frac_stream_white']*100:5.1f}%  "
                  f"({dt:4.1f}s)")
            if st["frac_analog_clip"] > 0.5:
                any_clip = True
            if (st["frac_analog_clip"] < 0.05
                    and st["frac_stream_white"] > 0.5):
                any_stream_white_no_clip = True

    # ── Step 3: verdict ──
    if any_clip and any_stream_white_no_clip:
        verdict = ("SCENARIO C (mixed): some combos hit analog-clip, others "
                   "don't but our stream decode whites them out anyway. "
                   "Need BOTH lower exposure/gain (or ND filter) AND a "
                   "smarter stream decode in leopard_stream_capture.py.")
    elif any_clip:
        verdict = ("SCENARIO A (analog clip): >50% of pixels are at the "
                   "sensor's analog ceiling (raw12 >= 4090). No software "
                   "fix on our side. Need lower analog gain (try Gain=0), "
                   "shorter exposure than firmware permits, or a physical "
                   "ND filter on the 35mm NIR-pass lens. The NIR-pass "
                   "filter passes a huge slice of the daylight spectrum "
                   "and IMX568 has high NIR QE.")
    elif any_stream_white_no_clip:
        verdict = ("SCENARIO B (our decode bug): the sensor is NOT analog-"
                   "clipped, but raw12 values >255 are being hard-clipped "
                   "to 255 in leopard_stream_capture.py:grab(). The "
                   "current `np.clip(raw12, 0, 255)` decode is calibrated "
                   "for very dark scenes (raw12 max ~255). Replace with "
                   "an AGC-percentile stretch (or right-shift 4 with "
                   "downstream re-AGC) so daylight scenes survive.")
    else:
        verdict = ("INCONCLUSIVE: no combo saturated >50% pixels and no "
                   "combo had >50% stream-white without analog clip. "
                   "Either the scene changed during capture, exposures "
                   "swept too low/high, or saturation is intermittent.")

    report["verdict"] = verdict
    print()
    print("VERDICT")
    print("-------")
    print(verdict)
    print()
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")
    print(f"[diag] wrote report → {report_path}")
    print(f"[diag] PNGs and raw .bin frames are alongside it in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
