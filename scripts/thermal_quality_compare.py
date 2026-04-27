"""
Thermal quality A/B comparison over a recorded JSONL session.

Decodes every ``thermal/frame`` JPEG from a recording, re-applies the
post-AGC enhancement chain configured in ``config/app_config.yaml``,
and writes a side-by-side BEFORE | AFTER MP4 the operator can scrub
through to verify the new image-quality settings are an improvement
on real footage. Also computes per-frame metrics (sharpness, local
contrast) and prints a one-line summary.

Limitations
-----------

JSONL recordings only archive the *post-AGC* 8-bit thermal JPEG
(``recording/encoders.py:encode_thermal``). The raw 16-bit frame is
not stored. So this tool can only validate enhancements that operate
on the 8-bit display image — gamma, bilateral denoise, unsharp mask,
CLAHE, and colormap re-selection. It cannot validate AGC percentile
re-tuning or dead-pixel median (those run pre-AGC, on raw16). For
those, ``python -m thermal.fake_thermal_source`` exercises the full
chain on synthetic raw16, and a follow-up morning step with the live
camera is the only way to evaluate them on real footage.

For colormap-pumped recordings (INFERNO, JET, MAGMA), recovering the
underlying grayscale by ``cvtColor(BGR, BGR2GRAY)`` is mildly lossy
because the colormap is many-to-one near saturation. For ``WHITE_HOT``
the recovery is exact (R == G == B). The output is still a perfectly
valid A/B comparison — it just isn't bit-identical to what the live
pipeline would have produced.

CLI
---

    python scripts/thermal_quality_compare.py --latest
    python scripts/thermal_quality_compare.py --path <jsonl> --limit 600
    python scripts/thermal_quality_compare.py --latest --layout stack
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import sys
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

# Add the repo root to sys.path so this script runs from anywhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from thermal.thermal_processor import (  # noqa: E402  — sys.path is just set
    ThermalEnhanceParams,
    apply_colormap,
    enhance_post_agc,
    from_config as enhance_from_config,
)


DEFAULT_DIR = "recordings"
OUT_SUBDIR = "thermal_compare"


# ──────────────────────────────────────────────────────────────────
# JSONL helpers (same shape as scripts/replay_inspect.py)
# ──────────────────────────────────────────────────────────────────
def find_latest(directory: str = DEFAULT_DIR) -> Optional[str]:
    pattern = os.path.join(directory, "seeker_*.jsonl")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    with io.open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ──────────────────────────────────────────────────────────────────
# Image metrics (laplacian-variance sharpness, std contrast)
# ──────────────────────────────────────────────────────────────────
def sharpness_laplacian_var(gray_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def local_contrast_std(gray_u8: np.ndarray) -> float:
    return float(gray_u8.std())


# ──────────────────────────────────────────────────────────────────
# Frame composition
# ──────────────────────────────────────────────────────────────────
def _label_strip(width: int, text: str, *, height: int = 28) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        strip, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return strip


def compose_side_by_side(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    *,
    metrics_text: str,
) -> np.ndarray:
    h, w = before_bgr.shape[:2]
    gap = 8
    canvas = np.zeros((h + 28 + 28, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:h, :w] = before_bgr
    canvas[:h, w + gap:w * 2 + gap] = after_bgr
    canvas[h:h + 28, :w] = _label_strip(w, "BEFORE  (recorded q=92)")
    canvas[h:h + 28, w + gap:w * 2 + gap] = _label_strip(
        w, "AFTER   (post-AGC enhance)"
    )
    canvas[h + 28:h + 56, :] = _label_strip(canvas.shape[1], metrics_text)
    return canvas


def compose_stacked(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    *,
    metrics_text: str,
) -> np.ndarray:
    h, w = before_bgr.shape[:2]
    gap = 4
    canvas = np.zeros((h * 2 + gap + 28 + 28, w, 3), dtype=np.uint8)
    canvas[:h] = before_bgr
    canvas[h + gap:2 * h + gap] = after_bgr
    canvas[2 * h + gap:2 * h + gap + 28] = _label_strip(w, "BEFORE / AFTER")
    canvas[2 * h + gap + 28:] = _label_strip(w, metrics_text)
    return canvas


# ──────────────────────────────────────────────────────────────────
# Pipeline reconstruction from the recorded session config
# ──────────────────────────────────────────────────────────────────
def params_for_compare(header: Dict[str, Any]) -> Tuple[ThermalEnhanceParams, str]:
    """Build params and the original colormap from the recording header.

    The recorded image was produced under whatever ``config_snapshot``
    is in the header. We rebuild ``ThermalEnhanceParams`` from the
    *current* live config (because that's the new state we're trying
    to validate); the original colormap from the header is used to
    re-colormap the AFTER frame when the user hasn't explicitly
    overridden it.
    """
    snap = (header.get("config_snapshot") or {}).get("thermal") or {}
    orig_colormap = (snap.get("agc") or {}).get("colormap") or "INFERNO"

    try:
        from common.config import load_config
        live_cfg = load_config().get("thermal", {})
    except Exception:
        live_cfg = {}
    live_params = enhance_from_config(live_cfg)
    return live_params, str(orig_colormap)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--latest", action="store_true", help="Use newest recordings/seeker_*.jsonl")
    src.add_argument("--path", type=str, help="Explicit JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N thermal frames (0 = all)")
    ap.add_argument("--out", type=str, default=None, help="Output MP4 path (default: recordings/thermal_compare/<basename>_compare.mp4)")
    ap.add_argument("--layout", choices=("side", "stack"), default="side")
    ap.add_argument("--colormap", type=str, default=None, help="Override AFTER colormap (default: same as recording)")
    ap.add_argument("--fps", type=float, default=0.0, help="Output MP4 fps (default: derive from recording timestamps)")
    args = ap.parse_args(argv)

    src_path = args.path or find_latest()
    if not src_path or not os.path.exists(src_path):
        print(f"[compare] no recording found at {src_path}", file=sys.stderr)
        return 2
    print(f"[compare] source: {src_path}")

    header = {}
    for rec in iter_records(src_path):
        if rec.get("channel") == "session/header":
            header = rec.get("msg") or {}
            break

    params, orig_colormap = params_for_compare(header)
    after_colormap = (args.colormap or orig_colormap).upper()
    print(f"[compare] live thermal enhancement params: {params}")
    print(f"[compare] colormap: BEFORE={orig_colormap} / AFTER={after_colormap}")

    # Discover frame size + estimate fps from the first ~50 thermal frames.
    first_frame_bgr: Optional[np.ndarray] = None
    timestamps = []
    sample_count = 0
    for rec in iter_records(src_path):
        if rec.get("channel") != "thermal/frame":
            continue
        m = rec.get("msg") or {}
        if not m.get("connected") or not m.get("jpeg_b64"):
            continue
        if first_frame_bgr is None:
            buf = base64.b64decode(m["jpeg_b64"])
            first_frame_bgr = cv2.imdecode(
                np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        timestamps.append(float(m.get("timestamp", 0.0)))
        sample_count += 1
        if sample_count >= 50:
            break

    if first_frame_bgr is None:
        print("[compare] no thermal frames in recording", file=sys.stderr)
        return 2

    h, w = first_frame_bgr.shape[:2]
    if args.fps > 0:
        fps_est = float(args.fps)
    elif len(timestamps) >= 2:
        dt = (timestamps[-1] - timestamps[0]) / max(1, len(timestamps) - 1)
        fps_est = max(1.0, min(60.0, 1.0 / max(1e-3, dt)))
    else:
        fps_est = 30.0
    print(f"[compare] frame size: {w}x{h}  estimated_fps: {fps_est:.1f}")

    # Build a probe canvas to learn the output dimensions.
    probe_after = first_frame_bgr.copy()
    if args.layout == "side":
        probe = compose_side_by_side(first_frame_bgr, probe_after, metrics_text="")
    else:
        probe = compose_stacked(first_frame_bgr, probe_after, metrics_text="")
    out_h, out_w = probe.shape[:2]

    # Output path
    out_path = args.out
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(src_path) or ".", OUT_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(out_dir, f"{base}_compare.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_est, (out_w, out_h))
    if not writer.isOpened():
        print(f"[compare] could not open writer at {out_path}", file=sys.stderr)
        return 3
    print(f"[compare] writing -> {out_path}")

    n_thermal = 0
    n_emitted = 0
    sum_sharp_before = 0.0
    sum_sharp_after = 0.0
    sum_std_before = 0.0
    sum_std_after = 0.0

    for rec in iter_records(src_path):
        if rec.get("channel") != "thermal/frame":
            continue
        m = rec.get("msg") or {}
        if not m.get("connected") or not m.get("jpeg_b64"):
            continue
        n_thermal += 1
        if args.limit and n_emitted >= args.limit:
            break

        try:
            buf = base64.b64decode(m["jpeg_b64"])
            before_bgr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
        if before_bgr is None or before_bgr.shape[0] == 0:
            continue
        if before_bgr.shape[:2] != (h, w):
            # Zoom changes mid-recording — skip frames whose size differs
            # to avoid resizing artefacts polluting the metric.
            continue

        # Recover an approximate AGC8 grayscale by averaging BGR.
        gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)

        # Apply the post-AGC enhancement chain configured today.
        try:
            enhanced = enhance_post_agc(gray, params)
        except Exception as e:
            print(f"[compare] enhance failed at frame {n_thermal}: {e}", file=sys.stderr)
            continue

        after_bgr = apply_colormap(enhanced, after_colormap)

        # Metrics on the grayscale equivalents.
        sb = sharpness_laplacian_var(gray)
        sa = sharpness_laplacian_var(enhanced)
        cb = local_contrast_std(gray)
        ca = local_contrast_std(enhanced)
        sum_sharp_before += sb
        sum_sharp_after += sa
        sum_std_before += cb
        sum_std_after += ca

        metrics_text = (
            f"sharp d {sa - sb:+7.1f}  ({sb:6.1f} -> {sa:6.1f})    "
            f"contrast d {ca - cb:+5.2f}  ({cb:5.2f} -> {ca:5.2f})    "
            f"frame {n_thermal}"
        )

        if args.layout == "side":
            canvas = compose_side_by_side(before_bgr, after_bgr, metrics_text=metrics_text)
        else:
            canvas = compose_stacked(before_bgr, after_bgr, metrics_text=metrics_text)
        writer.write(canvas)
        n_emitted += 1

    writer.release()

    if n_emitted == 0:
        print("[compare] WARNING: zero frames written", file=sys.stderr)
        return 4

    print(f"[compare] frames emitted: {n_emitted} (of {n_thermal} thermal records)")
    print(
        "[compare] mean sharpness — "
        f"before {sum_sharp_before / n_emitted:.1f}, "
        f"after {sum_sharp_after / n_emitted:.1f} "
        f"(d {(sum_sharp_after - sum_sharp_before) / n_emitted:+.1f})"
    )
    print(
        "[compare] mean contrast (std) — "
        f"before {sum_std_before / n_emitted:.2f}, "
        f"after {sum_std_after / n_emitted:.2f} "
        f"(d {(sum_std_after - sum_std_before) / n_emitted:+.2f})"
    )
    print(f"[compare] output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
