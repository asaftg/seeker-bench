"""
PROOF TOOL — 4-up render of AGC8 baseline vs Y16-global / Y16-roi /
Y16-gates on the same scene.

Reads a .npz from `_y16_vs_agc8_proof.py`, renders each Y16 frame
through three different AGC modes from `thermal.thermal_processor`,
and writes a 2x2 PNG + a 4-up MP4 to `recordings/thermal_compare/`.

The four panels (all WHITE_HOT colormap):
    A) AGC8 fallback (8-bit, camera AGC)        — current live baseline
    B) Y16 + global percentile  2/98             — what live seeker would
                                                   show after Patch 2A alone
    C) Y16 + ROI percentile (bottom 60%)         — proposed mode=roi default
    D) Y16 + operator gates (cold/hot in counts) — proposed mode=gates

Usage:
    python scripts/_phase23_render_modes.py
    python scripts/_phase23_render_modes.py --npz <path>
    python scripts/_phase23_render_modes.py --cold 19500 --hot 22500
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Tuple

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from thermal.thermal_processor import (  # noqa: E402
    apply_agc, apply_clahe_y16, apply_colormap,
    apply_gates_agc, apply_roi_agc,
)

OUT_DIR = os.path.join("recordings", "thermal_compare")
FULL_HFOV = 75.0


def _crop_zoom(bgr: np.ndarray, hfov: float, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Center-crop an image to emulate a narrower HFOV, upscale to original size."""
    if hfov >= FULL_HFOV:
        return bgr
    h, w = bgr.shape[:2]
    frac = float(np.tan(np.radians(hfov / 2)) / np.tan(np.radians(FULL_HFOV / 2)))
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    cx0 = (w - cw) // 2
    cy0 = (h - ch) // 2
    return cv2.resize(bgr[cy0:cy0 + ch, cx0:cx0 + cw], (w, h), interpolation=interp)


def _crop_zoom_u16(u16: np.ndarray, hfov: float) -> np.ndarray:
    if hfov >= FULL_HFOV:
        return u16
    h, w = u16.shape
    frac = float(np.tan(np.radians(hfov / 2)) / np.tan(np.radians(FULL_HFOV / 2)))
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    cx0 = (w - cw) // 2
    cy0 = (h - ch) // 2
    return cv2.resize(u16[cy0:cy0 + ch, cx0:cx0 + cw], (w, h),
                      interpolation=cv2.INTER_LINEAR)


def _newest_npz() -> str:
    pattern = os.path.join(OUT_DIR, "y16_vs_agc8_*.npz")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(
            f"no .npz found in {OUT_DIR} — run "
            f"scripts/_y16_vs_agc8_proof.py first"
        )
    return matches[-1]


def _label(width: int, text: str, *, height: int = 28) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def _laplacian_var(gray_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def _agc8_to_whitehot(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _y16_global(u16: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = apply_agc(u16, 2.0, 98.0)
    return g, apply_colormap(g, "WHITE_HOT")


def _y16_roi(u16: np.ndarray, top_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    g = apply_roi_agc(u16, top_frac, 2.0, 98.0)
    return g, apply_colormap(g, "WHITE_HOT")


def _y16_gates(u16: np.ndarray, cold: int, hot: int) -> Tuple[np.ndarray, np.ndarray]:
    g = apply_gates_agc(u16, cold, hot)
    return g, apply_colormap(g, "WHITE_HOT")


def _y16_clahe(u16: np.ndarray, clip_limit: float, tile_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    g = apply_clahe_y16(u16, clip_limit, tile_grid)
    return g, apply_colormap(g, "WHITE_HOT")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=str, default=None,
                    help="Path to .npz (default: newest in recordings/thermal_compare/)")
    ap.add_argument("--roi-top-frac", type=float, default=0.4,
                    help="ROI mode percentile band (bottom 1-top_frac)")
    ap.add_argument("--cold", type=int, default=None,
                    help="Gates mode cold_count. Default: auto from p10 of Y16 stack")
    ap.add_argument("--hot", type=int, default=None,
                    help="Gates mode hot_count. Default: auto from p95 of Y16 stack")
    ap.add_argument("--tag", type=str, default="modes",
                    help="Filename suffix")
    ap.add_argument("--hfov", type=float, default=75.0,
                    help="Software zoom HFOV applied to all 4 panels "
                         "(default 75 = full sensor; set to 37.5/18.75/12.5 "
                         "for wide/mid/narrow preset)")
    ap.add_argument("--clahe-clip", type=float, default=2.0,
                    help="CLAHE clip limit for the Y16-CLAHE panel")
    ap.add_argument("--clahe-tile", type=int, default=8,
                    help="CLAHE tile grid for the Y16-CLAHE panel")
    ap.add_argument("--panels", type=str, default="agc8,global,roi,clahe",
                    choices=["agc8,global,roi,gates", "agc8,global,roi,clahe",
                            "agc8,roi,gates,clahe", "agc8,global,gates,clahe"],
                    help="Which 4 modes to show (comma-sep)")
    args = ap.parse_args(argv)

    npz_path = args.npz or _newest_npz()
    print(f"[render] loading {npz_path}")
    data = np.load(npz_path)
    y16 = data["y16"]
    agc8 = data["agc8"]
    n = min(y16.shape[0], agc8.shape[0])
    h, w = 512, 640

    # Auto-tune gates if not explicit, from this capture's stats.
    flat = y16.reshape(-1)
    p2, p10, p50, p95, p98 = np.percentile(flat, [2, 10, 50, 95, 98])
    cold = int(args.cold if args.cold is not None else p10)
    hot = int(args.hot if args.hot is not None else p95)
    print(f"[render] Y16 stack stats: p2={int(p2)}, p10={int(p10)}, "
          f"p50={int(p50)}, p95={int(p95)}, p98={int(p98)}")
    print(f"[render] gates mode: cold={cold}, hot={hot}")
    print(f"[render] roi mode: top_frac={args.roi_top_frac}")

    # 2x2 canvas + label rows + footer metrics
    gap = 6
    cell_h = h
    cell_w = w
    top_label_h = 28
    bot_label_h = 30
    canvas_w = cell_w * 2 + gap
    canvas_h = (cell_h + top_label_h) * 2 + gap + bot_label_h

    fov_suffix = f"_hfov{int(round(args.hfov))}" if args.hfov < FULL_HFOV else "_hfov75"
    out_png = os.path.join(OUT_DIR, f"phase23_{args.tag}{fov_suffix}.png")
    out_mp4 = os.path.join(OUT_DIR, f"phase23_{args.tag}{fov_suffix}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, 20.0, (canvas_w, canvas_h))

    panels = args.panels.split(",")
    panel_renderers = {
        "agc8": (lambda u16, bgr: _agc8_to_whitehot(bgr),
                 "AGC8 fallback (current baseline)"),
        "global": (lambda u16, bgr: _y16_global(u16),
                   "Y16 + global percentile 2/98"),
        "roi": (lambda u16, bgr: _y16_roi(u16, args.roi_top_frac),
                f"Y16 + ROI bottom {int((1 - args.roi_top_frac)*100)}%"),
        "gates": (lambda u16, bgr: _y16_gates(u16, cold, hot),
                  f"Y16 + gates cold={cold} hot={hot}"),
        "clahe": (lambda u16, bgr: _y16_clahe(u16, args.clahe_clip, args.clahe_tile),
                  f"Y16 + CLAHE-on-raw clip={args.clahe_clip} tile={args.clahe_tile}"),
    }
    panel_keys = ["A", "B", "C", "D"]
    sums = {k: 0.0 for k in panel_keys}
    for i in range(n):
        renders = []
        for key in panels:
            fn, _label_text = panel_renderers[key]
            g, b = fn(y16[i], agc8[i])
            renders.append((g, b))
        a_g, a_b = renders[0]
        b_g, b_b = renders[1]
        c_g, c_b = renders[2]
        d_g, d_b = renders[3]

        # Apply FOV crop to each panel (after AGC) so all panels show
        # the same zoom level. AGC itself is computed on the FULL frame
        # (matches the live seeker which AGCs full then crops).
        if args.hfov < FULL_HFOV:
            a_b = _crop_zoom(a_b, args.hfov)
            b_b = _crop_zoom(b_b, args.hfov)
            c_b = _crop_zoom(c_b, args.hfov)
            d_b = _crop_zoom(d_b, args.hfov)
            a_g = _crop_zoom(a_g, args.hfov)
            b_g = _crop_zoom(b_g, args.hfov)
            c_g = _crop_zoom(c_g, args.hfov)
            d_g = _crop_zoom(d_g, args.hfov)

        sums["A"] += _laplacian_var(a_g)
        sums["B"] += _laplacian_var(b_g)
        sums["C"] += _laplacian_var(c_g)
        sums["D"] += _laplacian_var(d_g)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        labels_text = [panel_renderers[k][1] for k in panels]
        # Row 1: A (top-left), B (top-right) with labels above each
        canvas[:top_label_h, :cell_w] = _label(cell_w, f"A) {labels_text[0]}")
        canvas[:top_label_h, cell_w + gap:] = _label(cell_w, f"B) {labels_text[1]}")
        canvas[top_label_h:top_label_h + cell_h, :cell_w] = a_b
        canvas[top_label_h:top_label_h + cell_h, cell_w + gap:] = b_b

        # Row 2: C, D
        y2 = top_label_h + cell_h + gap
        canvas[y2:y2 + top_label_h, :cell_w] = _label(cell_w, f"C) {labels_text[2]}")
        canvas[y2:y2 + top_label_h, cell_w + gap:] = _label(cell_w, f"D) {labels_text[3]}")
        canvas[y2 + top_label_h:y2 + top_label_h + cell_h, :cell_w] = c_b
        canvas[y2 + top_label_h:y2 + top_label_h + cell_h, cell_w + gap:] = d_b

        # Footer: metrics
        ya = y2 + top_label_h + cell_h
        canvas[ya:ya + bot_label_h, :] = _label(
            canvas_w,
            f"sharpness  A {sums['A']/(i+1):5.1f}   B {sums['B']/(i+1):5.1f}   "
            f"C {sums['C']/(i+1):5.1f}   D {sums['D']/(i+1):5.1f}    frame {i+1}/{n}",
            height=bot_label_h,
        )

        writer.write(canvas)
        if i == n // 2:
            cv2.imwrite(out_png, canvas)

    writer.release()
    print(f"[render] still: {out_png}")
    print(f"[render] video: {out_mp4}")
    print(f"[render] mean Laplacian sharpness:")
    for key, panel_name in zip(panel_keys, panels):
        label = panel_renderers[panel_name][1]
        print(f"  {key}) {label:50s}: {sums[key]/n:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
