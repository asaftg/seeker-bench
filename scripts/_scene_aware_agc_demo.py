"""
PROOF TOOL — Scene-aware AGC concept demo (synthetic 16-bit data).

Synthesizes a 16-bit thermal scene that mimics the operator's bench
view: cool sky + warm road + hot trees in the corners + a vehicle on
the road slightly warmer than the pavement.

Renders FOUR ways for comparison, all going to the same WHITE_HOT
grayscale output:

    A) GLOBAL percentile AGC (current)       — what the live seeker
                                                does today on Y16
    B) ROI percentile AGC (bottom 60%)       — scene-aware: stretch
                                                based on the operator's
                                                region of interest;
                                                hot trees outside ROI
                                                saturate to white but
                                                the road + vehicle get
                                                the full 0-255 range
                                                allocated to them
    C) OPERATOR GATES (fixed raw counts)     — linear stretch between
                                                explicit cold/hot gates
                                                in the raw thermal
                                                counts; no histogram
                                                dependence, no scene
                                                "breathing"
    D) ROI + GATES (combined)                — ROI defines the working
                                                zone, gates define the
                                                color mapping inside it

Output: recordings/thermal_compare/scene_aware_agc_demo.png

NB: this is a SYNTHETIC demo on idealized data. The real proof is the
side-by-side capture from the actual sensor (run
`scripts/_y16_vs_agc8_proof.py` for that). This demo shows the
PRINCIPLE so you understand what the live change would do.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

OUT_DIR = os.path.join("recordings", "thermal_compare")


def synth_scene(h: int = 512, w: int = 640, seed: int = 42) -> np.ndarray:
    """Build a synthetic 16-bit thermal scene.

    Layout (top-to-bottom):
      rows 0-180:   sky band — low values 2500-3500 (cool)
      rows 180-300: tree band — base around 4000 + two HOT clusters
                    (sun-warmed canopies) at 18000-22000 in upper L/R
                    corners
      rows 300-410: road band — values 7500-9000 (warm pavement)
      rows 410-512: foreground curb/grass — values 5000-6500
      Vehicle: rectangle ~80x150 in the road band, values 11000-12500
               (warmer than road but cooler than the hot trees)

    All in raw u16 counts. Mimics what a Boson would output on Y16
    looking at a sunny outdoor scene with foliage.
    """
    rng = np.random.default_rng(seed)
    f = np.zeros((h, w), dtype=np.float32)

    # Sky
    f[:180, :] = 3000 + rng.normal(0, 80, (180, w))
    # Trees + canopy
    f[180:300, :] = 4500 + rng.normal(0, 120, (120, w))
    # Hot tree clusters in upper corners (sun-warmed canopy)
    cv2.circle(f, (60, 230), 55, 20000, -1)
    cv2.circle(f, (90, 250), 35, 22000, -1)
    cv2.circle(f, (570, 220), 60, 19500, -1)
    cv2.circle(f, (560, 260), 40, 21000, -1)
    cv2.circle(f, (605, 240), 30, 20500, -1)
    # Soften tree edges
    f = cv2.GaussianBlur(f, (15, 15), 6.0)

    # Road
    road = 8000 + rng.normal(0, 150, (110, w))
    f[300:410, :] = road
    # Foreground curb/grass
    f[410:, :] = 5800 + rng.normal(0, 100, (h - 410, w))

    # Vehicle on the road
    vx0, vy0, vw, vh = 240, 320, 160, 80
    vehicle = 11500 + rng.normal(0, 200, (vh, vw))
    f[vy0:vy0 + vh, vx0:vx0 + vw] = vehicle
    # Engine bay slightly hotter
    f[vy0 + 10:vy0 + 35, vx0 + 110:vx0 + 145] = 13000 + rng.normal(0, 200, (25, 35))
    # Cool windshield (reflecting sky)
    f[vy0 + 5:vy0 + 30, vx0 + 35:vx0 + 100] = 4500 + rng.normal(0, 100, (25, 65))

    f = np.clip(f, 0, 65535).astype(np.uint16)
    return f


def render_global_percentile(u16: np.ndarray,
                             lo_pct: float = 2.0,
                             hi_pct: float = 98.0) -> np.ndarray:
    """A) Global percentile AGC — current behavior on Y16."""
    lo, hi = np.percentile(u16, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(u16.shape, dtype=np.uint8)
    return np.clip((u16.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                   0, 255).astype(np.uint8)


def render_roi_percentile(u16: np.ndarray,
                          roi_top_frac: float = 0.4,
                          lo_pct: float = 2.0,
                          hi_pct: float = 98.0) -> np.ndarray:
    """B) ROI-percentile AGC.

    Compute percentiles over the bottom (1 - roi_top_frac) of the frame
    only, but apply the linear stretch to ALL pixels. Outside-ROI
    pixels may clip to 0 or 255 — that's the point.
    """
    h = u16.shape[0]
    roi_top = int(h * roi_top_frac)
    roi = u16[roi_top:, :]
    lo, hi = np.percentile(roi, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(u16.shape, dtype=np.uint8)
    return np.clip((u16.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                   0, 255).astype(np.uint8)


def render_operator_gates(u16: np.ndarray,
                          cold_gate: int = 6500,
                          hot_gate: int = 13500) -> np.ndarray:
    """C) Operator gates — fixed raw-count thresholds, no histogram."""
    return np.clip((u16.astype(np.float32) - cold_gate)
                   * (255.0 / (hot_gate - cold_gate)),
                   0, 255).astype(np.uint8)


def render_roi_with_gates(u16: np.ndarray,
                          roi_top_frac: float = 0.4,
                          lo_pct: float = 1.0,
                          hi_pct: float = 99.0) -> np.ndarray:
    """D) ROI percentiles AS gates — derive thresholds from ROI, apply
    them globally as fixed gates. This is the practical "auto-tune
    operator gates" mode."""
    h = u16.shape[0]
    roi = u16[int(h * roi_top_frac):, :]
    lo, hi = np.percentile(roi, [lo_pct, hi_pct])
    return np.clip((u16.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                   0, 255).astype(np.uint8)


def _label(width: int, text: str, *, height: int = 28) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def _draw_roi_box(bgr: np.ndarray, roi_top_frac: float = 0.4) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    y = int(h * roi_top_frac)
    cv2.line(out, (0, y), (w - 1, y), (0, 200, 200), 1)
    cv2.putText(out, "ROI ->", (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 200, 200), 1, cv2.LINE_AA)
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "scene_aware_agc_demo.png")

    print("[demo] synthesizing scene...")
    scene = synth_scene()

    print("[demo] rendering 4 modes...")
    a = render_global_percentile(scene)
    b = render_roi_percentile(scene)
    c = render_operator_gates(scene, cold_gate=6500, hot_gate=13500)
    d = render_roi_with_gates(scene)

    # Convert each to BGR for compositing
    bgrs = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in (a, b, c, d)]
    # Annotate ROI line on the ROI-based renderings (B and D)
    bgrs[1] = _draw_roi_box(bgrs[1])
    bgrs[3] = _draw_roi_box(bgrs[3])

    h, w = a.shape
    gap = 8
    # 2x2 grid
    canvas_w = w * 2 + gap
    canvas_h = h * 2 + gap + 28 * 2 + 30  # labels under each row + title

    canvas = np.zeros((canvas_h + 80, canvas_w, 3), dtype=np.uint8)
    # Title bar
    cv2.putText(canvas, "Scene-Aware AGC Demo (synthetic 16-bit thermal scene, WHITE_HOT)",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "vehicle on warm road, hot trees in upper corners, cool sky",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1, cv2.LINE_AA)

    y0 = 70
    # Row 1: A (top-left), B (top-right)
    canvas[y0:y0 + h, :w] = bgrs[0]
    canvas[y0:y0 + h, w + gap:] = bgrs[1]
    canvas[y0 + h:y0 + h + 28, :w] = _label(
        w, "A) GLOBAL percentile 2/98  (CURRENT)")
    canvas[y0 + h:y0 + h + 28, w + gap:] = _label(
        w, "B) ROI-percentile 2/98 over bottom 60%  (proposed default)")
    # Row 2: C, D
    yr2 = y0 + h + 28 + gap
    canvas[yr2:yr2 + h, :w] = bgrs[2]
    canvas[yr2:yr2 + h, w + gap:] = bgrs[3]
    canvas[yr2 + h:yr2 + h + 28, :w] = _label(
        w, "C) OPERATOR GATES  cold=6500 hot=13500 (raw counts, fixed)")
    canvas[yr2 + h:yr2 + h + 28, w + gap:] = _label(
        w, "D) ROI -> auto gates  (ROI 1/99 percentile becomes the gates)")

    cv2.imwrite(out_png, canvas)
    print(f"[demo] wrote {out_png}")

    # Numerical readout — vehicle vs road luma in each mode
    vy0, vx0, vh_, vw_ = 320, 240, 80, 160
    road_y, road_x = 400, 320  # road sample below the vehicle
    print()
    print("Vehicle bbox vs road sample mean luma (higher gap = better contrast):")
    for tag, img in zip("ABCD", (a, b, c, d)):
        veh = img[vy0:vy0 + vh_, vx0:vx0 + vw_].mean()
        rd = img[road_y - 10:road_y + 10, road_x - 30:road_x + 30].mean()
        print(f"  {tag}: vehicle={veh:6.1f}  road={rd:6.1f}  gap={veh - rd:+6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
