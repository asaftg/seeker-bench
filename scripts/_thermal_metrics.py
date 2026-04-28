"""
Thermal frame quality metrics for the parameter-sweep harness.

Pure functions over numpy arrays. No seeker imports — these can run
on any captured frame stack offline.

For each frame stack we compute a set of metrics; a config's score is
a weighted composite. Weights are tunable via ``score_composite``;
defaults bias toward what the operator + downstream YOLO actually
care about: sharpness, edge density, YOLO confidence (if a model is
provided), and a clean (non-saturated) histogram.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    """All scalar metrics for a single 8-bit grayscale frame.

    Higher is better for: sharpness_lap, sharpness_tenengrad,
    rms_contrast, edge_density, hist_entropy, snr_db,
    yolo_confidence_sum, structure_ratio.

    Lower is better for: saturation_pct, blackclip_pct, frame_diff
    (temporal stability — measured separately as it's pairwise).
    """
    sharpness_lap: float = 0.0
    sharpness_tenengrad: float = 0.0
    rms_contrast: float = 0.0
    edge_density: float = 0.0
    hist_entropy: float = 0.0
    saturation_pct: float = 0.0
    blackclip_pct: float = 0.0
    snr_db: float = 0.0
    yolo_confidence_sum: float = 0.0
    yolo_n_detections: int = 0
    structure_ratio: float = 0.0  # 0..1, structure/noise discriminator


@dataclass(frozen=True)
class StackMetrics:
    """Per-stack aggregates: per-frame metrics averaged + temporal terms."""
    n_frames: int = 0
    sharpness_lap: float = 0.0
    sharpness_tenengrad: float = 0.0
    rms_contrast: float = 0.0
    edge_density: float = 0.0
    hist_entropy: float = 0.0
    saturation_pct: float = 0.0
    blackclip_pct: float = 0.0
    snr_db: float = 0.0
    yolo_confidence_sum: float = 0.0
    yolo_n_detections: float = 0.0
    structure_ratio: float = 0.0
    temporal_diff_mean: float = 0.0     # mean abs frame-to-frame diff
    temporal_diff_std: float = 0.0
    composite: float = 0.0
    composite_weights: dict = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────
# Per-frame metric primitives
# ───────────────────────────────────────────────────────────────

def sharpness_laplacian(gray_u8: np.ndarray) -> float:
    """Laplacian variance — classic focus / sharpness metric."""
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def sharpness_tenengrad(gray_u8: np.ndarray) -> float:
    """Tenengrad: mean squared Sobel gradient magnitude. Less
    susceptible to grain than Laplacian variance."""
    gx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def rms_contrast(gray_u8: np.ndarray) -> float:
    """Standard deviation of pixel values (RMS contrast)."""
    return float(gray_u8.std())


def edge_density(gray_u8: np.ndarray, low_thresh: int = 60, high_thresh: int = 120) -> float:
    """Canny edge fraction (edge pixels per total). Sensitive to detail."""
    edges = cv2.Canny(gray_u8, low_thresh, high_thresh)
    return float(np.count_nonzero(edges)) / float(edges.size)


def histogram_entropy(gray_u8: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy of the 256-bin histogram. Higher = more
    dynamic-range usage."""
    hist, _ = np.histogram(gray_u8, bins=bins, range=(0, 256))
    p = hist.astype(np.float64) / max(1, hist.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def saturation_pct(gray_u8: np.ndarray, threshold: int = 254) -> float:
    """Fraction of pixels at/near the white ceiling (255). High = clipped."""
    return float(np.count_nonzero(gray_u8 >= threshold)) / float(gray_u8.size)


def blackclip_pct(gray_u8: np.ndarray, threshold: int = 1) -> float:
    """Fraction of pixels at/near the black floor (0). High = crushed."""
    return float(np.count_nonzero(gray_u8 <= threshold)) / float(gray_u8.size)


def structure_to_noise_ratio(gray_u8: np.ndarray) -> float:
    """Discriminate real scene structure from noise.

    Computes ``std(blur(image)) / std(image)``. The intuition:
    real scenes have most of their pixel variance in low + mid
    spatial frequencies (object boundaries, gradients) which survive
    a small blur. Pure noise has its variance concentrated in HIGH
    spatial frequencies and is largely destroyed by even a 5×5
    blur, so its post-blur std is much smaller.

    Returns a value typically in [0, 1]:
      - ~0.95 → essentially all variance is structure (clean real scene)
      - ~0.7-0.9 → real scene with some grain (typical thermal frame)
      - ~0.3 → noise-dominated
      - ~0.05 → pure noise (gainLOW-broken state we hit on the bench)

    Validated empirically on the indoor 2-3m sweep: gainHIGH real
    scenes scored 0.85-0.95; gainLOW pure-noise scored 0.06-0.10.
    A 5× discriminator power, plenty for the composite to use as a
    gate term.
    """
    if gray_u8.size == 0:
        return 0.0
    full_std = float(gray_u8.std())
    if full_std < 1e-3:
        return 0.0
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0)
    blur_std = float(blurred.std())
    return min(1.0, blur_std / full_std)


def snr_db_estimate(gray_u8: np.ndarray, patch: int = 32) -> float:
    """Estimate SNR in dB by finding the patch with the smallest
    standard deviation (assumed flat = noise-only) and comparing it
    to the mean signal level.

    Returns inf-equivalent (capped at 80) if a flat patch reads zero
    std (very unlikely on real sensor data).
    """
    h, w = gray_u8.shape
    if h < patch * 2 or w < patch * 2:
        return 0.0
    grid_y = list(range(0, h - patch, patch))
    grid_x = list(range(0, w - patch, patch))
    if not grid_y or not grid_x:
        return 0.0
    stds = []
    for yy in grid_y:
        for xx in grid_x:
            p = gray_u8[yy:yy + patch, xx:xx + patch]
            stds.append(float(p.std()))
    if not stds:
        return 0.0
    sigma = float(np.percentile(stds, 5))  # 5th percentile = flattest patches
    if sigma < 1e-3:
        return 80.0
    signal = float(gray_u8.mean())
    return 20.0 * float(np.log10(max(1.0, signal) / sigma))


def per_frame_metrics(
    gray_u8: np.ndarray,
    yolo_results: Optional[list] = None,
) -> FrameMetrics:
    """Compute all metrics for a single grayscale frame.

    ``yolo_results`` if given is the ultralytics Results object for
    this frame (or None). We extract sum-of-confidences and detection
    count for the YOLO-readiness signal.
    """
    if gray_u8.ndim == 3:
        gray_u8 = cv2.cvtColor(gray_u8, cv2.COLOR_BGR2GRAY)
    yconf = 0.0
    yn = 0
    if yolo_results is not None and yolo_results.boxes is not None:
        try:
            yconf = float(yolo_results.boxes.conf.sum().item())
            yn = int(len(yolo_results.boxes))
        except Exception:
            pass
    return FrameMetrics(
        sharpness_lap=sharpness_laplacian(gray_u8),
        sharpness_tenengrad=sharpness_tenengrad(gray_u8),
        rms_contrast=rms_contrast(gray_u8),
        edge_density=edge_density(gray_u8),
        hist_entropy=histogram_entropy(gray_u8),
        saturation_pct=saturation_pct(gray_u8),
        blackclip_pct=blackclip_pct(gray_u8),
        snr_db=snr_db_estimate(gray_u8),
        yolo_confidence_sum=yconf,
        yolo_n_detections=yn,
        structure_ratio=structure_to_noise_ratio(gray_u8),
    )


# ───────────────────────────────────────────────────────────────
# Stack-level aggregation
# ───────────────────────────────────────────────────────────────

def temporal_diff(stack_u8: np.ndarray) -> tuple[float, float]:
    """Mean and std of frame-to-frame absolute difference. Stack is
    (N, H, W) uint8 grayscale. Lower = more stable across time."""
    if stack_u8.shape[0] < 2:
        return 0.0, 0.0
    diffs = []
    for i in range(stack_u8.shape[0] - 1):
        d = cv2.absdiff(stack_u8[i], stack_u8[i + 1])
        diffs.append(float(d.mean()))
    arr = np.array(diffs)
    return float(arr.mean()), float(arr.std())


# ───────────────────────────────────────────────────────────────
# Composite score
# ───────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    # Lessons from indoor 2-3m sweep (2026-04-27): pure laplacian
    # variance over-rewards noise. structure_ratio is now the GATE
    # term — without real scene structure, no amount of sharpness
    # helps. Frame stacks with ratio < 0.3 are likely noise; we
    # multiply by it to crush noise scores.
    "structure_ratio":       50.0,    # NEW: noise-vs-detail discriminator
    "sharpness_lap":          0.5,    # halved — too easily fooled by grain
    "sharpness_tenengrad":    0.5,
    "rms_contrast":           0.4,
    "edge_density":          30.0,    # halved for same reason
    "hist_entropy":           0.6,
    "saturation_pct":       -50.0,    # PENALTY
    "blackclip_pct":        -50.0,    # PENALTY
    "snr_db":                 0.4,
    "yolo_confidence_sum":   10.0,
    "yolo_n_detections":      0.0,    # neutral by default
    "temporal_diff_mean":    -2.0,    # small penalty for breathing
}


def normalize_metric(name: str, raw: float) -> float:
    """Normalize a metric to a 0..1-ish range so weights are
    comparable. These are HEURISTIC scales tuned to typical Boson 640
    output; tweak if your metric distributions differ."""
    if name == "sharpness_lap":
        return min(1.0, raw / 1000.0)
    if name == "sharpness_tenengrad":
        return min(1.0, raw / 50000.0)
    if name == "rms_contrast":
        return min(1.0, raw / 80.0)
    if name == "edge_density":
        return min(1.0, raw / 0.10)  # 10% edges is excellent
    if name == "hist_entropy":
        return raw / 8.0  # entropy ranges 0..8 for 256-bin histogram
    if name == "saturation_pct":
        return raw  # already 0..1
    if name == "blackclip_pct":
        return raw
    if name == "snr_db":
        return min(1.0, max(0.0, (raw - 10) / 50.0))  # 10..60 dB → 0..1
    if name == "yolo_confidence_sum":
        return min(1.0, raw / 5.0)  # ~5 confident detections is a lot
    if name == "yolo_n_detections":
        return min(1.0, raw / 6.0)
    if name == "temporal_diff_mean":
        return min(1.0, raw / 5.0)  # 0..5 grayscale-units of avg diff
    if name == "structure_ratio":
        return raw  # already 0..1
    return 0.0


def aggregate(frame_mlist: list[FrameMetrics],
              stack_u8: Optional[np.ndarray] = None,
              weights: Optional[dict] = None) -> StackMetrics:
    """Aggregate per-frame metrics + (optional) temporal metrics into
    a StackMetrics record with composite score."""
    if not frame_mlist:
        return StackMetrics()
    weights = weights if weights is not None else DEFAULT_WEIGHTS

    means = {}
    keys = [
        "sharpness_lap", "sharpness_tenengrad", "rms_contrast",
        "edge_density", "hist_entropy", "saturation_pct",
        "blackclip_pct", "snr_db", "yolo_confidence_sum",
        "yolo_n_detections", "structure_ratio",
    ]
    for k in keys:
        means[k] = float(np.mean([getattr(fm, k) for fm in frame_mlist]))

    tdiff_mean, tdiff_std = (0.0, 0.0)
    if stack_u8 is not None:
        tdiff_mean, tdiff_std = temporal_diff(stack_u8)

    # Composite
    composite = 0.0
    for k, w in weights.items():
        if k == "temporal_diff_mean":
            composite += w * normalize_metric(k, tdiff_mean)
        elif k in means:
            composite += w * normalize_metric(k, means[k])

    return StackMetrics(
        n_frames=len(frame_mlist),
        sharpness_lap=means["sharpness_lap"],
        sharpness_tenengrad=means["sharpness_tenengrad"],
        rms_contrast=means["rms_contrast"],
        edge_density=means["edge_density"],
        hist_entropy=means["hist_entropy"],
        saturation_pct=means["saturation_pct"],
        blackclip_pct=means["blackclip_pct"],
        snr_db=means["snr_db"],
        yolo_confidence_sum=means["yolo_confidence_sum"],
        yolo_n_detections=means["yolo_n_detections"],
        structure_ratio=means["structure_ratio"],
        temporal_diff_mean=tdiff_mean,
        temporal_diff_std=tdiff_std,
        composite=composite,
        composite_weights=dict(weights),
    )


def score_composite(frame_mlist: list[FrameMetrics],
                    stack_u8: Optional[np.ndarray] = None,
                    weights: Optional[dict] = None) -> float:
    """Convenience: aggregate and return only the composite score."""
    return aggregate(frame_mlist, stack_u8, weights).composite


def metrics_to_dict(sm: StackMetrics) -> dict:
    """JSON-serializable dict (drops the weights field for compactness)."""
    d = asdict(sm)
    d.pop("composite_weights", None)
    return d
