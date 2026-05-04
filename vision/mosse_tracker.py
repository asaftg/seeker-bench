"""MOSSE — Minimum Output Sum of Squared Error tracker.

Pure-numpy implementation of the correlation-filter tracker from
Bolme et al., CVPR 2010 ("Visual Object Tracking using Adaptive
Correlation Filters"). Per-target tracker that takes an initial
bbox in frame N and returns the same target's bbox in frame N+1
by image correlation — does NOT need a detection in every frame.

Why we wrote this from scratch:
    1. opencv-python 4.13 (the wheel installed on this rig) does NOT
       expose ``cv2.legacy.TrackerMOSSE_create``. That class lives in
       ``opencv-contrib-python``, which we do not have on the Jetson
       target either. Custom numpy gets us across both environments.
    2. OpenCV's wrapper hides the internal Peak-to-Sidelobe-Ratio
       (PSR), which Bolme's paper identifies as the canonical
       "lost target" signal. Computing PSR ourselves gives gimbal_manager
       a direct quality metric to gate persistence on.
    3. ~100 lines is debuggable; the contrib build is not.

Algorithm (one-paragraph summary):
    The filter h is trained so that h*x ≈ g, where x is the target
    patch and g is a Gaussian centered on the target. Train in the
    frequency domain so convolution is multiplication: H = G ⊙ X̄
    (numerator) / (X ⊙ X̄ + ε) (denominator). For online tracking,
    take a new patch x', compute X' = FFT(x'), find the peak of
    real(IFFT(H ⊙ X')) — that's where the filter says the target is.
    Online learning: blend old (H_num, H_den) with new at rate η.

Performance on Jetson AGX Xavier:
    Per-tick MOSSE update for a 64×64 patch is ~1-3 ms (single
    Carmel core). We run at most one tracker per ByteTrack ID, so
    5 simultaneous tracks at 30 Hz ≈ 1 CPU core fully utilized.
    Compatible with throttling YOLO 30→10 Hz to free the GPU.

Sources:
    Bolme et al., "Visual Object Tracking using Adaptive Correlation
    Filters", CVPR 2010 — algorithm + PSR thresholds.
    https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ── PSR thresholds (from Bolme et al., Section 4) ────────────────
# PSR > 20      strong lock (clean track, training stable)
# PSR 10..20    acceptable
# PSR  7..10    uncertain — freeze online learning, keep tracking
# PSR < 7       likely occluded/lost — caller should treat as missing
PSR_STRONG_LOCK   = 20.0
PSR_GOOD          = 10.0
PSR_LOST_DEFAULT  =  7.0


@dataclass
class MosseUpdate:
    """One frame's worth of tracker output."""
    bbox_xywh: Tuple[int, int, int, int]   # (x, y, w, h) — top-left + size
    psr: float                              # peak-to-sidelobe ratio
    locked: bool                            # PSR above the lost threshold


# ── helpers ──────────────────────────────────────────────────────

def _preprocess(patch: np.ndarray) -> np.ndarray:
    """Bolme's preprocessing: log-normalize, mean-subtract, contrast-
    normalize, multiply by Hann (cosine) window to suppress edge
    discontinuities that the FFT would otherwise fold into the
    spectrum.
    """
    p = patch.astype(np.float32)
    p = np.log1p(p)                        # log(1 + x)
    p -= p.mean()
    s = p.std()
    if s > 1e-6:
        p /= s
    h, w = p.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    return p * win


def _gaussian_2d(h: int, w: int, sigma: float) -> np.ndarray:
    """2D Gaussian centered at the patch center, normalized so peak=1.
    This is the "ideal" correlation response the filter trains toward.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    g /= max(g.max(), 1e-6)
    return g


def _random_warp(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Affine warp for training-set augmentation. Bolme reports 8
    warps gives a tight, generalizing filter. Small rotations +
    translations only — large warps make the filter generic and
    drift-prone.
    """
    h, w = patch.shape
    # Tiny rotation in radians, tiny translation in px.
    angle = rng.uniform(-0.05, 0.05)         # ~3° max
    tx = rng.uniform(-2.0, 2.0)
    ty = rng.uniform(-2.0, 2.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # Build inverse map for warpAffine-equivalent (we use np.indices)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = cos_a * (xx - cx) - sin_a * (yy - cy) + cx + tx
    ys = sin_a * (xx - cx) + cos_a * (yy - cy) + cy + ty
    xs = np.clip(xs, 0, w - 1).astype(np.int32)
    ys = np.clip(ys, 0, h - 1).astype(np.int32)
    return patch[ys, xs]


def _crop_patch(frame: np.ndarray,
                cx: float, cy: float, w: int, h: int) -> Optional[np.ndarray]:
    """Extract a w×h patch centered on (cx, cy). Returns None if the
    patch falls outside the frame — caller treats as "lost"."""
    H, W = frame.shape[:2]
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    x1, y1 = x0 + w, y0 + h
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    return frame[y0:y1, x0:x1]


def _work_dims(box_w: int, box_h: int, max_dim: int) -> Tuple[int, int]:
    """Map a frame-coord bbox (box_w × box_h) to the FFT working size,
    capped at `max_dim` per axis. Preserves aspect ratio. Snapped to
    even lengths because numpy's FFT prefers even-length axes (small
    perf bump from radix-2). When the bbox is already <= max_dim, we
    return it unchanged so small targets pay zero scaling cost.
    """
    longest = max(box_w, box_h)
    if longest <= max_dim:
        return int(box_w), int(box_h)
    scale = float(longest) / float(max_dim)
    ww = max(8, int(round(box_w / scale)))
    hh = max(8, int(round(box_h / scale)))
    # Snap to even lengths.
    ww -= ww % 2
    hh -= hh % 2
    return ww, hh


def _peak_psr(response: np.ndarray) -> Tuple[Tuple[int, int], float]:
    """Find the peak of the correlation response and compute PSR.
    PSR = (peak - mean(sidelobe)) / std(sidelobe), where sidelobe is
    the response with an 11×11 region around the peak masked out.
    """
    py, px = np.unravel_index(int(np.argmax(response)), response.shape)
    peak = float(response[py, px])
    # Mask 11×11 around the peak (Bolme's recommendation).
    side = response.copy()
    h, w = side.shape
    y0, y1 = max(0, py - 5), min(h, py + 6)
    x0, x1 = max(0, px - 5), min(w, px + 6)
    side[y0:y1, x0:x1] = 0.0
    mu = float(side.mean())
    sd = float(side.std()) + 1e-8
    psr = (peak - mu) / sd
    return (py, px), psr


# ── tracker class ────────────────────────────────────────────────

class MosseTracker:
    """Per-target MOSSE tracker.

    Lifecycle:
        t = MosseTracker(frame, bbox_xywh)   # seed (or re-seed) on a
                                              # known-good detection
        upd = t.update(next_frame)           # call every subsequent frame
        if not upd.locked:                   # PSR fell below threshold
            ...                              # let YOLO re-acquire,
                                              # destroy this tracker

    The bbox SIZE is fixed at construction. MOSSE doesn't do scale
    adaptation; if the target's size changes by > ~30%, reseed the
    tracker on a fresh detection.
    """

    def __init__(self,
                 frame: np.ndarray,
                 bbox_xywh: Tuple[int, int, int, int],
                 *,
                 learning_rate: float = 0.125,
                 sigma: float = 2.0,
                 psr_lost: float = PSR_LOST_DEFAULT,
                 n_warps: int = 8,
                 max_patch_dim: int = 96,
                 rng_seed: Optional[int] = None) -> None:
        if frame.ndim == 3:
            frame = self._to_gray(frame)
        x, y, w, h = bbox_xywh
        # Snap to even sizes — FFT prefers even-length axes.
        w = max(8, w - (w % 2))
        h = max(8, h - (h % 2))
        # Frame-coord bbox dims (used for cropping the patch from the frame)
        # are decoupled from the FFT working dims. Big bboxes (close-range
        # vehicle) get downsampled to work-size before the FFT — without
        # this cap, a 300×200 close target ran 4 native FFTs per update at
        # ~10–20 ms each, dragging EO from 22 Hz to 2–5 Hz when the target
        # entered a fused lock. The bbox returned to callers is always in
        # frame coords.
        self._box_w, self._box_h = int(w), int(h)
        ww, hh = _work_dims(self._box_w, self._box_h, int(max_patch_dim))
        self._w, self._h = ww, hh
        self._scale_x = self._box_w / float(self._w)
        self._scale_y = self._box_h / float(self._h)
        self._cx = float(x) + self._box_w / 2.0
        self._cy = float(y) + self._box_h / 2.0
        self._lr = float(learning_rate)
        self._sigma = float(sigma)
        self._psr_lost = float(psr_lost)
        self._max_patch_dim = int(max_patch_dim)
        self._rng = np.random.default_rng(rng_seed)
        self._g = _gaussian_2d(self._h, self._w, self._sigma)
        G = np.fft.fft2(self._g)
        # Train the initial filter on the seed patch + warps.
        patch = self._crop_to_work(frame)
        if patch is None:
            raise ValueError("seed bbox extends outside frame")
        # Numerator = sum_i G * conj(X_i); Denominator = sum_i X_i * conj(X_i)
        num = np.zeros((self._h, self._w), dtype=np.complex128)
        den = np.zeros((self._h, self._w), dtype=np.complex128)
        for _ in range(n_warps):
            warped = _random_warp(patch, self._rng)
            X = np.fft.fft2(_preprocess(warped))
            num += G * np.conj(X)
            den += X * np.conj(X)
        self._H_num = num
        self._H_den = den

    # ── public ────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> MosseUpdate:
        """Locate the target in `frame` and (if PSR good) update the
        filter online. Returns the new bbox + PSR + locked flag.
        """
        if frame.ndim == 3:
            frame = self._to_gray(frame)
        patch = self._crop_to_work(frame)
        if patch is None:
            # Walked off the edge. Caller should drop this tracker.
            return MosseUpdate(self._current_bbox(), 0.0, False)
        X = np.fft.fft2(_preprocess(patch))
        H = self._H_num / (self._H_den + 1e-8)
        # Filter convention: H_num = G * conj(X), H_den = |X|².
        # Then H * X = G * |X|² / |X|² = G in the frequency domain
        # for the seed patch; for a new patch Z the peak of
        # IFFT(H * Z) lies where Z's pattern aligns with the trained
        # filter. (NOT IFFT(conj(H) * Z) — that flips the response
        # and PSR collapses to ~3 on a static target. Verified
        # empirically on synthetic frames in test_mosse_tracker.)
        R = np.real(np.fft.ifft2(H * X))
        (py, px), psr = _peak_psr(R)
        # Peak position is in WORK-pixel coords. Convert sub-pixel
        # translation back to FRAME-pixel coords using the per-axis
        # scale ratios so the centroid update is in the right space.
        dx_work = float(px) - (self._w - 1) / 2.0
        dy_work = float(py) - (self._h - 1) / 2.0
        self._cx += dx_work * self._scale_x
        self._cy += dy_work * self._scale_y
        locked = psr >= self._psr_lost
        if locked:
            # Online filter update at this new center.
            new_patch = self._crop_to_work(frame)
            if new_patch is not None:
                X_new = np.fft.fft2(_preprocess(new_patch))
                G = np.fft.fft2(self._g)
                lr = self._lr
                self._H_num = ((1.0 - lr) * self._H_num
                                + lr * (G * np.conj(X_new)))
                self._H_den = ((1.0 - lr) * self._H_den
                                + lr * (X_new * np.conj(X_new)))
        return MosseUpdate(self._current_bbox(), float(psr), bool(locked))

    def reseed(self, frame: np.ndarray,
                bbox_xywh: Tuple[int, int, int, int]) -> None:
        """Re-anchor the tracker on a fresh detection. Keep the same
        learning rate / sigma; rebuild the filter from scratch.
        Called when YOLO produces a confirmed detection that the
        controller wants the tracker to follow.
        """
        if frame.ndim == 3:
            frame = self._to_gray(frame)
        x, y, w, h = bbox_xywh
        w = max(8, w - (w % 2))
        h = max(8, h - (h % 2))
        self._box_w, self._box_h = int(w), int(h)
        ww, hh = _work_dims(self._box_w, self._box_h, self._max_patch_dim)
        # If the work-size changed (target growing/shrinking across the
        # cap boundary) we have to rebuild g, H_num, H_den at the new
        # array shape. Common case: same shape — keep the buffers.
        if (ww, hh) != (self._w, self._h):
            self._w, self._h = ww, hh
            self._g = _gaussian_2d(self._h, self._w, self._sigma)
        self._scale_x = self._box_w / float(self._w)
        self._scale_y = self._box_h / float(self._h)
        self._cx = float(x) + self._box_w / 2.0
        self._cy = float(y) + self._box_h / 2.0
        G = np.fft.fft2(self._g)
        patch = self._crop_to_work(frame)
        if patch is None:
            raise ValueError("reseed bbox extends outside frame")
        num = np.zeros((self._h, self._w), dtype=np.complex128)
        den = np.zeros((self._h, self._w), dtype=np.complex128)
        for _ in range(8):
            warped = _random_warp(patch, self._rng)
            X = np.fft.fft2(_preprocess(warped))
            num += G * np.conj(X)
            den += X * np.conj(X)
        self._H_num = num
        self._H_den = den

    @property
    def bbox_xywh(self) -> Tuple[int, int, int, int]:
        return self._current_bbox()

    # ── helpers ───────────────────────────────────────────────────
    def _current_bbox(self) -> Tuple[int, int, int, int]:
        # Bbox is reported in FRAME coords (using box_w/box_h),
        # not work coords. Centroid lives in frame coords too.
        x0 = int(round(self._cx - self._box_w / 2.0))
        y0 = int(round(self._cy - self._box_h / 2.0))
        return (x0, y0, self._box_w, self._box_h)

    def _crop_to_work(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Crop a box_w × box_h patch from `frame` centered on
        (cx, cy), then downsample to the FFT working size. Returns
        None if the bbox extends outside the frame. When the bbox
        is already at or below the max work dim, this collapses to
        a plain crop (no resize cost)."""
        patch = _crop_patch(frame, self._cx, self._cy,
                            self._box_w, self._box_h)
        if patch is None:
            return None
        if (self._box_w, self._box_h) == (self._w, self._h):
            return patch
        # cv2.resize takes (w, h). INTER_AREA is the right choice for
        # downsampling (anti-aliased). Local import to keep this module
        # importable in environments without cv2 — falls back to the
        # equivalent uint8 stride decimation below.
        try:
            import cv2  # type: ignore
            return cv2.resize(patch, (self._w, self._h),
                              interpolation=cv2.INTER_AREA)
        except Exception:
            # Pure-numpy fallback: stride-decimate, lossy but functional.
            ys = np.linspace(0, patch.shape[0] - 1, self._h).astype(np.int32)
            xs = np.linspace(0, patch.shape[1] - 1, self._w).astype(np.int32)
            return patch[ys[:, None], xs[None, :]]

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        # BGR or RGB → grayscale, weighted average. Cheaper than
        # importing cv2 just for the conversion.
        if frame.ndim == 2:
            return frame
        # Standard luminance weights.
        return (0.114 * frame[..., 0] + 0.587 * frame[..., 1]
                + 0.299 * frame[..., 2]).astype(np.uint8)
