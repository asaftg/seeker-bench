"""
Classical-CV heat blob detector.

Algorithm (simple but robust enough for Phase A):

    1. Estimate the spatial background via a large-kernel box
       filter. The background is the "local DC level" of the
       thermal frame. (We wanted medianBlur, but OpenCV only
       supports 8-bit median for kernels > 5; boxFilter works on
       float32 at any kernel size, and the MAD step below gives
       us the robustness we needed median for.)
    2. Compute residual = frame - background.
    3. Compute a robust noise floor via median + k * MAD on the
       residual. MAD (Median Absolute Deviation) is used instead
       of std because it's immune to the hot blobs themselves
       biasing the threshold upwards.
    4. Threshold → binary mask → connected components.
    5. Filter components by min/max area.
    6. Emit ThermalDetection for each surviving component.

This detector is deliberately independent of the classifier. If
YOLO crashes or isn't installed, detection still runs and the GUI
still shows orange heat boxes — just without a class label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from common.frames import BBox, ThermalDetection
from common.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class HeatDetectorConfig:
    threshold_k: float = 5.0         # residual > median + k*MAD is "hot"
    background_kernel: int = 21      # spatial background kernel (odd)
    min_blob_area_px: int = 3
    max_blob_area_px: int = 5000
    max_detections: int = 20         # cap per frame
    # Algorithm: 'boxfilter_mad' | 'tophat'
    #   boxfilter_mad — original: residual vs local mean, MAD threshold.
    #                   Best when the background is large, smooth and cold
    #                   (drone in sky). Fires on every bright edge indoors.
    #   tophat        — morphological white top-hat. Extracts compact bright
    #                   peaks and discards anything larger than `tophat_kernel`.
    #                   Much better on cluttered indoor scenes.
    algorithm: str = "tophat"
    # Phase 1 fix #5: 15 -> 11. Saves ~3 ms/frame on tophat morphology
    # (the largest single per-frame cost in the thermal pipeline). Targets
    # in the bench setup are < 11 px diameter, so the smaller kernel still
    # rejects everything that isn't a compact hot spot.
    tophat_kernel: int = 11          # > max target diameter in pixels


class HeatDetector:
    def __init__(self, config: HeatDetectorConfig | None = None) -> None:
        self.cfg = config or HeatDetectorConfig()
        if self.cfg.background_kernel % 2 == 0:
            raise ValueError("background_kernel must be odd")

    def detect(
        self,
        frame_u16: np.ndarray,
        agc8: np.ndarray | None = None,
    ) -> List[ThermalDetection]:
        """Find hot blobs in a raw 16-bit thermal frame.

        If the caller already produced an AGC-stretched uint8 version of
        this frame (e.g. via :func:`thermal.thermal_processor.apply_agc`),
        pass it as ``agc8`` to skip a duplicate ``np.percentile`` + cast.
        Measured saving: ~8 ms/frame on a 640×512 Boson, lifting thermal
        publish ~15 Hz → ~18-19 Hz on multi-target scenes (per the
        2026-05-04 thermal-pipeline audit).
        """
        if frame_u16.ndim != 2:
            raise ValueError(f"HeatDetector expects 2-D uint16, got {frame_u16.shape}")

        if self.cfg.algorithm == "tophat":
            return self._detect_tophat(frame_u16, agc8=agc8)
        return self._detect_boxfilter_mad(frame_u16)

    # ─────────────────────── algorithm: tophat ────────────────────
    def _detect_tophat(
        self,
        frame_u16: np.ndarray,
        *,
        agc8: np.ndarray | None = None,
    ) -> List[ThermalDetection]:
        if agc8 is not None and agc8.shape == frame_u16.shape and agc8.dtype == np.uint8:
            # Caller already AGC'd the frame — skip the redundant
            # percentile + float32 cast. The two paths produce
            # functionally equivalent uint8 frames for the morphological
            # top-hat downstream (the AGC chain uses [2, 98] and this
            # branch used [1, 99]; both stretch the bulk of the
            # histogram to span ~most of [0, 255], and TOPHAT only
            # cares about local bright peaks above the rolling
            # background).
            u8 = agc8
        else:
            # Standalone caller: do our own percentile stretch.
            f = frame_u16.astype(np.float32)
            lo, hi = np.percentile(f, (1.0, 99.0))
            if hi <= lo:
                hi = lo + 1.0
            u8 = np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)

        ks = max(3, self.cfg.tophat_kernel | 1)  # force odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        tophat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, kernel)

        med = float(np.median(tophat))
        mad = float(np.median(np.abs(tophat - med))) + 1e-3
        threshold = med + self.cfg.threshold_k * 1.4826 * mad
        mask = (tophat > threshold).astype(np.uint8)

        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: List[ThermalDetection] = []
        for lbl in range(1, num):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < self.cfg.min_blob_area_px or area > self.cfg.max_blob_area_px:
                continue
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            blob = tophat[y:y + h, x:x + w][labels[y:y + h, x:x + w] == lbl]
            contrast = float(blob.max()) if blob.size else 0.0
            detections.append(
                ThermalDetection(
                    bbox=BBox(x=x, y=y, w=w, h=h),
                    area_px=area,
                    contrast=contrast,
                    classification=None,
                )
            )
        detections.sort(key=lambda d: d.contrast, reverse=True)
        return detections[: self.cfg.max_detections]

    # ──────────────────── algorithm: boxfilter_mad ────────────────
    def _detect_boxfilter_mad(self, frame_u16: np.ndarray) -> List[ThermalDetection]:
        f32 = frame_u16.astype(np.float32)

        # 1. Background = local mean via box filter. Fast on float32
        #    at any kernel size, unlike medianBlur which is 8-bit only
        #    above kernel=5.
        k = self.cfg.background_kernel
        bg = cv2.boxFilter(f32, ddepth=cv2.CV_32F, ksize=(k, k),
                           normalize=True, borderType=cv2.BORDER_REFLECT)

        # 2. Residual (positive side only — we're looking for warm spots)
        residual = f32 - bg
        residual = np.clip(residual, 0, None)

        # 3. Robust threshold via median + k*MAD
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med))) + 1e-3
        threshold = med + self.cfg.threshold_k * 1.4826 * mad  # 1.4826 → std estimate

        mask = (residual > threshold).astype(np.uint8)

        # Morphology: close tiny gaps (prop dots) but don't merge far blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. Connected components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detections: List[ThermalDetection] = []
        # label 0 is background
        for lbl in range(1, num):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < self.cfg.min_blob_area_px or area > self.cfg.max_blob_area_px:
                continue
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])

            # Peak residual inside this blob = contrast signal
            blob_residuals = residual[y:y + h, x:x + w][labels[y:y + h, x:x + w] == lbl]
            contrast = float(blob_residuals.max()) if blob_residuals.size else 0.0

            detections.append(
                ThermalDetection(
                    bbox=BBox(x=x, y=y, w=w, h=h),
                    area_px=area,
                    contrast=contrast,
                    classification=None,
                )
            )

        # Always rank by contrast and cap at max_detections so the GUI
        # sees the strongest blobs first and noise never dominates.
        detections.sort(key=lambda d: d.contrast, reverse=True)
        return detections[: self.cfg.max_detections]
