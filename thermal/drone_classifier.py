"""
Two-tier target classifier.

    Tier 1 — YOLOv8n via `ultralytics`:
        Loaded lazily. If import fails, model file is missing,
        or a forward pass raises, we log a warning and fall back
        to the shape heuristic. The GUI never notices.

    Tier 2 — Classical shape heuristic:
        Aspect ratio + compactness + area buckets. Crude but
        guaranteed to work with zero external dependencies.

Both tiers produce `ClassificationResult` so the rest of the
pipeline is indifferent to which one ran.

Contract:
    classify(bgr_image, detections) -> list[ClassificationResult | None]
    The returned list is aligned with the input `detections` list
    position-by-position.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from common.frames import ClassificationResult, TargetClass, ThermalDetection
from common.logging_setup import get_logger

log = get_logger(__name__)


# ───────────────────────────────────────────────────────────────
# Shape heuristic (always available)
# ───────────────────────────────────────────────────────────────

def classify_by_shape(det: ThermalDetection) -> ClassificationResult:
    """Classify a single detection based purely on its bbox geometry.

    Conservative rules: only labels something HAND if the shape is
    clearly elongated (arm/hand). Everything else is UNKNOWN, which
    the GUI renders as the honest "HEAT DETECTED" banner.

    DRONE classification is intentionally NOT done here — a compact
    warm blob is not enough evidence. Use a fine-tuned YOLO model
    (seeker_thermal.pt) for reliable DRONE labels.
    """
    w, h = max(1, det.bbox.w), max(1, det.bbox.h)
    area = det.area_px
    aspect = max(w, h) / min(w, h)

    if area < 4:
        return ClassificationResult(
            target_class=TargetClass.NOISE,
            confidence=0.2,
            classifier_used="shape_heuristic",
        )

    # Without a fine-tuned YOLO model, every warm blob is just
    # "detected heat, identity unknown". The GUI renders this as
    # the orange HEAT DETECTED banner. HAND/DRONE labels only
    # become reliable once seeker_thermal.pt is trained and
    # deployed to models/.
    return ClassificationResult(
        target_class=TargetClass.UNKNOWN,
        confidence=0.3,
        classifier_used="shape_heuristic",
    )


# ───────────────────────────────────────────────────────────────
# YOLO wrapper (optional)
# ───────────────────────────────────────────────────────────────

class _YoloTier:
    """Wraps ultralytics YOLO. Safe no-op if not available."""

    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        trained_model_path: str = "models/seeker_thermal.pt",
        conf_threshold: float = 0.25,
        coco_to_target: Optional[dict] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.coco_to_target = coco_to_target or {0: "hand"}
        self._model = None
        self._model_path_used: Optional[str] = None

        # Only load YOLO if a fine-tuned thermal model exists.
        # Stock COCO weights (yolov8n.pt) are useless on thermal
        # imagery — they produce false DRONE/HAND labels on random
        # warm blobs. Shape heuristic is more honest until training.
        trained = Path(trained_model_path)
        stock = Path(model_path)
        if trained.exists():
            target = trained
        else:
            log.info(
                "No fine-tuned model at %s — YOLO disabled; "
                "using shape heuristic only. Train with "
                "thermal/training/ to enable YOLO.",
                trained_model_path,
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            log.warning("ultralytics not importable — YOLO tier disabled: %s", e)
            return

        try:
            self._model = YOLO(str(target))
            self._model_path_used = str(target)
            log.info("YOLO loaded from %s", target)
        except Exception as e:
            log.warning("YOLO failed to load (%s) — falling back to shape heuristic", e)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def classify_roi(self, roi_bgr: np.ndarray) -> Optional[ClassificationResult]:
        if self._model is None or roi_bgr.size == 0:
            return None
        try:
            results = self._model.predict(
                roi_bgr, conf=self.conf_threshold, verbose=False,
            )
        except Exception as e:
            log.warning("YOLO inference failed: %s", e)
            return None

        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None

        # Take the top-scoring box
        best = r.boxes[int(r.boxes.conf.argmax())]
        cls_id = int(best.cls.item())
        conf = float(best.conf.item())

        target_name = self.coco_to_target.get(cls_id, "unknown")
        try:
            target_cls = TargetClass(target_name)
        except ValueError:
            target_cls = TargetClass.UNKNOWN

        return ClassificationResult(
            target_class=target_cls,
            confidence=conf,
            classifier_used="yolo",
        )


# ───────────────────────────────────────────────────────────────
# Top-level classifier
# ───────────────────────────────────────────────────────────────

class Classifier:
    def __init__(
        self,
        enable_yolo: bool = True,
        model_path: str = "models/yolov8n.pt",
        trained_model_path: str = "models/seeker_thermal.pt",
        conf_threshold: float = 0.25,
        roi_padding_px: int = 16,
        coco_to_target: Optional[dict] = None,
    ) -> None:
        self.roi_padding_px = roi_padding_px
        self._yolo = _YoloTier(
            model_path=model_path,
            trained_model_path=trained_model_path,
            conf_threshold=conf_threshold,
            coco_to_target=coco_to_target,
        ) if enable_yolo else None

    @property
    def yolo_active(self) -> bool:
        return self._yolo is not None and self._yolo.available

    def classify(
        self,
        display_bgr: np.ndarray,
        detections: List[ThermalDetection],
    ) -> List[Optional[ClassificationResult]]:
        """Classify each detection.

        Returns a parallel list aligned with `detections`. Entries
        may be None if both tiers decline to classify.
        """
        out: List[Optional[ClassificationResult]] = []
        h, w = display_bgr.shape[:2]
        pad = self.roi_padding_px

        for det in detections:
            result: Optional[ClassificationResult] = None

            if self.yolo_active:
                # Extract a padded ROI from the display image for YOLO
                x0 = max(0, det.bbox.x - pad)
                y0 = max(0, det.bbox.y - pad)
                x1 = min(w, det.bbox.x + det.bbox.w + pad)
                y1 = min(h, det.bbox.y + det.bbox.h + pad)
                roi = display_bgr[y0:y1, x0:x1]
                result = self._yolo.classify_roi(roi)  # type: ignore[union-attr]

            if result is None:
                # Fallback — always produces a result
                result = classify_by_shape(det)

            out.append(result)

        return out
