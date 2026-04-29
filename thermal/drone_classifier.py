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

from pathlib import Path
from typing import List, Optional

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

    def _result_to_classification(self, r) -> Optional[ClassificationResult]:
        """Per-ROI YOLO Result -> ClassificationResult (or None on no-hit).

        Extracted so both ``classify_roi`` and ``classify_rois_batch``
        share identical post-processing.
        """
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return None
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
        return self._result_to_classification(results[0])

    def classify_rois_batch(
        self, rois: List[np.ndarray]
    ) -> List[Optional[ClassificationResult]]:
        """Run inference on a list of ROIs in ONE GPU call.

        Bit-equivalent to looping ``classify_roi`` per ROI (verified by
        scripts/_drone_classifier_batch_equivalence.py — 32/32 ROIs
        identical) but ~5x faster on a multi-blob frame because only
        one CUDA launch + one kernel sync per tick instead of N.

        Empty/zero-size ROIs are reported as None without ever hitting
        the model. If the whole list is empty, returns an empty list.
        """
        if self._model is None or not rois:
            return [None] * len(rois)
        # Map skip-indices (empty ROIs) so the batched call never sees them.
        valid_idx: List[int] = []
        valid_rois: List[np.ndarray] = []
        for i, r in enumerate(rois):
            if r is not None and r.size > 0:
                valid_idx.append(i)
                valid_rois.append(r)
        out: List[Optional[ClassificationResult]] = [None] * len(rois)
        if not valid_rois:
            return out
        try:
            results = self._model.predict(
                valid_rois, conf=self.conf_threshold, verbose=False,
            )
        except Exception as e:
            log.warning("YOLO batch inference failed: %s", e)
            return out
        for i, r in zip(valid_idx, results):
            out[i] = self._result_to_classification(r)
        return out


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

        Implementation: extracts ROIs for all non-synthetic detections,
        runs ONE batched YOLO call, then fills back into the parallel
        output list. Synthetic user-seeded detections bypass YOLO and
        keep their existing classification (running YOLO would
        overwrite the operator's "user" label with a random class).

        Batched inference is bit-equivalent to the legacy per-ROI loop
        (verified offline by scripts/_drone_classifier_batch_equivalence.py
        — 32/32 ROIs match across rounds) but ~5x faster at multi-blob
        frames because only one GPU launch per tick.
        """
        h, w = display_bgr.shape[:2]
        pad = self.roi_padding_px
        n = len(detections)
        out: List[Optional[ClassificationResult]] = [None] * n

        # First pass: handle synthetic targets, extract ROIs for the rest.
        rois: List[np.ndarray] = []
        roi_slot: List[int] = []  # output index each ROI maps back to
        for i, det in enumerate(detections):
            if getattr(det, "synthetic", False):
                out[i] = det.classification
                continue
            if self.yolo_active:
                x0 = max(0, det.bbox.x - pad)
                y0 = max(0, det.bbox.y - pad)
                x1 = min(w, det.bbox.x + det.bbox.w + pad)
                y1 = min(h, det.bbox.y + det.bbox.h + pad)
                rois.append(display_bgr[y0:y1, x0:x1])
                roi_slot.append(i)

        # Second pass: one YOLO call for all collected ROIs.
        if rois and self.yolo_active:
            yolo_results = self._yolo.classify_rois_batch(rois)  # type: ignore[union-attr]
            for slot, result in zip(roi_slot, yolo_results):
                out[slot] = result

        # Third pass: shape-heuristic fallback for any non-synthetic slot
        # that YOLO declined (or where YOLO is not active at all).
        for i, det in enumerate(detections):
            if out[i] is None and not getattr(det, "synthetic", False):
                out[i] = classify_by_shape(det)

        return out
