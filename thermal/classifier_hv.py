"""
Human + Vehicle YOLO classifier for thermal imagery.

Loads ``models/seeker_thermal_hv.pt`` (fine-tuned) when it exists.
Falls back gracefully to stock ``yolov8n.pt`` COCO weights with
automatic class remapping:

    COCO 0  (person)     → "person"
    COCO 2  (car)        → "vehicle"
    COCO 3  (motorcycle) → "vehicle"
    COCO 5  (bus)        → "vehicle"
    COCO 7  (truck)      → "vehicle"

This means the classifier produces useful labels immediately even before
fine-tuning on thermal data — COCO-trained YOLOv8n handles person/vehicle
detection well on AGC-colormapped thermal images.

Contract:
    classify(roi_image) -> {"class": str, "conf": float} | None

Returns None when no detection exceeds the confidence threshold, so
callers can treat None as "this ROI is unclassified" without special-casing.

Designed to run alongside (not replace) the drone classifier:
    - Drone classifier → "drone" labels via seeker_thermal.pt
    - This classifier  → "person" / "vehicle" labels
    - thermal_manager  → merges by highest confidence; tie-breaks to drone

Standalone smoke-test::

    python -m thermal.classifier_hv
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default COCO → seeker-class mapping used when no fine-tuned model is present
# ---------------------------------------------------------------------------
_COCO_FALLBACK: Dict[int, str] = {
    0: "person",      # person
    2: "vehicle",     # car
    3: "vehicle",     # motorcycle
    5: "vehicle",     # bus
    7: "vehicle",     # truck
}


class HumanVehicleClassifier:
    """Run YOLO h/v classification on a single ROI image.

    Parameters
    ----------
    model_path:
        Path to the fine-tuned thermal h/v model.  When the file does not
        exist the class falls back to stock ``yolov8n.pt`` with the COCO
        mapping above — giving immediate person/vehicle detection without
        requiring a fine-tuned model.
    fallback_model_path:
        Stock COCO model used when ``model_path`` is absent.
    conf_threshold:
        Minimum YOLO confidence to report a detection.
    classes:
        Expected class names.  Must match the label set of the loaded model
        when using a fine-tuned model.  Ignored in COCO-fallback mode
        (mapping is hard-coded to the COCO IDs above).
    """

    def __init__(
        self,
        model_path: str = "models/seeker_thermal_hv.pt",
        fallback_model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.40,
        classes: Optional[list] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self._model = None
        self._is_finetuned = False
        self._coco_map: Dict[int, str] = {}
        self._model_path_used: Optional[str] = None

        trained = Path(model_path)
        fallback = Path(fallback_model_path)

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            log.warning("ultralytics not importable — HumanVehicleClassifier disabled: %s", e)
            return

        if trained.exists():
            try:
                self._model = YOLO(str(trained))
                self._is_finetuned = True
                self._model_path_used = str(trained)
                log.info("HV classifier loaded fine-tuned model from %s", trained)
            except Exception as e:
                log.warning("Failed to load fine-tuned h/v model (%s) — trying COCO fallback", e)

        if self._model is None:
            # No fine-tuned model yet — fall back to COCO stock weights
            if fallback.exists():
                target = fallback
            else:
                # ultralytics will auto-download yolov8n.pt on first use
                target = Path("yolov8n.pt")
            try:
                self._model = YOLO(str(target))
                self._coco_map = dict(_COCO_FALLBACK)
                self._model_path_used = str(target)
                log.info(
                    "HV classifier using COCO fallback (%s). "
                    "Run scripts/train_yolo_hv.py to create seeker_thermal_hv.pt "
                    "for better thermal accuracy.",
                    target,
                )
            except Exception as e:
                log.warning("HV classifier: COCO fallback also failed: %s", e)
                self._model = None

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """True when a model is loaded and ready."""
        return self._model is not None

    # ------------------------------------------------------------------
    def classify(self, roi_image: np.ndarray) -> Optional[Dict[str, object]]:
        """Run inference on a single BGR ROI image.

        Parameters
        ----------
        roi_image:
            BGR uint8 array, any size (YOLO resizes internally).

        Returns
        -------
        ``{"class": str, "conf": float}`` for the highest-scoring detection,
        or ``None`` when nothing exceeds ``conf_threshold``.
        """
        if self._model is None:
            return None
        if roi_image is None or roi_image.size == 0:
            return None

        try:
            results = self._model.predict(
                roi_image,
                conf=self.conf_threshold,
                verbose=False,
            )
        except Exception as e:
            log.warning("HV classifier inference failed: %s", e)
            return None

        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None

        # Highest-confidence box
        best_idx = int(r.boxes.conf.argmax())
        best = r.boxes[best_idx]
        cls_id = int(best.cls.item())
        conf = float(best.conf.item())

        # Resolve class name
        if self._is_finetuned:
            # Fine-tuned model: use model's own names dict
            class_name = (r.names or {}).get(cls_id, "unknown")
        else:
            # COCO fallback: remap to our two-class scheme
            class_name = self._coco_map.get(cls_id)
            if class_name is None:
                return None  # COCO class not in our mapping → ignore

        return {"class": class_name, "conf": conf}


# ---------------------------------------------------------------------------
# Standalone smoke test: python -m thermal.classifier_hv
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("HumanVehicleClassifier standalone test")
    clf = HumanVehicleClassifier()
    print(f"  active: {clf.active}")
    print(f"  model : {clf._model_path_used}")
    print(f"  fine-tuned: {clf._is_finetuned}")

    if clf.active:
        # Blank frame (should return None or low confidence)
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        result = clf.classify(blank)
        print(f"  blank frame  → {result}")

        # Warm-blob frame (simulate a grey person-shaped region)
        warm = np.full((128, 64, 3), 80, dtype=np.uint8)
        result = clf.classify(warm)
        print(f"  warm frame   → {result}")

    print("Done.")
    sys.exit(0)
