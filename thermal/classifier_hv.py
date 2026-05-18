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
        imgsz: int = 640,
        per_class_conf: Optional[Dict[str, float]] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        # Optional per-class conf override applied AFTER predict.
        # YOLO doesn't natively support per-class thresholds, so we
        # request inference at min(conf_threshold) and then filter.
        # Example: {"person": 0.40, "vehicle": 0.40, "drone": 0.20}.
        self.per_class_conf: Dict[str, float] = dict(per_class_conf or {})
        # Lower the inference floor to the minimum of any class
        # threshold so the post-filter has something to filter.
        if self.per_class_conf:
            self.conf_threshold = min(
                self.conf_threshold, min(self.per_class_conf.values())
            )
        # Inference image size. Training was imgsz=640; running predict at 960
        # upscales the input so small/distant targets get more pixels.
        # Typical on A4000: 8ms @ 640 vs 14ms @ 960 — still ~70 Hz.
        self.imgsz = int(imgsz)
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
    def detect_full_frame(self, image: np.ndarray) -> list[Dict[str, object]]:
        """Run YOLO on the entire frame and return ALL detections above threshold.

        This is the primary detection path for humans and vehicles — unlike
        ``classify()`` which takes a crop from the heat blob detector, this
        method sees the whole image so it can find people/cars that produce
        little thermal contrast (e.g. a person in a warm room, a cold car).

        Parameters
        ----------
        image:
            BGR uint8 display frame (typically the AGC-colormapped thermal image).

        Returns
        -------
        List of dicts, one per detection::

            {"bbox": (x, y, w, h), "class": str, "conf": float}

        ``bbox`` is in pixel coordinates of ``image``.  Empty list when the
        model is not loaded or nothing exceeds the confidence threshold.
        """
        if self._model is None or image is None or image.size == 0:
            return []
        try:
            results = self._model.predict(image, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)
        except Exception as e:
            log.warning("HV full-frame inference failed: %s", e)
            return []
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        out: list[Dict[str, object]] = []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names or {}
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            if self._is_finetuned:
                class_name = names.get(int(cls_id), "unknown")
            else:
                class_name = self._coco_map.get(int(cls_id))
                if class_name is None:
                    continue
            # Per-class conf gate (post-filter — YOLO can't do this natively)
            cls_floor = self.per_class_conf.get(class_name)
            if cls_floor is not None and float(conf) < cls_floor:
                continue
            x = int(max(0, x1))
            y = int(max(0, y1))
            w = int(max(1, x2 - x1))
            h = int(max(1, y2 - y1))
            out.append({"bbox": (x, y, w, h), "class": class_name, "conf": float(conf)})
        return out

    # ------------------------------------------------------------------
    def track_full_frame(self, image: np.ndarray) -> list[Dict[str, object]]:
        """Run YOLO **with ByteTrack** on the full frame.

        Same return contract as ``detect_full_frame`` plus a stable
        ``track_id`` per object. ByteTrack lives inside ultralytics
        (``model.track(persist=True, tracker='bytetrack.yaml')``) and
        does two things we previously hand-rolled badly:

        - **Kalman motion model** — coasts each track between YOLO hits
          so fast camera pans / brief classifier misses don't spawn a
          new ID every frame.
        - **Low-confidence association** — detections below ``conf`` are
          still matched to existing tracks (but not used to birth new
          ones). A motion-blurred car that drops from 0.8 to 0.3 conf
          keeps its ID instead of flickering off.

        If tracking fails (e.g. ``lap`` package missing, model type
        incompatible), falls back to ``detect_full_frame`` and emits
        ``track_id=-1`` so callers can still render bboxes — they just
        won't have stable IDs that frame.

        See "Future work" in README for BoT-SORT (camera-motion
        compensated) upgrade path.
        """
        if self._model is None or image is None or image.size == 0:
            return []
        try:
            results = self._model.track(
                image,
                persist=True,                  # keep tracker state across calls
                tracker="bytetrack.yaml",      # shipped by ultralytics
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as e:
            # Most common cause: `lap` package missing or an older
            # ultralytics build. Fall back so the pipeline still runs.
            log.warning("HV track() failed (%s) — falling back to predict()", e)
            return [
                {**d, "track_id": -1}
                for d in self.detect_full_frame(image)
            ]
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        # track.id is None on the very first frame before ByteTrack
        # assigns IDs. Treat those as "unconfirmed" with id=-1 so the
        # downstream tracker can filter by track_id != -1 if it wants.
        if r.boxes.id is None:
            ids = [-1] * len(xyxy)
        else:
            ids = r.boxes.id.cpu().numpy().astype(int).tolist()
        names = r.names or {}

        out: list[Dict[str, object]] = []
        for (x1, y1, x2, y2), conf, cls_id, tid in zip(xyxy, confs, clss, ids):
            if self._is_finetuned:
                class_name = names.get(int(cls_id), "unknown")
            else:
                class_name = self._coco_map.get(int(cls_id))
                if class_name is None:
                    continue
            # Per-class conf gate. We allow ByteTrack to use ALL detections
            # for association (it does its own low-conf-association via
            # high/low track thresholds), but only emit a det to the
            # caller if it clears the per-class floor. This means a
            # blurred-into-noise car gets associated to its track for
            # ID continuity, but we don't render a low-conf box for it.
            cls_floor = self.per_class_conf.get(class_name)
            if cls_floor is not None and float(conf) < cls_floor:
                continue
            x = int(max(0, x1))
            y = int(max(0, y1))
            w = int(max(1, x2 - x1))
            h = int(max(1, y2 - y1))
            out.append({
                "bbox": (x, y, w, h),
                "class": class_name,
                "conf": float(conf),
                "track_id": int(tid),
            })
        return out

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
