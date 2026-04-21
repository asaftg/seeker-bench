"""EO full-frame human/vehicle classifier.

Thin shim over ``thermal.classifier_hv.HumanVehicleClassifier``. The
thermal h/v classifier already implements exactly what EO needs:
YOLOv8n full-frame detection with a COCO fallback that maps
person→person and car/truck/bus/motorcycle→vehicle. RGB webcam frames
are the *native* training domain of COCO YOLOv8n, so running the
stock weights here gives good results out of the box.

Key design point: we deliberately force the fallback path by pointing
``model_path`` at a file that does NOT exist. ``seeker_thermal_hv.pt``
is fine-tuned on THERMAL imagery — loading it for RGB would regress
accuracy badly. Stock ``yolov8n.pt`` is the correct choice for EO.

Drone is NOT wired here. COCO has no drone class; the airplane class
would false-fire on birds and distant aircraft. Proper EO drone
detection needs its own fine-tune — see TODO at bottom.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from common.logging_setup import get_logger
from thermal.classifier_hv import HumanVehicleClassifier

log = get_logger(__name__)


class EOClassifier:
    """Full-frame person+vehicle detector for RGB webcam frames.

    Parameters
    ----------
    fallback_model_path:
        Stock COCO YOLOv8n weights. Auto-downloaded by ultralytics on
        first use if missing.
    conf_threshold:
        Minimum YOLO confidence to report a detection.
    imgsz:
        Inference image size. 640 is a sensible default for 720p/1080p
        webcams; raising to 960 helps distant targets at the cost of
        ~2x compute.
    """

    def __init__(
        self,
        fallback_model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.40,
        imgsz: int = 640,
    ) -> None:
        # Force COCO fallback by pointing model_path at a guaranteed-
        # nonexistent path. HumanVehicleClassifier handles this cleanly:
        # when the trained path doesn't exist it loads fallback_model_path
        # and applies the stock COCO → seeker-class mapping.
        _forced_missing = Path("models/__eo_force_coco_fallback__.pt")
        self._hv = HumanVehicleClassifier(
            model_path=str(_forced_missing),
            fallback_model_path=fallback_model_path,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
        )

    @property
    def active(self) -> bool:
        return self._hv.active

    def detect(self, bgr_image: np.ndarray) -> list[dict]:
        """Run YOLO on a full BGR frame.

        Returns a list of dicts, one per detection::

            {"bbox": (x, y, w, h), "class": "person"|"vehicle", "conf": float}
        """
        if bgr_image is None or bgr_image.size == 0:
            return []
        return self._hv.detect_full_frame(bgr_image)

    def track(self, bgr_image: np.ndarray) -> list[dict]:
        """Run YOLO+ByteTrack on a full BGR frame.

        Returns the same shape as ``detect()`` plus a stable ``track_id``
        per object. Prefer this over ``detect()`` when the caller needs
        cross-frame identity (which is always, for overlays).
        """
        if bgr_image is None or bgr_image.size == 0:
            return []
        return self._hv.track_full_frame(bgr_image)


# ---------------------------------------------------------------------------
# TODO(ticket-3-followup): EO drone detection.
#
# Skipped this ticket per user decision. Real EO drone detection needs:
#   - Collect RGB drone footage at various ranges + conditions.
#   - Fine-tune yolov8n on a dataset like Det-Fly, Drone-vs-Bird, or
#     Anti-UAV-RGB. Cannot reuse seeker_thermal.pt (IR-only domain).
#   - Add a "drone" class to EOClassifier, or run a second YOLO head.
#   - Tune for false-positive suppression on birds at range — this is
#     the hard part for any EO drone detector.
# ---------------------------------------------------------------------------
