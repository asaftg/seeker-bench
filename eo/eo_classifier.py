"""EO full-frame human/vehicle (and drone) classifier.

Thin shim over ``thermal.classifier_hv.HumanVehicleClassifier``. We pick
the best available weight in this order:

  1. ``models/seeker_eo_v3.pt``  — close-targets fine-tune of v2.
     Same dataset, same 3 classes, but trained with aggressive
     `scale=0.9`, `hsv_s=0.95`, `mosaic=0.5` augmentations and
     `imgsz=960` to fix v2's blind spot on close-range vehicles
     viewed through the NIR-pass 35 mm lens (verified failure
     2026-04-25: foreground white Camry at ~10 m completely missed
     while a 30-px background car was confidently detected). v2
     remains on disk as the known-good rollback.
  2. ``models/seeker_eo_v2.pt``  — fine-tuned overnight on
     LLVIP-visible (low-light city pedestrians) + FLIR-ADAS RGB
     (driving vehicles + people) + seeker_hv. 3 classes:
     0=person, 1=vehicle, 2=drone. mAP50≈0.82 / mAP50-95≈0.55 on the
     mixed-source val set as of 2026-04-25.
  3. ``models/yolov8n.pt`` (stock COCO) — used if neither seeker_eo_*
     model is on disk. HumanVehicleClassifier remaps person→person
     and car/truck/bus/motorcycle→vehicle. No drone class in COCO so
     the drone label simply never fires under this fallback.

Why we *used* to force COCO: an earlier fine-tune was thermal-only
(``seeker_thermal_hv.pt``), and pointing EO at it regressed badly.
seeker_eo_v* are RGB-domain and trained for this exact use-case, so
they are now the preferred paths.

To roll back: delete or rename ``models/seeker_eo_v3.pt`` — the v2
fallback below picks up automatically without code changes.
"""
from __future__ import annotations

from pathlib import Path

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
        model_path: str | None = None,
        fallback_model_path: str = "models/yolov8n.pt",
        conf_threshold: float = 0.40,
        imgsz: int = 640,
    ) -> None:
        # Priority: v3 (close-targets fine-tune) > v2 (long-range only)
        # > COCO. We resolve the chain HERE so callers don't have to
        # know about versioning. Pass model_path explicitly to override.
        if model_path is None:
            v3 = Path("models/seeker_eo_v3.pt")
            v2 = Path("models/seeker_eo_v2.pt")
            if v3.exists():
                model_path = str(v3)
                log.info("EOClassifier: using seeker_eo_v3.pt (close-targets fine-tune)")
            elif v2.exists():
                model_path = str(v2)
                log.info("EOClassifier: using seeker_eo_v2.pt (v3 not present)")
            else:
                model_path = str(v2)  # let HV's missing-file fallback fire
        # HumanVehicleClassifier checks `model_path` first; if missing it
        # silently loads `fallback_model_path` with the COCO remap. So this
        # gracefully degrades whether or not seeker_eo_v*.pt is on disk.
        self._hv = HumanVehicleClassifier(
            model_path=model_path,
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
# EO drone class status (2026-04-25):
#   seeker_eo_v2.pt now ships a `drone` class (per-class mAP50-95 ≈ 0.54)
#   trained primarily on the seeker_hv split's drone annotations. Real-world
#   FP rate on birds at range is unknown and likely high — this val score
#   reflects in-distribution test images, not field operation.
#
# If field testing shows excessive bird/aircraft false alarms, the right
# fixes (in increasing cost):
#   1. Raise EO conf_threshold for the `drone` class only (post-filter).
#   2. Add a hard-negative bird/aircraft set to the next training round.
#   3. Two-stage: small classifier head over the EO YOLO crop to confirm
#      "drone vs bird vs plane vs UFO" before reporting `drone`.
# ---------------------------------------------------------------------------
