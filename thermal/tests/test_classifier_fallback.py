"""
Classifier fallback tests.

We explicitly do NOT load YOLO here — they're pure unit tests of
the shape heuristic and the wrapper's fallback logic.
"""
import numpy as np

from common.frames import BBox, TargetClass, ThermalDetection
from thermal.drone_classifier import Classifier, classify_by_shape


def _det(x, y, w, h, area, contrast=500.0):
    return ThermalDetection(
        bbox=BBox(x, y, w, h),
        area_px=area,
        contrast=contrast,
    )


def test_shape_noise_for_tiny_area():
    result = classify_by_shape(_det(0, 0, 1, 1, area=1))
    assert result.target_class == TargetClass.NOISE


def test_shape_hand_for_elongated_blob():
    # 60x20 → aspect 3.0, area 800 → HAND
    result = classify_by_shape(_det(0, 0, 60, 20, area=800))
    assert result.target_class == TargetClass.HAND
    assert result.classifier_used == "shape_heuristic"
    assert 0.5 <= result.confidence <= 1.0


def test_shape_drone_for_compact_small_blob():
    # 10x10 → aspect 1.0, area 80, fill=0.8 → DRONE
    result = classify_by_shape(_det(0, 0, 10, 10, area=80))
    assert result.target_class == TargetClass.DRONE


def test_shape_unknown_for_ambiguous():
    # 50x50 rectangle, area 500 (fill 0.2) — not drone-compact, not elongated
    result = classify_by_shape(_det(0, 0, 50, 50, area=500))
    assert result.target_class == TargetClass.UNKNOWN


def test_classifier_with_yolo_disabled_uses_fallback():
    clf = Classifier(enable_yolo=False)
    assert clf.yolo_active is False

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = [_det(10, 10, 60, 20, area=800), _det(40, 40, 10, 10, area=80)]
    results = clf.classify(img, dets)

    assert len(results) == 2
    assert results[0].target_class == TargetClass.HAND
    assert results[1].target_class == TargetClass.DRONE
    assert all(r.classifier_used == "shape_heuristic" for r in results)


def test_classifier_falls_back_when_yolo_model_missing(tmp_path):
    # Point to non-existent model paths — YOLO tier should decline
    clf = Classifier(
        enable_yolo=True,
        model_path=str(tmp_path / "nope.pt"),
        trained_model_path=str(tmp_path / "also-nope.pt"),
    )
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    dets = [_det(5, 5, 10, 10, area=80)]
    results = clf.classify(img, dets)
    assert len(results) == 1
    # Either falls back because YOLO declined, or YOLO was never
    # loaded; in both cases shape_heuristic owns the result.
    assert results[0].classifier_used == "shape_heuristic"
