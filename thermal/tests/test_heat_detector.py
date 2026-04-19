import numpy as np
import pytest

from thermal.heat_detector import HeatDetector, HeatDetectorConfig


def _cold_background(h: int = 96, w: int = 128, level: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(0)
    # Cold "sky" — low mean, low noise
    return (rng.normal(loc=level, scale=15.0, size=(h, w))
              .clip(0, 65535)
              .astype(np.uint16))


def _inject_hot_blob(frame: np.ndarray, cx: int, cy: int, radius: int, delta: int = 8000) -> None:
    import cv2
    cv2.circle(frame, (cx, cy), radius, int(frame.mean() + delta), -1)


def test_detects_single_large_blob():
    frame = _cold_background()
    _inject_hot_blob(frame, cx=60, cy=50, radius=6)
    det = HeatDetector().detect(frame)
    assert len(det) == 1
    b = det[0].bbox
    # Blob center should be inside the returned bbox
    assert b.x <= 60 <= b.x + b.w
    assert b.y <= 50 <= b.y + b.h
    assert det[0].contrast > 1000


def test_detects_multiple_blobs():
    frame = _cold_background()
    _inject_hot_blob(frame, 20, 30, 4)
    _inject_hot_blob(frame, 80, 70, 5)
    _inject_hot_blob(frame, 110, 20, 3)
    det = HeatDetector().detect(frame)
    assert len(det) == 3


def test_ignores_below_min_area():
    frame = _cold_background()
    _inject_hot_blob(frame, 50, 50, 1)  # 1-px radius → very small
    det = HeatDetector(HeatDetectorConfig(min_blob_area_px=50)).detect(frame)
    assert det == []


def test_empty_on_flat_cold_frame():
    frame = _cold_background()
    det = HeatDetector().detect(frame)
    # Pure noise should not generate many spurious detections
    assert len(det) <= 1


def test_respects_max_detections():
    frame = _cold_background(h=200, w=200)
    # Inject a grid of blobs
    for x in range(20, 200, 20):
        for y in range(20, 200, 20):
            _inject_hot_blob(frame, x, y, 3)
    det = HeatDetector(HeatDetectorConfig(max_detections=5)).detect(frame)
    assert len(det) <= 5


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        HeatDetector().detect(np.zeros((8, 8, 3), dtype=np.uint16))
