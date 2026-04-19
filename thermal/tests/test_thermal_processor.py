import numpy as np
import pytest

from thermal.thermal_processor import apply_agc, apply_colormap, raw16_to_display


def test_agc_output_is_uint8_with_full_range():
    # Noisy 16-bit frame with a cold corner and a hot corner
    rng = np.random.default_rng(42)
    frame = (rng.integers(4000, 6000, size=(64, 64))).astype(np.uint16)
    frame[0, 0] = 0
    frame[-1, -1] = 65535
    agc = apply_agc(frame)
    assert agc.dtype == np.uint8
    assert agc.shape == frame.shape
    # After percentile clipping the outliers should not dominate,
    # and the normal body of the histogram should span a wide range.
    assert agc.max() == 255
    assert agc.min() == 0


def test_agc_flat_frame_returns_zeros():
    frame = np.full((16, 16), 3000, dtype=np.uint16)
    agc = apply_agc(frame)
    assert agc.dtype == np.uint8
    assert (agc == 0).all()


def test_agc_rejects_wrong_shape():
    with pytest.raises(ValueError):
        apply_agc(np.zeros((4, 4, 3), dtype=np.uint16))


def test_colormap_returns_bgr():
    gray = np.full((32, 32), 128, dtype=np.uint8)
    bgr = apply_colormap(gray, "INFERNO")
    assert bgr.shape == (32, 32, 3)
    assert bgr.dtype == np.uint8


def test_white_hot_stays_grayscale_looking():
    gray = np.full((8, 8), 200, dtype=np.uint8)
    bgr = apply_colormap(gray, "WHITE_HOT")
    # All channels equal for WHITE_HOT (grayscale → BGR)
    assert np.array_equal(bgr[..., 0], bgr[..., 1])
    assert np.array_equal(bgr[..., 1], bgr[..., 2])


def test_raw16_to_display_pipeline():
    rng = np.random.default_rng(0)
    raw = rng.integers(1000, 5000, size=(16, 16)).astype(np.uint16)
    agc, bgr = raw16_to_display(raw)
    assert agc.dtype == np.uint8
    assert bgr.shape == (16, 16, 3)
