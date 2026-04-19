import numpy as np
import pytest

from thermal.digital_zoom import (
    PRESETS,
    apply_preset,
    center_crop,
    crop_fraction,
)


def test_presets_registered():
    assert "full" in PRESETS
    assert PRESETS["full"].hfov_deg == 75.0
    assert PRESETS["narrow"].hfov_deg == 12.5


def test_crop_fraction_full_is_one():
    assert crop_fraction(75.0, 75.0) == 1.0


def test_crop_fraction_half_angle():
    # tan(18.75)/tan(37.5) is well below 1
    frac = crop_fraction(75.0, 37.5)
    assert 0 < frac < 1
    assert frac == pytest.approx(np.tan(np.radians(18.75)) / np.tan(np.radians(37.5)))


def test_crop_fraction_rejects_zero():
    with pytest.raises(ValueError):
        crop_fraction(75.0, 0)


def test_center_crop_upscales_back_to_original():
    frame = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    out = center_crop(frame, full_hfov_deg=75.0, target_hfov_deg=12.5)
    assert out.shape == frame.shape  # upscaled back


def test_center_crop_no_upscale_is_smaller():
    frame = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    out = center_crop(frame, full_hfov_deg=75.0, target_hfov_deg=12.5,
                      upscale_to_original=False)
    assert out.shape[0] < 64 and out.shape[1] < 64


def test_apply_preset_unknown_raises():
    frame = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(KeyError):
        apply_preset(frame, 75.0, "bogus")


def test_apply_preset_full_is_identity():
    frame = np.arange(64, dtype=np.uint16).reshape(8, 8)
    out = apply_preset(frame, 75.0, "full")
    assert np.array_equal(out, frame)
