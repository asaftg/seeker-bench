import numpy as np
import pytest

from thermal.thermal_processor import (
    ThermalEnhanceParams,
    apply_agc,
    apply_bilateral_denoise,
    apply_clahe,
    apply_colormap,
    apply_dead_pixel_median,
    apply_gamma,
    apply_unsharp_mask,
    enhance_post_agc,
    from_config,
    raw16_to_display,
    raw16_to_display_with_params,
)


# ───────────────────────── existing AGC / colormap ──────────────

def test_agc_output_is_uint8_with_full_range():
    rng = np.random.default_rng(42)
    frame = (rng.integers(4000, 6000, size=(64, 64))).astype(np.uint16)
    frame[0, 0] = 0
    frame[-1, -1] = 65535
    agc = apply_agc(frame)
    assert agc.dtype == np.uint8
    assert agc.shape == frame.shape
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
    assert np.array_equal(bgr[..., 0], bgr[..., 1])
    assert np.array_equal(bgr[..., 1], bgr[..., 2])


def test_raw16_to_display_pipeline():
    rng = np.random.default_rng(0)
    raw = rng.integers(1000, 5000, size=(16, 16)).astype(np.uint16)
    agc, bgr = raw16_to_display(raw)
    assert agc.dtype == np.uint8
    assert bgr.shape == (16, 16, 3)


# ───────────────────────── dead-pixel median ────────────────────

def test_dead_pixel_median_removes_stuck_pixels():
    rng = np.random.default_rng(7)
    frame = rng.integers(2900, 3100, size=(64, 64)).astype(np.uint16)
    # Inject single-pixel hot defects far above the local mean.
    frame[10, 10] = 60000
    frame[30, 40] = 65535
    frame[50, 5] = 0  # stuck-cold pixel too
    cleaned = apply_dead_pixel_median(frame, ksize=3)
    # Defect pixels should be pulled back to the local distribution
    # (well below 60k, well above 0 for the cold one).
    assert cleaned[10, 10] < 5000
    assert cleaned[30, 40] < 5000
    assert cleaned[50, 5] > 1000
    # Bulk frame-mean should be roughly preserved (median is unbiased
    # on symmetric noise).
    assert abs(int(cleaned.mean()) - int(frame.mean())) < 100
    assert cleaned.dtype == np.uint16
    assert cleaned.shape == frame.shape


def test_dead_pixel_median_rejects_bad_kernel():
    f = np.zeros((4, 4), dtype=np.uint16)
    with pytest.raises(ValueError):
        apply_dead_pixel_median(f, ksize=2)  # even
    with pytest.raises(ValueError):
        apply_dead_pixel_median(f, ksize=1)  # too small


def test_dead_pixel_median_rejects_wrong_dtype():
    f = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_dead_pixel_median(f)


# ───────────────────────── CLAHE ────────────────────────────────

def test_clahe_preserves_shape_and_dtype():
    rng = np.random.default_rng(1)
    f = rng.integers(40, 200, size=(64, 64), dtype=np.uint8)
    out = apply_clahe(f, clip_limit=2.0, tile_grid=8)
    assert out.shape == f.shape
    assert out.dtype == np.uint8


def test_clahe_expands_contrast_on_low_contrast_input():
    # All pixels in a narrow band — CLAHE should widen the histogram.
    rng = np.random.default_rng(2)
    f = rng.integers(120, 140, size=(128, 128), dtype=np.uint8)
    out = apply_clahe(f, clip_limit=4.0, tile_grid=8)
    # Standard deviation should grow (more contrast).
    assert out.std() > f.std()


def test_clahe_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        apply_clahe(np.zeros((8, 8), dtype=np.uint16))


# ───────────────────────── gamma ────────────────────────────────

def test_gamma_one_is_identity():
    rng = np.random.default_rng(3)
    f = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    out = apply_gamma(f, gamma=1.0)
    assert np.array_equal(out, f)


def test_gamma_below_one_brightens_midtones():
    f = np.full((8, 8), 128, dtype=np.uint8)
    out = apply_gamma(f, gamma=0.85)
    # Midtone 128 with gamma 0.85 should map higher (lift).
    assert out[0, 0] > 128


def test_gamma_above_one_darkens_midtones():
    f = np.full((8, 8), 128, dtype=np.uint8)
    out = apply_gamma(f, gamma=1.4)
    assert out[0, 0] < 128


def test_gamma_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        apply_gamma(np.zeros((4, 4), dtype=np.uint16))


# ───────────────────────── bilateral ────────────────────────────

def test_bilateral_preserves_dtype_and_shape():
    rng = np.random.default_rng(4)
    f = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    out = apply_bilateral_denoise(f, d=5, sigma_color=15.0, sigma_space=15.0)
    assert out.shape == f.shape
    assert out.dtype == np.uint8


def test_bilateral_reduces_grain():
    # Noisy frame around a constant — denoised std should drop.
    rng = np.random.default_rng(5)
    f = (128 + rng.normal(0, 12, size=(128, 128))).clip(0, 255).astype(np.uint8)
    out = apply_bilateral_denoise(f, d=7, sigma_color=25.0, sigma_space=25.0)
    assert out.std() < f.std()


# ───────────────────────── unsharp mask ─────────────────────────

def test_unsharp_zero_amount_is_identity():
    rng = np.random.default_rng(6)
    f = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    out = apply_unsharp_mask(f, amount=0.0, radius=1.0)
    assert np.array_equal(out, f)


def test_unsharp_increases_edge_response():
    # Single-edge frame: 0 on left half, 200 on right half.
    f = np.zeros((32, 32), dtype=np.uint8)
    f[:, 16:] = 200
    out = apply_unsharp_mask(f, amount=0.6, radius=1.0)
    # Pixel adjacent to the step on the bright side should overshoot
    # the original 200 (or at minimum not undershoot it).
    assert int(out[16, 16]) >= int(f[16, 16])
    assert int(out[16, 17]) >= int(f[16, 17])


# ───────────────────────── ThermalEnhanceParams + from_config ───

def test_default_params_are_legacy_passthrough():
    """Unset config → params with every enhancement OFF, gamma 1.0 etc.

    `raw16_to_display_with_params` with defaults must equal the legacy
    `raw16_to_display` output bit-for-bit.
    """
    rng = np.random.default_rng(8)
    raw = rng.integers(2000, 6000, size=(32, 32)).astype(np.uint16)
    p = ThermalEnhanceParams()  # defaults
    enh, bgr_new = raw16_to_display_with_params(raw, p)
    agc_legacy, bgr_legacy = raw16_to_display(raw)
    assert np.array_equal(enh, agc_legacy)
    assert np.array_equal(bgr_new, bgr_legacy)


def test_from_config_handles_missing_sections():
    p = from_config({})
    assert p == ThermalEnhanceParams()  # all defaults
    p2 = from_config(None)
    assert p2 == ThermalEnhanceParams()


def test_from_config_reads_full_block():
    cfg = {
        "agc": {
            "low_percentile": 1.0,
            "high_percentile": 99.0,
            "colormap": "WHITE_HOT",
            "clahe": {"enabled": True, "clip_limit": 3.0, "tile_grid": 16},
        },
        "enhance": {
            "dead_pixel_median": {"enabled": True, "ksize": 5},
            "gamma": 0.85,
            "bilateral_denoise": {
                "enabled": True, "d": 7, "sigma_color": 22.0, "sigma_space": 18.0,
            },
            "unsharp_mask": {"enabled": True, "amount": 0.5, "radius": 1.5},
        },
    }
    p = from_config(cfg)
    assert p.low_percentile == 1.0
    assert p.high_percentile == 99.0
    assert p.colormap == "WHITE_HOT"
    assert p.clahe_enabled is True
    assert p.clahe_clip_limit == 3.0
    assert p.clahe_tile_grid == 16
    assert p.dead_pixel_median_enabled is True
    assert p.dead_pixel_median_ksize == 5
    assert p.gamma == 0.85
    assert p.bilateral_enabled is True
    assert p.bilateral_d == 7
    assert p.bilateral_sigma_color == 22.0
    assert p.bilateral_sigma_space == 18.0
    assert p.unsharp_enabled is True
    assert p.unsharp_amount == 0.5
    assert p.unsharp_radius == 1.5


def test_enhance_post_agc_is_no_op_on_default_params():
    rng = np.random.default_rng(9)
    f = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    out = enhance_post_agc(f, ThermalEnhanceParams())
    assert np.array_equal(out, f)


def test_full_chain_runs_end_to_end_with_all_enabled():
    rng = np.random.default_rng(10)
    raw = rng.integers(2000, 6000, size=(64, 64)).astype(np.uint16)
    raw[5, 5] = 65535  # dead pixel
    p = ThermalEnhanceParams(
        low_percentile=1.0,
        high_percentile=99.0,
        colormap="INFERNO",
        dead_pixel_median_enabled=True,
        dead_pixel_median_ksize=3,
        clahe_enabled=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid=8,
        gamma=0.9,
        bilateral_enabled=True,
        bilateral_d=5,
        bilateral_sigma_color=15.0,
        bilateral_sigma_space=15.0,
        unsharp_enabled=True,
        unsharp_amount=0.3,
        unsharp_radius=1.0,
    )
    enh, bgr = raw16_to_display_with_params(raw, p)
    assert enh.dtype == np.uint8
    assert enh.shape == (64, 64)
    assert bgr.shape == (64, 64, 3)
    assert bgr.dtype == np.uint8
    assert not np.isnan(enh).any()
