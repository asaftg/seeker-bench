"""Auto-calibrator for the EO pipeline.

The IMX568 + FX3 bridge gives us exactly one sensor-side knob that
actually affects the captured stream (LPCamera.ExposureExt). Everything
else worth tuning lives in software:

    AGC (percentile stretch low/high, gamma)
    Denoise (bilateral, NLM)
    Sharpen (unsharp mask amount + radius)
    Tone curve (CLAHE clip + tile)

This package finds the parameter vector that maximizes a composite
image-quality cost function — local SNR, sharpness, dynamic-range
coverage, target-mean match — and optionally matches a reference
image (Leopard CameraTool capture).

Entry point: ``python -m eo.auto_calibrate --duration 1200``
"""
