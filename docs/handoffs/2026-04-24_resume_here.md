# RESUME — auto-calibrator build

User goal (verbatim): "match leopard, expecting better. don't stop until perfect."
User has given full permission to act. Continue across sessions if context expires.

## State as of last checkpoint

**What works:**
- 32-bit Leopard SDK helper: `eo/leopard_sdk_helper.py` (CLI: `--exposure-ext N --ae off`, `--bits N`, `--sensor-mode N`, `--i2c-write sub:reg:val`, `--i2c-read sub:reg:n`, `--probe-setters`, `--probe-ranges`, `--inspect`)
- `LPCamera.ExposureExt` — only working brightness knob; eliminates AE breathing
- `LPCamera.I2CRegRW` reads work (chip ID 0x34:0x0000 returns [255, 15])
- `manual_exposure_ext: 1000` integrated into `eo/imx568_capture.py` and `eo/eo_manager.py`, on by default in `config/app_config.yaml`
- Pipeline runs end-to-end with the SDK prelude. Static-scene mean span 1.7 over 30s (was: full-saturation drift).

**What's blocked:**
- 24 candidate sensor I2C registers swept (`scripts/eo_find_sensor_registers.py`) — none move output. FX3 firmware overrides direct I2C with its own AE pipeline.
- `LPCamera.Bits` / `SensorMode` setters succeed at the object level but don't change USB output.
- Email sent to `support@leopardimaging.com` (template: `docs/leopard_support_email.md`) requesting register map.

## Calibrator plan (this is what to build now)

1. **`eo/auto_calibrate/metrics.py`** — pure-numpy image metrics:
   - `mean_target_cost(img, target=100.0)` → |mean - target|
   - `saturation_cost(img)` → fraction of pixels >= 254
   - `black_clip_cost(img)` → fraction <= 1
   - `histogram_entropy(img)` → Shannon entropy of 256-bin hist (higher = more dynamic range)
   - `local_snr(img, patch=32)` → median-of-flat-patches mean / std (higher = less grain)
   - `sharpness_laplacian_var(img)` → variance of cv2.Laplacian (higher = sharper)
   - `composite_cost(img, weights)` → weighted sum of normalized terms

2. **`eo/auto_calibrate/pipeline.py`** — parameterized post-processing:
   ```python
   @dataclass
   class CalibParams:
       exposure_ext: int = 1000
       agc_low_pct: float = 0.5
       agc_high_pct: float = 99.5
       gamma: float = 1.0
       bilateral_d: int = 0           # 0 = disabled
       bilateral_sigma_color: float = 25
       bilateral_sigma_space: float = 25
       sharpen_amount: float = 0.0    # 0 = disabled
       sharpen_radius: float = 1.0
   ```
   `apply(raw_y, params) -> processed_uint8`

3. **`eo/auto_calibrate/optimizer.py`** — differential evolution loop (`scipy.optimize.differential_evolution`). For each candidate vector:
   - Set ExposureExt via helper subprocess (ONLY when it changes)
   - Open PyAV, capture N=8 frames, take median
   - Apply non-sensor params from candidate
   - Evaluate cost
   - Save running best to `config/calibration.yaml` after every improvement (so a session interrupt doesn't lose progress)

4. **`eo/auto_calibrate/__main__.py`** — CLI:
   - `python -m eo.auto_calibrate --duration 1200` (20 min)
   - `--target-mean 100`
   - `--reference path/to/leopard_capture.png` (optional reference-matching cost)
   - Saves `config/calibration.yaml` + `eo_snapshots/calibration/before.png` + `after.png`

5. **Wire into seeker** — `eo/eo_processor.py` reads `config/calibration.yaml` on startup and applies AGC/denoise/sharpen params. Existing percentile-AGC code becomes a fallback when no calibration file exists.

## Resume-after-restart contract

Optimizer writes `eo_snapshots/calibration/state.json` after every cost evaluation:
```json
{
  "iter": 42,
  "best_cost": 0.123,
  "best_params": {...},
  "history": [...],
  "started_at": "2026-04-24T22:30:00",
  "last_update": "..."
}
```

A new session resuming finds this file, reads the best params, and either:
- Continues optimization from `best_params` if cost > target threshold
- Stops and reports converged result if cost <= threshold

## Definition of "perfect"

Composite cost reaches a plateau (no improvement > 1% across 50 consecutive iterations).
Final visual: mean within ±5 of target, saturation < 1%, sharpness Laplacian variance > 80% of theoretical max for the optical system, local SNR > 30 dB in flat patches.
