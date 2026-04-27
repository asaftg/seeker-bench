# Overnight thermal quality work — 2026-04-27

You said: "do EVERYTHING YOU CAN to improve the sensor quality" while
the sensor was off. Here's what shipped, what to check, and what's left.

## TL;DR

* Built a parameterized post-AGC enhancement chain on the thermal
  display path. **Detection still runs on raw16** — these knobs cannot
  shift detection statistics.
* Defaults that ship enabled: dead-pixel median (raw16), AGC 1/99
  percentiles (was 2/98), gamma 0.90 lift, bilateral denoise (d=3,
  σ=10), unsharp mask (amount=0.50, radius=1), digital-zoom upscale
  switched to cubic, JPEG quality 80→92, canvas image-smoothing
  quality high.
* Each stage is a single YAML toggle. CLAHE is implemented but
  shipped OFF (it amplifies grain on already-clean scenes; turn it on
  per-scene if you want).
* 42 thermal unit tests pass — 19 new ones cover every primitive and
  the full chain.
* Two side-by-side BEFORE | AFTER MP4s waiting for you in
  `recordings/thermal_compare/` (see below). On those, sharpness goes
  up ~20% and contrast holds within 3%.

## What to check before you reach for the keyboard

1. **The compare videos** — these are the offline preview. Open both:

   ```
   recordings/thermal_compare/seeker_2026-04-26_21-04-31_compare.mp4
   recordings/thermal_compare/seeker_2026-04-26_21-13-01_compare.mp4
   ```

   Each frame is `BEFORE  (recorded)` | `AFTER  (post-AGC enhance)`
   with a per-frame metric strip. If AFTER looks worse to you on any
   frame, dial knobs in `config/app_config.yaml::thermal.enhance` and
   re-run:

   ```
   python scripts/thermal_quality_compare.py --latest
   ```

   Caveat: the JSONL recording only stores the post-AGC JPEG, not
   raw16, so the offline tool cannot validate the AGC percentile
   change or the dead-pixel median. Both will only show their
   benefit when you bring the live camera back up.

2. **Run the app live** — once you plug the Boson back in, every
   knob takes effect on restart. No code changes needed; the YAML is
   the only switch surface.

   ```
   START_SEEKER.bat
   ```

   First thing to look at on the thermal panel: edges of vehicles /
   humans should look crisper than yesterday at the same zoom level.
   Single-pixel hot dots (sensor defects) should be gone.

3. **If the live image looks "etched"** (light/dark fringes parallel
   to bright edges), that's the unsharp mask going too hard. Drop
   `thermal.enhance.unsharp_mask.amount` from 0.50 to 0.30.

## Files touched (thermal scope only — nothing else)

* `config/app_config.yaml` — new `thermal.enhance` block, AGC
  percentiles loosened, JPEG quality bumped, zoom interpolation set.
* `thermal/thermal_processor.py` — new pure primitives:
  `apply_dead_pixel_median`, `apply_clahe`, `apply_gamma`,
  `apply_bilateral_denoise`, `apply_unsharp_mask`, plus
  `ThermalEnhanceParams` dataclass + `from_config()` adapter +
  `raw16_to_display_with_params()` driver.
* `thermal/thermal_manager.py` — builds `ThermalEnhanceParams` once,
  drives every frame through it, picks zoom-upscale interpolation
  from YAML.
* `thermal/tests/test_thermal_processor.py` — 19 new tests.
* `gui/static/js/thermal_view.js` — `imageSmoothingQuality = "high"`
  on the thermal canvas (Chrome default is `"low"`).
* `scripts/thermal_quality_compare.py` — new offline A/B tool over
  JSONL recordings.
* `docs/thermal_quality_tuning.md` — full operator reference.
* `OVERNIGHT_RESULTS_THERMAL.md` — this file.

I did NOT touch EO, fusion, gimbal, radar, recording-of-non-thermal,
or any GUI code outside the thermal canvas, per your explicit scope.
The git diff of this work is constrained to the files above.

## Defaults rationale

The EO comments in `app_config.yaml` (lines 273-331) document a hard-
won lesson: aggressive AGC + gamma + CLAHE + unsharp on a noisy
8-bit-from-12-bit FX3 input produced "oil-painting ridge artifacts"
on a low-light scene. I read those notes carefully and chose
defaults that respect the spirit of "honest passthrough":

* **CLAHE off by default** (the most-likely-to-misbehave stage).
* **Gentle bilateral** (d=3, not d=7+).
* **Modest unsharp** (0.50, not 1.0+).

The thermal input differs from EO in two important ways that make
the same stack safer:

1. Boson 640 raw is 16-bit with on-chip NUC — much cleaner than the
   FX3-bridged 8-bit-from-12-bit IMX568 stream EO sees.
2. AGC on thermal is a true bit-depth reduction (16 → 8), not a
   contrast pump on already-quantized data. So the noise-multiplier
   that bit EO does not exist on thermal.

If you want the previous look exactly back, the bottom of
`docs/thermal_quality_tuning.md` has the YAML revert block.

## What's queued for tomorrow if you want it

These need the live camera and weren't safe to do tonight:

* **Thermal auto-calibrator** — `eo/auto_calibrate/` has a metrics +
  optimizer pair that runs differential evolution over post-processing
  params. The same shape would work for thermal; the new
  `ThermalEnhanceParams` dataclass is already designed for it.
* **Dead-pixel mask** — a long average against a flat-temperature
  target gives an exhaustive defect list. Then replace the median
  with a targeted neighbor-interpolate on just the bad pixels (no
  global softening at all).
* **FFC trigger from the GUI** — the Boson SDK supports software-
  trigger FFC. Adding a button to the thermal panel takes ~1 hour.
* **Optional raw16 recording** — compressed PNG-per-frame at the
  recorder, behind a flag. Would unblock offline AGC tuning.

## How to commit progress checkpoints

Pushed in 1 commit so far:

* `6377971` thermal: parameterized image-quality pipeline (dead-pixel,
  gamma, denoise, sharpen, CLAHE)

A second commit will land with the offline tool + docs. Both push to
`session/phase-2-recap`. No history was rewritten; no
unrelated-to-thermal files were staged.
