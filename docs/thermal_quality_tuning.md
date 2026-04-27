# Thermal image quality — tuning guide

The thermal panel in the operator GUI now runs a parameterized
post-AGC enhancement chain. Every stage is controlled from
`config/app_config.yaml::thermal` and is independently toggleable.
Detection still operates on the raw 16-bit frame (pre-AGC) so changing
any of the display knobs cannot shift detection statistics.

This document is the operator's reference for what each knob does, what
it costs, and when to flip it. For the code, see
[`thermal/thermal_processor.py`](../thermal/thermal_processor.py).

---

## Pipeline order

```
raw16
  ├─► [detector — sees raw16, never the display]
  └─► dead-pixel median   (raw16 stage)
        └─► AGC percentile stretch (raw16 → u8)
              └─► CLAHE?                (u8)
                    └─► gamma           (u8)
                          └─► bilateral denoise (u8)
                                └─► unsharp mask  (u8)
                                      └─► colormap (u8 → BGR)
                                            └─► display + JPEG
```

`?` = stage is OFF by default; the rest are ON.

---

## Knobs

### `thermal.agc.low_percentile` / `high_percentile`
Default **1.0 / 99.0** (was 2.0 / 98.0 pre-2026-04-27).

Linear stretch from the percentile-clipped raw16 histogram into 0–255.
The legacy 2/98 clipped ~4% of pixels and visibly posterized midtones
on low-contrast outdoor scenes. 1/99 only nips genuine outliers
(sun-edge glints, single-pixel hot defects) while keeping the gradient
the operator reads details from.

Unlike EO (see `eo.agc` notes in `app_config.yaml`), this is a true
bit-depth reduction from a 16-bit raw input, not a contrast pump on
already-quantized data — so it does NOT 6× the noise floor. EO's
"oil-painting" failure mode does not apply here.

### `thermal.agc.colormap`
Default **INFERNO**. Choices: `INFERNO`, `IRONBOW`, `WHITE_HOT`, `JET`,
`MAGMA`. `WHITE_HOT` is true grayscale (R == G == B). On a
low-contrast scene a colormap quantizes the underlying image to a
discrete palette and can look posterized; if that's what the operator
is seeing, try `WHITE_HOT`.

### `thermal.agc.clahe.enabled` (default **off**)
Local-contrast equalization on the AGC'd 8-bit. Useful when the AGC'd
image is technically full-range but visually flat — long-range scenes
at thermal equilibrium where everything is "the same temperature."

⚠ Off by default because on a clean already-contrasty scene CLAHE
amplifies sensor grain. Toggle ON when the scene is genuinely flat and
pre-AGC dynamic range is lacking, NOT when the image just feels "soft."

`clip_limit: 2.0`, `tile_grid: 8` are reasonable defaults for thermal.
Bumping `clip_limit` past ~3 starts to look HDR-tortured.

### `thermal.enhance.dead_pixel_median.enabled` (default **on**)
3×3 median on raw16 before AGC. Boson 640 sensors typically ship with
FFC handling most bad pixels, but a few survive and look like bright
dots in INFERNO. Cheap (~0.3 ms at 640×512), edge-preserving on real
structure (a single pixel is below the kernel's plurality).

### `thermal.enhance.gamma`
Default **0.90**. Convention: `out = (in/255)^gamma * 255`. `1.0` =
identity. `<1` lifts midtones (brighter), `>1` sinks them.

`0.90` is a mild lift — brings the "warm but not hot" pixels (running
engines, daylight road surface) up out of the lower half of the
histogram where they otherwise sit after the percentile stretch.
`0.85` is more aggressive but still safe; below that the image looks
washed out.

### `thermal.enhance.bilateral_denoise.enabled` (default **on**)
Edge-preserving denoise. Default `d: 3, sigma_color: 10, sigma_space: 10`
("polish" — knocks out fine grain without softening vehicle silhouettes
or human outlines).

The initial 2026-04-27 default was `d: 5 / 15 / 15`, which on **live
raw16** input was fine but on the **JPEG'd recording** in the offline
A/B tool was visibly soft (sharpness Laplacian dropped 37%). The
tightened d=3 / σ=10 stays useful on both paths. Larger sigmas blur
real detail; do NOT bump these without comparing against a recording
first.

### `thermal.enhance.unsharp_mask.enabled` (default **on**)
Default `amount: 0.50, radius: 1.0`. Subtle edge-sharpen via
Gaussian-blur subtraction:

```
out = frame + amount * (frame - blur(frame))
```

`amount: 0.50` is sized to counteract the bilateral above and a touch
more — produces a visibly crisper edge on vehicle silhouettes / human
outlines. If the AFTER image looks "etched" (light/dark stripes
parallel to a bright edge) drop to `0.30`; if it still looks soft, bump
to `0.70` and pair with a higher-res zoom to verify the halo is
invisible.

### `thermal.digital_zoom.interpolation`
Default **cubic** (was `linear` hard-coded pre-2026-04-27). Choices:
`linear`, `cubic`, `lanczos4`, `area`. Used when a center-cropped FOV
is upscaled back to display dimensions (mid/narrow zoom presets).

`cubic` is visibly sharper than `linear` at 2-3× zoom for ~negligible
cost. `lanczos4` is marginally sharper still but ~3× the cost; use it
only on a static-platform scenario where 60 Hz isn't critical.

### `gui.thermal_jpeg_quality`
Default **92** (was 80). The colormapped 8-bit thermal image
compresses to a small file even at 92. The bandwidth cost over
localhost is ~+200 KB/s at 30 Hz, imperceptible. Drop back to 85 if
recording hits a write bottleneck on a slow disk.

---

## A/B testing the chain on a recording

```
python scripts/thermal_quality_compare.py --latest
```

Defaults to the newest `recordings/seeker_*.jsonl`. Writes a side-by-
side BEFORE | AFTER MP4 to `recordings/thermal_compare/` and prints a
sharpness / contrast delta summary.

Important caveat: JSONL recordings only archive the **post-AGC** 8-bit
JPEG (see [`recording/encoders.py`](../recording/encoders.py)). The raw
16-bit frame is NOT stored. So this tool can only validate
enhancements that operate on the 8-bit display image — gamma,
bilateral denoise, unsharp mask, CLAHE, colormap re-selection. It
cannot validate AGC percentile re-tuning or dead-pixel median (those
run pre-AGC, on raw16). For those, exercise the synthetic source:

```
python -m thermal.fake_thermal_source
```

— or use the live camera once it's back online.

CLI flags:

* `--path <jsonl>` instead of `--latest`
* `--limit N` — cap at N thermal frames (handy on large recordings)
* `--layout side|stack` — side-by-side (default) or vertical stack
* `--colormap NAME` — override the AFTER colormap
* `--fps N` — override output MP4 fps (default: derive from timestamps)

---

## Reference results (2026-04-27 overnight tuning)

Two recordings, current defaults (`gamma=0.90, bilateral d=3 σ=10/10,
unsharp 0.50/1.0`, CLAHE off, dead-pixel median on, 1/99 percentile):

| Recording                           | Frames | Sharpness Δ | Contrast Δ |
| ----------------------------------- | ------ | ----------- | ---------- |
| `seeker_2026-04-26_21-04-31.jsonl`  | 176    | +20%        | -3%        |
| `seeker_2026-04-26_21-13-01.jsonl`  | 340    | +22%        | -0.4%      |

Sharpness measured as Laplacian variance (high = more high-frequency
detail). Contrast measured as grayscale standard deviation. The small
contrast loss is the gamma 0.9 lift — it brightens dark areas without
changing lights, slightly compressing the global histogram. This is
the visible "warmth lift" the operator wants and is not a degradation.

Compare videos are at `recordings/thermal_compare/*.mp4`.

---

## What we are NOT doing yet (morning follow-ups)

These are deferred until the live camera is back online:

* **Auto-calibrator with a metrics + optimizer pair** like `eo/auto_calibrate/`
  has. Run a differential-evolution sweep over the enhancement params
  to find a per-scene optimum, save to `config/calibration.yaml`.
* **FFC / NUC against a flat plate.** Boson has built-in FFC; we don't
  override it, but a bench calibration on a uniform thermal target
  (warm cup of water at known distance) would let us subtract residual
  fixed-pattern noise.
* **Optional raw16 recording.** Compressed PNG-per-frame at the
  recorder. Big format change — defer until requested. Would let the
  offline A/B tool validate AGC re-tuning without the live camera.
* **Dead-pixel mask from a long average.** A few seconds of staring
  at a uniform thermal target produces an exhaustive defect list; we
  can replace the median with a targeted neighbor-interpolate on
  exactly the bad pixels.

---

## Reverting

Every knob has an `enabled: false` toggle (or, for gamma, set to
`1.0`; for unsharp, set `amount: 0.0`). To revert to pre-2026-04-27
behaviour entirely:

```yaml
thermal:
  agc:
    low_percentile: 2
    high_percentile: 98
    clahe:
      enabled: false
  enhance:
    dead_pixel_median:
      enabled: false
    gamma: 1.0
    bilateral_denoise:
      enabled: false
    unsharp_mask:
      enabled: false
  digital_zoom:
    interpolation: linear
gui:
  thermal_jpeg_quality: 80
```

Restart the app. The pipeline now produces bit-equivalent output to
the legacy code path. The heat detector / classifier / fusion paths
were never touched; only the display path runs the new chain.
