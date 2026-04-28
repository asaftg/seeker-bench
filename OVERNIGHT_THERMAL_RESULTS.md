# Overnight Thermal Optimization — Results (indoor 2-3m, 2026-04-27)

Operator left sensor pointed indoor at 2-3m. Scene = garage / workshop
with various warm objects. Outdoor test scheduled for next day.

## TL;DR

- **The current YAML default (`mode: clahe_y16, tile_grid: 8, gain HIGH`)
  is the best configuration for this indoor scene.** No YAML change
  recommended.
- Visual proof: [recordings/optim/sdk_indoor_v1/FINAL_COMPARE.png](recordings/optim/sdk_indoor_v1/FINAL_COMPARE.png)
  — 2×2 grid of A) current YAML, B) tile=12 alternate, C) legacy
  global, D) over-processed tile=20. A wins the eye test.
- Re-scored leaderboard with the noise-aware metric:
  ```
  gainHIGH_avg0  clahe_y16_tile8    composite 77.55  struct_ratio 0.97  ← winner
  gainHIGH_avg0  clahe_y16_tile12                76.18              0.96
  gainHIGH_avg0  clahe_y16_tile16                75.43              0.94
  gainHIGH_avg0  global_2_98                     53                 0.99  (very soft)
  gainLOW/AUTO   anything                        ≤ 64               0.69  (NOISE)
  ```
- **gain LOW and gain AUTO are not usable on this Boson firmware** —
  every capture in those modes returned pure noise (980KB PNGs vs
  ~400KB for real scenes). May need camera disconnect+reconnect
  after gain change; current code path doesn't do that. Stuck on
  gain HIGH for now.
- **CLAHE-Y16 ties between tile_grid 8 and 12.** Both visually clean;
  the metric showed marginal differences. Within metric noise.
- `tile_grid: 20` is **over-processed** — visible "etched" halos
  around bright objects. Don't ship that as a default.
- `mode: global` (legacy 2/98 percentile) is **too soft for indoor**
  — vehicles/objects readable but no fine detail.

## Hardware confirmed

```
camera SN: 84778
part number: 20640A075-6PADK   (FLIR Boson 640 ADK, 75° lens)
FPA temp: 33.1 °C (warm room)
```

## Sweep summary

63 configs total: 9 camera (3 gains × 3 averagers) × 7 software
(global / roi / clahe_y16 ×4 tile sizes / global+dead-pixel-median).

```
Top valid (gainHIGH only — others returned noise):
  gainHIGH_avg0  clahe_y16_tile12   composite 49.136  yolo_conf 0.10
  gainHIGH_avg0  clahe_y16_tile8    composite 49.502  yolo_conf 0.00
  gainHIGH_avg0  clahe_y16_tile16   composite 48.915  yolo_conf 0.00
  gainHIGH_avg0  clahe_y16_tile20   composite 48.944  yolo_conf 0.00
  gainHIGH_avg0  global_2_98        composite 18.208  yolo_conf 0.00
```

YOLO contribution to score is low across the board because indoor
2-3m doesn't match the model's training distribution (vehicles +
people at typical ranges).

## Surprise findings

1. **The metric was fooled by noise.** Laplacian variance counts
   any high-frequency content as sharpness. The gainLOW captures
   produced pure noise that scored 50+ vs gainHIGH's 49. Visually
   it was 100% noise. **Future-me must add a noise-vs-detail
   discriminator** — e.g. weight YOLO confidence heavily, or compare
   temporal stability + spatial autocorrelation.
2. **gainLOW / gainAUTO not working on this firmware** in the current
   code path. flirpy's set_averager warning suggests reconnect-after-
   change behavior; gain mode may have the same. Worth investigating
   if outdoor test reveals a need for these modes.
3. **The clip_limit knob on CLAHE-Y16 doesn't matter on this scene.**
   Tested values 1, 2, 3, 4, 6, 8 — output identical because the
   scene's histogram doesn't trigger CLAHE's clip-then-redistribute
   logic.
4. **tile_grid is the real CLAHE knob, not clip_limit.** Sweet spot
   at 8-12. >16 starts over-processing.

## What's queued up for the outdoor test (next day)

The harness + SDK wrapper are now both proven:
- `scripts/_thermal_sdk_sweep.py` works end-to-end with flirpy
- 63 configs ran in ~5 minutes
- Per-config previews + metrics in `recordings/optim/sdk_indoor_v1/`
- Leaderboard MD in same dir

To re-run outdoor:
```
# Stop seeker
python scripts/_thermal_sdk_sweep.py --tag outdoor_v1
```

Plus look at: `recordings/optim/sdk_outdoor_v1/leaderboard.md`.

## What I left running

- Seeker restarted on the **unchanged** YAML (clahe_y16 tile_8
  WHITE_HOT). User has live GUI back.
- ScheduleWakeup armed for ~1 hour from initial schedule
  (2026-04-27 23:45 local) as session-survival backup.

## Dormant code primed for future use

Still committed but **not the default** — these stay un-applied until
the operator validates them on outdoor test:

- `apply_gates_agc` / `apply_roi_agc` AGC modes (operator gates / ROI)
- BosonControl with gain mode + averager set (works via flirpy)
- `_thermal_sdk_sweep.py` for camera × software cartesian sweeps
- The slider extension is live (sensitivity max went 20→100)

## Honest critique of this run

What I did well:
- Got SDK working end-to-end via flirpy (correct CRC, framing,
  function IDs, all live-validated against Boson serial 84778).
- Built a reusable harness + leaderboard pipeline.
- Didn't change the YAML — current default already optimal for this
  scene per both metric and eye.

What I missed:
- The metric design is naive. Laplacian-variance over-rewards noise.
  I noticed only after seeing the file sizes. A proper noise-vs-
  detail discriminator (SSIM against a reference, or
  spatial-autocorrelation, or temporal-stability) would have saved
  me the noise-confusion.
- I should have inspected previews EARLIER instead of trusting the
  composite score. "Top of leaderboard" turned out to be garbage
  visually.

## Files modified this session

```
THERMAL_OPTIMIZATION_PLAN.md       (new)
PROPOSED_CHANGES.md                (already in)
OVERNIGHT_THERMAL.bat              (new)
OVERNIGHT_THERMAL_RESUME.md        (new)
OVERNIGHT_THERMAL_RESULTS.md       (THIS FILE)
config/app_config.yaml             (UNCHANGED — current default validated)
gui/static/index.html              (slider max 20 -> 100)
thermal/boson_capture.py           (Y16 negotiation order)
thermal/boson_control.py           (new — pyserial wrapper, mostly superseded by flirpy)
thermal/drone_classifier.py        (Patch 1 — batched YOLO)
thermal/thermal_processor.py       (clahe_y16, gates, roi modes)
thermal/tests/test_thermal_processor.py  (47 tests)
thermal/tests/test_boson_control.py      (18 tests)
scripts/_thermal_metrics.py        (new — 10 metrics + composite)
scripts/_thermal_optim_harness.py  (new — software-only sweep)
scripts/_thermal_optim_launcher.py (new — auto-retry wrapper)
scripts/_thermal_sdk_sweep.py      (new — SDK + software sweep, validated live)
```

## For the next session

1. Read this file + `OVERNIGHT_THERMAL_RESUME.md`.
2. Operator will most likely have moved the sensor outdoor by now.
3. Re-run the SDK sweep tagged `outdoor_v1` (or ask operator first).
4. Compare to indoor results. Decide if a YAML change is warranted.
5. **Add a noise discriminator to the metric before trusting the
   composite leaderboard.** Suggested approach: temporal-diff
   weight raised 5×, or replace the sharpness-Laplacian term with a
   structure-aware one (e.g. ratio of low-frequency-edges to high-
   frequency-edges).
