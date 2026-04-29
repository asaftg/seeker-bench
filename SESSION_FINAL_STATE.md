# Final Session State — Thermal Pipeline Work

End-of-session snapshot for clean handoff. Updated 2026-04-28.

## Live pipeline state (what's RUNNING)

The active YAML + code path is **byte-equivalent to chat-start
baseline** for the visible thermal image. Operator validated this
matches their original pre-work view.

```yaml
# config/app_config.yaml::thermal
agc:
  low_percentile: 2
  high_percentile: 98
  colormap: WHITE_HOT          # grayscale (operator preference)
digital_zoom:
  preset: wide                 # 37.5° HFOV
```

* Boson capture in **AGC8 fallback path** (raw16=False) — `boson_capture.py`
  uses legacy DSHOW property-set order which falls through to 8-bit BGR.
* `from_config()` reads no `mode:` key, defaults to `mode="global"`,
  triggering legacy `apply_agc(low=2, high=98)` byte-for-byte.
* AGC8 fallback path in `thermal_manager._process_and_publish` applies
  the YAML colormap (WHITE_HOT = grayscale-as-BGR, visually identical
  to camera BGR direct).

## Active improvements that DON'T change visible image

These are kept in place because they're either (a) operator-requested,
(b) frame-rate runway for future work, or (c) bug fixes.

| Change | File | Why kept |
|---|---|---|
| Drone classifier batched YOLO inference (Patch 1) | `thermal/drone_classifier.py` | ~30 ms/tick GPU saving. Pre-built path to 30+Hz once EO/WS sender is decoupled. Equivalence verified offline (32/32 ROIs identical). |
| SENSITIVITY slider max 20 → 100 | `gui/static/index.html` | Operator-requested — backend already accepts `threshold_k` up to 100 |
| Thermal Y16 widget on EO panel was never touched | — | n/a |

## Dormant infrastructure (in repo, not active in YAML)

These are **REACHABLE** via YAML edits. None are currently active.
Each is independently documented.

| Component | File | Activation |
|---|---|---|
| ROI percentile AGC | `thermal/thermal_processor.py::apply_roi_agc` | `agc.mode: roi` |
| Operator gates AGC | `thermal/thermal_processor.py::apply_gates_agc` | `agc.mode: gates` + cold/hot |
| CLAHE on raw Y16 | `thermal/thermal_processor.py::apply_clahe_y16` | `agc.mode: clahe_y16` |
| Dead-pixel median | `thermal/thermal_processor.py::apply_dead_pixel_median` | `enhance.dead_pixel_median.enabled: true` |
| Gamma / bilateral / unsharp | `thermal/thermal_processor.py` | `enhance.{gamma,bilateral_denoise,unsharp_mask}` |
| Y16 capture | `thermal/boson_capture.py` | Swap property-set order to WIDTH/HEIGHT → CONVERT_RGB → FOURCC |
| Boson serial control (FFC, gain, DDE, etc.) | `thermal/boson_control.py` + `flirpy` | Import `BosonControl` or `from flirpy.camera.boson import Boson` |
| Optimization sweep harness | `scripts/_thermal_optim_harness.py` | `python scripts/_thermal_optim_launcher.py --pose <tag>` |
| SDK + software cartesian sweep | `scripts/_thermal_sdk_sweep.py` | `python scripts/_thermal_sdk_sweep.py --tag <tag>` |
| Image quality metrics | `scripts/_thermal_metrics.py` | imported by harness |
| 4-up render comparison | `scripts/_phase23_render_modes.py` | `python scripts/_phase23_render_modes.py --npz <path>` |
| Phase capture utility | `scripts/_phase_capture.py` | `python scripts/_phase_capture.py --tag <tag>` |
| Y16 vs AGC8 proof tool | `scripts/_y16_vs_agc8_proof.py` | `python scripts/_y16_vs_agc8_proof.py` |

## Validated facts (won't need to re-discover)

* **Boson on COM3.** flirpy works: `Boson(port="COM3")` → camera SN
  84778, PN `20640A075-6PADK` (FLIR ADK 75°), FPA temp readable,
  `set_gain_mode`, `do_ffc`, `set_averager` all work.
* **gain LOW / gain AUTO produced pure noise** during the indoor
  sweep — possibly a firmware bug or needs disconnect-after-change.
  Not worth chasing without a clear use case.
* **Property-set order matters for Y16 on this laptop's DSHOW**:
  `WIDTH/HEIGHT → CONVERT_RGB → FOURCC` negotiates Y16; the legacy
  order silently falls through to AGC8.
* **CLAHE-Y16 visual win was marginal** (~4% composite at outdoor 37°)
  versus camera AGC8, AND came with slight processing artifacts.
  Operator chose to revert.
* **Camera AGC8 on this Boson is already competent** — most software
  enhancement layers we tried produced trade-offs, not strict wins.
* **Frame-rate bottleneck is the shared WS sender at gui/app.py:350**,
  not the thermal pipeline. Patch 1 saves ~30 ms but the shared
  publish loop bundles thermal+EO into one message. The path to 30+Hz
  is on the EO/WS side: separate WS topics or async publish.

## Cross-session resume

* Persistent task at `.claude/scheduled-tasks/thermal-optim-resume/SKILL.md`
  fires hourly (cron `17 * * * *`).
* Task NOOPs cleanly when no incomplete runs are in `recordings/optim/`.
* To stop hourly check: delete `.claude/scheduled-tasks/thermal-optim-resume/SKILL.md`
  or use `mcp__scheduled-tasks__delete_scheduled_task` from a Claude session.

## Test status

```
$ python -m pytest thermal/tests/ -q
82 passed
```

47 in `test_thermal_processor.py` + 18 in `test_boson_control.py` +
17 pre-existing in other thermal modules. All green.

## Branch / commit state

Active branch: `revert/structural-and-timing-fix`.

Last commits in this work:
```
1809cc5 thermal: colormap WHITE_HOT (grayscale) — operator's actual chat-start view
2c2beae thermal: revert to chat-start baseline (AGC8 fallback, INFERNO, mode=global)
8bf036e thermal: outdoor sweep winner — clahe_y16 tile_grid 8 -> 12  (now reverted)
b092020 thermal: add re-scored leaderboard + FINAL_COMPARE reference to results
8a52107 thermal_metrics: add structure_to_noise_ratio (gates noise out of composite)
ee81bfa thermal: SDK sweep harness + indoor 2-3m results
6ea0dc0 thermal: enable Y16 capture + CLAHE-Y16 AGC mode + slider headroom  (now reverted)
```

## Files NOT touched (other agents' work — left alone)

* `algorithms/track_predictor.py` + `algorithms/tests/test_track_predictor_world_input.py`
* `common/frames.py`
* `fusion/fusion_manager.py`
* `gimbal/gimbal_manager.py`
* `recording/encoders.py`

These remain in their pre-existing modified states. Coordinate with
the tracking/fusion agent before any change.

## Files changed by THIS session

```
config/app_config.yaml              (reverted to chat-start equivalent)
gui/static/index.html               (slider 20→100, kept)
thermal/boson_capture.py            (Y16 fix reverted, comments explain how to re-enable)
thermal/drone_classifier.py         (Patch 1 — batched YOLO, kept)
thermal/thermal_manager.py          (refactored to use ThermalEnhanceParams; default-equivalent to legacy)
thermal/thermal_processor.py        (new primitives added, all dormant under defaults)
thermal/tests/test_thermal_processor.py  (47 tests, 25 new)
thermal/tests/test_boson_control.py      (18 tests, all new)
```

## New files added by THIS session

```
PROPOSED_CHANGES.md                          — running log of in-flight patches
THERMAL_OPTIMIZATION_PLAN.md                 — parameter taxonomy + test strategy
OVERNIGHT_THERMAL.bat                        — Windows one-click overnight launcher
OVERNIGHT_THERMAL_RESUME.md                  — cross-session handoff doc
OVERNIGHT_THERMAL_RESULTS.md                 — indoor sweep results
SESSION_FINAL_STATE.md                       — this file
thermal/boson_control.py                     — pyserial FLIR FFC wrapper
scripts/_drone_classifier_batch_equivalence.py — Patch 1 verification test
scripts/_phase_capture.py                     — single-pose capture utility
scripts/_phase23_render_modes.py              — 4-up render comparison
scripts/_phase2_calibrate.py                  — heat-detector retune calibration
scripts/_scene_aware_agc_demo.py              — synthetic AGC mode demo
scripts/_thermal_metrics.py                   — image quality scoring
scripts/_thermal_optim_harness.py             — sweep orchestrator
scripts/_thermal_optim_launcher.py            — auto-retry wrapper
scripts/_thermal_sdk_sweep.py                 — SDK + software cartesian
scripts/_y16_vs_agc8_proof.py                 — Y16 vs AGC8 capture
recordings/optim/sdk_indoor_v1/               — indoor sweep artifacts (recordings)
recordings/optim/sdk_outdoor_v1/              — outdoor sweep artifacts (recordings)
recordings/optim/dryrun/                      — software-only dry-run results
```

The `recordings/optim/*` directories are not committed (they're large
PNG/MP4/npz). They're available locally for review.
