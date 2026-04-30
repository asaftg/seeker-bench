# Thermal Optimization Plan

How we systematically find the BEST thermal-pipeline configuration
across the full parameter space, including the Boson hardware controls
exposed via serial.

## The parameter space

**Camera-side (Boson, via serial command interface)** — currently
*untouched*. These ARE software parameters but they live one layer
down from us, in the camera's firmware:

| Knob | Type | Range | Notes |
|---|---|---|---|
| AGC mode | enum | Linear / Plateau / IGE / Manual | The actual algorithm AGC8 mode runs |
| DDE intensity | int | 0 (off) .. 9 | Camera's hardware Digital Detail Enhancement |
| SSO (Smart Scene Optim) | bool | on/off | Adaptive AGC behavior |
| Plateau value | int | 0..65535 | Plateau-mode AGC tuning |
| Gain mode | enum | High / Low / Auto | High-gain = more sensitive, less range |
| Brightness offset | int | -128..127 | Histogram shift |
| Spatial filter (ITT) | int | 0..N | Camera-level smoothing |
| Output mode | enum | Y16 / AGC8 / TLinear | We're already on Y16 |
| FFC mode | enum | auto / manual | Manual lets us trigger on demand |

**Software AGC (in-process)**:

| Knob | Type | Range |
|---|---|---|
| `thermal.agc.mode` | enum | global / roi / gates / clahe_y16 |
| global low/high pct | float | 0.5..50 / 50..99.5 |
| roi top_frac | float | 0..0.8 |
| gates cold/hot | int | uint16 raw counts |
| clahe_y16 clip_limit | float | 0.5..10 |
| clahe_y16 tile_grid | int | 4..32 |

**Software post-AGC enhancement chain**:

| Stage | Knobs |
|---|---|
| dead-pixel median | enabled, ksize ∈ {3, 5, 7} |
| CLAHE (post-AGC u8) | enabled, clip_limit, tile_grid |
| gamma | float 0.5..1.5 |
| bilateral denoise | enabled, d, sigma_color, sigma_space |
| unsharp mask | enabled, amount, radius |

**Display path**:

| Knob | Range |
|---|---|
| colormap | WHITE_HOT / INFERNO / IRONBOW / JET / MAGMA |
| digital_zoom interpolation | linear / cubic / lanczos4 |
| JPEG quality | 60..95 |

**Heat detector**:

| Knob | Range |
|---|---|
| threshold_k | 2..100 |
| min_blob_area_px | 3..50000 |
| background_kernel | odd 5..63 |
| tophat_kernel | odd 5..63 |
| algorithm | tophat / boxfilter_mad |

**Total: ~28 knobs.** Grid search is infeasible (~10^15 combos at modest
discretization). Need a structured strategy.

---

## Quality metrics

For each captured frame stack we compute:

1. **Sharpness** — Laplacian variance + Tenengrad gradient magnitude
2. **Local contrast** — RMS contrast over 64×64 patches, averaged
3. **Edge density** — Canny edges per kpix at adaptive threshold
4. **Histogram entropy** — full-range entropy (high = good dynamic range)
5. **Saturation** — % pixels at 0 or 255 (low = good)
6. **Temporal stability** — mean frame-to-frame absolute diff (low = stable)
7. **SNR** — variance in flat patches (low = clean)
8. **YOLO confidence proxy** — sum of HV classifier confidences across
   detections (uses the same `seeker_thermal_hv.pt` the operator uses)
9. **Heat-detector count** — detections per frame at fixed threshold_k
10. **Heat-detector spatial coverage** — area of frame covered by
    detection bboxes (low+balanced = good targeting; high = flooding)

Composite score: weighted sum, weights tunable. Default weights bias
toward (1) sharpness, (3) edge density, (8) YOLO confidence — what
the operator and downstream classifiers actually care about.

---

## Tiered test strategy

Combinatorial space is too big for full sweep. Tier the search so
each tier picks winners that go forward to the next tier.

### Tier 1 — Camera-only (FLIR SDK), AGC8 baseline

**Goal:** find the camera-side configuration that produces the best
AGC8 image quality on its own. This becomes the input quality "floor"
for everything downstream.

Hold software AGC at `mode: global` (legacy). Sweep:

* DDE intensity: 0, 2, 4, 6, 8 (5 settings)
* AGC algorithm: Linear, Plateau, IGE (3 settings)
* SSO: on, off (2 settings)
* Gain: Auto, High, Low (3 settings)

= 5 × 3 × 2 × 3 = **90 camera configs.** ~10s capture per config = 15 min/pose.

### Tier 2 — Camera-only, Y16 raw + software CLAHE

Take camera winner from T1, switch output to Y16, sweep software AGC:

* AGC mode: global, roi (top_frac=0.4), gates (cold=p10, hot=p95), clahe_y16
* For clahe_y16: clip_limit ∈ {1, 2, 3, 4, 6}, tile_grid ∈ {4, 8, 16}

= 4 modes + (5 × 3 = 15 clahe variants) = ~19 configs. ~3 min/pose.

### Tier 3 — Post-AGC enhancement on best of T2

* gamma: 0.85, 1.0, 1.15
* bilateral: off / (d=3, σ=10) / (d=5, σ=15)
* unsharp: off / (amount=0.3) / (amount=0.5)

= 3 × 3 × 3 = 27 configs. ~5 min/pose.

### Tier 4 — Heat detector tuning at best pipeline

Sweep `threshold_k` ∈ {5, 10, 20, 30, 50} × `min_blob_area_px` ∈ {100, 500, 1500} = 15 configs.
Score: detection rate vs known scene reference + false positive rate.

### Tier 5 — Multi-pose validation

Re-test top-3 configs from T1-T3 at 4 different gimbal poses to
confirm the winner generalizes.

**Total per pose:** Tier 1 (15 min) + T2 (3 min) + T3 (5 min) + T4 (3 min) = ~26 min.
**Total for 5 poses:** ~2.5 hours of automated capture.
**Plus warm-up + analysis time:** ~3.5 hours total. Comfortably overnight.

---

## Test scene

For consistency, all tiers run on a SINGLE primary pose first, then
T5 validates across 4 additional poses. The poses span the operator's
typical use cases — wide street view, narrow alley, sky-with-tree,
sky-only (for false-positive baseline).

Operator picks the poses up front; the harness drives the gimbal to
each via the existing WebSocket gimbal_absolute command.

---

## Output

Each tier produces:

1. `recordings/optim/tier{N}/<config_id>/raw_y16.npz` — captured frames
2. `recordings/optim/tier{N}/<config_id>/metrics.json` — all metrics
3. `recordings/optim/tier{N}/<config_id>/preview.png` — single frame
4. `recordings/optim/tier{N}/leaderboard.md` — sorted rankings

End-of-run summary: `recordings/optim/RESULTS.md` — top 3 configs
across the whole sweep, side-by-side previews, recommended live YAML.

---

## Failure handling for autonomous run

Each capture is a try/except. Camera command failure logs a warning
and skips that config. On 3 consecutive capture failures the harness
pauses for 30s and re-probes; on 5 it terminates with a partial-results
report.

Heartbeat file `recordings/optim/HEARTBEAT.txt` updated every 30s
with current tier, config index, ETA. Operator can check this from
any session.

State file `recordings/optim/STATE.json` written after each config so
auto-resume can continue from the last completed config rather than
starting over.

---

## Infrastructure required

* **`thermal/boson_control.py`** — pyserial wrapper for FLIR Boson
  command protocol. ~300 LOC. Implements: open/close, ping, set DDE,
  set GAO mode, set gain mode, set brightness, run FFC, get camera ID.
* **`scripts/_thermal_metrics.py`** — pure functions for all metrics.
  Imports nothing from seeker; just numpy + cv2 + the YOLO model.
* **`scripts/_thermal_optim_harness.py`** — orchestrator. Loops over
  tiers, drives camera + gimbal, captures, scores, saves state.
* **`scripts/_thermal_optim_resume.py`** — reads STATE.json, picks up
  where we left off.
* `recordings/optim/` directory tree.

---

## Auto-resume across sessions

Two redundant mechanisms:

1. **STATE.json** is the source of truth. At session start, the agent
   checks `recordings/optim/STATE.json`. If present and not "complete",
   resume from that config.

2. **Background `until` poll** kicks the harness if it's been more
   than 5 min since the last heartbeat (means either crashed or
   between-tier idle).

Both stop on `recordings/optim/STOP` sentinel file (operator can
abort by `touch`ing that path).

---

## Estimated wall-clock for overnight run

Assuming sensor in garage, ~6 hours uninterrupted:

* T1 (15 min × 5 poses): 75 min
* T2 (3 min × 5 poses): 15 min
* T3 (5 min × 5 poses): 25 min
* T4 (3 min × 5 poses): 15 min
* Analysis + report generation: 20 min
* Margin (camera FFC, settle times, retries): 90 min

**Total: ~4 hours.** Leaves plenty of headroom in a 6-hour window.

---

## Build order (in this offline session)

1. `boson_control.py` — needed before any of the camera-side knobs are
   reachable. Build + test against the live sensor when available.
2. `_thermal_metrics.py` — pure functions, can build + unit-test offline.
3. `_thermal_optim_harness.py` — orchestrator. Build offline; live-test
   when sensor is back.
4. Auto-resume plumbing — sentinel + state reader.
5. Dry-run the harness on a small subset (1 config × 1 pose) to
   verify end-to-end before committing to the full overnight run.

I will hold off the actual overnight launch until the operator says
go AND has verified by eye that the harness's first iteration looks
sensible.
