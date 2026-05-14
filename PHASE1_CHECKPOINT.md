# Phase 1 Checkpoint — 2026-05-14

Snapshot of the Seeker-01 codebase including all EO pipeline work from
2026-05-13 through 2026-05-14, plus the start of the human-detection
fine-tune pipeline.

## What's in this checkpoint

### EO pipeline improvements (live on Jetson)

- **Digital zoom (1× / 2× / 4× / 8×)** — operator-tunable display zoom.
  Buttons in EO panel header. `EOManager.set_zoom_level()` + center-crop +
  display rescale. (`eo/eo_manager.py`, `gui/static/index.html`,
  `gui/static/js/main.js`, `gui/app.py:/api/config/eo` accepts
  `zoom_level: 1|2|4|8`.)

- **SAHI tiling on FULL NATIVE at all zoom levels** — discovered (and
  shipped) that cropping the model input at high zoom collapses
  confidence. Architecture now: YOLO always sees the full 2472×2064
  native; zoom is display-only. Bboxes are filtered to the visible crop
  region and rescaled to display coords. Same compute as 1× regardless
  of zoom. (`eo/eo_manager.py` `_classifier_loop` + `_process_and_publish`.)
  Vehicle detection at 4×/8× went from 0 dets to working after this
  change.

- **Schmitt two-tier confirmation gate** — per-class HI (single-frame
  instant pass) + LO (multi-frame persistence via K-of-WINDOW). With
  COAST window after track loss. Filters out single-frame hallucinations
  without losing real targets. (`eo/eo_manager.py` `_classifier_loop` +
  `__init__`.)

- **DEV-tab SCHMITT TUNING sliders** — live-tunable per-class HI / LO +
  global K_PERSIST / WINDOW / COAST. POST `/api/config/eo_schmitt` and
  WS command `eo_schmitt_tune`. Hydrates from periodic WS payload.
  No restart needed to tune. (`gui/static/index.html`,
  `gui/static/js/main.js`, `gui/app.py`, `gui/sensor_bridge.py`.)

- **Track-id-aware containment-drop** — fixes stationary bbox flicker.
  When two overlapping bboxes contend, prefer the one whose track_id
  was IoU-matched from the previous tick (stable identity) over the
  conf-sort default. (`eo/eo_manager.py` `_classifier_loop`.)

- **Tighter `_fused_id_for_bbox` IoU fallback** — 0.30 → 0.50. Prevents
  stage-2 fused-id resolution from grabbing the wrong fused track when
  raw track_id briefly churns. (`gui/sensor_bridge.py`.)

- **Confidence EMA on Schmitt output** — smooths displayed conf so the
  label percentage doesn't flicker frame-to-frame when YOLO conf swings
  0.3 ↔ 0.9 on the same track. (`eo/eo_manager.py`.)

- **MOSSE empty-pool early-return** — skips BGR→GRAY when no trackers
  exist. ~1-2 ms/tick saved on empty-scene operation.
  (`vision/correlation_tracker_set.py`.)

### W1 slew compensation — DEFERRED

Multiple iterations on phase-correlation + MOSSE bypass left the bbox
overshooting/undershooting at various pan speeds. User decided this is a
larger tracking + gimbal-accuracy problem; pushed back to a separate
investigation. Current state has *some* slew code in `_process_and_publish`
that gates on cross-confirmation (BUS pose + phase-corr agree) but
visible tracking under slew is still ~30% accurate. To revisit.

### Human-detection fine-tune (in progress)

`eo/training/` (new):

- `extract_nightowls_subset.py` — uses `remotezip` to stream only the
  ~6,236 pedestrian-containing images from the 50 GB NightOwls
  validation zip (HTTP range requests). Total download ~6 GB.

- `convert_nightowls_to_yolo.py` — converts NightOwls COCO JSON →
  ultralytics YOLO format + grayscale-converts each image (mono NIR
  distribution match). 90/10 train/val split. Keeps the same class
  schema as `seeker_eo_v3.pt` (0=person, 1=vehicle, 2=drone).

- `fine_tune_eo_humans.py` — resumes from `seeker_eo_v3.pt` with
  `freeze=10` (backbone+neck frozen, head only trains). 10 epochs,
  AdamW @ lr=5e-4, mosaic + brightness aug. Preserves vehicle/drone
  class weights exactly. Promotes best to `seeker_eo_v4.pt`.

- `compare_weights.py` — runs baseline vs fine-tuned on the same 3
  visually-vetted human frames from a representative recording so we
  can directly read the conf delta.

Run order: `extract → convert → fine_tune → compare`.

## Diagnostic confirmation of why fine-tune is needed

Offline test on the full 1× native of a representative scene:
- Vehicle top conf **0.640** (model recognizes well)
- Person top conf **0.134** (model barely recognizes)

The model (`seeker_eo_hv_v2.pt` / `seeker_eo_v3.pt`) was trained on
FLIR-ADAS RGB daytime — driving scenes with many vehicles, fewer
pedestrians, and zero NIR-night samples. Person class is starved on
this distribution. Slider tuning won't bridge a 5× confidence gap;
retrain on NIR-night pedestrian data is the real fix.

## Recovery / continuation

To pick this up:

1. Pull `seeker_bench/` to the Jetson, replace `~/seeker-bench/`.
2. Restart seeker.
3. From `seeker_bench/eo/training/`:
   ```
   python extract_nightowls_subset.py    # ~6 GB download
   python convert_nightowls_to_yolo.py    # builds yolo_dataset/
   python fine_tune_eo_humans.py          # ~1-2 hr on RTX A4000
   python compare_weights.py              # baseline vs tuned
   ```
4. If conf delta is meaningful, copy `seeker_eo_v4.pt` to Jetson
   `~/seeker-bench/models/`, export TRT engine on Jetson, restart.

## Live Schmitt config (current Jetson state)

```yaml
eo.classifier:
  conf_threshold: 0.20   # model floor (NOT the slider gate)
  classes_conf:           # per-class LO (multi-frame floor)
    person:  0.25
    vehicle: 0.25
    drone:   0.20
  schmitt:                # global gate parameters
    conf_hi:     0.55
    conf_lo:     0.25
    k_persist:   4
    window:      20
    coast_ticks: 3
```

Sliders in DEV tab override these at runtime; YAML is the boot default.
