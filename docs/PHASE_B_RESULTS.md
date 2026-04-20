# Phase B — Thermal Classifier Results

Snapshot of where the thermal pipeline stands as of April 20, 2026, after
Tickets 1 and 2 and the overnight retraining of both YOLO models.

## TL;DR

Two YOLOv8n classifiers now run alongside the always-on heat-blob detector:

| Model | File | Classes | Mode | Val mAP50 | P / R |
|---|---|---|---|---|---|
| Human + Vehicle | `models/seeker_thermal_hv.pt` | person, vehicle | full-frame | **0.867** | 0.900 / 0.770 |
| Drone | `models/seeker_thermal.pt` | drone | ROI + full-frame | **0.945** | 0.982 / 0.920 |

Live behaviour verified on the FLIR ADK Boson 640 bench camera:

- HUMAN boxes fire reliably up to ~10-15 m, degrade gradually past that.
- VEHICLE boxes fire on cold-metal side-profile cars at 70-91 % confidence — the problem that started Phase B is solved.
- Drone model promoted; ready for aerial validation (no drone on hand at bench).

## Training corpora

### H/V model (`seeker_thermal_hv.pt`)

Final merged dataset at `datasets/seeker_hv/`:

| Dataset | Role | Train imgs | Val imgs | Classes contributed |
|---|---|---|---|---|
| HIT-UAV | aerial thermal from drone POV | 1,937 | 280 | person, vehicle (Bicycle dropped) |
| LLVIP | ground-level night surveillance | 12,023 | 3,463 | person |
| FLIR ADAS v2 | ground-level driving POV | 10,250 | 1,097 | person, vehicle (141k boxes!) |
| **Total** | | **24,210** | **4,840** | person, vehicle |

Training: `scripts/train_thermal_hv.py`, 50 epochs, yolov8n.pt transfer,
imgsz=640, batch=16, on RTX A4000.

**Runtime inference** bumped to `imgsz=960` on GPU (13 ms/frame @ 77 Hz on A4000)
to improve small-target recall. CPU deployments auto-fall back to 640.

### Drone model (`seeker_thermal.pt`)

Source dataset at `datasets/thermal_drone/`:

| Dataset | Role | Train imgs | Val imgs |
|---|---|---|---|
| Anti-UAV-RGBT (CVPR/ICCV challenge) | 318 IR video sequences, varied backgrounds | 29,658 | 28,954 |

Training: `scripts/train_thermal_drone.py`, 50 epochs yolov8n, imgsz=640, batch=16.

Not yet integrated: DUT Anti-UAV (3 zips, ~10k extra frames). Layout parser
needs a new handler for the DUT directory convention — deferred.

## Pipeline architecture

Per-frame flow in `thermal/thermal_manager.py`:

```
capture (cv2)
  → AGC + colormap                    (thermal_processor.py)
  → HEAT DETECTOR (classical CV)      (heat_detector.py)
      residual = frame - spatial_mean
      threshold = median + k*MAD
      → list[ThermalDetection] (orange boxes)
  → DETECTION TRACKER                 (detection_tracker.py)
      min_hits + max_misses + EMA smoothing
  → DRONE CLASSIFIER (ROI)            (drone_classifier.py)
      crops each heat blob → YOLO inference → TargetClass.DRONE
  → H/V CLASSIFIER (full-frame)       (classifier_hv.py)
      YOLO on whole AGC image → bbox + conf
      → persistence tracker (IoU matching, min_hits=2, max_misses=8)
      → IoU-merged into detections list or appended as new boxes
  → publish ThermalFrame on FrameBus
  → GUI bridge serializes + pushes to WebSocket at ~30 Hz
```

Key design choices captured in code:

- **Two-thread split** (`ThermalCapture` + `ThermalProcess`) so heavy YOLO work
  never blocks the camera grab loop. Process thread always works on freshest
  frame; stale frames silently dropped.
- **Config-driven auto-adaptation**: `classify_interval_frames: auto` and
  `classifier_hv_imgsz: auto` detect CUDA at startup and pick appropriate
  values (every-frame + imgsz=960 on GPU, every-6th + imgsz=640 on CPU).
- **Persistence trackers** on both classifier outputs kill single-frame
  flicker FPs. All knobs exposed in `config/app_config.yaml`.

## Relevant config knobs

```yaml
classifier:
  enabled: true

  # Drone classifier (ROI-based)
  model_path: models/yolov8n.pt
  trained_model_path: models/seeker_thermal.pt
  conf_threshold: 0.40
  roi_padding_px: 16

  classify_interval_frames: auto  # 1 on GPU, 6 on CPU

  # H/V classifier (full-frame)
  classifier_hv_enabled: true
  classifier_hv_model: models/seeker_thermal_hv.pt
  classifier_hv_conf: 0.55
  classifier_hv_min_bbox_px: 2500
  classifier_hv_min_hits: 2
  classifier_hv_max_misses: 8
  classifier_hv_bbox_ema: 0.15
  classifier_hv_imgsz: auto       # 960 on GPU, 640 on CPU
```

## Known limitations / open issues

1. **Long-range humans drop out.** Persons at 20+ m (pixel height ~25 px)
   are intermittent. FLIR ADAS dashcam data helped but not enough at
   sub-30 px scales. Candidates for Phase C:
   - Retrain h/v at `imgsz=960` natively (currently trained at 640,
     inferred at 960).
   - Upgrade to yolov8s (~3 MB → ~11 MB, 2-3x features) or yolov8m.
   - Add tiled inference: split frame into 2×2 overlapping tiles, run
     YOLO on each, merge with NMS. Triples inference cost but gives
     effective 2× resolution for small targets.
   - Add more in-domain data: record 200-500 bench frames of users at
     5-100 m, hand-label with `labelImg`, fine-tune.

2. **DUT Anti-UAV not integrated.** `prepare_antiuav_drone.py` has three
   code paths (image-dir sequences, video sequences, flat img/xml pairs)
   but DUT uses yet another layout. Needs a layout probe.

3. **Cold-object heat detector blind spot.** The heat detector only
   triggers on `residual > +k*MAD`. Thermally-inverted targets (cold
   cars against warm pavement) are invisible to the blob detector — they
   only get labeled because the full-frame YOLO catches them directly.
   Consider a bidirectional anomaly mode (`abs(residual) > k*MAD`) to
   give at least HEAT boxes on cold anomalies.

4. **No live hand/drone validation.** Bench camera has seen people and
   cars. No physical drone or phone-scale hot-blob tests yet.

## Scripts added this phase

All in `scripts/`:

| Script | Purpose |
|---|---|
| `prepare_hit_uav_hv.py` | HIT-UAV → YOLO h/v (drops Bicycle, merges Car+OtherVehicle) |
| `prepare_llvip_hv.py` | LLVIP VOC XML → YOLO (person only) |
| `prepare_flir_adas_hv.py` | FLIR ADAS v2 COCO JSON → YOLO (merges car/truck/bus/motor → vehicle) |
| `prepare_m3fd_hv.py` | M3FD VOC XML → YOLO (scaffolded; not yet run — no M3FD zip in place) |
| `prepare_cvc14_hv.py` | CVC-14 FIR TIFF + txt → YOLO person (scaffolded) |
| `prepare_antiuav_drone.py` | Anti-UAV (3 layouts) → YOLO drone |
| `prepare_seeker_hv.py` | merges HIT-UAV + writes `data.yaml` |
| `train_thermal_hv.py` | 50-epoch yolov8n train → `seeker_thermal_hv.pt` |
| `train_thermal_drone.py` | 50-epoch yolov8n train → `seeker_thermal.pt` |
| `download_public_datasets.py` | best-effort autonomous fetch (CVC-14, Anti-UAV GH mirror) |
| `overnight_retrain.py` | master orchestrator: polls Downloads for FLIR zip, preps, trains both models |
| `drone_retrain.py` | same but specifically polls for drone zips |
| `drone_prep_and_train.py` | skips polling — run when zips already in place |

All prep scripts are idempotent (skip work already done) and write with
dataset prefixes (`hit_*`, `llvip_*`, `flir_*`, `antiuav_*`, `rgbt_*`, etc.)
so they merge safely into the same `datasets/seeker_hv/` or
`datasets/thermal_drone/` tree.

## What Phase C should probably include

Aimed at closing the "long-range person" gap and adding bench realism:

1. **Native-imgsz retrain** — simple win, ~3 hr overnight.
2. **Model-size bump** — yolov8s or yolov8m. Re-benchmark inference times.
3. **Tiled inference plumbing** in `classifier_hv.py` as an optional mode.
4. **In-domain data capture** — `thermal/training/record_for_training.py`
   already exists; use it for a 500-frame bench session, label, fine-tune.
5. **Cold-anomaly mode** in `heat_detector.py` — bidirectional threshold.
6. **DUT Anti-UAV integration** — inspect layout, add 4th parser path.
7. **Drone live test** — once a test drone is available at the bench.
8. **PMM / radar integration** (separate Phase) — drone model will feed
   into the fusion tracker when radar is wired up.
