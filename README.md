# SEEKER-01 Bench Test

Real-time detection GUI for the Seeker-01 bench test.
**Phase A: thermal camera + GUI only.** Radar, fusion, and
recording are added in later phases.

---

## What this app does

- Reads frames from a FLIR ADK Boson 640 thermal camera (USB-C).
- Detects warm blobs against cold backgrounds (classical CV).
- Classifies blobs as `HAND`, `DRONE`, or `HEAT` using a
  pre-trained YOLOv8n model with a classical shape-heuristic
  fallback.
- Displays everything in a browser dashboard with live overlays.

---

## Running it — the short version

1. Plug the FLIR ADK into a USB-C port. That's it. No drivers.
2. Double-click `START_SEEKER.bat`.
3. Chrome opens to `http://localhost:8080`.
4. You should see a live thermal feed. Wave your hand in front
   of the camera — a yellow `HAND DETECTED` banner should appear.

If anything goes wrong, check `logs/seeker.log` — each module
writes its own log section.

---

## Running without a camera (dev mode)

To work on the GUI without plugging anything in:

```
python main.py --fake-thermal
```

This runs a synthetic thermal source that draws a hot blob
moving in a sine wave — great for testing the GUI on any laptop.

---

## Running each module standalone (debugging)

Every module is designed to run independently. If something is
broken, start here to isolate the failure:

| Command | What it does |
|---|---|
| `python -m thermal` | Live thermal feed in an OpenCV window, no GUI |
| `python -m thermal.fake_thermal_source` | Synthetic thermal feed in an OpenCV window |
| `python scripts/smoke_test.py` | End-to-end: fake thermal → bus → GUI JSON |
| `pytest` | Run all unit tests |

---

## Status lights — what they mean

| Light | Meaning |
|---|---|
| **THERMAL: CONNECTED** (green) | Camera plugged in and producing frames |
| **THERMAL: DISCONNECTED** (red) | Camera not found or unplugged. Check USB cable. |
| **RADAR: DISCONNECTED** (red) | Always red in Phase A. Radar is not built yet. |
| **HAND DETECTED** (yellow banner) | Warm blob classified as hand (indoor test mode) |
| **HEAT DETECTED** (orange banner) | Warm blob detected but not classified |
| **SCANNING** (gray banner) | No warm blobs above threshold |

---

## Building a .exe for another laptop

```
build_exe.bat
```

This calls PyInstaller with `seeker_bench.spec` and produces a
folder distribution at `dist/seeker_bench/`. It contains:

```
dist/seeker_bench/
├── seeker_bench.exe       ← double-click this
├── _internal/             ← bundled DLLs, Python runtime, torch, opencv
│   ├── gui/static/        ← HTML/CSS/JS
│   ├── config/            ← app_config.yaml
│   └── models/            ← yolov8n.pt (+ seeker_thermal.pt if present)
└── ...
```

**To distribute:** copy the entire `dist/seeker_bench/` folder
(not just the .exe) to another laptop and double-click
`seeker_bench.exe`. No Python or driver install needed — the FLIR
ADK works over UVC.

**To swap in a trained model without rebuilding:** drop your
`seeker_thermal.pt` into `dist/seeker_bench/_internal/models/`
and restart. The classifier prefers it over `yolov8n.pt`.

**Notes:**
- First build takes several minutes (PyInstaller walks torch + ultralytics).
- The .exe keeps a console window open by default so crashes are visible.
  Set `console=False` in `seeker_bench.spec` once you're happy with it.
- Output folder is ~1 GB because of torch. This is expected.

---

## Project layout

```
seeker_bench/
├── main.py                  Entry point
├── config/app_config.yaml   All tunables
├── common/                  Shared dataclasses + event bus
├── thermal/                 Thermal capture + processing + classifier
│   └── training/            Fine-tune YOLO on your own data
├── radar/                   (Phase B)
├── gui/                     FastAPI + vanilla JS + WebSocket
├── models/                  YOLO weights
├── datasets/                Captured training data
├── scripts/                 Smoke tests, utilities
└── logs/                    Runtime logs
```

---

## Phase A non-goals (coming later)

- Radar, fusion, EKF, PMM classification (Phase B+)
- HDF5 session recording (the Record tab is a stub)
- In-app YOLO labeling (use `labelImg` or Roboflow)
- Fine-tuned thermal YOLO weights (use `thermal/training/`)

---

## Future work

What's left after today's bench session. Listed in the order we plan
to tackle them. Each item has a short "ticket" tag so it's easy to
grep/issue-track later. None of these block the current EO + thermal
+ gimbal bench demo.

---

### 1. Mount the rig on the gimbal — full EO + thermal + gimbal tracking loop

**Status: blocked on mechanical integration — ticket: rig-on-gimbal.**

Right now the cameras sit on a tripod next to the gimbal. The gimbal
tracking math already has both modes wired (`gimbal.cameras_on_gimbal`
in `config/app_config.yaml`). Today we're running with
`cameras_on_gimbal: false` because the cameras are stationary and the
fused az/el is an absolute bench-frame bearing — gimbal commands
`pan = home_pan + az`, `tilt = home_tilt + el`.

When the rig physically goes on the gimbal:

- Flip `cameras_on_gimbal: true` in the config. The control law
  switches to the closed-loop off-boresight form:
  `pan_cmd = pan_cur + az_err`, and `az_err` shrinks to zero as the
  gimbal centers the target. No other code changes.
- Recalibrate the tilt endpoints with `scripts/gimbal_calibrate.py`
  once the load changes — the 500 µs / 2500 µs placeholders in
  `config/app_config.yaml::gimbal.tilt_calibration` are for an unloaded
  servo.
- Verify pan/tilt geometry: `invert: true` on both axes was set for
  the current servo orientation. Mounting changes may flip it.
- Add camera-motion-compensated tracking (see item 2 — BoT-SORT).
  ByteTrack assumes a roughly stationary camera; a panning gimbal
  breaks that assumption.

Verification plan: press TRACK on a static warm target, gimbal should
center it. Walk the target across the FOV — gimbal should slew
smoothly without oscillation. Metric: settling error within ±0.5° of
boresight, no limit-cycle > 0.3 Hz.

---

### 2. Tracking improvements

We already swapped the hand-rolled greedy-IoU matcher for ByteTrack
(via `ultralytics` `model.track(persist=True, tracker="bytetrack.yaml")`)
on both EO and thermal H/V. That killed the ID-spam problem on fast
camera pans. Further upgrades:

**2a. BoT-SORT with camera-motion compensation — ticket: tracker-botsort.**
BoT-SORT estimates global image motion between frames and subtracts
it before the Kalman predict step. Essential once the rig is on the
gimbal (item 1). One-line swap: change `tracker="bytetrack.yaml"` to
`tracker="botsort.yaml"` in `thermal/classifier_hv.py::track_full_frame`.
Costs ~1 ms/frame extra.

**2b. CSRT per-target correlation tracker on TRACK lock — ticket: tracker-csrt-on-lock.**
When the user presses TRACK on a target, attach an OpenCV CSRT
tracker (`cv2.legacy.TrackerCSRT_create()`) seeded from that bbox.
Run it every frame; re-seed from YOLO whenever YOLO's next detection
overlaps. Gives pixel-locked bboxes on the tracked target at full
frame rate *without* running YOLO at full frame rate. Main payoff is
edge deployment (item 5). CSRT is ~50 FPS single-target on a laptop,
so one per lock is free.

**2c. Optical-flow coast between detections — ticket: tracker-optical-flow-coast.**
When a classifier tick produces no detection on a tracked target,
advance the last bbox by median Lucas-Kanade flow inside it.
Complements ByteTrack: ByteTrack's Kalman is a motion model built
from past detections, optical flow is actual visual evidence between
detections. Useful for targets that rapidly change velocity (e.g. a
braking drone). `cv2.calcOpticalFlowPyrLK` at VGA res is under 1 ms.

---

### 3. Classification improvements

**3a. EO drone detection fine-tune — ticket: eo-drone-finetune.**
EO currently forces the COCO fallback (person / car / truck / bus /
motorcycle) because COCO has no drone class, and airplane false-fires
on birds. A proper EO drone detector needs its own fine-tune on RGB
drone footage. Candidate datasets: Det-Fly, Drone-vs-Bird, Anti-UAV-RGB.
Plumbing already exists — see the TODO at the bottom of
`eo/eo_classifier.py`. Work: collect data, label, run
`thermal/training/train.py`-style wrapper on EO data, drop the output
`best.pt` into `models/` under a new `eo_drone.pt` name, add a second
YOLO head in `EOClassifier` or extend the existing model's class set.

**3b. Thermal drone detector retrain with bench data — ticket: thermal-drone-v2.**
`models/seeker_thermal.pt` is the current fine-tune. Recapture data
once the rig is gimbal-mounted (wider variety of viewing angles),
re-run `thermal/training/record_for_training.py` → label → train.
Gate candidate models against a held-out validation set before
promoting via `thermal/training/promote_model.py`.

**3c. False-positive suppression on birds — ticket: classify-bird-rejector.**
The hard part of any aerial classifier. Options: size + persistence
gate in image space (birds are smaller and less persistent at typical
ranges), motion-signature classifier using the fusion track's az/el
velocity, or a dedicated bird class in the trained model so YOLO
learns to discriminate. Probably all three, gated by radar doppler
once available.

**3d. PMM (point-mass model) classification — ticket: classify-pmm.**
Was a stretch goal in the original work instructions. Uses the track's
kinematic profile (speed, accel, maneuver frequency) to separate
drone / bird / background. Needs radar (item 4) to produce reliable
velocity. Deferred until radar is in.

---

### 4. Radar full integration

**Status: blocked on hardware — ticket: radar-phase-B.**

The TI AWR2944 isn't plugged in yet. Everything in `radar/` is a stub.
When the hardware arrives:

- Wire the CFAR detector from `SeekerSim` (the user's prior simulator
  in `Desktop/SeekerSim/seeker_sim/detection.py`) into a new
  `radar/radar_manager.py` running on its own thread, publishing
  `RadarFrame` on `Topic.RADAR`.
- Fusion already supports three sensors by shape — add radar to the
  fusion manager's input list. Radar contributes angular az/el and,
  critically, doppler velocity that video can't produce.
- Promote radar to the primary identity source in fusion: radar's
  angular observations should keep fusion tracks alive through EO /
  thermal dropouts. Fusion's `max_misses` grace (currently 15 ticks /
  ~1 s) can probably drop to ~3.
- Integrate radar velocity into the PMM classifier (item 3d).
- Add radar visualization to the radar panel in the GUI (currently
  shows "RADAR: DISCONNECTED"). Range-doppler scope is the standard
  display; the mockup already has the panel reserved.

---

### 5. Edge device deployment

**Status: future work — ticket: edge-deploy.**

The current pipeline is designed to be edge-portable: all heavy work
is YOLO inference, and everything else (ByteTrack, fusion, GUI
backend) is pure Python / NumPy.

Target platforms, likely in order: **Jetson Orin Nano** (best balance
of TOPS / power / price for this workload), **Jetson Orin NX** if more
headroom is needed, **Raspberry Pi 5 + Hailo-8** (M.2 NPU, ~13 TOPS,
good if size/power are tight). Coral USB is too weak for YOLOv8n at
the FOVs we're using.

What needs to happen:

- **Swap ultralytics for a TensorRT / ONNX Runtime path.** Ultralytics
  can export directly: `yolo export model=seeker_thermal.pt format=engine`
  for Jetson (TensorRT) or `format=onnx` for ONNX Runtime / Hailo.
  Inference speedup: typically 3–5× over eager PyTorch on Jetson. The
  detect/track interface stays the same; only `thermal/drone_classifier.py`
  and `thermal/classifier_hv.py` need a pluggable backend.
- **Move the GUI off the device.** Currently the FastAPI server and
  the detection pipeline run in the same process. On edge we want
  detection running headless and the browser connecting over LAN.
  Already mostly the case — just need a config flag to skip the
  `open_browser` call and harden the websocket for wifi drops.
- **CSRT on TRACK lock (item 2b).** Running YOLO at 30 Hz on a Jetson
  Orin Nano is borderline at `imgsz=640`. Running CSRT at 30 Hz on the
  locked target and YOLO at 10 Hz for the rest of the scene is far
  more comfortable and visually indistinguishable.
- **Power budget.** Document actual draw under load: YOLO only,
  YOLO + ByteTrack, YOLO + ByteTrack + CSRT, full pipeline with
  gimbal + radar. Decide whether to cap FPS / imgsz for thermal duty.
- **PyInstaller → portable archive.** `build_exe.bat` already produces
  a folder distribution for Windows dev laptops. For Linux edge, use
  a plain venv + systemd unit; no PyInstaller needed.
- **Storage.** Recording (currently a stub) goes to HDF5 + ring buffer
  once implemented — sized for the device's eMMC / SD budget.

Non-goal for the first edge port: don't rewrite anything in C++. The
pipeline is already fast enough in Python once YOLO is on TensorRT /
Hailo. Save the rewrite for when profiling actually justifies it.
