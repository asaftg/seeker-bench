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
