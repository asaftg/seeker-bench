# seeker_v2 — Multi-Process Architecture Rewrite

Branched off `JETSON_BASELINE` (commit c539c28). Sibling to v1 seeker.
The v1 baseline at `~/seeker-bench/{eo,thermal,radar,fusion,gui}/` is
unchanged — `seeker_v2/` is parallel.

## Status: Phases 2.1 – 2.5 implemented (untested on Jetson)

Phase 2.1 – 2.5 code is committed. None of it has been exercised on
the actual Jetson hardware yet. The Python side runs on x86 dev hosts
for syntax / smoke testing; the C++ extensions only build on JetPack 5
(needs `linux/videodev2.h`, `NvJpegEncoder.h`, `NvInfer.h`).

## Architecture

```
EO_CAP ──┐                                       ┌── Inference (TRT, iGPU + DLA0)
THM_CAP ─┼── shm rings + small queues ── main ───┤
RDR_CAP ─┘                                       └── Fusion
                                                 └── GUI/WS (FastAPI + RAF)
```

Each capture is a separate OS process (own GIL). Frames cross process
boundaries via `multiprocessing.shared_memory.SharedMemory` with a
SPSC ring of `FrameDescriptor` records (see `processes/ipc.py`). JPEG
snapshots and detections cross via small `multiprocessing.Queue`s.

## Phase progression

| Phase | What lands                                                  | Files                                                       |
|-------|-------------------------------------------------------------|-------------------------------------------------------------|
| 2.1   | Multi-process Python scaffolding                            | `processes/{ipc,eo_capture,thermal_capture,radar_capture,inference,fusion}.py`, `main.py` |
| 2.2   | pybind11 C++ V4L2 backend (releases GIL during ioctl)        | `native/v4l2_capture.cpp`                                   |
| 2.3   | nvjpeg hardware JPEG encoder for thermal + EO snapshots      | `native/nvjpeg_encoder.cpp`, wired in `processes/thermal_capture.py` & `processes/eo_capture.py` |
| 2.4   | TensorRT runner with DLA0 binding for thermal classifier     | `native/dla_runner.cpp`, `scripts/build_dla_engine.sh`      |
| 2.5   | RAF render queue + JPEG snapshot endpoints + dark GUI        | `gui/index.html`, `gui/static/{css,js}/`, snapshot routes in `main.py` |

## Build & run

### 1. Build the C++ extensions (Jetson only)

```bash
cd ~/seeker-bench/seeker_v2/native
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Outputs land next to the source as importable Python modules:

  - `seeker_native.cpython-*-aarch64-linux-gnu.so`  ← Phase 2.2
  - `seeker_nvjpeg.cpython-*-aarch64-linux-gnu.so`  ← Phase 2.3
  - `seeker_dla.cpython-*-aarch64-linux-gnu.so`     ← Phase 2.4 (optional)

If a module is missing, the Python side falls back transparently:

  - `seeker_native` missing → pure-Python `RawV4L2Backend` (slower but functional)
  - `seeker_nvjpeg` missing → `cv2.imencode` (uses CPU + libjpeg)
  - `seeker_dla` missing → ultralytics loads the iGPU engine

### 2. (Optional) Build a DLA-bound thermal engine

```bash
# Export ONNX once, on Jetson:
yolo export model=models/seeker_thermal_hv_v2.pt format=onnx \
            imgsz=640 simplify=True opset=13 device=0

# Build a DLA0/FP16 engine:
bash seeker_v2/scripts/build_dla_engine.sh \
     models/seeker_thermal_hv_v2.onnx \
     models/seeker_thermal_hv_dla0.engine \
     0 640
```

`processes/inference.py` looks for `models/seeker_thermal_hv_dla0.engine`
and uses it if present, freeing the iGPU for the EO classifier.

### 3. Run

```bash
cd ~/seeker-bench
python3 -m seeker_v2.main --config config/app_config.yaml
```

Open the GUI at `http://<jetson_ip>:8081/`. v1 stays on 8080.

Useful flags:
  - `--no-eo` / `--no-thermal` / `--no-radar` — skip a sensor
  - `--no-inference` / `--no-fusion` — skip downstream stages
  - `--host 0.0.0.0 --port 8081` — bind override

### 4. Smoke checks (on x86 dev host)

Python side parses & imports without Jetson hardware:

```bash
python -c "import ast; [ast.parse(open(f).read()) for f in [
    'seeker_v2/main.py',
    'seeker_v2/processes/ipc.py',
    'seeker_v2/processes/eo_capture.py',
    'seeker_v2/processes/thermal_capture.py',
    'seeker_v2/processes/radar_capture.py',
    'seeker_v2/processes/inference.py',
    'seeker_v2/processes/fusion.py',
]]; print('OK')"
```

The IPC layer has its own self-test:

```bash
python -m seeker_v2.processes.ipc
```

## Reverting

The v1 baseline is at the `jetson` branch HEAD, tagged `JETSON_BASELINE`.

```
git reset --hard JETSON_BASELINE              # safe rollback to baseline
git reset --hard JETSON_SMALL_OPTIMIZATION    # rollback to Phase 1
```

The `jetson-v2-rewrite` branch will **never** be merged into `jetson`
or `main` without explicit user authorization.

## Performance targets

From the user (post-Phase-1 baseline):
  - EO: 25 Hz steady, no 7 Hz dips
  - Thermal: 30+ Hz (40 Hz stretch)
  - Radar: 20 Hz
  - WS @ 30+ Hz with no GUI stutter

Phase 1 baseline (`JETSON_SMALL_OPTIMIZATION`):
  - EO: 12-17 Hz with occasional 7 Hz dips
  - Thermal: 19-21 Hz (capped by 53 ms in-thread pipeline)
  - Radar: 20 Hz
