# `jetson-v2-rewrite` — Phase 2 Architecture Plan

Branch: `jetson-v2-rewrite` (off tag `JETSON_BASELINE`, commit `c539c28`)
NEVER merge to `jetson` or `main`. The baseline branches are sacred.

## Why a rewrite

Phase 1 (commit `0a8a0ad`, tag `JETSON_SMALL_OPTIMIZATION`) extracted
every measurable software-only optimization from the v1 single-process
Python architecture. Final rates after Phase 1:

```
target ↓     after Phase 1 ↓   gap ↓
EO 25 Hz     12-17 Hz          ~8-13 Hz short
Thermal 30+  19-21 Hz          ~10 Hz short
Radar 20 Hz  20 Hz             ✓ met
```

**The remaining gap is architectural.** The thermal pipeline alone is
~53 ms aggregate cost in Python (per per-stage analysis), which is a
mathematical 18.8 Hz cap on a single thread. EO V4L2 grab() is GIL-bound;
the backend can do 19 fps standalone but only 14-17 in seeker because
of contention with thermal/radar/AE/processor threads.

Phase 2 fixes this by:
1. **Separating capture from inference into different processes** —
   each gets its own GIL
2. **Replacing the V4L2 grab + AGC hot loop with a C++ extension** —
   releases GIL during the work
3. **Using Jetson hardware accelerators** — nvjpeg encoder, possibly
   NVMM zero-copy buffers, possibly DLA for secondary inference

User has explicitly authorized: *"absolutely full optimization from
scratch. you have full authority to write the ENTIRE APP FROM SCRATCH
(C++, multithread, whatever you want). but please - only write it in
a different branch so nothing will happen to the jetson baseline."*

## Hardware confirmed available (from cross-cutting agent)

| Accelerator | Device | Header | Use |
|---|---|---|---|
| nvjpeg encoder | `/dev/nvhost-nvjpg` | `/usr/src/jetson_multimedia_api/include/NvJpegEncoder.h` | EO + thermal JPEG encode at <1 ms |
| NVMM buffers | — | `nvbufsurface.h` | Zero-copy camera→GPU |
| VIC | `/dev/nvhost-vic` | (via GStreamer) | Resize / colorspace |
| DLA × 2 | `/dev/nvhost-ctrl-nvdla{0,1}` | TRT 8.5.2 | Secondary YOLO (thermal H/V?) |
| NVENC | `/dev/nvhost-nvenc1`, `/dev/nvhost-msenc` | (via GStreamer) | H.264/H.265 stream instead of JPEG |
| TensorRT 8.5.2 | — | C++ headers | iGPU inference |
| CUDA 11.4 | — | nvcc | Custom kernels if needed |

C++ toolchain on Jetson: g++ 9.4.0, cmake 3.16.3, ninja missing but
make works. Pybind11 viable. cuDNN 8.6.0 present.

## Goal architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          v2 ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │ EO Capture      │    │ Thermal Capture  │    │ Radar Capture    │
   │ (C++ subprocess)│    │ (C++ subprocess) │    │ (Python or C++)  │
   │ V4L2 mmap       │    │ V4L2             │    │ TLV parser       │
   │ RAW12->u16 view │    │ Y16/AGC8         │    │ DBSCAN (cKDTree) │
   │ AGC stretch     │    │ Tophat (CUDA?)   │    │ Kalman tracker   │
   │ nvjpeg encode   │    │ nvjpeg encode    │    │                  │
   │ shm_open frames │    │ shm_open frames  │    │ shm_open targets │
   └────────┬────────┘    └────────┬─────────┘    └────────┬─────────┘
            │                      │                       │
            │   shared memory      │                       │
            │   (latest frame)     │                       │
            ▼                      ▼                       ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │   Inference Process (Python + TRT)                              │
   │   - Reads frames from shm                                        │
   │   - Runs EO YOLO (every 2 frames @ imgsz=832)                    │
   │   - Runs Thermal HV YOLO (every 4 frames @ imgsz=640)            │
   │   - SAHI tiled inference for distant targets                     │
   │   - Writes detections to shm_dets / mp.Queue                     │
   └────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │   Fusion + Tracking Process (Python)                            │
   │   - Reads detections from queue                                  │
   │   - Cross-sensor association (angular IoU)                       │
   │   - MOSSE tracker for gap-bridging                               │
   │   - Persistent track manager                                     │
   │   - Writes fused tracks to shm_tracks                            │
   └────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │   Main Process / GUI Server (Python + FastAPI/aiohttp)          │
   │   - Async event loop only                                        │
   │   - Reads frames + tracks from shm                               │
   │   - Sends WS messages (binary fast path for JPEG, JSON for meta) │
   │   - Handles HTTP API                                             │
   │   - Coordinates lifecycle                                        │
   └─────────────────────────────────────────────────────────────────┘
```

Each top-level box is a separate OS process with its own GIL.
Shared memory is `multiprocessing.shared_memory.SharedMemory` for frames
+ a single-producer-single-consumer ring of frame indices via
`multiprocessing.Value` (lock-free via memory ordering).

## Phased plan (incremental, each phase shippable / revertable)

### Phase 2.1 — Multi-process Python (1-2 days)

Lowest-risk first move. Keep all current Python logic; just split into
processes. Frames flow through shared memory.

**Deliverables:**
- `seeker_v2/processes/eo_capture.py` — process target that runs
  RawV4L2Backend + AGC, writes BGR to `shm["eo_bgr_latest"]`,
  increments seq counter
- `seeker_v2/processes/thermal_capture.py` — same for Boson
- `seeker_v2/processes/radar_capture.py` — TLV parser + clustering
- `seeker_v2/processes/inference.py` — EO + thermal YOLO TRT
- `seeker_v2/processes/fusion.py` — fusion_manager logic
- `seeker_v2/main.py` — orchestrator, GUI/WS server, lifecycle

**Expected gain**: EO 14-17 → 18-22 Hz (escapes GIL contention).
Thermal 19-21 → 22-25 Hz (same). Radar unchanged.

### Phase 2.2 — Replace EO V4L2 backend with C++ extension (2-3 days)

Pybind11 module `seeker_v2.native.v4l2_capture` exposes:
```cpp
class V4L2Backend {
  Status open(const std::string& dev, int w, int h);
  Status close();
  // Returns numpy array reference to mmap buffer view, no copy.
  // Caller must call release(idx) when done.
  std::pair<py::array, int> grab();
  Status release(int idx);
  // XU IOCTL passthrough
  Status set_xu(int selector, py::bytes data);
  py::bytes get_xu(int selector, int size);
  // High-level controls (built on set_xu):
  Status set_exposure_ext(int value);
  Status set_trigger_disable();
};
```

The grab() releases the GIL during the kernel ioctl wait. Returns a
zero-copy numpy view. Caller does AGC (in C++ via cv2 NEON) and gets
an output BGR array.

**Expected gain**: EO 18-22 → 23-27 Hz. **Hits the 25 Hz target.**

### Phase 2.3 — nvjpeg encoder (1-2 days)

Replace `cv2.imencode(".jpg")` with NVIDIA's hardware encoder. For both
EO and thermal.

**Path**: pybind11 wrapper around `NvJpegEncoder.cpp` from
`/usr/src/jetson_multimedia_api/samples/`. ~200 lines C++ + 20 lines
pybind11.

**Expected gain**: ~5-8 ms/frame saved on EO, ~2 ms on thermal. Cleans
up the WS sender's TCP backpressure issue (smaller payloads encoded
faster). Phase 2.1+2.2 already gets us to target; this is gravy +
foundation for streaming H.265 later.

### Phase 2.4 — Move thermal classifier to DLA (optional, if needed)

Xavier has 2 DLAs that can run YOLOv8n (small model). If iGPU is still
contested between EO YOLO + thermal HV YOLO at this point, move thermal
HV to DLA0 (free up iGPU for EO + SAHI tiles).

**Cost**: 1 day to convert TRT engine for DLA + verify accuracy.
**Risk**: DLA may not support all ops; fallback layers run on iGPU
(slower than pure-iGPU). Profile before committing.

### Phase 2.5 — Frontend RAF render queue (the GUI agent's priority fix)

In JS: `onmessage` pushes JPEG bytes into a queue and returns immediately.
A `requestAnimationFrame` loop pulls from the queue and renders.
Decouples TCP draining from canvas redraw.

**Expected gain**: GUI thermal Hz indicator stops dropping by 5-6 Hz
when radar tracks active. (Server-side rates are already fine after
Phase 1.)

**Cost**: 2-4 hours. JS-only. Touches `gui/static/js/main.js`.

### Phase 2.6 — Pure C++ application (LAST RESORT, only if 2.1-2.4 fall short)

Replace `seeker_v2/main.py` with a C++ binary that does everything:
- C++ V4L2 capture (already from 2.2)
- TRT inference (C++ API, slightly faster than Python wrapper)
- Fusion in C++ (porting from Python — careful, tracking logic is dense)
- WebSocket server via `uWebSockets` or `Drogon`
- HTTP API likewise

**Expected gain**: Maybe 10-15% over 2.1+2.2 combined, mostly from
eliminating Python interpreter overhead in the orchestrator.
**Cost**: 1-2 weeks for feature parity. **Only if the user explicitly
prioritizes raw FPS over maintainability.**

## Execution order (revised after re-reading user message)

The user said *"continue to a full scope rewrite with ALL the things you
think are helping. absolutely full optimization from scratch."*

Interpretation: do the rewrite ambitiously. Take the architecture all
the way. But ship in phases so each step is testable.

**Tonight (autonomous):**
1. ✓ Branch created off JETSON_BASELINE
2. Skeleton repo layout for `seeker_v2/`
3. Phase 2.1 multi-process scaffolding — even if incomplete, a runnable
   skeleton makes morning-Asaf able to evaluate the approach

**Morning hand-off:**
- `PERF_REWRITE_PLAN.md` — this doc
- `seeker_v2/README.md` — how to build/run the v2 stack
- `seeker_v2/processes/` — at least one process skeleton per camera
- A measured-rates table showing what the v2 skeleton achieves vs
  baseline

**Subsequent rounds (with user input):**
- Phase 2.2 (C++ extension) — the biggest leverage win
- Phase 2.3 (nvjpeg)
- Phase 2.5 (RAF render queue) — should land for the user to feel a
  responsive GUI

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Multi-process IPC adds latency | Use shared memory for frames (no serialization), only queue small descriptors |
| Frame-drop semantics differ from single-thread | Single-slot publish (latest wins) — same as current |
| Process restart on crash gets messy | systemd-managed `seeker-v2.service` with `Restart=on-failure`; supervisor process for child lifecycle |
| Cross-process debugging harder | Each process logs to its own journal tag; aggregator reads them |
| User wants to revert quickly | `JETSON_BASELINE` tag is sacred. `git reset --hard JETSON_BASELINE` rolls back instantly |
| Pybind11 build complexity | Build on-device; CMake handles it. ~30 sec build time on Jetson |
| nvjpeg ABI mismatches | Use the version shipped with JetPack 5.1.5; don't update |
| TRT engines not portable | Already on-target; not a v2 concern |

## Out of scope for v2

- **Radar A/A mode optimization**: user explicitly deferred this.
- **Fundamental algorithm changes**: AGC, fusion association, MOSSE
  parameters — all kept identical to v1 baseline.
- **Schema changes to recording format**: v2 must read/write the same
  JSONL recording format so v1 recordings replay correctly.
- **GUI redesign**: same look/feel; only the WS/RAF rendering pipeline
  changes (Phase 2.5).
- **API breaking changes**: same `/api/*` endpoints with same shapes.
- **GMSL2 carrier board hardware swap**: separate hardware effort.

## Where the rewrite lives

```
seeker_v2/
├── README.md                 # build + run instructions
├── CMakeLists.txt            # top-level for native modules
├── pyproject.toml            # Python package setup
├── main.py                   # orchestrator
├── processes/
│   ├── eo_capture.py
│   ├── thermal_capture.py
│   ├── radar_capture.py
│   ├── inference.py
│   ├── fusion.py
│   └── ipc.py                # SharedMemory + Value seq counters
├── native/
│   ├── v4l2_capture.cpp     # Phase 2.2: C++ V4L2 backend
│   ├── nvjpeg_encoder.cpp   # Phase 2.3: HW JPEG
│   └── seeker_native.pyi    # type stubs for the .so
├── gui/
│   ├── app.py               # FastAPI server (mostly unchanged)
│   └── static/js/main.js    # Phase 2.5: RAF queue
└── config/
    └── app_config.yaml       # same schema as v1
```

The existing `seeker-bench/` directory at jetson HEAD remains the
canonical v1. v2 is a sibling.
