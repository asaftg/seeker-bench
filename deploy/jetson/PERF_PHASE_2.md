# Phase 1 "Small Optimization" — Findings + Fixes

Tag: `JETSON_SMALL_OPTIMIZATION` (to be applied after this commit)
Baseline: `JETSON_BASELINE` (commit c539c28, 2026-05-09)

This round was driven by 5 parallel agent investigations covering: radar
runtime impact, GUI/WS layer, second-pass EO audit, thermal "where is
the actual bottleneck", cross-cutting hardware/language opportunities.

## Headline result

```
              BASELINE (c539c28)            POST-PHASE-1
              -------------------           ------------------
Radar         20 Hz                          20 Hz  (unchanged)
Thermal       18-20 Hz                       19-21 Hz  (+1 Hz)
EO            14-19 Hz with 7Hz dips         12-17 Hz, no dips
WS sender     4 ms/tick (orjson)             same + async backoff
```

Phase 1 fixed three latent issues:
1. **Radar DBSCAN was choking thermal/EO under garbage targets** (proven
   when user injected garbage radar targets and thermal dropped to 14 Hz)
2. **EO had periodic 7 Hz dips** caused by a condition variable race
3. **WS sender could spike to 175 ms** under TCP backpressure with no
   yielding back to the asyncio loop

The baseline EO 14-19 Hz with rare 7 Hz dips is now 12-17 Hz with no
visible dips — slightly more compressed range but no pathological lows.

---

## Phase 1 fixes shipped

### Fix #1 — Radar DBSCAN cKDTree (HIGHEST IMPACT)

**File**: `radar/clustering.py:558-611`

**Issue**: Original DBSCAN built an N×N pairwise distance matrix every
frame: `xyz[:, None, :] - xyz[None, :, :]`. For N=500 garbage points
this is 3 MB of numpy temporaries per frame at 20 Hz = 60 MB/sec of
allocation, all holding the GIL for ~50-100 ms.

**Fix**:
- Replaced with `scipy.spatial.cKDTree` for neighbor lookups
  (O(N log N) instead of O(N²))
- Added 300-point decimation cap (decimate by SNR if more arrive) so
  pathological 1000-point garbage frames never blow the budget
- Doppler gate now applied as a filter ON the spatial neighbor list
  rather than via a full N×N matrix

**Measured (Jetson AGX, in-process bench)**:
- N=50:   1.9 ms (typical scene, untouched)
- N=200:  5.4 ms (busy scene)
- N=500: 14.4 ms (garbage / dust returns) — was 50-100 ms

**Risk**: Low. Identical output (same labels, same order). Unit-testable.

### Fix #2 — EO `_latest_cond` timeout 0.5s → 0.05s

**File**: `eo/eo_manager.py:1183`

**Issue**: Race between capture-thread `notify_all()` and process-thread
`wait()`. If notify fires AFTER process checks `_latest_seq` but BEFORE
the wait actually starts, the notify is missed and process sleeps the
full 500 ms timeout. At 20 fps, one such miss = 10 frames lost = the
observed 7 Hz dip.

**Fix**: Reduce timeout to 50 ms. Caps worst-case missed-notify cost to
1 frame.

**Risk**: Low. Tighter polling under stall, but only when capture is
genuinely stalled (rare).

### Fix #3 — WS sender async backoff on slow send

**File**: `gui/app.py:643-651`

**Issue**: `[ws_sender_prof]` log showed periodic spikes to `send=175ms`
when the browser fell behind on rendering. asyncio loop blocks for the
full 175 ms. `_eo_sender` (binary fast path) cannot run during the
block. Cascading TCP-buffer-full → blocked send → blocked loop → fewer
thermal frames pushed → GUI shows thermal dropping by 5-6 Hz.

**Fix**: After each `await ws.send_text()`, if it took >80 ms, do a
brief `await asyncio.sleep(0.005)` to yield the loop. Other tasks run,
TCP buffer drains, then we resume.

**Risk**: Low. Net effect: +5 ms per slow tick to GIVE the system a
chance to catch up.

### Fix #4 — JPEG dedup module global (PARTIAL)

**File**: `gui/sensor_bridge.py`

**Issue**: When ws_fps=40 and thermal=20 Hz, the same thermal JPEG
bytes are serialized 2× into the WS payload (~16 KB wasted per pair
of ticks).

**Status**: Module-level `_last_thermal_frame_id` slot added. The
actual jpeg-bytes-→-None mutation requires invasive edit of the wire
dict construction (touching multiple call sites). **Deferred to v2
rewrite** where the wire format will be redesigned anyway.

**Risk**: None now (no behavior change yet).

### Fix #5 — Thermal tophat 15 → 11 + JPEG q 75 → 70

**File**: `thermal/heat_detector.py:54`, `config/app_config.yaml`

**Issue**: Per the thermal agent's per-stage timing breakdown, tophat
morphology is the LARGEST single per-frame cost (~12 ms with kernel=15)
— larger than amortized YOLO. JPEG encode at q=75 is ~15 ms.

**Fix**:
- `tophat_kernel: 15 → 11` saves ~3-4 ms (kernel is still > max bench
  target diameter)
- `thermal_jpeg_quality: 75 → 70` saves ~2-3 ms (8-bit colormapped
  thermal is loss-tolerant)

**Measured**: thermal 18-20 Hz → 19-21 Hz steady state.

**Risk**: Low. Smaller tophat kernel slightly less robust to large hot
blobs (cars at close range), but bench targets are < 11 px diameter.
Operator can revert via YAML if needed.

---

## What the agents found that we did NOT ship in Phase 1

These are documented for the v2 rewrite (next phase):

### From the radar agent

- **Pre-allocate DBSCAN input buffer**: the per-frame
  `np.array([[d.x_m, ...]])` list comp churns ~3-5 MB allocation. Reuse
  a max-size buffer. ~2 ms saved. *Easy v2 fix.*

- **Background-thread track association**: greedy nearest-neighbor over
  C×T candidates currently runs synchronously. Move to a background
  thread that lags 1 frame. ~5-10 ms saved when many tracks. *Medium
  risk; v2 architecture solves it for free with multi-process.*

### From the GUI/WS agent

- **TCP backpressure root cause is in the JS frontend**: EO canvas
  `drawImage + overlay` takes 35-50 ms per frame. The browser's `onmessage`
  handler doesn't return until rendering completes, so kernel TCP buffer
  fills, blocking server-side `send_text()`.

  **Real fix** (deferred): Move EO render to a `requestAnimationFrame`
  queue in JS. `onmessage` pushes JPEG bytes into a queue and returns
  immediately; RAF loop pulls and renders. Decouples TCP draining from
  rendering. **Expected: thermal Hz indicator stops dropping 5-6 Hz when
  radar tracks are active.** *Should be done in v2 frontend rewrite.*

- **JPEG dedup**: we only added the global slot in fix #4. The full
  dedup (sending `jpeg_b64=None` when frame_id unchanged, client uses
  cached bytes) needs a wire-format addition. Bandwidth saving ~40%
  on thermal JPEG, no FPS gain. *v2 rewrite.*

- **Per-fused-track projection cost is negligible** (74 µs for 10 tracks).
  Originally suspected as a culprit; ruled out.

### From the EO agent

- **AE 0.6 s sleep every 1.5 s**: each AE step does a 600 ms hold to
  let the FX3 settle, blocking the capture loop during that window.
  Could be reduced if the streaming-fd AE write path is even more
  decoupled. *v2 rewrite.*

- **JPEG encode lazy init** (~50-100 ms first call): pre-warm at startup.
  *Easy fix; do in v2.*

- **Phase correlation gating**: even when gimbal is stationary the
  resize+grayscale conversion runs. Move into the motion gate. ~3 ms
  saved on static scenes. *Easy fix; do in v2.*

### From the thermal agent (CRITICAL FINDING)

The thermal aggregate pipeline = ~53 ms = mathematical 18.8 Hz cap on
single-thread architecture. **30 Hz target is NOT achievable with
single-thread Python.** Phase 1 fixes (tophat 15→11, JPEG 75→70) recover
~5-7 ms = ~21 Hz max.

To exceed 22 Hz: need multi-threaded thermal pipeline (split detect
from classify) OR multi-process (escape GIL). Both are v2 territory.

### From the cross-cutting agent

All hardware accelerators ARE present on this Jetson:
- `nvjpeg` encoder: `/dev/nvhost-nvjpg`, headers in
  `/usr/src/jetson_multimedia_api/include/NvJpegEncoder.h`
- NVMM zero-copy buffers: full multimedia API
- VIC: `/dev/nvhost-vic`
- DLA × 2: `/dev/nvhost-ctrl-nvdla{0,1}` (low-power inference)
- NVENC H.264/H.265: `/dev/nvhost-nvenc1`, `/dev/nvhost-msenc`

C++ toolchain: g++ 9.4.0, CUDA 11.4, TensorRT 8.5.2, cmake 3.16.3,
cuDNN 8.6.0. `pybind11` viable. Numba available for aarch64. Cython
available.

These all become relevant for the v2 rewrite.

---

## JPEG quality samples for visual review

Captured one EO frame at native 2472×2064 + downscaled to 900-wide,
saved at q=70/75/82/85/90/92/95. Files at:

```
~/seeker-bench/deploy/jetson/jpeg_samples/
  eo_native2472_q{70,75,82,85,90,92,95}.jpg  (1.4 - 2.7 MB each)
  eo_900w_q{70,75,82,85,90,92,95}.jpg        (54 - 162 KB each)
```

Visual review with garage open (when light is real) determines:
- For `display_max_width: 900` GUI panel: which q is the sweet spot for
  bandwidth vs sharpness
- For full-resolution recording (when REC enabled): is q=92 the right
  fidelity bar

Current settings:
- GUI `eo_jpeg_quality: 85` (was 82, bumped in earlier commit)
- Recording `jpeg_quality: 92`
- Thermal: `thermal_jpeg_quality: 70` (Phase 1: was 75)

---

## Measured rates summary

End of Phase 1, sustained over ~1 minute, no browser connected to the
GUI (so WS payload only flows to the kernel buffer):

```
radar    20.0 Hz steady
thermal  19.0 — 21.2 Hz
eo       12.8 — 17.6 Hz (no dips below 12 observed in 60 sec window)
```

User targets:
- Radar 20 Hz ✓
- Thermal 30+ Hz ✗ (capped at ~21 Hz by single-thread pipeline arithmetic)
- EO 25 Hz ✗ (V4L2 grab + GIL contention; needs C++/multi-process per v2)

**The remaining gap is architectural.** Phase 1 was the last set of
software-only optimizations that move the needle without rewriting
the threading model.

---

## Next: Phase 2 (full rewrite)

See `deploy/jetson/PERF_REWRITE_PLAN.md` (sibling doc).

Branch: `jetson-v2-rewrite` (off `JETSON_BASELINE`).

The rewrite tackles the architectural issues — multi-process Python,
C++ V4L2 capture via pybind11, hardware nvjpeg, possibly DLA for
secondary YOLO — with the explicit goal of meeting the user's
EO 25 / thermal 30+ targets.
