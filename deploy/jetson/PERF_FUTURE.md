# Jetson Performance — Deferred Optimizations

State after the 2026-05-08/09 optimization round (commits f5f2346..67bbf04):

| Sensor / Stage | Was | Now | Target | Status |
|---|---|---|---|---|
| EO producer rate | 1 Hz (stuck) | 14–19 Hz | **25 Hz** | ~75% there |
| Thermal producer rate | 12–13 Hz | 18–20 Hz | **30+ Hz** | ~65% there |
| Radar | 20 Hz | 20 Hz | 20 Hz | ✅ done |
| WS sender JSON | 0.3 ms/tick | 0.1 ms/tick | n/a | ✅ orjson |
| EO grab() standalone | 88 ms | 50–55 ms | <33 ms | ~60% there |

## Why we stopped where we did

The remaining gap to user targets is no longer about *single-threaded
work per frame* — that's been compressed. The next-level fixes are
**architectural** and carry real implementation risk. This document
preserves the analysis so future work picks up cleanly.

---

## Bottleneck #1 — V4L2 grab() in Python is GIL-bound

**Symptom**: backend hits 19 fps standalone but only 14–17 fps inside
seeker. That ~3 fps gap is GIL contention with thermal + AE + GUI
threads.

**Numbers** (per-frame profile, Jetson AGX, 2472×2064 YUYV → RAW12 BGR):

| Stage | Cost (ms) | GIL? |
|---|---|---|
| `fcntl.ioctl(VIDIOC_DQBUF)` | 4–14 (avg 10) | released during kernel wait |
| `np.frombuffer(...).copy()` | 8–10 | GIL held (Python loop over memcpy) |
| stats: `np.percentile(stride[::8])` | 4.7 (cached every 5th frame) | GIL held |
| `cv2.convertScaleAbs(u16, alpha, beta)` | 9 | released by cv2 (NEON SIMD) |
| `cv2.cvtColor(GRAY→BGR)` | 3 | released by cv2 |
| Other overhead | ~2 | mixed |
| **Total** | **30–40 ms** | mixed |

19 fps = 52 ms/frame. So ~12 ms/frame is "lost" to scheduling. With 8+
Python threads contending for one GIL, that's the cost.

### Option A — Move the V4L2 backend to a separate process

Use `multiprocessing.shared_memory` to share frames + a `multiprocessing.Queue`
for control signals. The child process owns `/dev/video0`, runs the
grab loop unimpeded, and the parent reads decoded frames from shared
memory.

**Effort**: 1–2 days. **Risk**: medium (process lifecycle, signal handling,
ctypes XU IOCTLs through a shared-memory queue). **Expected gain**: EO 17 → 25–30 fps.

**Implementation sketch:**
```python
# eo/_v4l2_grab_proc.py — child process
def grab_proc(shm_name, ctrl_queue, status_queue):
    import multiprocessing.shared_memory as shm
    cap = RawV4L2Backend("/dev/video0", 2472, 2064)
    cap.open()
    buf = shm.SharedMemory(name=shm_name)
    bgr_view = np.frombuffer(buf.buf, dtype=np.uint8).reshape(H, W, 3)
    seq = 0
    while True:
        # check ctrl_queue non-blocking for AE/exposure cmds
        try:
            cmd = ctrl_queue.get_nowait()
            if cmd[0] == "exposure":
                cap.set_exposure_ext(cmd[1])
        except queue.Empty:
            pass
        bgr = cap.grab()
        if bgr is not None:
            np.copyto(bgr_view, bgr)  # one memcpy to shm
            seq += 1
            status_queue.put(("frame", seq, cap.last_raw_stats))
```

### Option B — Rewrite grab() in C++ as a Python extension

`pybind11` extension with the V4L2 ioctls + AGC + cvtColor entirely in
C++. Releases GIL during the work, returns the BGR ndarray.

**Effort**: 2–3 days (build system + Jetson cross-compile). **Risk**: low
(small surface area, easy to test). **Expected gain**: EO 17 → 25–30 fps,
same as Option A. **Bonus**: faster JPEG via libjpeg-turbo.

User suggested: *"I can also be VERY radical like writing the entire code
in C++ if that helps."* — this is the smallest piece worth rewriting.
~400 lines, owned scope. Bigger ports (the whole eo_manager) would be
much harder to justify.

### Option C — opencv-cuda for AGC stretch

`cv2.cuda_*` runs the AGC + cvtColor on the iGPU. Saves ~12 ms/frame +
zero CPU time.

**Effort**: 0.5 day to compile opencv-with-cuda for L4T. **Risk**: low
(stable API). **Caveat**: the PyPI `opencv-python` aarch64 wheel does NOT
include CUDA support. Have to compile from source on-device or grab a
JetPack-specific build.

**Hidden cost**: Frame must be uploaded GPU→CPU at least once for the
JPEG encoder, which only has CPU codepaths. So the win is partial
(~10 ms saved on grab, +1 ms upload, net ~9 ms).

### Option D — Read-twice + zero-copy view in V4L2 grab()

Today we copy the whole 10MB mmap buffer per frame because the kernel
will refill it after our `VIDIOC_QBUF`. Alternative: do the AGC stretch
**before** the QBUF, while we still own the buffer. Then return the
cv2-allocated BGR (which is independent). Zero-copy on the input.

**Effort**: 0.5 day. **Risk**: low. **Expected gain**: ~5–8 ms/frame.

Already prepared in this round but reverted as unsafe — needs careful
restructuring of read()/grab() so the AGC happens between DQBUF and
QBUF.

---

## Bottleneck #2 — Thermal stuck at ~20 Hz despite YOLO amortized

After Fix #2 (classify_interval=4 + MOSSE), thermal YOLO is amortized
~5× lower load. We expected 28–35 Hz, got 18–20.

**Hypothesis**: the thermal process_loop has a per-frame cost that's
dominant once YOLO is throttled. Candidates:
- AGC percentile on 640×512 uint16 (~8 ms) — agent claimed already
  cached, verify
- tophat 31×31 morphology (3 ms)
- detection_tracker Kalman update (2 ms)
- JPEG encode (1 ms after fix #6)
- MOSSE updates per active track (1–3 ms × N tracks)

Total ~15–20 ms/frame matches the observed 50 ms total budget at 20 Hz.

**Next probe**: instrument `thermal_manager._process_loop` with
per-stage timings, log over 60 ticks. Same `[ws_sender_prof]` pattern
the GUI already uses.

**Likely fix**: cache the AGC percentile (same trick as EO fix #5).
Tophat is already 3 ms (cv2-optimized). Can also reduce MOSSE
`max_patch_dim` from 96 → 64 if needed.

---

## Bottleneck #3 — Linux YUV→BGR conversion may be wasteful

The IMX568 is mono. We replicate Y → BGR for downstream YOLO compat.
That's a 2× memory copy + 2× JPEG encode size + 2× WS bandwidth for
no information gain.

**Option**: change the entire pipeline to work in single-channel grayscale.
Requires touching `eo_processor.py`, `eo_manager.py`, the YOLO predict
calls (need imgsz_in_channels override), and the JPEG encoder.

**Effort**: 1 day. **Risk**: medium (silent breakage if any path assumes
3-channel). **Gain**: ~3 ms/frame + 10 MB allocation/frame + ~30% JPEG size.

---

## Bottleneck #4 — Process-pool for inference threads (system-wide)

The runtime audit found **36 Python threads** in seeker, all sharing
one GIL. Capture threads release GIL during I/O, but the post-capture
processors (YOLO pre/post, AE, fusion association) hold it.

**Big win architectural change**: move thermal and EO processors to
separate processes. Keep the bus + GUI in main. Each sensor runs in
its own GIL.

**Effort**: 2–3 days. **Risk**: high (frame serialization across processes,
shared memory ownership, lifecycle). **Expected gain**: 30–50% latency
reduction system-wide; both EO and thermal could likely hit 30 Hz+.

---

## Bottleneck #5 — JPEG encoder is single-thread libjpeg

The thermal + EO JPEG encodes both run on the process thread. After
Fix #6, EO at 900px wide is ~5–8 ms encode. Thermal at 640px is ~1 ms.

**Option**: hardware JPEG encoder via `nvjpeg` (Jetson Multimedia API).
Encodes on the iGPU/VIC at < 1 ms.

**Effort**: 1 day to wire up via cython/ctypes. **Risk**: medium (requires
JetPack 5.x SDK; not pip-installable). **Gain**: 4–7 ms/frame on EO.

---

## Bottleneck #6 — AE step period

`auto_exposure.step_period_s` defaults to 1.5 s. AE convergence takes
8–12 steps = 12–18 s. During convergence, the camera looks wrong.

**Option**: drop to 0.5–0.75 s with a tighter dead-band on stable scenes.
Risk: AE overshoot on rapid scene changes. Tunable in YAML.

**Effort**: 30 min config + 1 hour bench testing.

---

## Bottleneck #7 — Radar A/A mode (out of scope per user)

User explicitly said: *"the radar in A/A is VERY not optimized — but
let's leave it out of scope for now. if i put it in A/A mode - it
will suffocate the entire system."*

DCA1000 raw-ADC + AA mode pipeline lives in `radar_dca/`. The PMM
detector + symmetric-sideband matched filter is mentioned as "shipped
but tuning open" in the v1.0.0 release notes.

**When picked up**: profile `radar_dca/dca_pipeline.py`, the AA mode
specifically. Likely candidates: numpy FFT chains that should be
batched, threading model, packet drop policy.

---

## Suggested order if continuing

1. **Profile thermal process_loop** → find the actual per-frame cost.
   Cheap, high-information.
2. **Cache thermal AGC percentile** (mirror EO fix #5). 30 min.
3. **Option D — zero-copy V4L2 grab restructure**. 0.5 day, ~5 ms/frame
   on EO.
4. **Drop BGR replicate** for EO (3-channel → 1-channel pipeline).
   1 day, requires careful downstream review.
5. **opencv-cuda compile** for AGC stretch on iGPU. 0.5 day install +
   2 lines code change.
6. **C++ rewrite of `_v4l2_raw_backend`** via pybind11. 2 days.
   This is the path to 25 Hz EO.
7. **nvjpeg encoder** for both EO and thermal. 1 day.
8. **Process pool architecture** — biggest change, do last when other
   wins are confirmed. 2–3 days.

---

## What NOT to do

- **Reduce display_max_width below 640**: browser canvas is 640px, going
  smaller costs sharpness with no encode benefit.
- **Increase JPEG quality above 90**: marginal visual gain, exponential
  size cost.
- **Remove the trigger-disable auto-recovery** in `RawV4L2Backend.grab()`:
  it's been observed to silently re-arm; the recovery is cheap (~0.1 ms
  every 300 frames) and saves debugging.
- **Touch the radar A/A pipeline** before user signals readiness.

---

## Reference: per-fix benefit measured 2026-05-09

| Fix | Before | After | Gain | Notes |
|---|---|---|---|---|
| f5f2346 — RAW12 reinterp + trigger-disable | EO white-saturated | clean BGR | quality fix | Foundation for everything else |
| 4e9946b — cv2.convertScaleAbs AGC | grab 88 ms | grab 60 ms | 32% | Replaced numpy LUT |
| 78a641a — single fd / no 2nd open | EO 1 Hz cliffs | 5–10 Hz steady | huge | UVC kernel throttle eliminated |
| 99634b6 — stream-fd AE + MOSSE on | thermal 12–13 Hz, EO 5–10 Hz | thermal 18–20, EO 14–19 | 50–100% | Fixed periodic AE-tick stalls |
| 67bbf04 — orjson + stats cache + JPEG | as above | thermal 18–20, EO 14–19 | small | WS overhead now negligible |
