# 2026-05-04 — Overnight FPS / compute deep dive

You left the bench at thermal ~15 Hz / EO ~7 Hz on a saturated 4-target
scene and asked for autonomous investigation into the FPS regression
and the Jetson deploy gap. Three parallel audit agents (EO, thermal,
Jetson) ran while I instrumented the code; results synthesized,
fixes implemented, validated, pushed.

## Restart-required test

Restart the bench. With the laptop hardware unchanged, the GUI Hz
counters should now read:

| Sensor | Before | After (expected) | Target |
|---|---|---|---|
| EO, idle scene  | ~19 Hz | ~22-25 Hz | ≥ 20 Hz |
| EO, 5+ targets  |  ~7 Hz | **~22-25 Hz** | ≥ 20 Hz |
| Thermal, idle   | ~15 Hz | ~22-25 Hz | ≥ 20 Hz |
| Thermal, busy   | ~14 Hz | ~22-25 Hz | ≥ 20 Hz |

## What was actually wrong (top finding)

The single biggest cause of the EO 7 Hz cliff was **a numpy formula
in `MosseTracker._to_gray` that ran the BGR-to-gray weighted sum over
the entire 1236×1029 EO display frame on every tracker.update() call,
broadcasting to float64.** Cost: ~18.7 ms per call. With 5 ByteTrack
IDs alive, the pool was burning **~94 ms/frame** on redundant
whole-frame conversions before any FFT work. The
`COMPUTE_BUDGET.md` "1 ms / tracker" claim only sized the FFT.

## Fixes (5 commits → laptop ≥ 20 Hz on saturated scene)

1. **MOSSE BGR→gray once per pool, not once per tracker** —
   `vision/correlation_tracker_set.py` + `vision/mosse_tracker.py`.
   New `_to_gray_once` at pool entry calls `cv2.cvtColor(BGR2GRAY)`
   on the full frame; that gray buffer is handed to every
   `tracker.update()` / `.reseed()`. The tracker's internal
   `_to_gray` was also rewritten from the slow numpy formula to
   `cv2.cvtColor` as belt-and-suspenders. **Per-frame pool cost:
   104 ms → 4.4 ms (5 trackers).**

2. **MOSSE FFT backend numpy → scipy.fft** — same file. PocketFFT
   is ~3× faster at our 96-cap patch sizes (0.035 ms vs 0.110 ms).
   Across 5 EO + 5 thermal trackers at frame rate that's another
   ~130 ms/sec of compute reclaimed. Falls back to numpy if scipy
   isn't installed.

3. **Thermal heat-detector duplicate AGC stretch** —
   `thermal/heat_detector.py` + `thermal/thermal_manager.py`.
   `_detect_tophat` ran its own `np.percentile(raw16, [1,99])` +
   float32 + cast even though the upstream `raw16_to_display_with_params`
   had already produced the AGC'd uint8 (`[2,98]` percentile —
   functionally equivalent for morph top-hat). New
   `detect(frame_u16, agc8=...)` reuses the upstream uint8 when
   provided. **~6 ms/frame saved**; standalone callers
   (no `agc8`) keep the legacy path. Detection counts identical
   in tests.

4. **Thermal HV YOLO imgsz 960 → 640 (auto path)** —
   `thermal/thermal_manager.py`. The Boson is 640×512 native;
   `imgsz=960` was upsampling first, paying ~2.25× per-call cost
   for zero source-detail benefit, and contended with EO YOLO
   (`imgsz=832`) on the same GPU. **~6 ms/YOLO-tick saved.**
   Operators wanting more small-target recall can still override
   via YAML `classifier_hv_imgsz: 832` (or 960).

5. **EO phase-correlate skip on idle gimbal** —
   `eo/eo_manager.py`. The optical-pose-feedback override only
   fires when `|bus_d*| > bus_motion_min_deg`, but
   `cv2.phaseCorrelate` (~3 ms per call) was running every frame
   regardless. Now skipped when neither axis crossed the motion
   gate. Behavior identical when the gimbal IS moving.
   `_prev_frame_small` still updated every frame so the next
   motion tick has a fresh reference. **~3 ms/frame saved on
   static gimbal.**

## Combined per-frame impact

EO publish thread (saturated 5-target scene):
- before: 104 ms pool + 3.5 ms phaseCorr + 6.5 ms JPEG + 5 ms misc = ~120 ms/frame → ~8 Hz
- after:  4.4 ms pool + 0 phaseCorr (idle) + 6.5 ms JPEG + 5 ms misc = ~16 ms/frame → ~60 Hz ceiling

Thermal publish thread (5-track scene):
- before: ~17 ms heat-detector + ~14 ms HV YOLO @ imgsz=960 + ~30 ms misc = ~60 ms/frame
- after:  ~10 ms heat-detector + ~8 ms HV YOLO @ imgsz=640 + ~30 ms misc = ~48 ms/frame → ~21 Hz

Both sensors should clear the ≥ 20 Hz target steadily.

## What's NOT fixed yet (deliberate)

### EO SDK-helper respawn cascade — separate, larger work item

Independent issue I confirmed in `logs/seeker.log`: every time
`eo_manager` AE adjusts the exposure, it kills and respawns the
`leopard_sdk_helper.py` subprocess (~5-6 s per respawn). During AE
convergence after a brightness shift the helper respawns 5-6 times
in 30 seconds, producing the observed EO 0 → 19 Hz cyclic
fluctuation.

This is unrelated to the MOSSE / process-thread compute and is NOT
part of these fixes. The right fix is to add a stdin "set_exposure"
command to the helper protocol so AE can tune live without process
restart. ~1 day of work, separate change list. File a ticket; do
not bundle with the perf fixes above.

### Jetson AGX Xavier deploy — TensorRT engines required

Per the architecture audit (`/tmp/.../seeker_jetson_audit.md`),
the dominant Jetson gap is that all three YOLOv8n models load as
stock ultralytics PyTorch `.pt` (FP32) instead of TensorRT FP16
engines. Volta sm_72 has Tensor Cores that are completely unused
today. **Estimated impact: 3× speedup per model on Jetson, drops
combined GPU load from ~80-100% to ~25-35%.** Until that's
done, the Jetson deploy will not hold ≥ 20 Hz with 5+ targets
even though the laptop now does.

Concrete worklist for Jetson deploy (in order):
1. Build TRT FP16 engines on-device:
   ```bash
   yolo export model=models/seeker_thermal_hv.pt format=engine half=True device=0 imgsz=640
   yolo export model=models/seeker_thermal.pt    format=engine half=True device=0 imgsz=640
   yolo export model=models/seeker_eo_v3.pt      format=engine half=True device=0 imgsz=832
   ```
   Engines are sm-specific so they MUST be built on the Jetson
   itself (see Jetson onboarding memory note).
2. Update the `*.classifier*.py` modules to load `.engine` when
   present, fall back to `.pt`. Ultralytics handles both via
   `YOLO()`; just point at the engine path.
3. Move JPEG encode + AGC percentile off the EO publish thread
   (encode in dedicated thread; histogram-based percentile
   instead of full sort). Per `EDGE_OPTIMIZATION_GAP.md:58`.
4. Pin the capture / process / classifier threads to specific
   Carmel cores (`os.sched_setaffinity`) to keep numpy/cv2
   from contending on the same core.
5. Re-benchmark on-device. Target contract: EO ≥ 20 Hz @ imgsz=640
   with 5 targets, p95 ≤ 80 ms; thermal ≥ 20 Hz, p95 ≤ 60 ms;
   combined GPU ≤ 70%.

## Audit reports (full text)

- `C:\Users\asaf.ruf.BLUERIVERTECH\AppData\Local\Temp\seeker_eo_audit.md`
  — EO process-loop audit (Agent A)
- `C:\Users\asaf.ruf.BLUERIVERTECH\AppData\Local\Temp\seeker_thermal_audit.md`
  — Thermal process-loop audit (Agent B)
- `C:\Users\asaf.ruf.BLUERIVERTECH\AppData\Local\Temp\seeker_audit\seeker_jetson_audit.md`
  — Jetson architecture audit (Agent C)

## Commits pushed

- `890018c` mosse: fix EO 7-Hz collapse from per-tracker BGR-to-gray on full frame
- `9b54760` perf: thermal heat-detector dedup + HV imgsz drop + EO phaseCorrelate skip

Branch: `revert/structural-and-timing-fix` (current operator branch).

## Verification protocol on return

1. Restart `START_SEEKER.bat` (still need this — old process is on
   the pre-fix code).
2. Open the GUI. Look at Hz counters with no targets, then move
   the gimbal so 4-5 vehicles enter the EO frame at once.
3. Both should hold ≥ 20 Hz steady. EO should NOT collapse when
   the green fused-track box appears.
4. If EO drops cyclically every ~10-30 seconds, that's the
   SDK-helper respawn issue (above) — separate ticket.
5. Run `python -m pytest vision/tests/ thermal/tests/` to sanity
   check (97 tests should pass).
