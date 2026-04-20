# Edge Deployment Optimization — Gap Analysis (DEFERRED)

**Status:** Documented, not acted on. User decision April 20, 2026 —
prioritize EO sensor + pan-tilt gimbal integration before any
optimization work. Revisit before a real hardware port.

## Why this doc exists

The thermal pipeline currently runs on an RTX A4000 dev laptop. The
eventual target is an **edge device — Raspberry Pi preferred, Jetson
as fallback**. The system will also carry an EO camera and radar,
so the thermal pipeline cannot monopolize the compute budget.

An audit was run on April 20, 2026 to identify where the cycles go.
Findings captured here so the next Claude session doesn't have to
re-derive them.

## Where compute goes today (per frame)

Relative cost on an ARM CPU (Pi 4/5 class, no accelerator):

| Stage | ARM cost share | Notes |
|---|---|---|
| **H/V YOLO full-frame @ 960** | ~80% | the elephant — dominates everything |
| Tophat 31×31 morphology | ~8% | always runs, every frame |
| AGC percentile + colormap | ~4% | every frame; `np.percentile` is a full sort |
| Drone YOLO on heat-blob ROIs | ~3% | cheap — already ROI-gated |
| JPEG encode @ q80 | ~3% | GUI path |
| Tracker / bus / copies | ~2% | already tight |

On the A4000 this all fits in 25–35 ms. Naively ported to a Pi it
would not come close to real-time.

## Currently well-tuned — do NOT change during optimization

- **Two-thread split** (`ThermalCapture` + `ThermalProcess`) in
  `thermal/thermal_manager.py` — capture never blocks on inference.
- **Drone classifier is ROI-gated** — YOLO only runs on heat blobs,
  not the full frame. This is the pattern H/V should eventually
  follow.
- **Lazy model loading** — both classifiers skip cleanly if weights
  are missing, no crash.
- **Persistence trackers** on both classifier outputs kill flicker
  FPs and are cheap.
- **Config-driven adaptation** — `classify_interval_frames: auto`
  and `classifier_hv_imgsz: auto` already branch on CUDA presence.

## The real levers (in the order I'd pull them)

Not doing any of this now. Listed so it's cheap to pick up later.

1. **Instrument first.** Add per-stage timing counters (logs/perf.csv
   or an Engineering-tab readout) so the next optimizer sees real
   numbers on the real device, not A4000 benchmarks. ~30 min.
2. **Expose every throttle as a config knob.** Let `app_config.yaml`
   force `classify_interval_frames`, add `heat_detect_interval_frames`,
   `agc_recompute_interval_frames`. No algorithm changes, all fallback.
3. **Cheap AGC.** Replace per-frame `np.percentile` with a histogram
   estimate on a 160×128 downsample, recomputed every ~10 frames.
   Visually indistinguishable, ~10× cheaper.
4. **Smaller / reused tophat.** Drop kernel 31→21, run every 2nd
   frame, reuse mask. Roughly 2× speedup on that stage.
5. **ONNX Runtime backend for both YOLOs** (with XNNPACK EP on ARM).
   Typically ~2× PyTorch on ARM, drop-in. Opens the door to int8.
6. **ROI-gated H/V.** Same pattern as the drone classifier. Requires
   a **cold-anomaly mode in `heat_detector.py`** (bidirectional MAD
   threshold `abs(residual) > k*MAD`), otherwise cold-vehicle-on-warm-
   pavement targets become invisible to the blob stage and never get
   an ROI handed to the classifier. This is a hard prereq, not
   optional — documented as open issue #3 in `PHASE_B_RESULTS.md`.
7. **Optional: NCNN backend** behind the classifier interface.
   Another ~2× over ORT on Pi, but adds a dependency. Only if 1–6
   aren't enough.
8. **Only then** revisit the "long-range humans" model work
   (Phase C §1–4 in `PHASE_B_RESULTS.md`). A yolov8s-at-960 that
   can't run on the target device is a negative deliverable. The
   right artifact is something like yolov8n-int8-at-416 that hits
   20+ FPS on a Pi with acceptable recall.

## Open architectural questions (need user input before work starts)

These are the forks that make the optimization strategy diverge.
**Not answered yet.** The next Claude session should ask before
designing.

1. **Target device, really.** Pi 4? Pi 5? Jetson Orin Nano?
   "Unknown, assume worst case" is also a valid answer — it just
   pushes us to the Pi-CPU path, which is the strictest.
2. **Accelerator budget?** A $25 Coral USB or $70 Hailo-8L changes
   the whole strategy (int8 YOLO at 60+ FPS becomes realistic).
   Without one we're stuck on CPU YOLO — viable but constrained.
3. **Full-frame H/V — keep or drop?** The full-frame path is what
   catches cold vehicles the heat detector can't see. Options:
   - (A) Drop full-frame, go ROI-only, add cold-anomaly heat mode.
   - (B) Keep full-frame but shrink aggressively (imgsz=320,
     throttle, quantize). Loses distant-person recall.
   - (C) Hybrid — full-frame at 320 every ~6 frames plus ROI-gated
     every frame. Most code but best coverage.
4. **GUI co-located on the Pi, or remote?** If remote, JPEG encode +
   WebSocket push becomes the bottleneck instead of YOLO — totally
   different optimization.

## Cost of deferring

Low, provided the next additions (EO sensor, gimbal, radar, fusion)
keep using the same `FrameBus` + dataclass-at-boundary pattern. If
they do, optimization stays a local concern inside each module and
can be done later without architectural rework.

The one thing to **avoid during EO / gimbal work** is introducing a
second full-frame always-on neural net on the EO stream. If EO gets
object detection, gate it the same way the drone classifier is
gated — run detection on ROIs, not on the whole frame — so we don't
dig a deeper hole before the optimization pass.
