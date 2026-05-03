# MOSSE Tracker — Compute Budget on Jetson AGX Xavier

Sizing for adding the per-target MOSSE correlation tracker on top
of the existing 3-sensor + 2-YOLO-pipeline pipeline running on the
target hardware.

## Hardware

- Jetson AGX Xavier (JP5.1.5)
- 8× Carmel ARMv8 cores @ 2.26 GHz
- 512-core Volta GPU + 64 Tensor Cores
- OpenCV 4.x without contrib (we use a pure-numpy MOSSE)

## Current per-frame budget (no MOSSE)

| Pipeline | Component | Hz | Per-frame ms | Hardware |
|---|---|---|---|---|
| EO | YOLOv8n + ByteTrack @ imgsz=832 | 30 | ~30-40 | GPU |
| Thermal | drone classifier (YOLOv8n) | 60 | ~10-30 (small frame) | GPU |
| Thermal | HV classifier (YOLOv8n) | 60 (every classify_every) | ~10-30 | GPU |
| Radar | DCA1000 ingest + DBSCAN + Kalman | 15 | ~5-15 | CPU |
| Fusion | cross-sensor matching | 15 | ~5 | CPU |
| Gimbal | predictor + closed-loop + bus I/O | 60 | ~3-6 | CPU |

**GPU near-saturated** at ~80-100% during simultaneous EO+thermal-HV+thermal-drone
inference. **CPU has slack** (~30-40% of one core typical, plenty of headroom on
the 8-core part).

## Adding MOSSE — per-target per-frame cost

| Target patch size | MOSSE update | Hardware |
|---|---|---|
| 30×30 (small distant) | ~0.5-1 ms | CPU (numpy FFT) |
| 100×100 (typical car) | ~1-3 ms | CPU |
| 200×200 (close-range) | ~3-7 ms | CPU |

Pure-numpy + scipy FFT — no opencv-contrib dependency, no CUDA. Each tracker is
independent; running multiple trackers serially on one core is fine.

## Operating modes

### Mode A — TRACK-engaged only (current default, when enabled)

MOSSE runs only while operator has TRACK locked on a fused-track ID. Pool
contains exactly the per-sensor track IDs that map to the engaged target.

| Sensor scope | # MOSSE | Per-frame load |
|---|---|---|
| 1 EO ByteTrack ID at 30 Hz | 1 | ~3 ms × 30 = 90 ms/s ≈ 9% of one core |
| 1 thermal ByteTrack ID at 60 Hz | 1 | ~1.5 ms × 60 = 90 ms/s ≈ 9% of one core |
| **Total** | **2** | **~0.18 of one core (2.3% of total CPU budget)** |

Negligible. Fits comfortably alongside everything else.

### Mode B — All ByteTrack IDs (alternative, more permissive)

MOSSE runs for every ByteTrack ID in scene, regardless of operator engagement.

| Sensor scope | Typical # IDs | Per-frame load |
|---|---|---|
| EO @ 30 Hz, ~3 IDs in scene | 3 | ~9 ms × 30 = 270 ms/s ≈ 27% of one core |
| Thermal @ 60 Hz, ~3 IDs in scene | 3 | ~4.5 ms × 60 = 270 ms/s ≈ 27% of one core |
| **Total** | **6** | **~0.55 of one core (6.9% of total)** |

Still well under any compute concern. Mode B is the simpler integration; Mode A
is the right call for a cleaner architecture (only do work when operator asks).

## YOLO throttle interaction

The MOSSE pool's purpose includes "fill the gaps when YOLO is throttled." If we
keep YOLO at 30 Hz on EO, MOSSE runs every frame but just reseeds on the YOLO
output most of the time — small win.

The big win comes from **throttling YOLO 30 Hz → 10 Hz** (classify_interval_frames=3
on EO). MOSSE fills the 20 Hz gap at frame rate. Net effect:

| Component | Before | After |
|---|---|---|
| EO YOLOv8n GPU load | ~30 ms × 30 = 900 ms/s ≈ 90% GPU | ~30 ms × 10 = 300 ms/s ≈ 30% GPU |
| EO MOSSE CPU load | 0 | ~9-27% of one core |
| **Net** | **GPU saturated** | **GPU 60% freed; CPU +0.27 cores** |

A clear net win — and the GPU headroom freed can either accommodate higher
imgsz on YOLO (better recall on small/distant targets) or run thermal HV at
higher rate or both. **YOLO throttle is the biggest lever in this whole picture;
without it, MOSSE doesn't really pay off.**

## Failure modes to watch

1. **MOSSE drifts to background** (clouds for sky-targets, asphalt for ground
   vehicles). Mitigation: aggressive PSR gate (default `psr_lost: 7`), reseed
   on every YOLO hit (kills accumulated drift).
2. **Bbox size doesn't adapt.** If target's pixel size changes >30% (e.g. drone
   approaching), MOSSE can't shrink/grow the patch. YOLO's reseed handles this
   — bigger bbox from YOLO replaces the old MOSSE patch. As long as YOLO runs
   at least every ~1 second, size adapts.
3. **Multiple targets in tight cluster.** ByteTrack IDs can swap if association
   gets confused; MOSSE reseeds whichever ID YOLO assigned to whichever bbox.
   No worse than legacy.

## Validation strategy

- Unit tests: algorithmic invariants (FFT, peak detection, gaussian, lifecycle
  state machine). 15 tests in `vision/tests/`.
- Live tests: enable `eo.correlation_tracker.enabled: true`, run TRACK on a
  static target during a manual slew, verify bbox stays alive through the slew.
  Then a moving target, verify the bbox follows smoothly between YOLO ticks.
- Recording-replay validation: TBD — would need a per-frame "what bbox did
  fusion publish" inspector against the EO-frame jpegs. Not a blocker.
