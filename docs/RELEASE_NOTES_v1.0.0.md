# Seeker-01 — v1.0.0 Release Notes

**Tag:** `v1.0.0`
**Date:** 2026-04-29
**Branch fast-forwarded:** `main` ← `revert/structural-and-timing-fix`
**Purpose:** First field-deployable cut. Hand-off baseline for the
Jetson Xavier AGX port and for in-field DJI FPV / Shahed-class drone
testing.

---

## What V1.0 contains

### Phase A — Thermal + GUI (production)
- FLIR Boson 640 Y16/AGC8 capture (DSHOW), two-thread split.
- Tophat heat detector with persistence tracker (min_hits, max_misses, EMA).
- AGC + colormap (live: WHITE_HOT, AGC8 fallback, `mode=global`).
- Y16 + ROI/gates AGC infrastructure dormant in tree (one-line YAML flip).
- FastAPI + WebSocket GUI on `127.0.0.1:8080`.

### Phase B — Classification (production)
- `models/seeker_thermal_hv.pt` — person + vehicle (mAP50 0.867).
- `models/seeker_thermal.pt` — drone (mAP50 0.945).
- ROI-gated drone classifier + full-frame H/V classifier.
- `classify_interval_frames` and `classifier_hv_imgsz` auto-branch on CUDA.

### Phase 2 — Late fusion (production)
- Radar is a first-class `FusionManager` contributor.
- 3-way cross-sensor association (EO ↔ thermal, then radar attach by IoU).
- Per-sensor decay — `sensors_now` reflects who's currently seeing it.
- Class-lock: born `RADAR_TARGET`, locks to camera class once EO/thermal joins.
- Unified single-overlay-per-target rendering; cyan duplicate path removed.
- World-frame fusion port shipped (default ON, A/B-able via flag).

### Phase 2.5 — Tracking + gimbal (production)
- Recorder + replay infrastructure (`recording/`, `scripts/replay_*.py`).
- Predictor tuning replay-validated against regression recordings.
- TRACK button → closed-loop pixel-error control (default ON).
- Maestro auto-reconnect — fixes the silent "device does not recognize
  the command" lockup that was killing live sessions.
- Stage A optical-residual diagnostic (LK on EO + thermal anchor).
  Stage B integrator deferred (default OFF in YAML).

### Phase 3 — DCA1000 raw-ADC (validated, partial)
- Single firmware (`mmw_demoDDM`), single jumper config, single cfg push.
- TLV + raw ADC stream simultaneously over UART + LVDS.
- Three host-side modes: `stock` / `ag` / `aa`. Mode picker switches without chip restart.
- AA architecture validated 2026-04-29 (119 MB raw ADC capture).
- PMM detector (symmetric-sideband matched filter) **shipped but tuning
  open** — see "Known incomplete" §1.
- Recorder add-ons (this release): `meta.yaml` per session +
  `dca_index.csv` per-packet seek index.

### EO calibration
- **V1 — Focus assist** (`scripts/eo_focus_assist.py`) shipped.
- **V2 — Intrinsic + stereo calibration** scaffold landed
  (`calibrate_thermal_intrinsics.py`, `calibrate_eo_thermal_stereo.py`,
  `calibration_capture.py`). Runtime defaults to v1; v2 path activates
  when calibration files are populated. Verification gate documented in
  `docs/CALIBRATION_V2_VERIFICATION.md`.
- **V3 — Wire intrinsics into `fusion/angular.py`** still gated on
  regression replay, NOT in V1.0.

### Cross-cutting infrastructure
- ID chain unified (`E#` / `T#` / `R#` / `#`), backend stamps `fused_id`
  on every raw det/target.
- Software extrinsic sliders (radar/thermal az/el bias).
- Recorder + 5 replay tools + REPLAY badge / clock.
- 183 tests passing.

---

## Recording — what gets captured when REC is pressed

```
recordings/<name>/
  <name>.jsonl              ← bus stream: eo/frame, thermal/frame,
                              radar/frame, fusion/tracks, gimbal/state,
                              events, session/header (config snapshot)
  <name>.meta.yaml          ← schema_version, mode, profile_name,
                              awr_cfg_path + sha256, awr_cli_port/baud,
                              dca endpoints, frame_dims, pmm knobs,
                              capture (host_ip, socket_recv_buffer)
  <name>_radar.bin          ← raw ADC byte stream (aa mode only)
  <name>_radar.csv          ← per-packet index for .bin seek:
                              ts_host_ns, byte_offset, payload_len,
                              seq_num, chunk_offset
```

The `meta.yaml` skips `awr_firmware.*`, `dca.fpga_version`,
`capture.host_nic`, `capture.mtu` (best-effort fields, not obtainable
without extra subprocess overhead at REC-press). Field-side analyst
can fill these in `notes.md` or run `DCA1000EVM_CLI_Control fpga`
manually if needed.

A 3-minute aa-mode recording is approximately 10.8 GB of `.bin` data
plus ~300 MB of JSONL/CSV. Verify host SSD sustained write rate
before long captures.

---

## Known incomplete (NOT blocking V1.0 — explicitly deferred)

### P0 — Required before reliable in-field results
1. **PMM detector tuning** — currently 100% of frames produce hits,
   blade-freq stuck at 52.7 Hz (band-edge artifact). Fix in
   `radar_dca/pmm_detector.py`: mask range bins 0..2 (TX leakage),
   subtract slow-time mean per range bin (clutter), raise threshold
   6→12 dB, add absolute-power gate. See `docs/MORNING_HANDOFF.md`.
2. **Radar bench-cal** — az/el extrinsic biases still 0°. Class-promotion
   (`RADAR_TARGET → vehicle`) won't fire until the projected radar bbox
   sits on the EO bbox of the same target. See `docs/NEXT.md` P0.
3. **PyInstaller build of V1.0 dist** — see "Build status" below.

### P1 — Quality-of-life
4. **Persist extrinsic sliders to YAML** — `NEXT.md` P0.5. Currently
   restart reverts to YAML defaults. Need SAVE button + `ruamel.yaml`
   patch.
5. **GUI 3-mode (stock/ag/aa) dropdown** — `BackendFactory` ready,
   control not wired.
6. **`cameras_on_gimbal: true`** + `gimbal_calibrate.py` re-run when
   the rig physically goes on the gimbal.

### P2 — Future work (DO NOT add to V1.0)
- World-frame fusion port (structural fix for YOLO id-swap during slew).
- Cold-anomaly heat-detector mode (bidirectional MAD) — hard prereq
  for ROI-only edge port.
- EO drone fine-tune (Det-Fly / Drone-vs-Bird / Anti-UAV-RGB).
- Long-range humans retrain (yolov8s + tiled inference).
- Radar profiles AA-2..AA-5 (only AA-1 ships in V1.0; others built
  when fault modes show up — see `docs/PHASE_3_AA_PROFILES.md`).
- BoT-SORT camera-motion compensation (tracker swap when on gimbal).
- HDF5 recording (JSONL is the canonical format for V1.0).

---

## Edge port — handoff notes for the Jetson agent

1. **Read first:** `docs/EDGE_OPTIMIZATION_GAP.md` — Asaf's canonical
   per-stage CPU cost map and lever order. Don't re-derive.
2. **Target:** Jetson Xavier AGX, JetPack 5.1.5, sm_72 Volta, 32 TOPS.
   Engines must be built on-device.
3. **Staging:** `c:\jetson-stage\` has the runbook + L4T archives + creds.
4. **Lever order in priority:** instrument → throttle knobs → cheap AGC
   → smaller tophat → ONNX/TensorRT export → INT8 → ROI-gate H/V
   (gated on cold-anomaly mode landing first).
5. **Don't break the bench while porting** — develop the Jetson port
   on a separate branch off `v1.0.0`, never push to `main` until the
   port stabilizes. The Windows/laptop bench remains the V1.0 reference.
6. **PyInstaller is Windows-only** — for Linux/Jetson use a plain venv +
   systemd unit per `README.md` §"Edge device deployment".

---

## Build status

PyInstaller dist build status is recorded at the bottom of this file
during the V1.0 release autonomous run. If the build succeeded,
`dist/seeker_bench/seeker_bench.exe` is the runnable artifact for
Windows laptops that match the bench. The Jetson port does NOT use
this — it builds from source on-device.

(See `_BUILD_STATUS_v1.0.0.txt` in this directory for the autonomous
build log.)

---

## Migration notes

- No breaking config changes. Existing `config/app_config.yaml`
  files from `revert/structural-and-timing-fix` work as-is on V1.0.
- `config/calibration.json` — calibration agent's V2 schema is
  forward-compatible (v1 still works when v2 keys are absent or
  zeroed).
- Recordings made before V1.0 (no `.meta.yaml`) are still replayable;
  the new files are additive.

---

## Verification at tag-time

- `pytest --ignore=tools -q` → 183 passed in 135s
- `pytest recording/tests/ -q` → 13 passed (7 meta + 6 dca_index)
- `python scripts/_smoke_id_chain.py` → PASS (5/5 sensor link chains)
- `python scripts/_smoke_recorder.py` → OK (all 7 channels)

---

## Branches & cleanup

- `main` is now V1.0.0 (was 82 commits behind active dev — fast-forwarded
  cleanly, no divergence).
- `revert/structural-and-timing-fix` remains the active dev branch.
- `feature/phase-3-dca1000` was deleted (work integrated; redundant).
- `seeker_bench_phase3` worktree was removed (work superseded).
- 11 historical handoff `.md` files moved from repo root to
  `docs/handoffs/` (with full git history preserved).
- `docs/INDEX.md` is the documentation entry point.

---

## Credit

Built across multiple parallel Claude agent sessions on the
`asaftg/seeker-bench` repo. Recorder add-ons + repo cleanup +
release tag landed in the autonomous V1.0 push on the night of
2026-04-29 → 30.
