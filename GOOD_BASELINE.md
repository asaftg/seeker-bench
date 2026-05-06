# Good baseline — Seeker-01 working state 2026-05-05

This file captures the **known-good** revert chain for the lock-mode
work. Three checkpoints exist on origin:

| Tag | Commit | What it captures |
|---|---|---|
| `good-baseline-v1` | `59c4a4f` | Pre-lock-mode. Operator graded better than 7/10. All FPS / hunting / engagement-reset fixes shipped, no lock mode at all. |
| `pre-lock-v2` | `d51c9be` | Lock mode v1 in the tree but DISABLED via YAML kill switch. Manual gimbal softened from 120 to 45 dps. Lock-mode v1 had multi-target swap bugs in dense scenes (recordings/lock poorly.jsonl). |
| `lock-v2` | (this commit) | Lock mode v2: ID-only auto-reseed (no IoU/class search), GUI suppresses duplicate fused box for engaged ID, distinctive corner-bracket render with `LOCK #ID` label, recorder serializes lock fields, diagnostic events emitted. Default `enabled: false`. |

The upcoming lock-mode tracker (`vision/lock_tracker.py`) is additive
code — if it doesn't deliver the rock-solid behavior promised, the
baseline is recoverable in two ways.

## How to revert

### Soft revert (config-only, preferred)

Set in `config/app_config.yaml`:

```yaml
gimbal:
  lock_mode:
    enabled: false
```

and restart the bench. All lock-mode code paths are gated by this
flag; `false` disables the per-frame lock tracker, the lock-bbox
fields stay None, the GUI falls back to the projected fused bbox
exactly as in the baseline. This is the recommended revert path
because it preserves any other improvements that may have shipped
on top.

### Hard revert (git)

```bash
git checkout good-baseline-v1
```

This pins the entire repo to the tagged commit. Use only if the
soft revert isn't enough (i.e. some non-lock-mode change after this
baseline regressed something).

## Capabilities at this baseline

### Performance
- EO publish rate: ~20-22 Hz (was 7 Hz on saturated scene before
  the MOSSE `_to_gray` fix; was 19 Hz before the SDK helper improvements)
- Thermal publish rate: ~22-25 Hz (was 14 Hz before the heat-detector
  AGC dedup + HV imgsz drop + scipy.fft swap)
- Both sensors hold ≥20 Hz with 5+ active targets

### Tracking + control
- MOSSE pool (`vision/correlation_tracker_set.py`) keeps per-target
  bboxes alive between YOLO ticks for both EO and thermal
- ff-gate (`gimbal_manager.py:1494`): the velocity lookahead is
  zeroed when `|err| < zero_band` so a centered target with noisy
  velocity estimates doesn't drive a setpoint walk
- kd-gate (`gimbal_manager.py:1488`): the kd brake is also zeroed
  inside the deadband so it stops fighting the proportional output
- Slew cap (`gimbal.track_slew_cap_dps: 18.0`): engaged-track motion
  is capped at 18 dps to keep EO motion blur within YOLO's recovery
  envelope
- Engagement reset (`gimbal_manager.set_track_target`): switching
  from one tracked target to another resets predictor + smooth
  target + last setpoint so velocity state from the previous target
  doesn't leak in
- Settled-gate hysteresis (`track_predictor.PredictorParams.
  settled_hysteresis_ratio: 0.6`): debounces near-threshold
  gimbal_dps oscillation
- Track grace 60 ticks (4 s at 15 Hz fusion): legitimate slews
  with motion blur don't immediately drop the lock

### GUI + Fusion
- Pose-synced fused bbox (`gui/sensor_bridge.fused_to_wire`): the
  green box pose-syncs to each panel's frame-at-capture instead of
  fusion's publish-time pose, so it doesn't lag the image during
  fast slews
- World-frame fusion (`fusion.world_frame_fusion: true`): fused
  tracks carry world az/el so per-panel re-projection is timing-
  invariant

### Sensor configs (night-friendly)
- EO `conf_threshold: 0.25`: low-SNR night targets survive brief
  motion-blur dips below the classifier threshold
- Thermal MOSSE `lost_frames: 15`: per-target tracker coasts ~250 ms
  through slew gaps before pruning
- EO MOSSE `lost_frames: 12`: similar, ~480 ms at 25 Hz

## Known limitations at this baseline

These are the gaps the lock-mode work targets:

1. **Brief bbox-vanish at engagement** — when the operator clicks
   TRACK, the slew burst can briefly drive YOLO confidence below
   threshold. ByteTrack drops the EO ID, fusion has no EO observation
   for that engagement, the green box flickers or vanishes for
   ~400-600 ms before recovery.
2. **Mid-track classifier dropouts** — recorded in
   `recordings/better but bbs still disapper.jsonl` track #38,
   the engaged target had zero observations from any sensor for
   ~17 seconds while still visible to the operator. Fusion's
   max_misses (300 ticks ≈ 20 s) kept the lock alive but only as a
   stale projection.
3. **No graceful re-acquisition** — if all classifiers miss the
   engaged target for >grace_ticks, the lock dies and the operator
   must click TRACK again on a new fused-track ID.

## Lock-mode design (incoming)

See `vision/lock_tracker.py` and the upcoming commit. Summary:

- When the operator presses TRACK, snapshot the bbox content on EO
  and thermal at engagement
- Spawn a dedicated MOSSE tracker per sensor seeded on those patches
- Run the lock tracker every frame regardless of YOLO/heat/fusion
  state. Returns either a bbox (ACTIVE/COASTING) or None (LOST)
- GUI renders the lock bbox with priority over the projected
  fused-track bbox while engaged
- Auto-reseed: when a fused track lands within `reseed_search_
  radius_deg` of the locked position with a matching class, refresh
  the MOSSE template from that observation
- 3-state lifecycle:
  - ACTIVE (PSR ≥ threshold) → solid green box
  - COASTING (PSR < threshold for `lost_frames`) → amber box,
    filter stops learning, watch for re-acquisition for
    `coast_window_s`
  - HARD_RELEASED (coast window expired, no re-acquisition) → drop
    the lock, return to manual

## Files affected by lock mode

NEW:
- `vision/lock_tracker.py` — LockTracker class + LockState enum
- `vision/tests/test_lock_tracker.py` — unit tests

MODIFIED:
- `common/frames.py` — `lock_bbox` field on EOFrame + ThermalFrame
- `eo/eo_manager.py` — call lock.update() per frame, stamp lock_bbox
- `thermal/thermal_manager.py` — same
- `gimbal/gimbal_manager.py` — own LockMode coordinator, spawn on
  set_track_target, auto-reseed hook on fusion publish
- `gui/sensor_bridge.py` — surface `lock_bbox` + `lock_state` on the
  WS payload
- `gui/static/js/eo_view.js`, `thermal_view.js` — render lock box
  with priority + amber for COASTING
- `config/app_config.yaml` — new `gimbal.lock_mode.*` section,
  `enabled: true` default, kill switch documented

NONE of the existing files have their behavior changed when
`lock_mode.enabled: false`. The lock-mode code paths are entered
only when the config flag says so. Soft-revert is a one-line YAML
edit.
