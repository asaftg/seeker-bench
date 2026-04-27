# Session summary — 2026-04-26 → 2026-04-27

Two-day session that built a JSONL recorder + replay infrastructure
from scratch and used it to debug live tracking. Below is what
shipped, what we learned from each replayed recording, and what's
explicitly OPEN for the next session.

## What's new

### Recorder + replay infrastructure
- `recording/jsonl_recorder.py` — daemon-per-topic JSONL recorder.
  One file per session, line-delimited JSON `{ts_ns, channel, msg}`.
- `recording/encoders.py` — per-frame JSON encoders (thermal/EO/radar/
  gimbal/fused). q=92 JPEGs by default, configurable.
- `common/events.py` — `emit(type, payload)` + `subscribe(callback)`
  for in-process event broadcast. Events route to the recorder
  synchronously so two emits in the same tick don't overwrite each
  other (FrameBus is latest-only — subtle but mattered).
- `common/frames.py` — added `Topic.EVENTS`.
- `recording/README.md` — full format spec + event vocabulary +
  "if you're an LLM agent reading this..." canonical commands.

### Replay tools (`scripts/`)
- `replay_inspect.py` — `--latest --summary` is the canonical first
  command. Also `--track-id N`, `--grep <regex>`, `--events-since`,
  `--channels`, `--head`. Auto-discovers newest recording.
- `replay_algo.py` — re-runs `algorithms/track_predictor.step` against
  captured fused observations. Variant overrides via CLI for A/B.
  Validated parity on default params.
- `replay_of.py` — re-runs `DetectionTracker` against captured JPEG
  frames + synthetic-target events. Parity verified at 1.4 px max
  bbox-centre drift on the tree session.
- `replay_server.py` — FastAPI/WS that re-streams a JSONL file
  through the existing GUI at `http://localhost:8081/?speed=N&from=Ms`.
- `_smoke_recorder.py` — self-contained pipeline test, no rig.

### Pure-function tracking predictor
- `algorithms/track_predictor.py` — alpha-beta state estimator
  extracted from `gimbal_manager._tick`. Live + replay share the same
  code path. State is caller-owned `PredictorState`, params are
  `PredictorParams` (dataclass).

### GUI
- **REPLAY badge + clock** — solid red pill in the topbar, lazily
  created when the WS envelope carries `replay:true`. Clock shows
  `mm:ss.s` from session start. Stable tabular-nums monospace so
  you can quote it directly to the agent.
- **REC pill** — drives the live JSONL recorder. Click while ON
  prompts for an optional name (`rename_to:"<name>"` on the WS),
  backend renames `recordings/seeker_*.jsonl` → `recordings/<name>.jsonl`.
- `REPLAY_LATEST.bat` next to `START_SEEKER.bat`. Double-click,
  optional first arg = playback speed.

### Per-rig event vocabulary (in JSONL)
User actions: `recording_started/stopped`, `track_engaged/released`,
`track_heat_engaged/released`, `synthetic_target_drawn/cleared`,
`gimbal_manual_input`, `gimbal_absolute_input`, `gimbal_home_pressed`,
`eo_exposure_set`, `eo_lowlight_toggled`, `extrinsic_tune`,
`extrinsic_saved`, `radar_tune`, `device_changed`,
`zoom_preset_changed`, `heat_detector_set`.

System: `ae_converged`, `fused_track_born/dropped`,
`track_grace_expired`, `tilt_saturated_enter/exit`,
`pan_saturated_enter/exit`.

Algo diagnostics: `track_predictor_step` (full state per tick),
`track_settled_change`.

## Tracking predictor changes (gimbal/gimbal_manager.py + algorithms/track_predictor.py)

These are the ones that survived all the recording-driven A/B testing.

| Change | Default | What it fixes |
|---|---|---|
| `vel_clip_dps` 60→30 | 30 | Single-tick observation arriving on a brief settle no longer spikes smoothed velocity past the cap |
| `vel_decay_halflife_s` (new) | 0.20 | Cached velocity fades fast when fresh obs stop — predictor doesn't coast on a stale spike |
| `no_obs_lead_zero_after_s` (new) | 0.30 | Hard-zero the lead extrapolation past 0.3 s of staleness — collapses residual hunting |
| `tilt_saturated` (new) | False | When the gimbal is at the tilt floor/ceil and the obs would push further, zero `world_el_dot` and clip `sp_tilt`. Pan-only behaviour matches the existing heat-track path |
| `synth_slew_window_s` / `synth_slew_dps` (new) | 0.6 / 25 | After a synthetic target is auto-locked, cap the gimbal setpoint advance for 0.6 s at 25°/s so OF features can keep up |

**Validated against captured recordings** (no rig):

```
HUMAN #175 stale region:    hunting LIVE=5.00°  NEW=2.31° (54% smaller)
                            collapses to 0° at age=0.3s
FREAK-OUT #281 stale:       hunting LIVE=5.00°  NEW=3.00° (40% smaller)
                            collapses to 0° at age=0.3s
TREE bbox post-slew NCC:    -0.087 (off-target)
                            slew-dampen fix needs fresh recording
```

## Tilt-saturated event flicker fix
`gimbal_manager._tick` previously passed `el_obs_for_sat = 0.0` on
every stale tick. Saturation flag flipped True↔False at 60 Hz → 12
enter/exit pairs in 0.4 s in the mistrack recording. Fix: when stale,
use `predictor_state.world_el − cur_tilt` (last-known el-error).
State stays stable; events fire only on real transitions.

## Sticky `coasting=True` flag for synthetic targets
`thermal/detection_tracker.py` previously reported `coasting = (misses > 0)`
for synthetic tracks. Synthetic tracks never call `_merge_detection`
(no real-detection match), so `misses` only ever increments — flag
got stuck True forever after one OF miss. Fix: per-tick
`coasted_last_tick` boolean; synthetic uses that, real tracks keep
legacy semantics.

## Fusion robustness
- `fusion.max_misses` 15 → 30 (~1 s → ~2 s of grace at 15 Hz). Lets
  EO-only person tracks survive YOLO hiccups during fast slews.
- `fused_track_born/dropped` events emitted for the JSONL stream.

## REVERTED (kept here so they don't get re-introduced)

These were attempted, validated against the WRONG recording, and
removed after the next recording exposed the regression:

### Radar world-frame rotation (`gui/static/js/radar_view.js`)
- Tried `R(-p)` originally, then `R(+p)`, then NO ROTATION.
- A/B against `radar_opposite.jsonl` (43 s of pure pan, 5
  persistent target IDs) showed NO ROTATION gives the smallest
  world-x std deviation by 2-6× over either rotation.
- Conclusion: this rig's radar reports (x,y) that are already
  world-stable. `_rotateToWorld` is now a pass-through. Wedge /
  boresight / FOV labels still rotate so the operator sees where
  the radar is currently pointing.

### Fusion gimbal-pose compensation + 3° centroid fallback gate
- Introduced morning of 2026-04-27 to fix YOLO id-swap during fast
  slews (fused track #42 in `Human_and_vehicle_mistrack.jsonl` died
  when ByteTrack reassigned id 71 → 80 over a 0.4 s gap).
- `gimbal_not_tracking_static.jsonl` (afternoon) showed BOTH changes
  caused fused track #20 to merge two distinct vehicles — bbox
  angular size bouncing between (3.27, 1.76) and (5.05, 4.34) as
  the IoU "match" alternated between them. Predictor saw 9-13°
  jumps in `obs_world_az`, gimbal swung 17° hunting the average.
- Reverted both. Comments document the chain so a future agent
  knows which knobs are dangerous in dense multi-vehicle scenes.

## Open issues (for the next session — tracking optimization)

1. **YOLO id-swap during slew kills fused-track lock.** Documented
   in `Human_and_vehicle_mistrack.jsonl` — every TRACK lock died at
   ~1.5 s when ByteTrack reassigned the EO id mid-slew. The morning
   fusion fix attempted this, was reverted. The proper fix is
   probably to track in **world frame** (store `world_az = cam_az +
   gimbal_pan` per observation, match in world frame). Structural
   change worth a fusion-replay tool first.
2. **Multi-vehicle scene density.** IoU=0.15 + wide vehicle bbox
   (~3°) = nearby vehicles can match the wrong fused track. Without
   the (reverted) compensation, base fusion is fragile here. Not
   yet observed in a recording; the static-vehicle case died for a
   different reason (the morning fix's fallback gate). Worth keeping
   in mind.
3. **Synthetic target slew-dampen needs fresh recording.** OF replay
   showed the previous behaviour: bbox NCC=-0.087 between draw-time
   and post-slew (target was off-target). The fix (cap setpoint
   advance for 0.6 s at 25°/s) is in place but cannot be replay-
   validated — needs a live re-record of the tree case.
4. **Single-sensor EO person tracks die when YOLO loses confidence.**
   Thermal h/v classifier isn't running on this rig at the moment so
   there's no sensor fallback. Either thermal h/v or radar
   association would help. Out of scope for tracking optimization
   per se but documented.

## Files of interest for the next agent

Read first:
- `recording/README.md` — format + canonical commands + event vocab
- `OVERNIGHT_NOTES.md` — overnight changes from 2026-04-26
- `algorithms/track_predictor.py` — pure-function predictor
- `gimbal/gimbal_manager.py:_tick` — fused-track branch (calls predictor)
- `fusion/fusion_manager.py:_update_tracks` — IoU matcher (kept simple)

Canonical command for any future debug:
```
python scripts/replay_inspect.py --latest --summary
python scripts/replay_inspect.py --latest --track-id <N> --out track.csv
python scripts/replay_algo.py --latest --algo predictor --variant <K=V,...>
python scripts/replay_of.py --latest --variant <K=V,...>      # synth/heat only
python scripts/replay_server.py --latest                       # visual playback
```
