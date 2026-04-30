# Morning addendum — 2026-04-27 09:30

After the `Human_and_vehicle_mistrack.jsonl` recording arrived I dug
deeper. The previous predictor fixes ARE active in that recording
(`fusion.max_misses=30`, `track_vel_clip_dps=30`, etc.) but the
underlying tracking still failed for two reasons that I couldn't see
until I had this richer recording. Plus the radar bug you flagged.

## What the recording showed (track #42, 1.5 s lock)

```
EO YOLO id=71  visible from t=-0.48 to t=+0.44   (~0.92 s, smooth)
                            ↓ 0.4 s gap (gimbal slewing fast) ↓
EO YOLO id=80  visible from t=+0.85 onwards     (NEW id, same person)
```

ByteTrack lost confidence on id=71 during the slew, then re-detected
the same person but assigned id=80. **Fusion had no way to associate
id=80 back to fused track #42** because it was matching new EO
observations to existing fused tracks **in CAMERA-frame angles**, and
during the 0.4 s gap the gimbal slewed by 25° → the same world target's
camera-az shifted ~25° → IoU=0 → birth a NEW fused track (#52). The
TRACK lock was on the dead #42, so it grace-expired.

## Fixes shipped this morning

### Radar world-frame rotation — `gui/static/js/radar_view.js`
- **Sign was inverted on the world rotation** — was applying `R(−p)`
  instead of `R(+p)`. Result: stationary world targets appeared to
  move WITH the gimbal pan instead of staying put. Now correct.
- **Wedge / boresight / FOV-edge labels** had matching sign bugs —
  pan-positive=LEFT in our calibration but the wedge swung canvas-RIGHT
  for positive pan. Introduced `panRadCanvas = -gimbalPanRad` so all
  three render consistently. Edge labels swapped to match world angles.

### Fusion gimbal-motion compensation — `fusion/fusion_manager.py`
- Each fused track now stores `pose_at_update` (the gimbal pan/tilt at
  its last observation match). The matcher compensates the track's
  stored camera-az by `(cur_pan − last_pan)` before computing IoU /
  centroid distance against new candidates. A 0.4 s slewed gap no
  longer kicks the same target out of the gate.
- Added a soft-match fallback: even when IoU < 0.15, a candidate
  within 3° angular centroid of the (gimbal-compensated) track
  position with matching class will associate. Catches the case
  where YOLO returns a slightly different bbox shape on re-acquisition.
- Snapshot `cur_gimbal_pose` once per fusion tick and reuse — no
  per-track BUS lookup.

### Tilt-saturated event flicker fixed — `gimbal/gimbal_manager.py`
- Previous code passed `el_obs_for_sat = 0.0` on every stale tick
  (no fresh fused obs). When the gimbal was AT the tilt floor with a
  cached `world_el = -1.16°`, the saturation flag flipped every other
  60 Hz tick → 12 enter/exit event pairs in 0.4 s in the recording.
- Now: when stale, use `predictor_state.world_el − cur_tilt` (the
  last-known el-error) for the saturation check. State stays stable
  across the tick boundary, events emit only on real transitions.

# Overnight changes — 2026-04-26 → 2026-04-27

You went to sleep with the request "improve EVERYTHING you are seeing
in the replays." Below is what changed, what was validated against
the JSONL recordings you captured, and what to try first when you
wake up.

## What I validated against your recordings (no rig needed)

| Recording | Replay tool | Old behaviour | New behaviour |
|---|---|---|---|
| `seeker_2026-04-26_21-09-30.jsonl` (radar+EO freak-out, track #281) | `replay_algo.py predictor` | hunting amplitude 5.00° in the no-fresh-obs window, persists forever | **3.00° initially, collapses to 0° at age=0.3s** |
| `seeker_2026-04-26_21-05-17.jsonl` (5× human attempts, track #175) | `replay_algo.py predictor` | hunting amplitude 5.00° in the no-fresh-obs window | **2.31° initially, collapses to 0° at age=0.3s** |
| `seeker_2026-04-26_21-13-01.jsonl` (tree synthetic target #32) | `replay_of.py` + NCC analysis | bbox content NCC vs draw-time = **-0.087** (uncorrelated → bbox is OFF the tree) | (slew-dampen fix below cannot be validated from captured frames; need fresh recording) |

## All changes (in code)

### Tracking predictor — `algorithms/track_predictor.py`
1. **`vel_clip_dps` 60 → 30.** Single fresh observation arriving on a brief settle between fast slews used to give a 49 dps raw_dot that smoothed to +17.8 dps and pushed the setpoint past the cap. Tighter clip kills the spike.
2. **Velocity decay on stale obs (new field `vel_decay_halflife_s`, default 0.20 s).** When `fresh_fused=False`, `world_*_dot` decays toward 0 — predictor doesn't coast on a stale velocity spike. Per-tick increment, not cumulative-from-last-obs (correct half-life behaviour at 60 Hz).
3. **Zero-lead-on-stale (new field `no_obs_lead_zero_after_s`, default 0.30 s).** Above 0.30 s since last fresh observation, the lead-time extrapolation is hard-zeroed. Decay alone left a residual hunting; this collapses it.
4. **Tilt-saturated awareness (new field `tilt_saturated`).** Zeroes `world_el_dot` when the gimbal is mechanically against floor/ceil, so observations the gimbal physically can't follow don't contaminate the velocity estimate.

### Gimbal manager — `gimbal/gimbal_manager.py`
5. **Pan-only when tilt saturated (fused-track path).** The fused-track predictor branch now calls `_pan_only_if_tilt_saturated` BEFORE `predictor.step`, passes `tilt_saturated` into params, and emits `tilt_saturated_enter{kind:"fused", cur_tilt, el_err, tracked_id}` / `tilt_saturated_exit{kind:"fused"}` events. Pan continues independently — exactly the "below 0° → pan only" behaviour you flagged. Previously this only worked on the heat-track path.
6. **Synthetic-target slew dampening.** New `_synth_lock_t` timestamp + `synth_slew_window_s` (0.6 s) + `synth_slew_dps` (25°/s) knobs. For 0.6 s after a synthetic target is auto-locked, gimbal setpoint advances are capped at 25°/s so the per-frame scene motion stays under LK's tracking limit. Reset on synth-clear / track-end.
7. **Pan-saturation events (symmetric with tilt).** New `pan_saturated_enter{cur_pan, sp_pan, edge}` / `pan_saturated_exit` events fire when the desired setpoint pushes the gimbal against the mechanical pan envelope. Previously pan saturation was silent.
8. **`track_predictor_step` carries the new diagnostic fields** (settled, lead, age, world_*_dot, etc.) — was already there from the earlier session.

### Fusion — `config/app_config.yaml`
9. **`fusion.max_misses` 15 → 30** (~1 s → ~2 s of grace at 15 Hz). Replay analysis showed every human-tracking attempt had `sensors=['eo']` only, and YOLO loses the person briefly during fast slews. With max_misses=15 the fused track id died after ~1 s of YOLO drop and your TRACK lock grace-expired soon after. With 30, the angular position survives long enough for YOLO to re-acquire on the same target, keeping the same fused-track id and your lock alive. Coupled with the new `no_obs_lead_zero_after_s=0.3`, the gimbal holds last-observed position cleanly during the extended grace — no runaway, just a pause.

### Heat tracker — `thermal/detection_tracker.py`
10. **Sticky `coasting=True` flag for synthetic targets fixed.** New `_Track.coasted_last_tick` field; synthetic tracks now report coasting based on the most recent tick's match status, not the cumulative `misses` counter that never resets. Real tracks unchanged.

### Configuration — `config/app_config.yaml`
All new knobs surfaced for live tuning without code changes:
```yaml
gimbal:
  track_lead_time_s: 0.30
  track_vel_alpha: 0.30
  track_predict_warmup_n: 5
  track_predict_cap_deg: 5.0
  track_gimbal_settled_dps: 15.0
  track_extrap_horizon_s: 2.0
  track_vel_clip_dps: 30.0                 # was 60
  track_vel_decay_halflife_s: 0.20         # NEW
  track_no_obs_lead_zero_after_s: 0.30     # NEW
  synth_slew_window_s: 0.6                 # NEW (synth-target slew dampening)
  synth_slew_dps: 25.0                     # NEW
fusion:
  max_misses: 30                           # was 15
```

## New tooling

* **`scripts/replay_of.py`** — re-runs the heat tracker against
  captured JPEG frames + synthetic-target events from a JSONL
  recording. A/B test OF parameters (`--variant max_dist_px=80,
  of_max_features=40,...`) without re-running the rig. Parity to
  live = max 1.4 px bbox-centre drift on the tree session.
* **Pixel-content NCC diagnostic** (inline above) showed the tree
  synthetic-target bbox lands on visually-different content after
  the slew (NCC = −0.087). This is the data point that justified the
  slew-dampen fix.

## What to do when you wake up

1. **Validate the synth slew-dampen fix with a fresh recording.** The
   replay tools can't simulate the new gimbal motion against the old
   frames. Re-record the tree case: draw a bbox, watch whether the
   bbox stays on the tree as the gimbal slews. If yes → fix worked.
   If still drifting → bump `synth_slew_dps` lower (try 15) or extend
   `synth_slew_window_s` (try 1.0).

2. **Re-record the radar+EO scene** (the freak-out one) and watch the
   gimbal during the no-fresh-obs window. With the predictor fixes,
   the hunting should be visibly milder for the first ~0.3 s and then
   the gimbal should HOLD position cleanly. If you see runaway again,
   `replay_algo.py --algo predictor --variant <override>` against the
   new recording will show me why.

3. **Re-record the human-tracking case.** With `fusion.max_misses=30`
   the lock should survive YOLO hiccups significantly longer (~2 s
   per drop, vs 1 s before). Plus the predictor's zero-lead-on-stale
   means the gimbal doesn't hunt during the wait for re-acquisition.
   Click TRACK once on a person row — you should not need to re-press.

4. **For me to look at any of these:** quote the replay clock from
   the GUI's red REPLAY badge (e.g. "0:12.3"). I'll run
   `scripts/replay_inspect.py --latest --events-since <ts>` or
   `--track-id <N> --out` and follow the trace. Or rename the file
   when you stop recording (the prompt I added) and tell me the
   name.

## Summary of replay-validated improvements

```
HUMAN #175 stale region:    hunting LIVE=5.00°  NEW=2.31°   (54% smaller)
FREAK-OUT #281 stale:       hunting LIVE=5.00°  NEW=3.00°   (40% smaller)
Both:                       hunting collapses to 0° at age=0.3s
                            (LIVE oscillates forever until grace)
TREE bbox post-slew NCC:    -0.087 (off-target) → fix needs fresh test
```

Smoke tests pass. All imports clean. Nothing committed — your call
on whether to merge.
