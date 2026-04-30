# TRACK button on fused tracks — plan

## What you said

- TRACK is broken; gimbal doesn't center as well as manual-BB synth lock.
- Same algorithm for static AND dynamic targets. No velocity gating, no
  per-target mode switching.

## Why TRACK is currently worse than manual BB

Three different tracking paths in `gimbal_manager._tick`:

1. **Synth (manual BB)** — one-shot world-angle lock on first observation,
   then setpoint is a constant world angle until lock is released. No
   correction noise. Result on the bench: 0.01-2° residual.

2. **Real heat blob** (the `elif fresh_heat:` branch, lines ~986-1044) —
   closed-loop pixel-error controller: `d_pan = kp * az`, `d_tilt = kp *
   el`, soft deadband around image centre, low-pass filter on the error.
   Setpoint = `cur_pan + d_pan`. Per-axis, every tick a fresh thermal
   detection arrives. Result: stable for both static and slow-moving
   thermal blobs.

3. **Fused track** (the `elif tracked_id is not None:` branch, lines
   ~1077-1247) — alpha-beta predictor (`algorithms.track_predictor`).
   Predictor maintains world-frame `(world_az, world_el)` smoothed +
   `(world_az_dot, world_el_dot)` velocity. Setpoint = `world + lead_time
   × world_dot` with caps and decay.

Path #3 is what runs when you click TRACK on the targets list. It was
designed for fast-moving targets where camera latency means we need to
*lead* the target. The lookahead is the problem on the bench:

- For a static target, `world_dot` should be 0. In practice it has
  noise floor from detection jitter (YOLO bbox shifts ±1 px/frame on
  the same person at 5 m → ±0.05° az noise → after alpha-beta vel
  filtering, residual `world_dot` of ~0.5-1 °/s).
- `lead_time = 0.30 s`, so `lead_time × 0.5-1 °/s` = 0.15-0.30°
  noise on the setpoint *every tick*.
- That noise reaches the controller as a continuously-changing target,
  the gimbal can never settle. Visible to the operator as wobble or
  drift.

The synth and heat-blob paths don't have this because neither extrapolates
forward. Synth just holds a constant world angle; heat-blob just
proportional-corrects to the current pixel error.

## What "one algorithm for both" means

The cleanest universal control law is **closed-loop pixel-error
proportional control** — exactly path #2.

```
d_pan  = smooth_proportional(trk.az_deg)         # soft deadband + kp gain
d_tilt = smooth_proportional(trk.el_deg)
sp_pan  = cur_pan  + d_pan
sp_tilt = cur_tilt + d_tilt
```

For a **static** target: `trk.az_deg` and `trk.el_deg` go to 0 as the
gimbal centres on the target → `d_pan, d_tilt → 0` → setpoint stops
changing → camera holds. Same end-state as synth lock.

For a **moving** target: every fresh observation gives the target's
current angular position relative to the camera. The controller pushes
the gimbal toward that position. There IS some lag (the gimbal arrives
at where the target *was* at the time of the observation, not where it
is *now*) — for indoor / slow targets this is invisible.

For very fast targets, lag matters. That's a separate problem we can
revisit in Phase 3 (dynamic tracking) if it actually shows up. The
current predictor's lookahead was added pre-emptively, before we had
data showing it was needed; data so far suggests it's hurting more
than helping.

## What I will and won't change

**Will change** (this plan):

1. Add a config flag `fused_track_use_closed_loop` (default **false**
   initially so the existing predictor is the baseline, flip to true
   once you've A/B'd live).
2. When the flag is true, the fused-track tick branch uses the same
   closed-loop control as the heat-blob path:
   - `_lp_filter_error(trk.az_deg, trk.el_deg)`
   - `_smooth_proportional(...)` with `kp_track`, `track_zero_band_deg`,
     `track_full_band_deg`.
   - `_pan_only_if_tilt_saturated` for the mechanical tilt limit.
   - `max_step_deg` per-tick cap.
   - Setpoint = `cur_pan + d_pan, cur_tilt + d_tilt`.
3. Keep the predictor running in parallel for diagnostics and replay
   parity. Just don't feed its output to the controller in the new
   mode.
4. Emit `fused_closed_loop_step` events so we can compare residuals
   between the two modes via replay.

**Won't change** (preserve hard-won earlier work):

- The predictor module itself. Stays as-is.
- The synth path. Already works.
- The heat-blob path. Already works.
- Tilt-saturation handling, pan-saturation events. Reused.

## Validation criteria (live, when you're back)

Same scene as the manual-BB test (pan ≈ -20°, tilt ≈ +4°, looking at
the tree). Restart bench. Then:

1. **Detect** a target on the targets list (any class — radar, EO, fused).
2. Click TRACK on it.
3. Compare:
   - With `fused_track_use_closed_loop: false` (existing predictor):
     measure final residual via `replay_optical_truth.py`.
   - With `fused_track_use_closed_loop: true` (new closed loop):
     measure final residual.
4. Pass criteria for the new closed-loop mode:
   - Final residual ≤ same as synth lock at the same zoom (~0.5° at narrow,
     1-2° at mid).
   - No visible setpoint wobble (operator-level "feels stable" check).
   - Lock survives at least 30 s on a static target without drift events.

If new mode passes: flip the default to `true`, ship.

If new mode fails (e.g. settling oscillation, undershoot due to
deadband): tune knobs (`kp_track`, `track_zero_band_deg`,
`track_full_band_deg`) — same knobs the heat-blob path already uses, so
any tuning here transfers across.

## What I'll do in the next 2 h while you're away

1. Code the change as described, behind the config flag.
2. Smoke test imports + the existing 9 unit tests.
3. Restart bench (sensors are off — bench will start in the disconnected
   state, that's fine for static smoke testing).
4. Add a small pure-function unit test that drives the new closed-loop
   handler with mock fused observations and verifies the setpoint
   trajectory.
5. Commit + push, leave the flag default OFF so you can A/B with one
   YAML edit when you're back.
6. Stop. Wait for live verification.

## What I won't do

- Won't enable the flag by default until you've live-tested.
- Won't touch the predictor's internals.
- Won't pre-emptively fix Phase 3 (dynamic) or Phase 4 (radar) until
  this lands.
