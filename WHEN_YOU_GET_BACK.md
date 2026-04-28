# When you're back — short version

## What I did while you were away

### TRACK button → closed-loop control (pushed, default ON)
Commit [84fe2a8](https://github.com/asaftg/seeker-bench/commit/84fe2a8) on
`session/phase-2-recap`.

Same algorithm for static and dynamic targets, no velocity gating
(per your ask). The fused-track tick branch now uses the same closed-
loop pixel-error proportional control as the existing real-heat-blob
path:

    az_in, el_in = lp_filter(trk.az_deg, trk.el_deg)
    d_pan  = smooth_proportional(az_in, kp_track, zero_band, full_band)
    d_tilt = smooth_proportional(el_in, kp_track, zero_band, full_band)
    sp_pan = cur_pan + clip(d_pan, max_step_deg)

For a static target az/el → 0 as the camera centres → setpoint stops
changing → matches the synth lock end-state. For a moving target
each fresh observation refreshes az/el and the gimbal tracks (with
some lag, accepted for now).

The alpha-beta predictor still runs in parallel for replay parity +
diagnostics. Its lookahead output is just no longer fed to the
controller in this mode (that's where the noise came from on static
targets — `lead_time × velocity_noise = 0.3 deg` of setpoint jitter
per tick).

Knobs reused from the heat-blob path (no new tuning surface):
`kp_track`, `track_zero_band_deg`, `track_full_band_deg`,
`track_min_step_deg`, `max_step_deg`, `err_lp_alpha`, `tilt_sat_eps_deg`.

To A/B revert: edit `config/app_config.yaml`, set
`fused_track_closed_loop: false`, restart bench.

Full design rationale in `TRACK_BUTTON_PLAN.md`.

### Radar pipeline mapped (NOT YET TOUCHED)
I traced the radar pipeline end-to-end so we can target a fix once
you tell me what specifically is miscalculated. Map below.

## When you're back, two things to do

### 1. Live-validate the closed-loop TRACK

Bench up, sensors warm. Click TRACK on a target on the targets list.
Compare the centering vs the manual-BB synth lock at the same scene
(pan ≈ -20°, tilt ≈ +4°, tree).

- If TRACK now matches the synth lock quality (≤0.5° at narrow zoom):
  flip is good, leave defaults.
- If TRACK still feels worse: tell me how (oscillation? lag? drift?)
  and I'll tune the closed-loop knobs.
- If TRACK is now WORSE than before: revert with one config edit
  (`fused_track_closed_loop: false`), tell me what you saw.

### 2. Tell me which radar issue you're seeing

I have a ranked candidates list (below). Pick the one(s) that match
your observation, or describe what you see and I'll match it for you.

Most likely candidates, ranked:

| # | Symptom | What's wrong | Where to fix | Effort |
|---|---------|--------------|--------------|--------|
| 1 | Radar bbox is too big or too small on the EO/thermal overlay | Slant range vs ground range mix-up in the size→angle conversion | `fusion_manager.py:411-413`, `clustering.py:412-413`, or floor at 0.4° | small |
| 2 | Radar tracks appear at wrong elevation (above when they should be below or vice versa) | Sign flip in `el = atan2(z, horiz)` or its bias | `fusion_manager.py:406-409` | tiny |
| 3 | Radar tracks offset to one side from thermal/EO targets even when bias slider is at 0 | az/el bias applied in wrong direction (added when it should be subtracted somewhere) | `fusion_manager.py:405,409`, `sensor_bridge.py:185-186` | small |
| 4 | Radar tracks appear at wrong position when gimbal pans | The reverted world-frame rotation issue may need more thought | NOTE: do not re-add rotation; the morning A/B already proved no-rotation wins on this rig | TBD |
| 5 | Radar bboxes never below 0.4° even for distant clusters that should look smaller | Floor at 0.4° is too aggressive | `fusion_manager.py:412-413`, `clustering.py:412-413` | tiny |
| 6 | Radar associates with the wrong EO/thermal track (cross-sensor merge errors) | IoU thresholds too permissive | `fusion_manager.py:233, 305` | small |
| 7 | Coasting (Kalman-only) radar boxes draw at stale positions / sizes | Coast handling in `sensor_bridge.py:299-317` | `sensor_bridge.py` radar projection | medium |
| 8 | Radar reports targets at completely wrong distance (e.g. behind the sensor) | Radar firmware profile mismatch or mount orientation | `radar_manager.py` device init / `clustering.py` filter | medium |

**Pick one** and I'll write the targeted fix.

## Reference: full radar pipeline map

End-to-end, with files and conventions:

```
1. RAW INGEST
   radar/tlv_parser.py:143-148
   Firmware emits (x, y, z, doppler) as float32 per point.
   Convention: x=right, y=forward (boresight), z=up. TI mmwave standard.

2. CLUSTERING + TRACKING
   radar/clustering.py:292-409
   DBSCAN in 4D (x,y,z,doppler). Centroid + 1.5*std half-extents.
   Constant-velocity Kalman per cluster. M-of-N confirmation gate.
   Coast for max 30 frames (~2.3s @ 13 Hz) before drop.
   Clamp cluster size to [0.25, 3.0] m so single tiny clusters can't dominate.

3. CARTESIAN -> ANGULAR (fusion + GUI overlay)
   fusion/fusion_manager.py:372-419 (for fusion)
   gui/sensor_bridge.py:160-223 (for overlay)
   az = atan2(x, y) + az_bias_deg            # +az = right
   el = atan2(z, sqrt(x*x + y*y)) + el_bias_deg  # +el = up
   slant = sqrt(x*x + y*y + z*z)
   ang_w = max(0.4, 2*atan(size_x_m / slant))
   ang_h = max(0.4, 2*atan(size_z_m / slant))

4. WORLD-FRAME ROTATION
   gui/static/js/radar_view.js:_rotateToWorld
   PASS-THROUGH on this rig (per 2026-04-27 A/B test against
   radar_opposite.jsonl: no-rotation wins by 2-6x std dev).
   The radar already reports world-stable (x,y).
   *Do not re-introduce rotation without a fresh A/B*.

5. SOFTWARE EXTRINSIC (radar -> EO alignment)
   radar/radar_manager.py:85-86          (live-tunable storage)
   fusion/fusion_manager.py:120-121      (fusion's copy)
   radar/extrinsic block in config/app_config.yaml (initial values)
   GUI extrinsic_tune handler keeps both copies synced.
   Bias is ADDED in fusion + GUI projection.

6. CROSS-SENSOR ASSOCIATION
   fusion/fusion_manager.py:_tick (lines 199-341)
   EO -> thermal:    angular IoU >= 0.15 to merge candidates
   Radar -> camera:  angular IoU >= 0.05 (looser, cluster-extent-derived)
   Greedy best-match. Class is wildcard for radar.

7. PERSISTENCE
   fusion/fusion_manager.py:_update_tracks (lines 453-580)
   IoU >= 0.15 to attach a new candidate to an existing fused track.
   EMA smoothing on bbox angular pose (alpha=0.4).
   max_misses=30 (~2s grace) before drop.
   sensor_grace_ticks=5 for the per-sensor "active" set.
```

## Status of the bigger plan (Phase 3 + 4)

Phase 3 (dynamic tracking) is partly addressed by the closed-loop
TRACK change (the same algorithm now serves dynamic targets too,
just with some lag). After live validation we may need to revisit
fast-target lookahead — but only if data shows it's needed.

Phase 4 (radar) is unblocked once you pick one of the symptoms above.
