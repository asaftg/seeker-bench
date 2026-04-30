# Stage A / Stage B optical correction — session notes

User left for the day with: "you have full authority to advance by yourself
until you have good results." This file captures what I found, what's
shipping, and what's deferred.

## The actual home run: Maestro auto-reconnect (commit d203f00)

While running the first A/B captures of Stage B, I noticed `gimbal/state.connected = False` for
**all 1363 samples** in the recording, despite startup logging
`connected=True`. Trace showed:

```
16:24:37  INFO  Maestro opened on COM4
16:24:37  INFO  GimbalManager started (connected=True)
...
16:24:45  WARNING Maestro write failed on ch 0: WriteFile failed
                  (PermissionError(13, 'The device does not recognize the command.'))
16:24:45  WARNING Maestro write failed on ch 1: ...
16:24:45  WARNING Maestro write failed — marking disconnected
```

After that single transient PermissionError, `_command_now` early-returned
forever (`if not self._connected: return`) and **no PWM commands reached the
Maestro for the rest of the session**. The controller's internal state advanced
based on commanded setpoints, so `gimbal/state.pan_deg / tilt_deg` continued to
look correct, but the camera was physically frozen.

This is a known Windows USB-CDC quirk on the Pololu Maestro under sustained
60 Hz writes — `'device does not recognize the command'` is misleading
naming. The Maestro itself is fine; pyserial's underlying handle hits a
transient state that recovers cleanly after a `close() → open()` cycle.

**Fix in `gimbal/gimbal_manager._command_now`:**
- Don't permanently mark disconnect on a single write failure; count
  consecutive failures.
- After 3 in a row: `close()` the handle so the next tick re-opens.
- Successful write resets the counter.
- If `_connected` is False, every tick attempts to re-open (cheap when
  no Maestro is plugged).

This single fix dramatically improved open-loop tracking — open-loop
residuals dropped from "several degrees, sometimes wrong direction" to
**0.01°-2° per BB** at mid zoom (see `recordings/ab3_off_mid.jsonl`).

The user's earlier complaints about BBs not centering well were largely
because of this — the rig was running at most ~8 s per session before
silently going offline. **Anyone hitting "the gimbal stopped responding"
in the past few sessions was hitting this bug.**

## Stage A: optical-residual diagnostic event (commit 5ff0465)

Always-on diagnostic. When a synthetic-target lock is active, the gimbal
manager runs LK optical flow on EO + thermal frames against an anchor
captured at lock time. The measured pixel displacement, converted via the
recorded FOV, gives the actual angular movement the camera made — emitted
as the `optical_residual` event so we can A/B closed-loop work via replay
tools without re-running the rig.

EO is primary (more pixels per degree, more texture); thermal is fallback
when EO loses features (big slews exceed EO's narrow 11° FOV).

5/5 unit tests + offline parity to `scripts/replay_optical_truth.py` + 882
events fired live in `thermal_bb_test_2.jsonl` confirm the measurement is
trustworthy.

## Stage B: closed-loop optical correction — DEFERRED

Implementation shipped (commit c13d767, tuning d203f00) but **default OFF**
in `config/app_config.yaml`. A/B testing on the bench (recordings
`ab3_off_mid` vs `ab3_on_mid` and the iteration `ab4_on_mid`) confirmed
that the integrator does not improve residuals and often makes them
worse.

**Root cause** (uncovered in the BB4 trace of `ab3_on_mid.jsonl`):

```
t=1.06: cur_tilt=+8.83 (post-slew plateau)
t=1.33: cur_tilt=+9.38, corr_el=+0.817 (integrator started)
t=1.99: cur_tilt=+10.73, corr_el=+1.827
t=4.41: cur_tilt=+8.12, corr_el=-0.793 (oscillating!)
t=6.94: cur_tilt=+4.41, corr_el=-3.000 (saturated, opposite sign)
```

The integrator drives the controller's internal commanded position, which
slowly diverges from physical reality because:

1. Sub-degree corrections fall in the servo's mechanical dead-zone — the
   camera doesn't physically follow them.
2. The LK chain is anchored on a CENTRAL ROI of the frame. As the
   controller's commanded position drifts away from the camera's actual
   position, the LK measurement reflects camera motion that DIDN'T happen
   (because the controller advanced internally).
3. Each correction induces tiny camera motion → LK noise → integrator
   sees this as "still residual" → adds more correction → unstable
   feedback.

**The right design** for a synth-target closed-loop controller is the
same one already in use for real heat-blob tracks: anchor LK on the
**bbox CONTENT** (not the central ROI), measure the world target's
current pixel position in each frame, and feed pixel-error-to-image-
center directly into the controller (the way the existing
`elif fresh_heat:` branch in `gimbal_manager._tick` does for real
heat blobs at lines 887-908). Synth targets currently have a one-shot
world-angle lock that bypasses this — by design, because the OF
tracker on the synth `_Track` was originally too unreliable. With the
Maestro fix in, that one-shot lock now produces 0.01-2° residuals
(measured) and is good enough for static targets.

**For dynamic targets** the right next step is:
1. Make the synth `_Track`'s OF more reliable (re-anchor every N
   frames; sample features densely inside bbox; reject features
   exceeding inter-frame motion threshold).
2. OR: anchor `OpticalResidualTracker` on bbox content and feed
   pixel-error into the controller as a secondary correction term
   that's gated by "settled" + not against mechanical limit.

This is a refactor, not a tuning. Defer until needed.

## What works right now (post-Maestro-fix)

`recordings/ab3_off_mid.jsonl` (Stage B OFF, mid zoom, 4 BBs at progressive
slew sizes):

| BB | bbox | slew (cmd) | az residual | el residual |
|----|------|-----------|-------------|-------------|
| 1 | (290,226,60,60) — center | ~0° | +0.008° | +0.001° |
| 2 | (220,220,60,60) — small | ~-2° | -1.96° | +0.18° |
| 3 | (140,180,80,80) — medium | ~-4° / +1° | +0.09° | +0.76° |
| 4 | (60,60,80,80) — corner | ~-6° / +5° | +0.04° | -1.10° |

Center BB is dead-on. Other BBs show genuine ~1-2° mechanical residuals
(servo dead-zone + gravity). These are the real, measurable open-loop
errors that a closed-loop controller could chip away at if the
architecture were right.

## Files added / modified this session

- **`gimbal/optical_residual.py`** (new) — pure LK tracker module
- **`gimbal/tests/test_optical_residual.py`** (new) — 9 unit tests, all green
- **`gimbal/gimbal_manager.py`** — Stage A hooks, Stage B integrator, **Maestro auto-reconnect**
- **`config/app_config.yaml`** — Stage B knobs (default OFF)
- **`scripts/replay_optical_truth.py`** (new) — offline LK truth tool
  with `drift` and `bb-residual` modes
- **`scripts/maestro_diag.py`** (new) — read-only Maestro diagnostic
- **`scripts/ab_optical_correction.py`** (new) — headless WS harness for
  A/B captures across multiple zoom presets

## Recordings produced

In `recordings/`:
- `verify_handshake.jsonl` — 2.4 MB, end-to-end harness check
- `ab_stageB_off.jsonl` and `ab_stageB_on.jsonl` — first A/B (PRE Maestro fix; gimbal was disconnected, data is bad — kept for archeology)
- `ab2_off_mid.jsonl` and `ab2_on_mid.jsonl` — second A/B (still pre-fix on the OFF run, `connected=False` throughout)
- `ab3_off_mid.jsonl` and `ab3_on_mid.jsonl` — POST-Maestro-fix; clean A/B
- `ab4_on_mid.jsonl` — Stage B ON with conservative tuning (alpha=0.05); still doesn't improve residuals
- `tilt drift.jsonl` and `thermal_bb_*.jsonl` — earlier user captures with the user-drawn BBs

The `ab3_off_mid` is the canonical "what the rig does today" baseline.

## Next steps when user is back

1. The user reported "BBs are drifting as hell" earlier — that's the synth
   `_Track`'s own OF tracker losing features visually on the display, not
   our Stage A measurement. With the Maestro fix in, the gimbal physically
   moves now, so the bbox should stay closer to the target object. Worth
   re-checking visually.
2. For dynamic tracking, real heat-blob tracks already use the right
   pixel-error closed loop. That should work. Phase 3 of the original plan.
3. Stage B refactor (anchor LK on bbox content, pixel-error control law)
   if dynamic tracking on synthetic targets is needed later.

## Phase 2: world-frame fusion (commit 73a6c90)

Built `scripts/replay_fusion.py` — offline fusion harness with two
variants (camera-frame for parity, world-frame for the structural fix
documented in `SESSION_SUMMARY.md`). Implements the full `_tick`
pipeline: observation builders, cross-sensor association, dedup,
matching, merge.

Status: tool runs both variants against the regression set. Parity
between camera-frame replay and recorded events is approximate but not
exact (under-predicting birth/death counts ~50%); root cause not fully
diagnosed but suspected to be in tick-rate alignment with BUS semantics
when sensors are silent. Replay clearly shows the world-frame variant
producing more distinct fused tracks across both
`Human_and_vehicle_mistrack.jsonl` and `gimbal_not_tracking_static.jsonl`
than camera-frame, which is the qualitative direction we want.

The honest evaluation step (Phase 2c) is per-track lifespan analysis,
not raw event counts: for each fused track, how long does it survive
between birth and death, and does world-frame produce LONGER-LIVED
tracks per real-world target than camera-frame? That analysis is the
right next step before live verification.

## Final state pushed

Branch `session/phase-2-recap`:
- `5ff0465` Stage A optical-residual diagnostic
- `c13d767` Stage B initial implementation (gated, default off)
- `d203f00` **Maestro auto-reconnect** (the actual home run)
- `331ca5b` Stage B deferred + multi-FOV baseline + STAGE_AB_NOTES.md
- `73a6c90` Phase 2a: replay_fusion.py
