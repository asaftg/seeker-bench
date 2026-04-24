# Next up

Living roadmap — tomorrow's priorities on top, parked items below.

---

## Tomorrow (2026-04-24)

### P0 — Finish calibration workflow
- **Persist extrinsic sliders to YAML.** Right now `extrinsic_tune`
  only updates the live managers — a restart reverts to YAML values.
  Options:
  - Add a "SAVE" button in the DEV card that POSTs a patch back into
    `config/app_config.yaml` (write via `ruamel.yaml` to preserve
    comments/formatting), OR
  - Auto-write on debounced settle (risk: clobbers manual YAML edits
    someone made in parallel).
  - Lean toward the explicit SAVE button.
- **Calibrate on the bench.** Put a target in the scene, line up the
  EO bbox, drag radar az/el until the cyan radar bbox sits on top,
  then drag thermal az/el until the thermal bbox agrees. Capture the
  numbers and commit them to YAML.

### P1 — Overlay screen polish
- **Verify thermal-only / EO-only fused tracks** behave correctly on
  the new source-centric filter (a single-sensor fused track projected
  onto the other panel should disappear when its source sensor is
  toggled off).
- **Toggle indicators on canvases.** Right now there's no visual cue
  on the thermal/EO panel that an overlay is suppressed. Consider a
  small "overlays: R T E" legend in the panel corner with dimmed
  letters for off sensors.

### P2 — Phase 2 late fusion (radar contributor)
Start the real work: extend `FusionManager` to consume `RadarTarget`s
alongside EO/thermal detections, producing unified `FusedTrack`s where
`sensors` can include "radar". This is the actual late-fusion (vs.
today's projection overlay).

**Why late (not early/mid):** all three sensors already produce tracked
outputs — plugging radar into `FusionManager`'s existing track-to-track
association is a local extension, not a rewrite. Early fusion discards
radar's temporal smoothing; mid fusion (radar-points-into-image +
shared detector) is a real win for detection at range but needs a
joint model we don't have yet — revisit as Phase 3.

Requires:
- Observation model for radar (az, el from pos; range as extra gating
  feature the EO/thermal path doesn't have).
- Track-to-track association gate for radar vs. existing tracks.
- GUI row rendering already handles multi-sensor "sensors" lists, so
  minimal UI churn.

### P1.5 — Gimbal tracking / PID tuning (separate ticket, non-radar)
Target-lock tracking is functional but not tight. Action items:
- Profile current pan/tilt error response with a step input (slew
  onto a target, log error over time) to see actual rise/settle.
- Tune PID gains (currently softened for synthetic-target scenes —
  see commit `6e3e39c`). Real-target behavior may want different
  balance between aggression and overshoot.
- Consider splitting gains per-axis (tilt often needs more damping
  than pan due to gravity-loaded servo).
- Check deadband + slew-rate limiter interaction — mechanical
  stops at pitch limits shouldn't trigger integrator windup.

---

## Parked — waiting for DCA1000 hardware

- **Range extension.** Retune chirp profile for longer unambiguous
  range; needs raw ADC capture to validate.
- **Drone / PMM work.** Small-RCS target tuning, possibly CFAR changes,
  micro-Doppler signature analysis.

---

## Nice-to-have backlog

- **Gimbal click-on-panel to aim.** Click a point in the EO or
  thermal canvas → convert pixel to az/el via the panel's FOV →
  send as a gimbal goto command. Useful during bench testing.
- **Recording pipeline (HDF5).** The REC pill currently just toggles
  a stub. Wire it to actually log fused tracks + raw frames for
  offline analysis.
- **Extrinsic persistence UX.** If we go with the SAVE button,
  consider showing a "modified" indicator on the card title when
  sliders diverge from the last-saved YAML values.
