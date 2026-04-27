# Next up

Living roadmap — top priorities first, parked items below.

---

## Recently shipped

- **2026-04-23** — Radar overlay on EO/thermal, software extrinsic
  calibration sliders, unified OVERLAY SCREEN. (`b2debe1`)
- **2026-04-26** — **Phase 2 late fusion landed.** Radar is a
  first-class FusionManager contributor; per-sensor decay; coasting
  filter; unified single-overlay-per-target rendering.
  (`8361dc9`, `107cb81`, `1d0bcc3`, `98005eb` — see
  `docs/DAY_LOG_2026-04-26.md`)

---

## Now

### P0 — Bench-calibrate the radar
The radar az/el bias is still 0° — it has to be dialled in before
class-promotion (`radar_target` → `vehicle`) will fire reliably.
Once a radar target's projected bbox sits on top of the same
real-world target's EO bbox, the cross-sensor IoU will exceed
`radar_iou_gate=0.05` and tracks will merge.

- Park a known target in the FOV. Drag radar az/el sliders until
  the dashed `RADAR_TARGET` bbox sits on the EO image of the same
  target. Then drag thermal az/el until the thermal box agrees.
- Verify class-promotion fires: radar-only born → EO sees → label
  flips from "RADAR TARGET" to "VEHICLE" on the same row id.
- Once happy with values, save them (see P0.5).

### P0.5 — Persist extrinsic sliders to YAML
Live `extrinsic_tune` updates the managers but restart reverts to
YAML defaults. Pick:
- SAVE button in the DEV card that POSTs a patch back into
  `config/app_config.yaml` via `ruamel.yaml` (preserve comments).
- Or auto-write on debounced settle (risk: clobbers manual YAML
  edits made in parallel).
Lean toward the explicit SAVE button.

### P1 — Overlay screen polish
- **Verify thermal-only / EO-only fused tracks** behave correctly on
  the new source-centric filter (a single-sensor fused track projected
  onto the other panel should disappear when its source sensor is
  toggled off).
- **Toggle indicators on canvases.** Right now there's no visual cue
  on the thermal/EO panel that an overlay is suppressed. Consider a
  small "overlays: R T E" legend in the panel corner with dimmed
  letters for off sensors.

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
