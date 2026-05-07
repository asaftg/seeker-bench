# Seeker-01 — Session Recorder + Replay

This is the offline-iteration loop for Seeker-01. You record a session
once on the bench, then debug ANY captured signal — tracking, fusion,
classifier, FPS, future radar algos — without re-running the rig.

## If you're an LLM agent reading this…

The user's expected workflow is:

> "I recorded the bug. The gimbal slewed 40° right and never recovered."

…with **no file paths, no CLI flags from them**. Run these in order:

```
python scripts/replay_inspect.py --latest --summary
```

That picks the newest `recordings/seeker_*.jsonl`, prints session
duration, channel counts, event types, and the top tracked ids. From
there:

```
# Per-tick CSV for one tracked id (gimbal pose, predictor internals,
# observed az/el, world position, settled gate, lead time)
python scripts/replay_inspect.py --latest --track-id <ID> --out /tmp/track.csv

# All events of a kind ("predictor", "track", "tilt_sat", etc.)
python scripts/replay_inspect.py --latest --grep predictor

# Time slice
python scripts/replay_inspect.py --latest --events-since 2026-04-26T18:12:30

# Channel breakdown
python scripts/replay_inspect.py --latest --channels

# First few raw lines (jpeg payloads trimmed)
python scripts/replay_inspect.py --latest --head 20
```

For algorithm A/B (currently: tracking predictor):

```
# Parity check vs. live recording — Δsp_pan should be ~0 for default variant
python scripts/replay_algo.py --latest --algo predictor --variant default

# Try a different lead time / smoothing alpha
python scripts/replay_algo.py --latest --algo predictor \
    --variant lead_time_s=0.5,vel_alpha=0.4 --out /tmp/variant_a.csv
```

For visual playback into the existing GUI:

```
python scripts/replay_server.py --latest
# → user opens http://localhost:8081/  (?speed=2.0 to fast-forward)
```

These three scripts are the entire surface area. They auto-discover
the latest recording. Don't ask the user where the file is.

## Format

Line-delimited JSON. One record per line:

```json
{"ts_ns": 1715000000123456789, "channel": "thermal/frame", "msg": {…}}
```

* `ts_ns` — `time.time_ns()` at message ingest by the recorder
* `channel` — free-form string. New channels appear by being written to.
* `msg` — JSON-able dict, schema-less. Recorder doesn't inspect it.

The first line of every file is `channel="session/header"`:

```json
{"ts_ns": …, "channel": "session/header",
 "msg": {"version": 1, "started_at": …, "started_at_ns": …,
         "jpeg_quality": 92, "config_snapshot": {…}}}
```

`config_snapshot` is the YAML config at recording time — replay tools
read FOVs, calibration biases, and gimbal tuning out of this.

## Channels (initial)

| Channel | Source | Rate | Key fields |
|---|---|---|---|
| `session/header` | recorder | once | `version, started_at, jpeg_quality, config_snapshot` |
| `thermal/frame` | `Topic.THERMAL` | ~30 Hz | `frame_id, jpeg_b64, hfov_deg, vfov_deg, detections, heat_tracks` |
| `eo/frame` | `Topic.EO` | ~20 Hz | `frame_id, jpeg_b64, hfov_deg, vfov_deg, detections` |
| `radar/frame` | `Topic.RADAR` | ~15 Hz | `frame_id, points, targets, max_range_m, fov_half_deg` |
| `gimbal/state` | `Topic.GIMBAL` | ~60 Hz | `pan_deg, tilt_deg, mode, target_pan_deg, target_tilt_deg, tracked_target_id` |
| `fusion/tracks` | `Topic.FUSED` | ~15 Hz | `tracks: [{id, target_class, sensors, primary, az_deg, el_deg, …}]` |
| `events` | `Topic.EVENTS` | event-driven | `type, payload` |

Adding a new channel = add an entry in `recording/encoders.py:_channels_table`
and write to `BUS.publish(Topic.NEW, msg)`. No schema, no migrations.

## Event vocabulary

`type` strings the JSONL stream may contain. Not exhaustive — new
sites add more without touching the recorder.

User actions (emitted in `gui/app.py` WS / REST handlers):
- `recording_started{path}`, `recording_stopped`
- `track_engaged{target_id}`, `track_released{reason?}`
- `track_heat_engaged{heat_id}`, `track_heat_released{reason?}`
- `synthetic_target_drawn{bbox, tid}`, `synthetic_target_cleared`
- `gimbal_manual_input{dpan, dtilt}`, `gimbal_absolute_input{pan, tilt}`
- `gimbal_home_pressed`
- `eo_exposure_set{mode, value}`, `eo_lowlight_toggled{enabled}`
- `extrinsic_tune{<bias fields>}`, `extrinsic_saved{biases, path}`,
  `extrinsic_tune_done{…}`
- `radar_tune{<knob=value>}`
- `device_changed{sensor, idx}`, `zoom_preset_changed{sensor, preset}`
- `heat_detector_set{threshold_k, min_blob_area_px, max_detections}`

System events (emitted by managers):
- `ae_converged{p99, frac_clip, exposure_ext}` — EO software AE
- `fused_track_born{id, class, primary, az, el, sensors, conf}`
- `fused_track_dropped{id, hits, misses, reason}`
- `track_grace_expired{kind, tracked_id, miss_ticks}`
- `tilt_saturated_enter{cur_tilt, el_err}`, `tilt_saturated_exit`

Algorithm diagnostics (emitted by `gimbal_manager` per tick):
- `track_predictor_step{tracked_id, now, cur_pan, cur_tilt, gimbal_dps,
   settled, fresh_fused, obs_world_az, obs_world_el, world_az, world_el,
   world_az_dot, world_el_dot, obs_count, age, confidence, lead,
   shift_az, shift_el, sp_pan, sp_tilt}`
- `track_settled_change{settled, gimbal_dps}`

## Recording

From the GUI: click the REC pill in the top bar. A new file is created
in `recordings/seeker_YYYY-MM-DD_HH-MM-SS.jsonl`. Click again to stop.

Unattended / CI / remote: `python main.py --auto-record`.

Configurable in `config/app_config.yaml` under `recording:`:
- `enabled` — master kill switch (false = REC button is a no-op)
- `auto_start` — begin at app launch
- `output_dir` — where files land
- `jpeg_quality` — 92 default, drop to 85 if recordings stutter
- `channels.<name>` — per-channel disable

## Adding a new event type

```python
from common.events import emit
emit("my_new_thing", {"foo": 1, "bar": "x"})
```

Done. The recorder picks it up via its events subscription. Type
strings are free-form; document new ones in this README so future
agents can grep for them.

## Adding a new replayable algorithm

1. Write `algorithms/my_algo.py` with a pure function that takes
   captured inputs and returns its output. No I/O, no globals.
2. Have the live manager call it (so live + replay share code).
3. Emit a per-step diagnostic event (`my_algo_step{…}`) so the
   inputs/outputs are captured in the JSONL stream.
4. Add an adapter to `scripts/replay_algo.py:ALGOS` keyed by name —
   pull the right channels from the JSONL, feed `my_algo.step`,
   write a CSV.

The `predictor` adapter is the canonical example.

## What is NOT captured

Honest list — so you know what to add if a future debug needs it:

- **Thermal raw16.** Only the 8-bit AGC-normalized display image lands
  in the JSONL. Raw 16-bit thermal counts are ~1 MB/frame at 30 Hz =
  ~1.8 GB/min — not worth recording by default. If you ever need
  absolute temperatures, add a `thermal/raw16` channel + a config knob
  to enable it; the encoder slot is reserved.
- **Sensor connect/disconnect transitions** as discrete events. Each
  frame still carries a `connected` flag, so the replay tools can
  derive the transition timeline. Add an explicit event in the
  manager if the data point becomes important.
- **YOLO sub-threshold detections.** Only confirmed detections (above
  the manager's confidence floor) ride along. The classifier's full
  score table is dropped on the floor — fine for tracker debug, not
  fine for classifier ROC tuning. Add a dev-mode `eo/raw_detections`
  channel if/when that becomes the work.
- **Gimbal driver-level µs values.** Recorded fields are degrees only.
  If the µs↔deg calibration itself comes into question, add a
  `gimbal/driver` channel exposing `pan_us, tilt_us, last_cmd_age_ms`.
- **AE bracket internals.** Only the one-shot `ae_converged` event
  fires. Mid-bracket steps aren't captured. Easy add: emit
  `ae_bracket_step{p99, frac_clip, exposure_ext}` per controller tick.
- **Live wire-format projections** (radar/fused boxes onto thermal/EO
  pixel grids). Replay rebuilds these on the fly from raw angles, so
  some panels look slightly different than live. Acceptable for
  debug; if you need pixel-perfect parity, the projection helpers
  in `gui/sensor_bridge.py` can be lifted into the replay server.

## Coverage snapshot (what each channel records)

| Channel | Per-message fields |
|---|---|
| `thermal/frame` | `frame_id, timestamp, connected, width, height, hfov_deg, vfov_deg, zoom_preset, jpeg_b64, detections[bbox, area_px, contrast, classification, synthetic], heat_tracks[id, bbox, hits, misses, age, confirmed, coasting, synthetic]` |
| `eo/frame` | `frame_id, timestamp, connected, initializing, width, height, hfov_deg, vfov_deg, source_device, jpeg_b64, detections[bbox, track_id, target_class, confidence]` |
| `radar/frame` | `frame_id, timestamp, connected, profile, max_range_m, fov_half_deg, num_points, num_targets, points[x,y,z,v,snr,r,az,el,tid], targets[tid, pos, vel, size, conf, src, np, coasting, hits, misses]` |
| `gimbal/state` | `timestamp, connected, pan_deg, tilt_deg, mode, target_pan_deg, target_tilt_deg, tracked_target_id, error` |
| `fusion/tracks` | per track: `id, target_class, confidence, sensors, primary, az_deg, el_deg, ang_w_deg, ang_h_deg, hits, misses` |
| `events` | `type, payload` (free-form; vocabulary above) |
| `session/header` | `version, started_at, started_at_ns, jpeg_quality, config_snapshot` (full YAML — every tunable parameter from every manager at record-start) |

If something the user reports doesn't appear in this list, that's
the gap to file. Adding it = one entry in `recording/encoders.py`
or one `emit("name", {...})` call.

## Why not MCAP / Foxglove?

JSONL gives:
- Plain-text grep-ability
- Zero new dependencies (stdlib `json`)
- Trivial diff'ability across sessions
- Forward-compat: `jsonl_to_mcap.py` is a ~50-line afternoon if anyone
  ever wants Foxglove Studio integration. We just don't right now.

The format is intentionally schema-less. Don't introduce one.
