# Calibration v2 — verification runbook

This is the gate the v2 calibration code must pass before being trusted in
the field. Two phases: a **v1-byte-identical replay** (proves the new
code didn't regress the legacy path) and a **v2 live multi-range test**
(proves the new path actually fixes the user-visible alignment problem).

## Phase 1 — v1 replay regression guard

The dangerous-reverts memory says fusion math has regressed at least three
times in this codebase. Anything new that touches fusion gets this guard.

### Goal
Prove that with **no v2 calibration loaded** (just biases in
`config/calibration.json`), the new fusion code produces a fused-track
output stream **byte-identical** to current main. Drift here = bug in the
v2 wiring leaking into the v1 path.

### Procedure

1. **Capture a fresh JSONL recording on current main**, before deploying
   the v2 code:
   ```
   git checkout main
   # Run seeker_bench, click record, exercise: a static target, a
   # gimbal slew, a radar-only target, and a few seconds of empty
   # FOV. Save the recording.
   ```

2. **Run the recording through replay_fusion under main**, dumping the
   fused-track stream to a file:
   ```
   python -m scripts.replay_fusion <recording.jsonl> > /tmp/main_tracks.jsonl
   sha256sum /tmp/main_tracks.jsonl
   ```

3. **Switch to the v2 branch**:
   ```
   git checkout calibration-v2
   ```

4. **Wipe `config/calibration.json` to bias-only (v1) form** so v2
   loading doesn't fire:
   ```json
   {"radar": {}, "thermal": {"az_bias_deg": <existing>, "el_bias_deg": <existing>}}
   ```
   (Whatever bias values existed before the change — the v2 code preserves
   them on read.)

5. **Replay against the v2 branch**:
   ```
   python -m scripts.replay_fusion <recording.jsonl> > /tmp/v2_tracks.jsonl
   sha256sum /tmp/v2_tracks.jsonl
   ```

6. **Compare hashes.** They MUST be equal. If not:
   ```
   diff /tmp/main_tracks.jsonl /tmp/v2_tracks.jsonl | head -50
   ```
   Find the first divergence; it's a v1-path leak in the v2 wiring. Block
   the merge until fixed.

## Phase 2 — v2 live multi-range test

This is the test that maps directly to your stated symptom: "not 100% for
different distances / objects / areas in the picture."

### Goal
With v2 calibration loaded, thermal→EO overlay error must be <10 px max
across a 3-range × 3-position grid, vs. ≥30 px at off-axis or off-design
ranges with v1.

### Setup

- Load full v2 calibration into `config/calibration.json` (eo K+dist,
  thermal K+dist+R+t, biases zero).
- Place a known **hot object** (a person, a heated metal disc, a coffee
  thermos with ε≈1 tape on it — anything the Boson detects as a hot
  bbox at all three ranges).

### Capture the 9-position grid

For each of the 9 cells (3 ranges × 3 image positions):

| Range \ Position | Frame center | Frame left edge | Frame right edge |
|------------------|--------------|------------------|------------------|
| 5 m              | shot_C5      | shot_L5          | shot_R5          |
| 30 m             | shot_C30     | shot_L30         | shot_R30         |
| ~100 m           | shot_C100    | shot_L100        | shot_R100        |

For each cell, hold the rig still and capture a single synchronized
EO+thermal frame with the recorder running. Note the EO bbox of the
hot object and the **fused** bbox.

### Measure overlay error

For each cell:

```
err_px = ||(EO_bbox_center) − (fused_bbox_center_projected_to_EO)||
```

The fused bbox center is what the GUI overlay draws on EO. With v2
working, this is the projected-via-K-dist-R-t pixel; with v1 it's the
scaled-FOV pixel.

### Acceptance criteria

- **Per-cell**: `err_px < 10` for all 9 cells.
- **Vs. v1**: at off-axis cells (L/R) and off-design ranges (5 m and 100 m
  if calibration was at 30 m), v1 should show err_px ≥ 30; v2 should not.
- **Edge-perf**: `_tick` cost on Jetson Xavier AGX must not regress vs.
  v1 by more than 0.5 ms. Update `docs/EDGE_OPTIMIZATION_GAP.md` with the
  measured cost.

If v2 fails the per-cell gate but improves over v1 substantially, that's a
sign the **inter-sensor parallax** at short range still bites — v2's
current cut uses far-field projection at thermal-observation time. The
follow-up refinement (re-project at radar range when association assigns
one) addresses the residual but is out of scope for this cut. Document the
remaining error and ship v2 anyway — it's strictly an improvement over v1.

## Rollback

If either phase fails and the cause isn't an obvious v2-side bug, revert
to v1 immediately:

```
# Wipe v2 keys from calibration.json — leave bias-only form.
# The runtime auto-detects v1 and skips the new projection path.
```

No code revert needed; the v1 path is preserved.
