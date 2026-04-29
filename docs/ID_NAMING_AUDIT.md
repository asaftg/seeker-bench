# ID Naming Audit & Fix Plan

> Goal: ONE physical target → ONE ID the operator sees, consistently
> across every panel, the targets list, and the gimbal lock state.

## Current state — every ID namespace in the system

| Namespace | Mint site | Used for | Wire field | GUI label |
|---|---|---|---|---|
| **Fused** (`#N`) | `fusion_manager.py:_next_id` (starts at 1) | Cross-sensor track. Canonical "TRACK lock" target. | `fused[].id`, `top_targets[].id`, `tracked_target_id` | `#42`, `2×#42` (when ≥2 sensors), `#42 ← EO` (projection) |
| **Heat track** (`H#N`) | `detection_tracker.py:_next_id` (starts at 1) | Thermal heat-blob persistence tracker (DBSCAN-style with Kalman + OF bridge). Used for dev-mode "track this heat blob" lock AND for synthetic-target tracks (user "DRAW TARGET"). | `thermal.heat_tracks[].id`, `tracked_heat_id` | `H#7` (dev mode rows + heat-track box overlay) |
| **EO ByteTrack** (`E#N`) | ultralytics ByteTrack inside `eo_classifier.py` | Per-sensor persistence on EO YOLO detections. Transient: re-keyed when YOLO loses confidence > a few frames. | `eo.detections[].track_id` | `E#15` (raw EO box, fallback when no fused id) |
| **Radar Kalman** (`tid`) | `radar/clustering.py:_next_tid` (starts at 0) | Persistent radar target tracker. | `radar.targets[].tid` | **NOT DISPLAYED** anywhere in the GUI |
| **Radar DBSCAN cluster** (`tid`) | `clustering.py` cluster-label path | Frame-local DBSCAN cluster id, ephemeral (255 = unassigned). | `radar.points[].tid` | not displayed (raw points just rendered as dots) |
| **Synthetic target** | shares `detection_tracker._next_id` | User "DRAW TARGET" bbox. | `thermal.heat_tracks[].id` (with `synthetic:true`) | `H#N` — visually identical to a real heat track |

## What's wrong with this picture

### 1. `tracked_target_id` is OVERLOADED in `GimbalState`
```python
# common/frames.py:GimbalState
tracked_target_id: Optional[int] = None
```
Holds **either** a fused id **or** a heat id depending on which lock is
live. The WS boundary (`gui/sensor_bridge.py`) splits it back into
`tracked_target_id` (fused) + `tracked_heat_id` (heat) by reading both
`gstate.tracked_target_id` AND a separately-tracked `tracked_heat_id`.
But GimbalState itself can't distinguish. If a future consumer reads
GimbalState directly they get an int with no namespace tag.

### 2. `tid` is double-namespaced inside the radar pipeline
- `RadarDetection.tid` = DBSCAN cluster id (frame-local, ephemeral)
- `RadarTarget.tid` = Kalman tracker id (persistent across frames)

Same field name, two completely different meanings. A consumer that
joins points to targets by `tid` gets accidentally-correct results
within a single frame and accidentally-wrong results across frames.

### 3. Inconsistent linking from raw det → fused track
We added `FusedTrack.eo_track_id` so the GUI can match an EO YOLO
raw det (`E#15`) to its fused track (`#42`) by id rather than by
EMA-drifted bbox IoU. **No equivalent for thermal or radar:**
- `FusedTrack.thermal_heat_id` — should link to `heat_tracks[].id`
- `FusedTrack.radar_tid` — should link to `radar.targets[].tid`

So a thermal heat track that's been promoted into a fused track shows
`H#7` on the thermal panel and `#42` in the targets list — two ids,
no way to correlate. Same problem we just solved for EO, only
unsolved for thermal/radar.

### 4. No id label on radar boxes
RadarTargets carry a Kalman `tid` but the GUI never renders it. You
can't tell whether the box you see now is the same radar target as
two seconds ago. Should be `R#N`.

### 5. Synthetic targets are visually indistinguishable from heat tracks
Both render as `H#N` on the thermal panel. Synthetic gets a magenta
dashed "USER TARGET" overlay sometimes, but the *id label* is the
same shape as a real heat blob. A separate `S#N` namespace (or just
`H*#N` with an asterisk) would make it obvious which is which.

### 6. Targets list mixes namespaces
In dev mode the TARGETS panel lists `#42` and `H#7` rows together.
Different namespaces, same column. Easy to misclick.

### 7. The link metadata is one-directional
`FusedTrack.eo_track_id` lets the GUI go fused → EO det. But there's
no field on `EODetection` saying "I'm currently fused as #42." So
the GUI has to do the lookup. Same for any future thermal/radar
links. A bidirectional link (both sides carry the other's id) would
let any panel render the right label without a lookup.

### 8. `track_id` vs `id` vs `tid`
- `EODetection.track_id` — int, may be -1 for unconfirmed
- `FusedTrack.id` — always set, monotonic
- `RadarTarget.tid` — int, monotonic, starts at 0
- `HeatTrackDebug.id` — int, monotonic, starts at 1
- `RadarDetection.target_id` — int, 255 = none (different sentinel!)

Three different names for the same concept. Three different
"unassigned" sentinels (`-1`, `None`, `255`).

## Proposed fix — phased

The fix is a rename + link-chain consolidation. Phased so we can
land each step independently and revert if it breaks something.

### Phase A — names without behaviour change (safe)

A1. **Rename `RadarDetection.tid` → `RadarDetection.cluster_id`** (it's
    a cluster id, not a tracker id). Wire field stays `tid` for
    backwards compat for now; just the Python field gets the right name.

A2. **Rename `EODetection.track_id`'s sentinel from `-1` → `None`** for
    consistency with `FusedTrack.eo_track_id` (already `Optional[int]`).
    Update `eo_manager.py` to skip `tid is None` instead of `tid < 0`.

A3. **Document the single canonical convention in `common/frames.py`:**
    - `id` — primary key of a tracker-managed object (FusedTrack.id,
      HeatTrackDebug.id)
    - `track_id` — link to a per-sensor tracker (EODetection.track_id,
      RadarTarget.track_id once renamed)
    - `cluster_id` — frame-local cluster (RadarDetection.cluster_id)
    - 255 / -1 sentinels are forbidden; use `Optional[int] = None`

### Phase B — link-chain symmetry (mostly fusion-side)

B1. **Add `FusedTrack.thermal_heat_id`** — populated when a fused
    track was last updated by a thermal observation. Mirror of
    `eo_track_id`. Lets the thermal panel match raw-det → fused id by
    id rather than by IoU (same drift problem we just fixed for EO).

B2. **Add `FusedTrack.radar_tid`** — same idea for radar contributions.

B3. **Wire all three through `_observations_from_*` →
    candidate dict → `_update_tracks` (birth + update) → FusedTrack
    construction → `fused_to_wire`** (mechanical extension of what we
    already did for EO).

### Phase C — gimbal-lock disambiguation

C1. **Replace `GimbalState.tracked_target_id: Optional[int]` with**
    ```python
    @dataclass
    class GimbalLock:
        kind: Literal["fused", "heat", None]
        id: Optional[int]
    GimbalState.lock: GimbalLock
    ```
    Keeps the int payload but tags the namespace. Wire side becomes
    `lock: {kind, id}`; old `tracked_target_id` / `tracked_heat_id`
    become derived.

C2. **`set_track_target()` and `set_track_heat()` set `_lock` with the
    correct `kind`** instead of two separate fields that can both be
    None or one-of-each.

### Phase D — display unification

D1. **Add `R#N` labels to radar boxes** (uses `RadarTarget.track_id`
    after the A1 rename).

D2. **Show fused id on thermal raw-det boxes** the same way EO now
    does — using `FusedTrack.thermal_heat_id` from Phase B. Fall back
    to `H#N` (heat-tracker id) if no fused match yet.

D3. **Synthetic targets render as `S#N`** in both the panel and the
    targets list, so the operator can't confuse a user-drawn target
    with a real heat blob.

D4. **Targets list groups rows by namespace:**
    ```
    TARGETS (fused)
      #42  VEHICLE  ...
      #43  PERSON   ...
    HEAT (dev only)
      H#7  ...
    SYNTHETIC
      S#1  ...
    ```
    instead of the current single mixed list.

### Phase E — one-shot self-test

E1. **Extend `scripts/_smoke_eo_track_id.py`** into
    `_smoke_id_chain.py` — synthetic frames for each sensor, asserts
    every link survives every stage. Locks the chain so a future
    blanket-revert can't silently break id propagation again.

## Where I'd start

**Phase A2 + B1 + B2** unblock the visible UX issue (thermal raw det
shows wrong id when fused track exists, exactly the same way EO did
before today). Phases C–D are quality-of-life: nicer separation,
clearer labels, less ambiguity. Phase E locks it all in.

## What NOT to do (lessons from today)

- Don't touch sensor capture/processing code as part of an ID
  rename. Sensor pipelines are stable; only the metadata flowing out
  needs to change.
- Don't make sweeping renames + behaviour changes in one commit. Each
  phase above should be one PR / one commit.
- Don't ship without the smoke test passing for every changed link.

## Open questions (for the operator)

1. Are the `H#N` rows in the targets list actually useful in
    production, or only for dev-mode? If dev-only, hide them entirely
    in the operator-facing list.
2. Should the radar `R#N` label be visible always, or only in dev
    mode? Risk: radar IDs churn (especially with id-swap on coast),
    might add visual noise.
3. Synthetic targets — `S#N` or just visual styling (magenta dashed
    is already there)? I lean toward `S#` *and* the existing styling,
    so the operator knows at a glance.
