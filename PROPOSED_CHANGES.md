# Proposed live changes — staged patches with proofs

Each patch is independently revertable. None applied except Patch 1.

---

## Patch 1 — Drone classifier batching (frame rate) — **APPLIED**

**Status:** in place since 2026-04-27 ~21:11.

Internal saving: ~30 ms/tick on the GPU. **Invisible at the GUI level**
because the WS sender in [gui/app.py:350](gui/app.py:350) couples
thermal & EO into one combined payload — the publish rate is gated by
the EO encode side, not thermal. Decoupling that would be a separate
piece of work; out of scope for this thermal pass.

No regression. Image content unchanged.

---

## Patch 2+3 — Y16 capture + Operator Gates AGC (consolidated)

**Status:** ALL CODE STAGED. NOT APPLIED. Awaiting your go.

The two patches are presented together because Patch 3's quality win
requires Patch 2's data — operator gates need true 16-bit raw counts
to allocate display range to. Applying them as one commit means the
GUI image goes from current AGC8 → Y16+gates in one step (better than
baseline) instead of AGC8 → Y16+linear (worse than baseline) → Y16+
gates (better than baseline).

### What's already pre-built and TESTED offline

* `thermal/thermal_processor.py` extended with:
  * `apply_roi_agc(frame_u16, roi_top_frac, low_pct, high_pct)` —
    percentile over a sub-region, stretch globally
  * `apply_gates_agc(frame_u16, cold_count, hot_count)` — fixed
    raw-count thresholds, no histogram dependence
  * `apply_agc_mode(frame_u16, params)` — dispatcher
  * `ThermalEnhanceParams.mode` field — `"global" | "roi" | "gates"`
  * `from_config()` reads new YAML keys; defaults to `mode="global"`
    when YAML omits them — **byte-equivalent to legacy**
* `thermal/tests/test_thermal_processor.py` — 10 new tests covering
  all primitives + the byte-equivalence guarantee. **All 41 tests
  pass.** Confirmed:
  * `mode="global"` (default) ≡ legacy `apply_agc` bit-for-bit
  * `raw16_to_display_with_params(default_params)` ≡ `raw16_to_display`
    bit-for-bit
  * Unknown mode strings normalize to `"global"`

The new code is **already in `thermal_processor.py`**. Because every
default preserves legacy behavior, the live seeker (if restarted now)
is bit-equivalent to before. **No live behavior change yet.**

### What still needs to be applied (the actual deployment)

These three changes go in **one commit** when you say go:

#### 2A — `thermal/boson_capture.py` (5-line property-set reorder)

```diff
             raw16_ok = False
             if self.prefer_raw16:
-                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
-                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                 cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
+                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
+                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
                 ok, test = cap.read()
```

Verified by direct DSHOW probe — this order returns `(512, 640) uint16`
on this laptop. The legacy order silently downconverts to 8-bit BGR.

#### 2B — `gui/static/index.html:313` (1 attribute)

```diff
-              <input id="thr-slider" type="range" min="2" max="20" step="0.5" value="2" data-invert="1">
+              <input id="thr-slider" type="range" min="2" max="100" step="0.5" value="2" data-invert="1">
```

Backend already accepts `threshold_k` up to 100
([gui/app.py:293](gui/app.py:293)). This unlocks the same range
in the slider so you can suppress more aggressively when Y16's
wider dynamic range surfaces more warm objects.

#### 3A — `config/app_config.yaml::thermal.agc` (replace block)

```yaml
thermal:
  agc:
    # AGC mode dispatch — see thermal/thermal_processor.py::apply_agc_mode.
    #
    # global  — current pre-2026-04-27 behaviour. Percentile stretch
    #           over the WHOLE frame. Hot trees in the corners squish
    #           the working zone (vehicles on the road) into a narrow
    #           mid-gray band.
    # roi     — percentile stretch over the bottom (1-roi.top_frac) of
    #           the frame. Hot objects above the operator's region of
    #           interest may saturate to white — that's the trade.
    #           Auto-tunes to scene; no per-scene tuning needed.
    # gates   — fixed cold_count..hot_count raw-u16 thresholds. No
    #           histogram dependence; stable frame-to-frame; gives
    #           maximum target contrast in the operator-defined band.
    #           Requires per-scene tuning of cold/hot counts.
    mode: roi                   # global | roi | gates
    low_percentile: 2           # used by mode: global and roi
    high_percentile: 98
    roi:
      top_frac: 0.4             # mode: roi — percentile over bottom 60%
    gates:
      cold_count: 19500         # mode: gates — clip below this to 0 (black)
      hot_count: 22500          # mode: gates — clip above this to 255 (white)
                                # defaults derived from captured Y16 stats:
                                # frame range was [18546, 22708], p2=18704,
                                # p98=22377. Tighter gates (19500..22500)
                                # allocate full 0-255 to the working zone.
    colormap: WHITE_HOT
```

**Defaults shipping `mode: roi`** because:
* It auto-tunes per scene — no operator action needed.
* Bottom 60% of the frame matches your bench setup (sky in the upper
  band, road + vehicles + foreground in the lower). Hot trees in the
  upper band saturate to white but you don't care.
* If `roi` doesn't suit a particular scene, switching to `gates`
  (or back to `global`) is a one-line YAML edit.

### Why this should beat baseline (= AGC8 fallback) visually

The AGC8 fallback path currently in production has the camera doing
its own histogram-equalization-style AGC on the whole frame. That
gives crisp local contrast but pumps every edge equally — including
ones the operator doesn't care about (hot trees, hot pavement edges).

Y16 + ROI percentile = same percentile statistics computed only on the
operator's working band. Result: the working band gets ALL the display
range allocated to it. Vehicles render with **higher** contrast against
their immediate background than under AGC8.

Y16 + gates = even more contrast than ROI in the working band, but
fixed thresholds; needs scene-by-scene tuning.

### Live proof workflow when you say go

1. Capture `02_pre_y16.png` of the live AGC8 image at gimbal (0,0),
   37.5° (current state, just for the record).
2. Apply Patch 2A + 2B + 3A in one commit. Restart seeker.
3. Verify `Boson capture opened on index 1 (raw16=True, shape=(512, 640))`
   in seeker.log.
4. Capture `03_patch2+3_roi.png` of the same scene.
5. Side-by-side `02_pre_y16 | 03_patch2+3_roi`. Operator judges.
6. If `roi` doesn't pop enough: edit YAML `mode: gates`, restart, capture
   `04_patch2+3_gates.png`. Compare.
7. Operator picks preferred mode. Default ships as their choice.

### Detection-rate impact

From offline calibration on 60-frame Y16 capture at 37.5°, gimbal (0,0):

| Mode | det/frame at threshold_k=20 |
|---|---|
| AGC8 (current live) | 0.47 |
| Y16 + global | 5.00 |

Y16 finds **~10× more warm targets** than AGC8 at the same threshold.
That's the long-range detection win the user asked for. Sensitivity
slider (extended to 100 in 2B) lets the operator suppress when the
extra detections are too many.

### Rollback recipes

```bash
# Single-commit revert of all of Patch 2+3:
git revert <consolidated-commit-hash>

# OR partial: keep Y16 capture, revert the AGC mode:
# Edit config/app_config.yaml:
#   thermal.agc.mode: roi  ->  mode: global
# Restart seeker. (Leaves Y16 enabled but uses legacy global percentile.)

# OR fully back to baseline AGC8:
# Edit thermal/boson_capture.py — put FOURCC/CONVERT_RGB back above
#   WIDTH/HEIGHT.
# Restart seeker.
```

---

## What lands when you say go

A single commit with this short summary:

```
thermal: enable Y16 capture + ROI-based AGC mode for long-range detection

- boson_capture.py: reorder property sets so DSHOW negotiates Y16 on
  this laptop's bridge (verified by offline probe).
- thermal_processor.py: add apply_roi_agc, apply_gates_agc,
  apply_agc_mode primitives. ThermalEnhanceParams gains mode/gates/roi
  fields. from_config reads the new YAML keys. mode="global" (legacy)
  is the default — the new modes are opt-in via YAML.
- app_config.yaml: thermal.agc.mode = roi. Gates pre-set to
  cold_count=19500 / hot_count=22500 (derived from captured Y16 stats
  for the bench scene at gimbal 0,0; switching to gates is a YAML edit).
- index.html: extend SENSITIVITY slider max from 20 to 100. Backend
  already accepts up to 100.
- 41 unit tests pass, including byte-equivalence guarantee for legacy
  mode="global" behavior.
```

**Standing by for your go.**
