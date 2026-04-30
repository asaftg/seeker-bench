# Thermal Optimization — Overnight / Cross-Session Handoff

For the next Claude Code session OR the operator coming back to a
partially-complete run.

---

## What's running tonight

Sensor in the garage. Operator's plan: leave it pointing at a stable
scene, run `OVERNIGHT_THERMAL.bat`, walk away. Harness sweeps the
software pipeline and writes a ranked leaderboard to
`recordings/optim/<pose>/`.

Default pose tag (when no arg passed to the .bat): `garage_overnight`.

---

## How to check status

1. **Heartbeat:** `recordings/optim/<pose>/HEARTBEAT.txt` — most-
   recent harness activity, updated every config (~5-15s).
2. **State:** `recordings/optim/<pose>/STATE.json` — current tier,
   list of completed configs, top-5 leaderboard.
3. **Leaderboards:**
   `recordings/optim/<pose>/leaderboard_t1t2.md`,
   `leaderboard_t3.md`, `leaderboard_overall.md` — written at end
   of each tier, sorted by composite score.
4. **Previews:** `recordings/optim/<pose>/<config_name>/preview.png`
   — one preview per config.

If the .bat is still running (cmd.exe window open), it's actively
working. If the window closed and `STATE.json::tier == "complete"`,
the run finished.

---

## How to stop a run cleanly

```
echo. > recordings\optim\<pose>\STOP
```

The launcher checks this sentinel between configs and exits the loop.
The harness writes a final state and the leaderboard up to where it
stopped.

---

## How to resume a run

The launcher already does this automatically. If you killed the .bat
mid-run, just re-run it:

```
OVERNIGHT_THERMAL.bat <pose>
```

`STATE.json::completed_configs` is the source of truth. The launcher
passes `--resume` to the harness, which skips configs that are
already in the list.

---

## How to start a NEW pose / scene

Each pose tag gets its own subdirectory under `recordings/optim/`.
To run a different scene:

1. Reposition the gimbal in the GUI.
2. (Re)stop seeker so the camera handle is free.
3. `OVERNIGHT_THERMAL.bat <new_pose_tag>`

---

## What the harness is doing — quick map

`scripts/_thermal_optim_launcher.py` (auto-retry wrapper, ~5 retries on transient failure)
   └─ `scripts/_thermal_optim_harness.py` (one orchestration pass)
        ├─ captures Y16 + AGC8 stacks (~30 frames each, ~2 min)
        ├─ tier 1+2 sweep (~12 configs at 5s each = ~1 min on GPU)
        ├─ tier 3 if T1+T2 winner is CLAHE-Y16 (~18 configs)
        └─ writes leaderboards + previews

Total wall-clock per pose: roughly 5 minutes if YOLO model loads,
~10 minutes if YOLO inference runs on every config.

---

## What to do if the harness crashes repeatedly

Most likely causes, in order of frequency:

1. **Boson handle held by seeker** — kill any python.exe that's
   holding the camera, then re-run.
2. **`models/seeker_thermal_hv.pt` missing or wrong path** — the
   harness logs "YOLO model not loaded", then runs without YOLO
   scoring (composite drops by the YOLO weight). Not fatal.
3. **Permission error on COM port for Boson** — only relevant if
   we're using `BosonControl` (currently NOT in the harness;
   software-only sweep is the default).

Output to look at: stdout from the .bat window + `HEARTBEAT.txt`.

---

## How a fresh Claude Code session should resume

1. Read this doc.
2. `cat recordings/optim/<pose>/STATE.json` to see current tier and
   completed-config count.
3. If `tier != "complete"`, run:
   ```
   python scripts/_thermal_optim_launcher.py --pose <pose>
   ```
4. Wait for it to finish, then summarize the leaderboard for the
   operator.
5. If `tier == "complete"`, no resumption needed — report the top-5
   from `STATE.json::leaderboard_top` and ask if the operator wants
   to apply the winner's config to YAML (it ships behind a one-line
   YAML edit: `thermal.agc.mode: clahe_y16` already, plus the
   winning `tile_grid` value).

The optimization plan and full parameter taxonomy is in
[THERMAL_OPTIMIZATION_PLAN.md](THERMAL_OPTIMIZATION_PLAN.md).
The full proposed-changes record is in
[PROPOSED_CHANGES.md](PROPOSED_CHANGES.md).

---

## Key invariants the next session must respect

* **Don't touch EO, fusion, gimbal, radar, or any non-thermal code.**
  The operator has been clear about this throughout. Other agents
  may be working those areas — coordinate via the operator before
  any cross-domain change.
* **Default mode is `clahe_y16`** in `config/app_config.yaml`. The
  legacy `global` mode is one YAML edit away if the operator ever
  wants to compare. Keep it that way.
* **Patch 1 (drone classifier batching) is in place.** Don't undo.
* **Boson Y16 capture is in place.** The `boson_capture.py` property-
  set order is the verified-working one. Don't revert.
* **The Boson SDK function IDs are tagged IDD-VERIFIED-PENDING.**
  Don't trust them in a state-changing command without first
  validating against an actual Boson IDD or a successful test
  against the live sensor.

---

## Files / directories to know

| Path | Role |
|---|---|
| `THERMAL_OPTIMIZATION_PLAN.md` | parameter taxonomy + test strategy |
| `PROPOSED_CHANGES.md` | running list of patches in flight |
| `scripts/_thermal_optim_harness.py` | sweep orchestrator |
| `scripts/_thermal_optim_launcher.py` | retry wrapper |
| `scripts/_thermal_metrics.py` | scoring functions |
| `thermal/boson_control.py` | FLIR FFC pyserial wrapper (NOT yet wired into harness) |
| `OVERNIGHT_THERMAL.bat` | one-click overnight launcher |
| `recordings/optim/<pose>/` | per-pose results |
