# Plan for next session — 2026-05-08

User mandate: detection at 20–40 m on airborne1 must work. The user
believes it's a bug, not physics, and has rejected "give up at 2 RX".
You must close the gap.

**CRITICAL USER UPDATE (2026-05-07 late night):**
- "the recordings may not be good candidates if we don't have real
  received electrons. don't bias toward them. you will end up in this
  optimization loop against recordings that aren't real state."
- Translation: do NOT spend the session tuning algorithms against the
  existing 2-RX recordings. They may simply not contain a recoverable
  signature because RX2/RX3 weren't streamed. The bug to find is in
  FW + SW preventing 4-RX capture. Stage 1 (the FW/SW investigation)
  is the priority. Stage 2 (signal enhancement on existing recordings)
  is a sanity check only — useful to confirm the bug, NOT to chase
  detection at all costs.
- Beam budget: ±30° (60° total) is the default. Narrow only if it
  empirically improves SNR. Don't pre-commit to narrow.

Wake-ups confirmed (all enabled in the scheduled-tasks store):
- `radar-pmm-resume-2026-05-08` — 5/8 01:20 PDT (primary)
- `radar-pmm-followup-2026-05-08-noon` — 5/8 02:00 PDT (early backup, 40min after primary)
- `radar-pmm-followup-2026-05-09-morning` — 5/9 08:00 PDT
- `radar-pmm-followup-2026-05-10-final` — 5/10 08:00 PDT

Read first:
- `docs/SESSION_2026-05-07_radar2.md` (state)
- `docs/HANDOFF_2026-05-06_radar.md` (agent #1's handoff)
- `runs/rd_movie_drone_fly_5_15/WINDOW_SUMMARY.png` (proof method works)
- `runs/rd_movie_airborne1_30_40_v2/WINDOW_SUMMARY.png` (problem case)

## Current verified facts

1. drone_fly works. RD plot shows the drone clearly at 5 m, near-zero Doppler with broadband prop modulation. herm_v2 detector fires at bin 0–1 with 1374 Hz blade-pass, MAD 5 Hz, SNR 13–16 dB, 100% rate. Background is clean.
2. airborne1 30–40 s with 2-RX coherent sum + slot-0 decimation + BG-subtraction shows **nothing** moving in the 20–220 m corridor. Drone is reportedly at ~44 m / 2.8 m/s receding (per GUI screenshot at t=0:38).
3. Layout bug at `radar_dca/dca_pipeline.py:541` confirmed: live pipeline reshapes RX-major when wire is sample-major. Fix is one-line. The "every 24 bin chip artifact" notch is a band-aid for that bug — the artifact disappears with correct reshape.
4. RX2 and RX3 are 100% zero in the recording. Confirmed via `firmware/mmw_demoDDM_patched/mss/mmw_lvds_stream.c:234` setting `lvdsLaneEnable = 0x3U` (2 lanes).

## Stage 1 — End-to-end FW + SW audit. ANY bug that degrades the data.

**LATEST USER UPDATE (2026-05-07 late night):**
"not only the 4-rx bug. if there is any other bug that prevents me
from getting real, good raw data. I want to fix it. I'm sure we can
find this pmm drone, but we just need a functioning radar."

→ Stage 1 is no longer scoped to LVDS lanes. It's a full chain audit:
chip cfg → ADC → IF filters → LVDS → DCA1000 → host UDP → bin file →
parser → range FFT → Doppler FFT → CFAR. Every link.

### 1.0 End-to-end audit checklist (do all of these)

For each link, document the configured value, the expected behavior,
the actually-observed behavior (from the .bin or live captures), and
flag any mismatch as a bug to fix. Save the audit to
`docs/FW_SW_AUDIT_2026-05-08.md` as a checklist.

**A. Chip RF / IF chain (cfg + firmware):**
- profileCfg: idleTime, rampEndTime, freqSlope, numAdcSamples, digOutSampleRate. Compute sampled BW, range res, max range. Match meta.yaml.
- profileCfg HPF: hpfCornerFreq1, hpfCornerFreq2 (currently 0 = disabled). Is this killing target return at low IF (close range)?
- profileCfg rxGain: 164 (=16.4 dB internal scale). Is this saturating on close clutter? Check int16 saturation rate in airborne1 (we saw 27 saturated samples — anomaly).
- adcCfg: 16-bit, REAL output. Confirm.
- adcbufCfg: chanInterleave=0 (sample-major). Confirmed.
- channelCfg: rxMask=0xF, txMask=0xF (all 4 enabled). Firmware overrides to 2 lanes — Stage 1.1.
- chirpCfg: 6 chirp slots, all 4 TX active each. ddmPhaseShiftAntOrder. Confirm phase code per TX per slot is what TI documents.
- frameCfg: 128 loops × 6 chirps = 768 chirps/frame. Frame periodicity 50 ms (20 fps). Confirm.

**B. Firmware behavior (read every relevant .c file):**
- `firmware/mmw_demoDDM_patched/mss/mmw_lvds_stream.c` — lvdsLaneEnable, channel-output ordering.
- `firmware/mmw_demoDDM_patched/mss/mssgenerated/ti_drivers_config.c` — maxLVDSLanesSupported.
- `firmware/mmw_demoDDM_patched/dss/` — DSP-side processing. ANY chip-side filtering or scaling applied to ADC samples before LVDS streaming?
- `firmware/mmw_demoDDM_patched/mss/mmw_demoDDM_mss.c` — main task. Look for any clutter-removal, decimation, scaling, gain-control loop, or anything that touches data before LVDS.
- Search for "AGC", "MTI", "filter", "decimate", "scale" — anything that could attenuate signal.
- Look at any patches applied vs the stock TI demo. The README mentions "five surgical patches" — what are they exactly?

**C. LVDS / DCA1000 transport:**
- DCA1000 control commands sent (`radar_dca/control_port.py`). Verify lane mode, packet size, etc.
- Host UDP receive path (`radar_dca/data_port.py`). Verify socket buffer (8 MB fallback observed), no dropped packets in recordings (already confirmed clean), no byte reordering.

**D. Bin file / parser:**
- Already confirmed: wire is sample-major RX-interleaved, RX2/RX3 zero. Live `dca_pipeline.py:541` has wrong reshape — fix is one-line.
- Check the int16 endianness assumption. We assume little-endian. Verify by checking adjacent samples have realistic magnitude differences (they do).
- Check whether ADC output is signed or unsigned int16. We assume signed; values range −473 to +523, so signed two's complement is correct.

**E. Range FFT:**
- Hann window applied. Verify it's the right length.
- DC subtract per chirp. Verify it's not over-subtracting (e.g., if drone return dominates DC, subtracting it kills the drone).
- rfft positive-bin scaling by 2x (analytic-signal equivalence). Verify this matches expected for real-ADC.

**F. Doppler FFT / DDMA un-mix:**
- Slot-0 decimation we did is a workaround. Check whether proper per-TX phase decode would give different/better results.
- Verify slot 0 is actually where TX0 has phase 0 (per the DDMA scheme).

**G. Gimbal / pose:**
- Did the gimbal move during airborne1 30-40s? If yes, range bin migration could be from gimbal motion + drone motion, not just drone. Check `gimbal_pan_at_capture` and `gimbal_tilt_at_capture` in the JSONL. If they vary mid-recording, that's another bug source.

**H. Cfg ↔ chip ↔ recording consistency:**
- Verify `awr_cfg_sha256` in meta.yaml matches the live cfg file. If mismatch, the recording was made with a different cfg than we think we're parsing.
- Run the live pipeline once on a fresh capture (or use an existing recording) and compare its dca_pipeline range FFT output to mine via `_common_fixed.py`. Confirm the only difference is the reshape.

### 1.1 LVDS lane investigation (subset of 1.0)

### 1.1 Fix the live pipeline reshape (REQUIRED — applies to ANY scenario)
File: `radar_dca/dca_pipeline.py:541-545`
```
# OLD (BUGGY):
real_cube = (
    raw.reshape(d.n_chirps, d.n_rx, d.n_samples)
       .transpose(0, 2, 1)
       .astype(np.float32)
)
# NEW:
real_cube = (
    raw.reshape(d.n_chirps, d.n_samples, d.n_rx)
       [:, :, :2]                # only RX0+RX1 are populated; RX2/RX3 are zero
       .astype(np.float32)
)
```
After this, ALSO comment out or remove the body of `_notch_harmonic_artifact` at line 552. The artifact does not exist with correct parsing; the notch was zeroing real range bins and was specifically zeroing the drone's expected range at 70 m.

### 1.2 Investigate the 4-lane LVDS situation (REQUIRED if 4-RX is unlocked)
File: `firmware/mmw_demoDDM_patched/mss/mmw_lvds_stream.c:234` and `mss/mssgenerated/ti_drivers_config.c:212`
- Read both files. Understand what changes for 4-lane.
- Check whether the AWR2944PEVM + DCA1000 board physically supports 4 lanes. Look at `docs/STUDIO_BRINGUP*.md`, `docs/repo_inventory.md`, any schematic in `firmware/`. WebFetch the AWR2944P EVM and DCA1000 datasheet/quickstart from TI if needed.
- If the board supports 4 lanes: write a tools script `tools/lvds_4lane_audit.py` that proposes the firmware patch (do not apply it — flag for the user).
- If the board only supports 2 lanes: that's a hardware limitation. Document and proceed with the 2-RX gain plan in Stage 2.

### 1.3 Verify recordings are usable end-to-end
- Confirm `tools/diagnose_pmm/_common_fixed.py` parser works on all 3 recordings (drone_fly, airborne1, background). Already smoke-tested.
- Run the corrected layout against the live pipeline path: write `tools/diagnose_pmm/05_live_pipeline_recheck.py` that mimics the live pipeline's CFAR + Doppler chain but uses correct reshape. Compare to dca_pipeline output to confirm the bug really is the only thing different.

## Stage 2 — Sanity-check signal enhancement (NOT optimization)

PRIORITY DEMOTED per the user's late-night update. Stage 1 is the win
condition. Stage 2 is here only to confirm "with everything we have +
2 RX, the drone is still invisible at 30-70 m" — which becomes
evidence that supports re-flying with 4-RX. Do not chase detection.

If Stage 2's first pass shows the drone IS visible after enhancement,
great. If after one round of (BG subtract + multi-frame + EO oracle)
it still isn't visible at the EO-known cell, STOP enhancing and move
to Stage 1's firmware fix. Do NOT iterate further on the existing
recordings.

Beam budget per user: ±30° default (60° total). Narrow ONLY if a
bench experiment shows narrowing helps (run both broadside-sum and
narrow-beam, compare SNR at the EO-known cell, pick whichever wins).

### 2.1 RX0+RX1 phased beam — narrowband AoA (cheap)
With only 2 RX, you have one baseline. The angular resolution is poor but you CAN steer the beam by phase-shifting one RX before the sum. For a target at azimuth θ:
- Phase shift between RX0 and RX1 = 2π·d·sin(θ)/λ
- d = 1.95 mm (half wavelength at 77 GHz)
- λ = 3.9 mm
- For θ = 0 (broadside): no phase shift → just sum (current code)
- For θ = ±10°: phase shift = 2π · 0.5 · sin(10°) = 0.546 rad

What you can do:
- Compute a small "beam bank" of N=5 phase shifts spanning the EO-known azimuth ±10°.
- For each beam: compute RD map.
- Pick the beam with the highest target SNR.
- Expected gain: marginal at 2-RX (only 2 elements), but ~+1.5 dB if the drone is off-broadside.

Implementation: extend `tools/diagnose_pmm/_common_fixed.py` with an `integrate_rx_beam(rfft_cube, theta_deg)` helper. Add a `--azimuth-deg` flag to all detection scripts.

### 2.2 Multi-frame coherent integration WITH motion compensation
This is the biggest lever in the <1 sec budget.

Naive coherent integration of N consecutive frames will smear the drone's signal across range bins as the drone moves. At v = 2.8 m/s and 50 ms framePeriodicity, the drone moves 14 cm per frame = 0.05 range bins per frame. Over 20 frames (1 second) that's 1 range bin of motion. So coherent over 1 second is BORDERLINE — but only if we DON'T motion-compensate.

Two approaches:
1. **Range-bin coast** — track the drone's range bin across frames using the slow-time spectrum, then for each k, shift frame k's range axis by the predicted bin shift, then coherently sum. Requires an initial detection (a seed track).
2. **Range-rate estimation in the RD map** — compute single-frame RD maps; track a moving target as a slope in the (range-bin, frame-idx) plane; do "stretch processing" to coherently sum along that slope.

Implementation outline:
- Compute per-frame range-FFT cubes for all 200 frames in 30-40s.
- Compute per-frame slot-0 RD maps (no integration yet).
- Stack residuals (after BG subtraction) into a (range, doppler, time) cube.
- For each candidate (range_0, range_rate) pair, sum along the diagonal in (range, time) space and FFT in time. The summed power as a function of range_rate is the "stretched" RD map; peaks indicate moving targets at that range_rate.
- This is essentially a Hough transform on the residual data. Cheap — 200 frames × O(N_range × N_range_rate) ops.

Expected gain: +10·log10(20) = +13 dB over a single frame. That's massive.

### 2.3 Background subtraction in COMPLEX domain
Currently I subtract magnitude. Better: subtract the COMPLEX RD map of a background recording (or pre-drone window). This cancels static clutter exactly (not just on average) and reduces the noise pedestal that was burying the drone.
- Compute the complex slot-0 RD map averaged over background.bin's full 50 seconds.
- For each airborne1 frame in 30-40 s: subtract the (complex) average.
- This is COHERENT clutter cancellation — much stronger than magnitude subtraction.
- Note: this assumes the radar pose, gimbal, and chip phase calibration are identical between background and airborne1. If they aren't, this step adds noise instead of cancelling. Verify by running first on background-frame_N minus background-frame_M; should give ~0 if conditions match.

### 2.4 Bin-aligned EO oracle
The EO classifier reports drone azimuth/range at sampled times. Use it as ground truth:
- Read `recordings/airborne1_v5+thermalv2_replay.jsonl` for thermal/EO drone detections.
- For each EO drone detection in 30-40s: get (az_deg, range_estimate_m).
- Look at the radar RD map ONLY at that (range_bin, azimuth_beam) cell.
- Plot the slow-time spectrum at that cell over time.
- If we still see nothing, the radar SNR is genuinely insufficient — that's the proof the user wants of "physics or bug".

This is the cleanest test. It bypasses search noise entirely.

## Stage 3 — Generate the user-validatable artifact

Per the user's request: a "movie" of distance-vs-doppler heatmaps that shows where the drone is across t=0:30-0:40.

After Stage 2 lifts the SNR enough that something pokes out:
- Re-render the RD movie (`tools/rd_movie_airborne1_v2.py`) with the new pipeline (BG-subtracted complex residuals, narrow-beam at EO azimuth, coherent multi-frame).
- The track view (right panel: range-vs-time max-over-Doppler) should show a smooth rising line from ~30 m at t=30s to ~70 m at t=65s.
- This is the evidence the user wants.

If even after all of Stage 2, the airborne1 30-40s shows nothing:
- The user's prediction was right and we still need the firmware fix to enable 4 RX. Document the negative result with all details.
- Compute the noise floor at the EO-pinpointed cell. Compare to the expected drone return level (link budget calculation: 77 GHz, RCS −18 dBsm, range 30-70 m, 2-RX gain). If the predicted return is below the measured noise, conclude "SNR-limited at 2-RX, need 4-RX".

## Stage 4 — Final write-up

Document everything in `docs/FINAL_REPORT_radar_v2.md`:
- Bug list (live pipeline reshape, 24-bin notch, possibly LVDS lanes).
- Algorithm changes that worked (slot-0 decimation, BG subtraction, multi-frame, EO-oracle).
- What 4-RX would change.
- Reproduction commands for every figure.

## Code state

Already in place (working, read-only against existing recordings):
```
tools/diagnose_pmm/_common_fixed.py        — corrected layout, RX0+RX1 only
tools/diagnose_pmm/01_seq_audit.py         — packet-drop audit
tools/diagnose_pmm/04_spectrogram_grid.py  — older, uses BUGGY common
tools/rd_movie_airborne1.py                — RD movie v1 (no BG subtract)
tools/rd_movie_airborne1_v2.py             — RD movie v2 (BG-subtracted, track view)
tools/inject_herm_v2_tracked.py            — JSONL augmenter (track-filtered)
radar_dca/herm_detector_v2.py              — slot-0 blade-pass detector (WORKS on drone_fly)
radar_dca/herm_replay_v2.py                — herm_detector_v2 replay over a recording
runs/rd_movie_drone_fly_5_15/WINDOW_SUMMARY.png    — drone_fly RD proof
runs/rd_movie_airborne1_30_40_v2/WINDOW_SUMMARY.png — airborne1 RD (currently empty)
runs/herm_v2/SUMMARY_drone_detection.png   — drone_fly final result
docs/SESSION_2026-05-07_radar2.md          — live state
```

To-write next session:
```
tools/diagnose_pmm/_common_fixed.py        — add integrate_rx_beam(theta_deg)
tools/lvds_4lane_audit.py                   — Stage 1.2
tools/diagnose_pmm/05_live_pipeline_recheck.py — Stage 1.3
tools/multi_frame_motion_compensated_rd.py  — Stage 2.2 (the big lever)
tools/complex_bg_subtract_rd.py             — Stage 2.3
tools/eo_oracle_radar_lookup.py             — Stage 2.4
tools/rd_movie_airborne1_v3.py              — Stage 3 (final movie)
docs/FINAL_REPORT_radar_v2.md               — Stage 4
```

## Rules for the autonomous session

- The user is asleep until ~mid-day 2026-05-08.
- DO NOT ask permission. Run the entire plan.
- DO NOT modify the live pipeline (`radar_dca/dca_pipeline.py`); apply fixes via parallel diagnostic helpers and document the patches in the FINAL_REPORT for the user to apply.
- DO NOT commit to git unless the user has explicit prior authorization (check `~/.claude/settings.json` for committed permissions).
- ALWAYS run with `set PYTHONIOENCODING=utf-8` (Windows console default cp1252 chokes on unicode).
- Use the Agent tool for any large parallel investigation (firmware decode, web research on AWR2944P + 4-lane DCA1000, OMP rotor-rate dictionary).
- BEFORE EXITING: if work is not complete, schedule another wake-up via `mcp__scheduled-tasks__create_scheduled_task` and update `docs/SESSION_2026-05-08_radar2.md` so the next session can pick up cleanly.

## What success looks like

The user must see one of these by end of day 2026-05-09:
- A range-vs-time track plot for airborne1 30-65s showing a smooth line from ~30 m to ~70 m.
- OR a clean negative-result writeup with link-budget math showing 2-RX is below the noise floor for this RCS/range, with a specific firmware patch to apply for 4-RX retry.

The user has indicated strong belief in option (1). Treat option (2) as the fallback only after exhausting the Stage 2 signal-enhancement options.
