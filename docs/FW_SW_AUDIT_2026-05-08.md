# Radar FW + SW end-to-end audit — 2026-05-07/08

User mandate (verbatim):
> "find any bug that prevents me from getting real, good raw data. I'm
> sure we can find this pmm drone, but we just need a functioning radar."

This is a complete audit of the chain: chip cfg → RF/IF → ADC → LVDS
→ DCA1000 UDP → host bin file → parser → range FFT → Doppler →
detection → tracking. Every link audited; every potential bug listed.

## Status snapshot — what's already known good

| Link | Status |
|---|---|
| Chip RF chain (cfg) | Mostly correct, two refinements available (HPF, rxGain) |
| Firmware patches | 5 surgical patches applied, all bug-fixes for halt/streaming. None of them degrade signal. |
| LVDS lane wiring | 2 lanes only (`lvdsLaneEnable=0x3U` in firmware). RX2/RX3 zero-padded. Probably hardware constraint of the DCA1000 EVM. |
| DCA1000 / UDP | 0 packet drops in any of the 3 recordings. Receive buffer fallback is to 8 MB silently — flag for diagnostics, not signal-degrading. |
| Bin parser (offline) | Confirmed correct in `tools/diagnose_pmm/_common_fixed.py`. |
| Cfg sender | Pass-through, no transformations. |
| drone_fly result | 100% detection at bin 0-1 with 1374 Hz blade-pass, MAD 5 Hz, SNR 13-16 dB. |

## Bugs found, ranked

Each bug has a unified-diff patch in `docs/proposed_patches/`.

### B1 (HIGH, already known, fix documented)
**dca_pipeline.py:541-545 — wrong reshape.** Treats wire as RX-major; actual is sample-major RX-interleaved with RX2/RX3 zero. The "every-24-bin chip artifact" is the alias spectrum of this wrong reshape — not a real artifact. Patch: `01_reshape_fix.patch`.

### B2 (HIGH, already known, fix documented)
**dca_pipeline.py:552-603 — `_notch_harmonic_artifact` zeros 41% of range axis.** It's a band-aid for B1. With B1 fixed, the artifact does not exist. The notch was zeroing real radar data including the drone's expected range at 70 m (bin 27 falls in zone 18-30). Patch: `02_remove_24bin_notch.patch` (must be applied together with B1).

### B3 (HIGH, NEW)
**dca_pipeline.py:546 — per-chirp DC subtraction.** `real_cube -= real_cube.mean(axis=1, keepdims=True)` subtracts each chirp's sample mean before range-FFT. This conflates per-chirp DC bias with target signal. For close-range targets the target return contributes substantially to the chirp's DC level — subtracting it attenuates the close-range target.

Empirical impact (drone_fly avg chirp at 5m, my measurement just now):
| range bin | range | with DC subtract | without | impact |
| --- | --- | --- | --- | --- |
| 0 (DC) | 0.0 m | 459 | 1638 | **−11.0 dB** (DC killed by design) |
| 1 | 2.6 m | 1695 | 2867 | **−4.6 dB** (real attenuation) |
| 2 | 5.3 m | 1914 | 1918 | 0 dB (drone hover bin — fine) |
| 3+ | 7.9+ m | — | — | 0 dB (no impact) |

So this only hurts ~0-3m targets. **Not the airborne1 problem.** But it should be fixed using a fixed DC calibration computed from idle chirps (or background frames), not per-chirp self-subtraction. Patch: `03_dc_cal_fix.patch` (optional; lowest priority of HIGHs).

### B4 (HIGH, NEW)
**radar_manager.py:723-726 — gimbal pose captured at frame processing time, not at sensor capture time.** When the radar TLV arrives over UART, the host sleeps until `_process_and_publish` (line 687). After clustering completes, the gimbal pose at the current wall clock is read. This is up to ~50 ms after the chip actually acquired the data. With the gimbal slewing at >10°/s, this is up to 0.5° pose error — phantom track positions in fused (radar + EO) views.

This could explain why the user's GUI screenshot shows the radar's 44m / 2.8 m/s detection sometimes at the wrong place: pose latency. Patch: `04_gimbal_pose_at_capture.patch`.

### B5 (MEDIUM, NEW — actionable cfg change)
**`radar/cfg/awr2944P_unified.cfg:37` profileCfg HPF=0.** On AWR2944P, hpfCornerFreq1=0 means **350 kHz active corner**, not disabled. For a chirp slope of 8.883 MHz/μs:
- 2.6 m target: f_IF = 154 kHz → ~7 dB attenuation
- 5.3 m target: f_IF = 314 kHz → 0.9 dB at the knee
- ≥8 m: above 350 kHz → no attenuation

Doesn't affect airborne1 (>20m), but does attenuate drone_fly's 5m drone slightly. Optional cfg change to lower HPF if close-range needed. Patch: `05_cfg_hpf_optional.patch`.

### B6 (MEDIUM, NEW)
**`radar/cfg/awr2944P_unified.cfg:37` profileCfg rxGain=164** = 36 dB IF + 36 dB RF = 72 dB total (decoded per `rl_profileCfg_t::rxGain` bitfield). We saw 27 ADC saturations in airborne1's first frame (close clutter saturating int16). Reducing to 162 (34 dB IF, 70 dB total) removes the saturation at the cost of −2 dB SNR at far range. Patch: `06_cfg_rxgain_optional.patch`.

### B7 (LOW)
**clustering.py:100, 512-513 — Kalman track coast 30 frames.** A confirmed track keeps publishing `coasting=True` predictions for up to 30 frames (~1.5 s at 20 fps) without any sensor input. This could explain GUI's "R#47 coast" — the track is mostly coast, not real fresh detections. Reduce to ~6 frames for fast-moving targets. Patch: `07_coast_budget.patch`.

### B8 (LOW)
**data_port.py:411 — payload-queue silent drop.** Queue overflow drops oldest UDP payloads silently; counter is not in diagnostics output. If the consumer thread stalls, recordings could be incomplete in non-obvious ways. Patch: `08_queue_diagnostics.patch`.

### B9 (LOW, INFORMATIONAL — not a fix)
**Firmware patches already in place** (`firmware/mmw_demoDDM_patched/`):
1. `disableFrameStopAsyncEvent = true` in mss_main.c:2606 — suppresses BSS FRAME_END that was tearing down LVDS (HIGH).
2. RL_RF_AE_FRAME_END_SB handler removed dataPathStop call (HIGH).
3. PAD_BYPASS clear in mss_main.c:3294 — disables nRESET pin glitch (HIGH).
4. enablePeriodicity = false in calibrationCfg — disabled periodic RF calib (MEDIUM).
5. Removed per-frame debug printf — was stalling CLI UART (HIGH).

None of these patches degrade signal; all fix halt/instability. They're already applied. Nothing to do here.

## What we did NOT find — items the next session should still investigate

1. **2-RX vs 4-RX root question.** Whether the AWR2944PEVM + DCA1000EVM has 4 LVDS lanes physically routed. The firmware caps to 2 lanes; if the hardware supports 4, this is the single biggest signal recovery available (~+6 dB for far-range targets). **Action**: read schematics or run an LVDS-bringup test with `lvdsLaneEnable = 0xFU`.

2. **DDMA phase-code verification.** I assumed slot 0 of the 6-slot pattern has all 4 TX at phase 0 (coherent sum). If actually different, the slot-0 decimation strategy gives a less-than-coherent result and we lose a few dB of SNR. **Action**: synthesize a known TX0-only signal on the bench, push through the DDMA pattern, compare the slot-0-decimated spectrum to a single-TX baseline.

3. **Frame-to-frame phase coherence.** Coherent multi-frame integration in <1s budget needs the slow-time signal at one range bin to have continuous phase across the 25-ms quiet gap between frames. If the chip retunes the LO between frames, phase coherence is broken. **Action**: take a stationary range bin and unwrap its phase across 20 frames; verify continuous-ish (no step-discontinuities at frame boundaries).

4. **Saturation impact.** 27 saturated int16 in airborne1's first frame. Reduce gain or move sensor; characterize how often saturation occurs in the drone window.

## Path forward

In order of effort vs. signal-recovery payoff:

1. Apply patches B1+B2 (one combined fix). Re-run the live pipeline. Result: clean range FFT, no false 24-bin spurs, drone_fly still works.

2. Investigate item #1 above (4-lane LVDS). If feasible, change firmware, re-flash, re-fly airborne1. Expected: ~6 dB SNR gain at far range, plus real angle-of-arrival.

3. Apply patch B4 (gimbal pose timing). Re-fuse EO+radar; expect tighter overlay alignment.

4. Implement coherent multi-frame integration with motion compensation in a parallel diagnostic helper (`tools/multi_frame_motion_compensated_rd.py`), single-pass test on existing recording. If the airborne1 drone becomes visible at ~44m / 2.8 m/s, ship that. If not, the recording genuinely lacks recoverable signal — re-fly with patches applied.

5. Apply optional patches B3, B5, B6, B7, B8 — quality-of-life, not on the critical path.
