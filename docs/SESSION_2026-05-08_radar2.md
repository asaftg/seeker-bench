# Session — 2026-05-08 (radar agent #2 continuation)

## Summary

This session ran ~3 hours autonomously on 2026-05-07 night before the
user's weekly token limit reset. Completed:

1. End-to-end FW + SW audit. Three parallel agents (firmware decode,
   cfg deep audit, host pipeline audit). All findings consolidated in
   `docs/FW_SW_AUDIT_2026-05-08.md`.
2. Empirical verification of two HIGH-severity findings (HPF
   attenuation, per-chirp DC subtract impact). Both are real but
   only affect <5 m range — neither explains airborne1.
3. Wrote five unified-diff patches in `docs/proposed_patches/`:
   - `01_reshape_and_notch_fix.patch` — fix dca_pipeline reshape, remove false 24-bin notch.
   - `02_dc_cal_idle_chirp.patch` — replace per-chirp DC subtract with median.
   - `03_gimbal_pose_at_arrival.patch` — capture gimbal pose at packet arrival.
   - `04_cfg_optional_close_range.patch` — optional cfg HPF/rxGain tweaks.
   - `05_lvds_4lane_firmware.patch` — enable all 4 LVDS lanes (REQUIRES HW VERIFICATION).
4. Stage 2 single sanity pass on airborne1 30-40 s with all
   enhancements applied:
   - Corrected layout (RX0+RX1)
   - Complex-domain BG subtraction using background.bin
   - 1-second coherent multi-frame integration
   - EO oracle look at known drone cell (44 m, ±1429 Hz)
   - Result: drone bin is **2.2 dB quieter than control bins** — clean
     negative result. Drone genuinely below noise floor with 2-RX SNR.
5. Final report at `docs/FINAL_REPORT_radar_v2.md`.

## Key conclusion

User's hypothesis ("it's a bug, not physics") is partially confirmed:
SW bugs found and patched. The remaining gap to airborne1 detection
is the **2-lane LVDS firmware constraint** (`mmw_lvds_stream.c:234`).
Removing this unlocks all 4 RX, gives ~+6-9 dB effective SNR via
coherent gain + beamforming. **This requires verifying the EVM
hardware physically routes 4 LVDS lanes** before patching the
firmware.

## What the next session should do

1. Read `docs/FINAL_REPORT_radar_v2.md`. It's the deliverable.
2. **Hardware verification — TOP PRIORITY**: WebFetch / web-search the
   official TI AWR2944EVM Hardware Reference Guide. Look up the
   "LVDS connector pinout" or "Samtec connector pin assignment" or
   similar — typically section 4 or 5. Confirm whether 4 LVDS lane
   pairs (LVDS_0_P/N through LVDS_3_P/N) are physically routed to
   the DCA1000 60-pin Samtec connector. If yes: green-light the
   `05_lvds_4lane_firmware.patch`. If no: 2-RX is a hardware ceiling
   on this EVM; document that in the FINAL_REPORT.
3. **Phase coherence sanity check** (Tier 3 item from final report):
   verify across-frame phase continuity at a stationary range bin.
   This determines whether the 1-second coherent integration in
   stage2_sanity_eo_oracle.py was actually coherent or partially
   incoherent. A stronger result possible here would change the
   negative-result interpretation.
4. **DDMA phase code verification**: confirm the slot-0 decimation
   really does give all 4 TX at phase 0 (coherent sum). Synthesize
   a known TX0-only point target through the cfg's DDMA pattern
   numerically; verify our slot-0 decimation recovers it without
   ghost copies.
5. Update `docs/FINAL_REPORT_radar_v2.md` with the hardware
   verification outcome and any new findings.

DO NOT redo anything in the "Summary" section above. All five patches
are written; the audit doc is complete; Stage 2 sanity test is
already run with negative result.

## Wake-up cadence

- 2026-05-08 02:00 PDT (next, in <1h after user's limit resets at 01:20)
- 2026-05-09 08:00 PDT
- 2026-05-10 08:00 PDT (final report polish)
