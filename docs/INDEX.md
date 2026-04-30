# docs/ — Index

Living documentation for the Seeker-01 bench. Living roadmap is
`NEXT.md`; everything else is historical or design-stage.

## Living docs (read these)

| File | Purpose |
|---|---|
| [`NEXT.md`](NEXT.md) | Living roadmap. P0/P1 priorities + parked items. |
| [`PHASE_3_DCA1000_PLAN.md`](PHASE_3_DCA1000_PLAN.md) | DCA1000 hardware, firmware, profile catalog, milestones. |
| [`PHASE_B_RESULTS.md`](PHASE_B_RESULTS.md) | Thermal H/V + drone classifier corpora, perf, open issues. |
| [`EDGE_OPTIMIZATION_GAP.md`](EDGE_OPTIMIZATION_GAP.md) | Per-stage CPU cost map + lever order for the Jetson port. |
| [`ID_NAMING_AUDIT.md`](ID_NAMING_AUDIT.md) | ID namespaces + phased fix plan (A→E). |
| [`CALIBRATION_V2_VERIFICATION.md`](CALIBRATION_V2_VERIFICATION.md) | v1-byte-replay + 9-cell live test gates for the V2 calibration cut. |
| [`repo_inventory.md`](repo_inventory.md) | Generated module/class/config map (last refreshed 2026-04-19; re-run when stale). |
| [`radar_setup.md`](radar_setup.md) | Radar bring-up notes. |
| [`leopard_support_email.md`](leopard_support_email.md) | EO vendor communication trail. |

## Day logs (chronological)

| Date | File | What happened |
|---|---|---|
| 2026-04-23 | [`DAY_LOG_2026-04-23.md`](DAY_LOG_2026-04-23.md) | Phase 2 late-fusion landed. |
| 2026-04-26 | [`DAY_LOG_2026-04-26.md`](DAY_LOG_2026-04-26.md) | Radar promoted to first-class fusion contributor; per-sensor decay. |
| 2026-04-27 | [`SESSION_2026-04-27.md`](SESSION_2026-04-27.md) | ID-chain refactor wrap-up. |

## plans/ — design docs that became implementations

| File | Status |
|---|---|
| [`plans/track_button_plan.md`](plans/track_button_plan.md) | Implemented in `84fe2a8`. Closed-loop pixel-error TRACK. |
| [`plans/thermal_optimization_plan.md`](plans/thermal_optimization_plan.md) | Sweep complete; AGC8 baseline kept. Results in `handoffs/2026-04-27_overnight_thermal_sweep_results.md`. |

## handoffs/ — historical session-end notes

Dated by the session they came from. Useful for archeology
("why did we revert X?") but not part of the current development
roadmap.

| File | Why kept |
|---|---|
| `handoffs/2026-04-24_resume_here.md` | Earlier resume note. |
| `handoffs/2026-04-26_overnight_predictor_fixes.md` | Predictor / radar-rotation / fusion pose-comp morning fixes. The pose-comp & 3° gate were later REVERTED (see ID_NAMING_AUDIT.md and `feedback_seeker01_dangerous_reverts` memory). |
| `handoffs/2026-04-27_overnight_thermal_resume.md` | Cross-session handoff for the overnight thermal sweep. |
| `handoffs/2026-04-27_overnight_thermal_sweep_results.md` | Indoor + outdoor sweep leaderboard. CLAHE-Y16 winner reverted by operator. |
| `handoffs/2026-04-27_proposed_y16_roi_changes.md` | Y16 + ROI-AGC patch proposal. Patch 1 applied; 2+3 reverted. |
| `handoffs/2026-04-27_session_summary_recorder.md` | Two-day recorder+replay session, lists 4 open tracking issues. |
| `handoffs/2026-04-27_stage_ab_optical_correction.md` | Stage A optical-residual diagnostic + Stage B closed-loop (deferred). Also captures the Maestro auto-reconnect home-run. |
| `handoffs/2026-04-27_when_you_get_back.md` | TRACK live-validate plan + 8 ranked radar-fix candidates. |
| `handoffs/2026-04-28_session_final_state_thermal.md` | Thermal end-of-session live state + dormant infrastructure. |
