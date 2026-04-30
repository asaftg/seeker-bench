# Phase 3 morning handoff — 2026-04-29

## TL;DR

**Architecture works as agreed.** Single firmware (`mmw_demoDDM` already
in QSPI), single jumper config (current state, no changes), one cfg
push gives both TLV (existing on-chip DSP path) AND raw ADC (over
LVDS to DCA1000) flowing simultaneously. Mode toggling between
stock / AG / AA is now purely host-side: same chirps, same chip
state, host picks which data plane to consume.

Validated on bench at 10:34 AM — see `recordings/20260429_103459_aa_validation/`.

## What changed today (the journey)

I went down some wrong paths before landing on the right one. Honest
record so we don't repeat:

1. **mmw_demoDDM with `lvdsStreamCfg -1 1 1 0`** (header=1) → chip
   crashes at `sensorStart` with `Init Calibration Status = 0xffe / Error -1`.
   Confused this for "firmware corruption" overnight. **It wasn't.**
2. **Read the firmware source** (`mss_main.c:419`): "Note that only HW
   data without HSI header is supported as of now." `header=1` is
   simply unsupported. The fix is `header=0`.
3. **Tried mmWave Studio path** as a workaround — needed jumper change
   (J18 in for development mode), Studio's `JsonImport` failed because
   missing Matlab Runtime, the lower-level `ar1.PowerOn` timed out
   waiting for chip handshake. **Rabbit hole. Do not redo.**
4. **Pivoted to "just fix the cfg"**: stock chirp profile +
   `lvdsStreamCfg -1 0 1 0` (header=0). **First try → 76 MB raw ADC
   in 3 seconds.** Architecture was always there; we just had a
   single bit wrong in the cfg.
5. **Final validation**: the same cfg streams TLV (21 KB with magic
   word) AND raw ADC (119 MB) to the host simultaneously, on a
   single power-cycle.

## Current state

- AWR2944P firmware: **stock `mmw_demoDDM`** in QSPI. Untouched.
  No reflash performed today and none needed.
- AWR2944P jumpers: **J17 out, J18 out, J20 in** (binary `001` =
  QSPI flash boot mode). Same as before. No physical changes ever
  again for normal operation.
- DCA1000: 5V powered, FPGA v2.9, system_connected. Working.
- Repo state: `feature/phase-3-dca1000` branch, ready for new
  commits (this morning's changes).

## The mode architecture (final)

One cfg pushed to chip on Seeker startup: `radar/cfg/awr2944P_aa.cfg`.
This cfg includes `lvdsStreamCfg -1 0 1 0`. Chip emits:

- **TLV stream** on COM10 (UART, 3.125 Mbps) — existing on-chip DSP
  pipeline output, what `radar/radar_manager.py` consumes today.
- **Raw ADC stream** on UDP 192.168.33.30:4098 via LVDS+DCA — new path
  that `radar_dca/dca_pipeline.py` consumes.

Mode dispatch is host-side (no chip change between modes):

| Mode | Backend | Consumes | Behaviour |
|------|---------|----------|-----------|
| `stock` | `RadarManager` | TLV from COM10 | Identical to today's Seeker. Discards UDP. |
| `ag`    | `RadarManager` + filter | TLV from COM10 | Same TLV path, narrow-FoV + dynamic-only filter applied to RadarTargets host-side. |
| `aa`    | `DCAManager` + `DCAPipeline` | Raw ADC over UDP | Range FFT → PMM detector → RadarTargets. PMM hits become `class=DRONE`. |

`backend_factory.name_of(manager)` already infers the mode from the
profile_name. The GUI mode picker switches the Manager class without
restarting the chip (no `sensorStop`+power-cycle needed because
the chip is always in the same state).

## What still needs work (in priority order)

### 1. PMM tuning against real data — TODAY's bench test
Replay output on the bench capture (no drone in scene):

```
frames processed:  25
frames with hits:  25 (100%) — way too many
total hits:        4336
best hit:          range=0.00 m, blade=52.7 Hz, SNR=45.7 dB
```

Every range bin fires, blade-freq stuck at 52.7 Hz (band-edge
artifact). The PMM detector is being fooled by:

- **TX→RX leakage at range bin 0** dominating the spectrum on every range row.
- **No clutter suppression** — static returns leak prop-like sidebands.
- **Noise floor estimate too aggressive** at low SNR ranges.

Fixes to apply in `radar_dca/pmm_detector.py` and `replay.py`:
- Mask range bins 0..2 (TX leakage / near-field).
- Subtract slow-time mean per range bin (clutter cancellation).
- Raise threshold from 6 dB → 12 dB once clutter is suppressed.
- Add a "min absolute power" gate so the detector doesn't fire on
  low-power range bins where any peak passes the SNR threshold.

### 2. Driveway test with the DJI FPV
Once PMM tuning is done:
- Set radar pointing at open driveway / sky cut.
- Fly FPV at 30 m → 50 m → 100 m → 150 m hover.
- Capture 5-10 seconds at each range.
- Replay through PMM, check for blade-freq hits at the actual drone
  range.

The chirp profile in `awr2944P_aa.cfg` still uses TI's stock
parameters (slope=30 MHz/μs, ramp=20.81μs → 24 cm range bins, 92 m
max range). For longer range, dial slope and ramp at the .cfg
level — see `docs/PHASE_3_AA_PROFILES.md` for candidate profiles.

### 3. Wire DCAPipeline to consume real bytes
The pipeline's M4.1 stub (raw-ADC parser) is now demonstrably
solvable: `radar_dca/bin_parser.py` already does the byte→complex
reshape correctly (validated against real capture). Lift that
parser into `dca_pipeline._process_frame_stub` so it works in real
time, not just offline replay.

### 4. GUI 3-mode toggle
Wire the existing `radar.backend` config knob to a GUI dropdown.
The `BackendFactory` is ready; only the GUI control needs
hooking up.

## Files that landed today

New:
- `radar/cfg/awr2944P_stock_plus_lvds.cfg` — diagnostic baseline
  (stock + corrected lvdsStreamCfg). Kept for reference; identical
  in chirp behaviour to `awr2944P_aa.cfg` minus the FoV narrowing.
- `studio_capture/stock_chirp.mmwave.json` — JSON dims for
  `bin_parser.dims_from_mmwave_json` matching our AA chirp profile.
- `studio_capture/{hello_test, probe_full_api, fpv_capture_v2}.lua`
  — Studio integration scripts. NOT used in the final architecture
  but kept in tree for future reference if we ever need Studio for
  high-numLoops integration that mmw_demoDDM can't do.

Updated:
- `radar/cfg/awr2944P_aa.cfg` — corrected `lvdsStreamCfg -1 0 1 0`
  (was `-1 1 1 0`); reverted numLoops/framePeriod/compression to
  stock values. Now identical to stock cfg + lvdsStreamCfg + narrow
  aoaFovCfg. Comments document the architecture.
- `scripts/run_studio_capture.py` — fixed Run Script path syntax
  (no more `dofile()` wrapping).

## Things NOT to redo

- Don't reflash the AWR. The chip is fine. Stock firmware does
  everything we need.
- Don't change SOP jumpers. J17 out / J18 out / J20 in is the
  permanent operating state.
- Don't bother with mmWave Studio. It needs Matlab Runtime, jumper
  changes, and adds nothing the existing firmware can't do.
- Don't add `enableHeader=1` to `lvdsStreamCfg`. Only `0` works.
- Don't push numLoops > 128 in the .cfg. The chip's HWA Doppler-FFT
  buffer can't fit it. (Long integration is done host-side on raw
  ADC, not on-chip.)
