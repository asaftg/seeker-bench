# mmw_demoDDM_patched

Forked from `C:/ti/mmwave_mcuplus_sdk_04_07_02_01/ti/demo/awr2x44P/mmw_ddm/`
on 2026-05-05 to apply five surgical patches that fix the LVDS streaming
halt while keeping the on-chip CFAR / clustering / TLV emitter intact.

## Why this fork exists

The previous direction (`firmware/mmw_studio_cli/`) stripped the on-chip
DSP entirely so the host could do all signal processing in Python. That
broke human/vehicle detection — there was no on-chip CFAR producing the
TLV point cloud the bench's `RadarManager` is built around. Two weeks of
host-side work didn't recover it because the host pipeline doesn't unfold
the chip's DDMA modulation (the cfg sets `ddmPhaseShiftAntOrder 0 2 3 1`,
but `dca_pipeline.py` does a 4-RX AoA, not a 16-virtual-element AoA).

The hybrid architecture: keep the chip's on-chip CFAR + TLV (humans,
vehicles), and add a host-side PMM scan on raw ADC for drones via DCA1000.
mmw_demoDDM gives us the first half for free; we just need to fix its
sustained-streaming halt.

## Patches applied vs the SDK source

| # | File | Change |
|---|---|---|
| 1 | `mss/mss_main.c:2606` | `disableFrameStopAsyncEvent = false` → `true`. Suppresses BSS-emitted FRAME_END so a transient timing-monitor trip doesn't tear down the data path. |
| 2 | `mss/mss_main.c:2147-2156` | `RL_RF_AE_FRAME_END_SB` handler: removed `MmwDemo_dataPathStop()` call. Belt-and-suspenders for patch 1 — if the suppress somehow misses, the handler logs and continues instead of shutting LVDS down. |
| 3 | `mss/mss_main.c:2264` | Removed `MmwDemo_LVDSStream_PeriodicCycle()` call site. The prior agent's CBUFF re-cycle empirical sweep (v3=85s burst, v4=30s, v5=inconclusive, v6=disabled) proved cycling REGRESSED the halt window — chip-side CBUFF wraparound is NOT the failure mode. |
| 4 | `mss/mss_main.c:3294` | Inserted PAD_BYPASS write after `Board_driversOpen()` — disables external nRESET pad so the FT4232H's pin-drift glitches can't reset the chip. Verified working in mmw_studio_cli over multi-hour uptime. |
| 5 | `mss/mmw_lvds_stream.c:618` | Deleted per-frame `[SEEKER] HW frame %u` debug printf. It floods the CLI UART that `RadarManager` polls for TLV; under sustained streaming the printf can stall the CLI ring buffer and look like a chip halt from the host side. |

The prior agent's calibration-disable patch (`mss_main.c:2956-2964`,
`enablePeriodicity = false`) is retained as-is — it was correct.

## Build

```
build.bat mmwDemoDDM
```

Requires `ti-cgt-armllvm_4.0.2.LTS` (R5F + M4 builds). The C66 DSS build
is currently skipped because that compiler isn't installed on this
bench — the appimage is repacked using the SDK's pre-built DSS RPRC,
which is fine because the patches don't touch `dss/`. If you need to
rebuild DSS, install `ti-cgt-c6000_8.3.13` and rerun `build.bat`.

After build, repack the appimage manually if only MSS+M4 were rebuilt:

```bash
node $MCU_PLUS_INSTALL_PATH/tools/boot/out2rprc/elf2rprc.js \
    awr2x44P_mmw_demo_mssDDM.xer5f
node $MCU_PLUS_INSTALL_PATH/tools/boot/out2rprc/elf2rprc.js \
    awr2x44P_mmw_demo_dss_cm4DDM.xem4
node $MCU_PLUS_INSTALL_PATH/tools/boot/multicoreImageGen/multicoreImageGen.js \
    --devID 55 --out awr2x44P_mmw_demoDDM.appimage \
    awr2x44P_mmw_demo_mssDDM.rprc@0 \
    awr2x44P_mmw_demo_dss_cm4DDM.rprc@2 \
    awr2x44P_mmw_demo_dssDDM.rprc@1 \
    $MMWAVE_AWR294X_DFP_INSTALL_PATH/firmware/radarss/xwr2x4xp_radarss_metarprc.bin@3
```

## Flash

```
python ../mmw_studio_cli/one_shot_flash.py \
    --appimage firmware/mmw_demoDDM_patched/awr2x44P_mmw_demoDDM.appimage \
    --cfg firmware/mmw_demoDDM_patched/flash.cfg
```

Operator action required: disconnect the DCA1000 60-pin ribbon from the
AWR HD-60 connector before flashing. The DCA's FT4232H drives SOP[2:0]
through the ribbon and overrides the AWR-side electronic SOP control.

## Post-flash verify on COM11 @ 115200

Expect to see:
- `[BOOT] PAD_BYPASS disable: WARM_RESET_CONFIG ... -> ...`  (proves patch 4)
- mmw_demoDDM banner + `Init Calibration Status` line
- After Seeker pushes the unified cfg: continuous streaming with no
  `BSSEV !! FRAME_END` lines (proves patches 1+2)
- No `[SEEKER] HW frame N` lines (proves patch 5)
- No `[SEEKER] CBUFF cycle at frame N` lines (proves patch 3)

30-minute soak: `last_packet_age_s` in `aa_diagnostics` never crosses
2 seconds; TLV `frame_id` increments monotonically at ~20 Hz.
