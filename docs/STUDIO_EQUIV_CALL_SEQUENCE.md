# Studio-Equivalent Call Sequence (mmw_ddm reference)

Source: TI mmw_ddm demo for AWR2944P, MCU+SDK 10.02.00.04, mmwave_mcuplus_sdk
04.07.02.01, mmwave_dfp 02.04.18.01. System-level call graph for the four
paths we will reuse or strip in our custom app.

---

## 1. BSS firmware load at boot

mmw_ddm itself does NOT call `rlDeviceFileDownload`. On AWR2944P the BSS R4
firmware is delivered as one core inside the multicore appimage (alongside
MSS R5, DSS C66, M4SS) and is loaded by the SBL. mmw_ddm's job is to attach
to an already-running BSS through the mmwavelink mailbox.

SBL sequence (canonical across QSPI/UART/CAN/ENET SBL variants):

1. `Bootloader_socConfigurePll`, `System_init`, `Drivers_open`.
2. `Bootloader_socLoadHsmRtFw` (HSM Runtime FW must be in place before any
   core load).
3. `Bootloader_open` -> `Bootloader_parseMultiCoreAppImage` to discover
   which cores the appimage carries.
4. For each present core (RSS_R4, C66SS0, M4SS0_1, R5FSS0_0): set
   `cpuInfo.clkHz` then `Bootloader_loadCpu` (or `Bootloader_loadSelfCpu`
   for R5FSS0_0). Order: BSS, DSP, M4, self.
5. `Bootloader_runCpu(RSS_R4)` then
   `Bootloader_socConfigurePllPostApllSwitch`. This kicks BSS reset release.
6. `SOC_rcmWaitBSSBootComplete()` is the async gate. SBL must block here
   before unhalting R5FSS0_0; otherwise app-side `rlDevicePowerOn` sees a
   half-booted BSS.
7. `Bootloader_rprcImageLoad` then `Bootloader_runSelfCpu` for R5FSS0_0.

App-side attach in mmw_ddm MSS:

1. `MMWave_init` registers async event callback and CRC config.
2. `MMWave_sync` synchronizes MSS-side mmwavelink with the BSS.
3. Inside `MMWave_initLink` -> `MMWave_initMMWaveLink` -> `rlDevicePowerOn`,
   the driver populates `rlClientCbs_t` (mailbox open/read/write, OSAL
   mutex/sem, CRC, async event) and powers the link. The demo then polls
   `RSS_CR4_BOOT_INFO_REG0` bit 18 to confirm BSS idle loop reached.
4. `MMWave_open` issues channel cfg, ADC out cfg, low-power cfg, and
   triggers BSS RF init calibration. Async event `RL_RF_AE_INITCALIBSTATUS_SB`
   is the gate that completes `MMWave_open`.

| Action | File | Function | Line |
|---|---|---|---|
| SBL: load BSS+DSP+M4+R5 RPRCs from multicore image | C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04\examples\drivers\boot\sbl_qspi\awr2x44p-evm\r5fss0-0_nortos\main.c | main / Bootloader_loadCpu | ~140-160 |
| SBL: kick BSS R4 reset release | C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04\examples\drivers\boot\sbl_uart\awr2x44p-evm\r5fss0-0_nortos\main.c | Bootloader_runCpu(RSS_R4) | 184 |
| SBL: gate on BSS boot complete | C:\ti\mcu_plus_sdk_awr2x44p_10_02_00_04\examples\drivers\boot\sbl_uart\awr2x44p-evm\r5fss0-0_nortos\main.c | SOC_rcmWaitBSSBootComplete | 203 |
| App: install client callbacks and power on link | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\control\mmwave\src\mmwave_link_mailbox.c | MMWave_initMMWaveLink / rlDevicePowerOn | 399 / 474 |
| App: poll BSS idle bit 18 of CR4_BOOT_INFO_REG0 | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\control\mmwave\src\mmwave_link_mailbox.c | (inline post-power-on check) | 487 |
| App: top-level init / sync | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MMWave_init / MMWave_sync | 3398 / 3413 |

`rlDeviceFileDownload` lives in `C:\ti\mmwave_dfp_02_04_18_01\ti\control\mmwavelink\src\rl_device.c`. It is the path for older standalone-radar parts where BSS firmware is pushed over SPI from a separate host MCU. AWR2944P is fused-cascade MCU+RF, so we do not need it.

---

## 2. mmwavelink command router

mmw_ddm's CLI is a text parser that accumulates state into a single global
`gCLIMMWaveControlCfg` then calls a small set of mmwavelink wrappers. Text
dispatch is `CLI_MMWaveExtensionHandler`: strcmp `argv[0]` against
`gCLIMMWaveExtensionTable[]`, invoke matching `CLI_MMWave*` handler. Each
handler does `atof/atoi` and packs an `rl*Cfg_t` struct. None of the
handlers talk to the BSS directly; they only build config.

The dispatch surface (functions that actually push state to BSS):

- Per-profile / per-chirp build-up: `MMWave_addProfile`, `MMWave_addChirp`,
  `MMWave_addPhaseShiftChirp`, `MMWave_addAdvChirp`. Called from inside CLI
  handlers as profiles and chirps are parsed.
- Whole-config push: `MMWave_config(handle, ctrlCfg, ...)`, called once per
  sensor start from `MmwDemo_configSensor`. Walks the `MMWave_CtrlCfg` union
  (frame or advFrame), issues `rlSetFrameCfg` / `rlSetAdvFrameConfig` plus
  profile and chirp commands.
- Lifecycle: `MMWave_start(handle, calibrationCfg, ...)` and `MMWave_stop`.
- Direct mmwavelink calls that bypass the MMWave layer:
  `rlRfCalibDataStore`, `rlRfPhShiftCalibDataStore`, `rlRfRxIfSatMonConfig`,
  `rlRfRxSigImgMonConfig`, `rlRfAnaMonConfig`. Headers in
  `mmwave_dfp/ti/control/mmwavelink/include/rl_*.h`.

Replacement: keep `MMWave_init`, `MMWave_sync`, `MMWave_open` (with hardcoded
`MMWave_OpenCfg`). Replace `cli.c` + `cli_mmwave.c` + `mmw_cli.c` with a raw
UART receiver that deserializes our binary frames into `rlProfileCfg_t` /
`rlChirpCfg_t` / `rlFrameCfg_t` / `rlAdvFrameCfg_t` and drives the same
`MMWave_addProfile` / `MMWave_addChirp` / `MMWave_config` / `MMWave_start`.

| Action | File | Function | Line |
|---|---|---|---|
| Tokenize text command, dispatch to handler | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\utils\cli\src\cli_mmwave.c | CLI_MMWaveExtensionHandler | 2047 |
| Build profile cfg, push to MMWave | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\utils\cli\src\cli_mmwave.c | CLI_MMWaveProfileCfg / MMWave_addProfile | 688 / 774 |
| Build chirp cfg, attach to profile | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\utils\cli\src\cli_mmwave.c | CLI_MMWaveChirpCfg / MMWave_addChirp | 803 / 860 |
| Build frame cfg into ctrlCfg | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\utils\cli\src\cli_mmwave.c | CLI_MMWaveFrameCfg / CLI_MMWaveAdvFrameCfg | 1258 / 1334 |
| Demo glue: run sensor open+config+start | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mmw_cli.c | MmwDemo_CLISensorStart | 151 |
| Push accumulated ctrlCfg to BSS | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MmwDemo_configSensor / MMWave_config | 2898 / 2908 |
| Trigger frames | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MmwDemo_startSensor / MMWave_start | 2939 / 2971 |

---

## 3. LVDS data path setup

Three stages: one-time SoC HSI clock mux, one-time CBUFF init that owns the
LVDS PHY and EDMA wiring, and per-subframe HW session config.

Stage 1 (board init, before `MMWave_init`): mmw_ddm writes `0x333` directly
to `HSI_CLK_SRC_SEL` in MSS top RCM. Field value 3 selects
`DPLL_DSS_HSDIV0_CLKOUT2` per HSI lane. The DSS PLL and its hsdivider are
set up earlier by the SBL via syscfg-generated `Bootloader_socConfigurePll`;
copy that block verbatim. TODO confirm exact divider values match our DCA1000
target rate.

Stage 2 (`MmwDemo_LVDSStreamInit`, once from `MmwDemo_dataPathOpen` after
`MMWave_open`): `CBUFF_open(&initCfg)` with LVDS interface, 16-bit format,
two lanes enabled, DDR mode, MSB first, max 2 sessions (one HW for ADC raw,
one SW for user header). Then `HSIHeader_init`, then EDMA channel/param
alloc via `EDMA_allocDmaChannel` / `EDMA_allocParam`.

Stage 3 (`MmwDemo_LVDSStreamHwConfig`, per subframe before sensor start):
builds `CBUFF_SessionCfg` from parsed `lvdsStreamCfg` dataFmt, opens HW
session, arms it. CBUFF then streams ADC straight from chirp buffer to
LVDS on every frame trigger without MSS involvement.

NOTE on lane count: mmw_ddm sets `lvdsLaneEnable = 0x3` which is 2 lanes.
AWR2944P has 2 physical LVDS lanes. The DCA1000 "4-lane DDR" wording refers
to older xWR1xxx parts; do not assume 4 lanes here. TODO confirm target
rate against `docs/EDGE_OPTIMIZATION_GAP.md` and DCA1000EVM user guide.

| Action | File | Function | Line |
|---|---|---|---|
| Mux HSI clock source to DSS PLL | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MmwDemo_BoardInit | 3544 |
| Open CBUFF for LVDS, 2 lanes, DDR, 16-bit | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mmw_lvds_stream.c | MmwDemo_LVDSStreamInit / CBUFF_open | 211 / 239 |
| Init HSI header module | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mmw_lvds_stream.c | HSIHeader_init | 254 |
| Allocate CBUFF EDMA channels | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mmw_lvds_stream.c | MmwDemo_LVDSStream_EDMAInit / allocateEDMAChannel | 138 / 81 |
| Per-subframe HW session open and arm | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mmw_lvds_stream.c | MmwDemo_LVDSStreamHwConfig | 719 |
| Wire LVDS to datapath | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MmwDemo_dataPathOpen + lvdsStreamCfg branch | 1494 / 2014 |

CBUFF init values from mmw_ddm (copy these verbatim for our app):

```
initCfg.enableECC               = 0
initCfg.crcEnable               = 1
initCfg.maxSessions             = 2
initCfg.enableDebugMode         = false
initCfg.interface               = CBUFF_Interface_LVDS
initCfg.outputDataFmt           = CBUFF_OutputDataFmt_16bit
initCfg.lvdsCfg.crcEnable       = 0
initCfg.lvdsCfg.msbFirst        = 1
initCfg.lvdsCfg.lvdsLaneEnable  = 0x3   (lanes 0 and 1)
initCfg.lvdsCfg.ddrClockMode    = 1
initCfg.lvdsCfg.ddrClockModeMux = 1
HSI_CLK_SRC_SEL register write   = 0x333
```

---

## 4. DSP / TLV path to strip

On-chip detection runs in two places: the DSS application
(`dss/dss_main.c`) on C66x hosts the DPC, and the DPC modules under
`ti/datapath/dpc` implement range FFT, Doppler FFT, DDMA decode, CFAR.
MSS pulls detected-object lists out of DPM result buffers and packs them
as TLVs over the data UART (`MmwDemo_transmitProcessedOutput` ->
`UART_write`). We strip all of this; we want raw ADC over LVDS only.

DSS side (entire DSS C66 + DSS-CM4 build target, exclude from build):

| Module | Path (under C:\ti\mmwave_mcuplus_sdk_04_07_02_01) |
|---|---|
| DSS application (C66 entry point) | ti\demo\awr2x44P\mmw_ddm\dss\dss_main.c |
| DSS Cortex-M4 helper application | ti\demo\awr2x44P\mmw_ddm\dss_cm4\dss_cm4_main.c |
| Object-detection DPC (DDMA variant used here) | ti\datapath\dpc\objectdetection\objdethwaDDMA\src\objectdetection.c |
| Object-detection DPC (TDM variant, also in tree) | ti\datapath\dpc\objectdetection\objdethwa\src\objectdetection.c |
| Range FFT DPU | ti\datapath\dpu\rangeprocDDMA |
| Doppler FFT DPU | ti\datapath\dpu\dopplerprocDDMA |
| Range + CFAR DPU | ti\datapath\dpu\rangecfarprocDDMA |
| AoA DPU | ti\datapath\dpu\aoaproc |
| CFAR DPU | ti\datapath\dpu\cfarproc |

MSS side (do not call, do not link):

| Action | File | Function | Line |
|---|---|---|---|
| TLV packer | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MmwDemo_transmitProcessedOutput | 1123 |
| DPM init/sync (talks to DSS) | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | DPM_init / DPM_synch | 3465 / 3479 |
| DPM result report fxn (TLV trigger) | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MmwDemo_DPC_ObjectDetection_reportFxn | 2230 |
| DPC pre-start ioctl plumbing | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | MmwDemo_dataPathConfig | 1723 |
| UART data-export task | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\mss\mss_main.c | mmwDemo_mssUartDataExportTask | (called at 3504) |
| TLV type definitions | C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\include\mmw_output.h | (whole file) | n/a |

Keep on MSS:

- `MmwDemo_BoardInit` (HSI clock mux).
- `MMWave_init` / `MMWave_sync` / `MMWave_open` / `MMWave_config` /
  `MMWave_start` / `MMWave_stop`.
- `MmwDemo_LVDSStreamInit` plus CBUFF and EDMA wiring in
  `mmw_lvds_stream.c`. HW session path only; cut the SW session (it carries
  the user header that the TLV decoder expects).
- New raw-UART receiver replacing `cli.c` + `cli_mmwave.c` + `mmw_cli.c`,
  driving the same `MMWave_*` calls from section 2.

DSS, DPM, all DPCs, all DPUs, `MmwDemo_transmitProcessedOutput`,
`mmwDemo_mssUartDataExportTask`, `mmw_output.h` are all out. Do not call
`DPM_init`. Without DPM, the DSS image is never expected to be present and
the multicore appimage does not need a DSS RPRC.
