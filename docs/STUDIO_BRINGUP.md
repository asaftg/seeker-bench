# Studio bring-up — radar-only firmware path on AWR2944PEVM

This is the migration plan + runbook for replacing the SDK
`mmw_demoDDM` firmware with TI mmWave Studio's radar-only firmware path
on the AWR2944P (variant `xWR2x4xP`). Goal: continuous LVDS streaming
without the demo firmware's "halts after 5-30 s" behaviour that TI has
confirmed is by design.

Date: 2026-05-01.
Reference: SPRUJ22C (AWR2944PEVM user guide), SPRUIM4 (mmWavelink user
guide), `C:/ti/mmwave_dfp_02_04_18_01/ti/control/mmwavelink/include/`.

---

## 1. SOP strap procedure (functional / SOP4 — radar-only dev mode)

The AWR2944PEVM has three SOP (Sense-On-Power) jumpers — `SOP0`, `SOP1`,
`SOP2` — on header `J3` (silkscreen `SOP_CONFIG`) on the underside of
the board near the on-board XDS110 emulator. The mode is the binary
value SOP[2:1:0] sampled on `nRESET` rising edge.

Per SPRUJ22C §3.4 ("SOP Configuration"):

| Mode    | SOP2 | SOP1 | SOP0 | Purpose                                               |
|---------|------|------|------|-------------------------------------------------------|
| SOP 0   |  0   |  0   |  0   | Functional (boot from QSPI flash)                     |
| SOP 2   |  0   |  1   |  0   | Flashing (mmWaveStudio writes QSPI)                   |
| **SOP 4** | **1** | **0** | **0** | **Functional dev / mmWavelink download** (this is what we want) |
| SOP 7   |  1   |  1   |  1   | Reserved / debug                                      |

For Studio mode (radar-only firmware downloaded over UART by host):

```
SOP2 = 1   (jumper INSTALLED)
SOP1 = 0   (jumper REMOVED)
SOP0 = 0   (jumper REMOVED)
```

After reseating jumpers, **press the `nRESET` button** (S2 on the EVM,
top-right corner) so the device re-samples SOP. Power-cycling also
works.

### Verification

With the EVM in SOP4, `radar_dca/studio_bringup.py --sanity --load-only`
exits cleanly (no firmware download, just confirms the host side is
correct). The actual firmware download path can be verified by watching
the XDS110 UART (typically `COM4` at 115200 8N1) — you should see the
boot banner from the radar bootloader waiting for a download.

### Going back to demoDDM (old behaviour, if needed for a regression run)

1. Re-flash `mmw_demoDDM_xwr2x4xp.bin` via UniFlash (SOP2 mode).
2. Reset SOPs to all zeros (SOP0 mode = boot from flash).
3. Power-cycle.

The Phase-3 unified cfg + DCA UDP path still works against demoDDM —
that's our fallback if Studio mode reveals a blocker.

---

## 2. What changes for Seeker

### What stays (unchanged)

| File | Why it survives |
|------|----------------|
| `radar_dca/dca_control.py` | DCA1000 FPGA control is over Ethernet, unrelated to chip firmware. |
| `radar_dca/data_port.py` | UDP listener — same wire format from DCA → host. |
| `radar_dca/dca_pipeline.py` | Range-FFT / AoA / PMM all run host-side off raw ADC. |
| `radar_dca/bin_parser.py` | DCA packet → IQ assembly is unchanged. |

The chip emits **the exact same waveform** in Studio mode as in demoDDM
mode (same profileCfg / chirpCfg / frameCfg parameters), so the
host-side raw-ADC processing is bit-identical.

### What changes

| File | Change |
|------|--------|
| `radar/cfg_sender.py` | **Bypassed in Studio mode.** No CLI port to push commands to. |
| `radar/radar_manager.py` | **Bypassed in Studio mode.** No TLV stream from MSS. The `RadarManager` instance becomes a stub: `start()` is a no-op; the TLV consumer thread doesn't exist. |
| `radar/composite_manager.py` | New `studio` mode toggle. When on: skips `RadarManager.start()`, replaces it with a single `studio_bringup.bringup()` call. |

### What's new

| File | Role |
|------|------|
| `radar_dca/studio_bringup.py` | The mmWavelink ctypes wrapper. Pushes BSS+MSS firmware, configs the device, calls `rlSensorStart`. |

---

## 3. Composite manager integration

Studio mode is a third axis on `CompositeRadarBackend`, orthogonal to
the existing `stock` / `ag` / `aa` filter modes. The mode determines
**how the chip is brought up**, not what the host displays.

```
   radar_firmware = "demoDDM"  (current default)
       chip emits TLV (UART) + raw ADC (LVDS)
       composite_manager starts: RadarManager (TLV) + DCA pipeline
       set_mode() switches host-side filters

   radar_firmware = "studio"   (new)
       chip emits raw ADC only (LVDS) — no TLV path
       composite_manager starts: studio_bringup.bringup() + DCA pipeline
       set_mode() still switches filters, but stock/ag operate on the
       host-side raw-ADC pipeline output (which the existing dca_pipeline
       can produce — no new processing needed).
```

### Edits to `composite_manager.py` (sketch)

```python
class CompositeRadarBackend:
    def __init__(self, ..., radar_firmware: str = "demoDDM"):
        self._fw_mode = radar_firmware  # "demoDDM" | "studio"
        ...

    def start(self) -> None:
        if self._fw_mode == "studio":
            from radar_dca.studio_bringup import StudioDLL, bringup
            self._studio_dll = StudioDLL()
            bringup(self._studio_dll, mss_bin=..., bss_bin=...)
            log.info("Composite: chip booted via Studio mmWavelink path")
            # NOTE: no self._radar.start() — there is no TLV stream.
        else:
            self._radar.start()
        # DCA path is identical either way.
        if self._dca_control is not None and ...
            self._dca_control.reset_fpga()
            ...

    def stop(self) -> None:
        # Reverse: stop DCA first, then sensor.
        if self._dca_pipeline is not None: self._dca_pipeline.stop()
        ...
        if self._fw_mode == "studio":
            from radar_dca.studio_bringup import stop as studio_stop
            studio_stop(self._studio_dll)
        else:
            self._radar.stop()
```

In Studio mode the GUI still calls `set_mode("ag")` etc; those mode
switches need a host-side TLV-equivalent on top of `dca_pipeline` output.
That's a follow-up — for now `stock` and `ag` modes degrade to "raw-ADC
detections passed through the same FoV / speed-min filter" by routing
the pipeline output through the existing `RadarManager.set_tuning`
filter logic in standalone form.

---

## 4. Configuration parameter mapping

`studio_bringup.py` derives its profile/chirp/frame values from
`awr2944P_unified.cfg` to keep the waveform bit-identical:

| Cfg line                                              | mmWavelink struct + field                                       |
|-------------------------------------------------------|-----------------------------------------------------------------|
| `channelCfg 15 15 0`                                  | `rlChanCfg_t` `rxChannelEn=15`, `txChannelEn=15`, `cascading=0` |
| `adcCfg 2 0`                                          | `rlAdcOutCfg_t.fmt` `b2AdcBits=2`, `b2AdcOutFmt=1`              |
| `lowPower 0 0`                                        | `rlLowPowerModeCfg_t.lpAdcMode=0`                               |
| `profileCfg 0 77 12 7 20.81 0 0 8.883 0 384 30000 0 0 164` | `rlProfileCfg_t` (see `PROFILE_PARAMS` in studio_bringup.py)    |
| `chirpCfg 0 5 0 0 0 0 0 15`                           | `rlChirpCfg_t` (see `CHIRP_PARAMS`)                             |
| `frameCfg 0 5 128 65535 384 50 1 0`                   | `rlFrameCfg_t` (`framePeriodicity = 50ms / 5ns = 10_000_000`)   |
| `lvdsStreamCfg -1 0 1 0`                              | `rlDeviceSetHsiConfig` + `rlDeviceSetHsiClk` (LVDS 4-lane DDR 600 Mbps) |

Lines in `awr2944P_unified.cfg` that exist only because the SDK demo
parses them (`guiMonitor`, `cfarCfg`, `compressionCfg`, `intfMitigCfg`,
`localMaxCfg`, `ddmPhaseShiftAntOrder`, `antGeometryCfg`,
`antennaCalibParams`, `analogMonitor`, `calibData`, `aoaFovCfg`,
`measureRangeBiasAndRxChanPhase`) are **dropped in Studio mode** —
those configure the SDK's MSS DSP processing chain, which we don't
have. The host now does all that work in `dca_pipeline.py`.

The `idleTime 7→12 µs` comment in the cfg refers to MMWSDK-2560 — that
LVDS-vs-Ethernet rate workaround is **still required** in Studio mode
because it's a chip-level constraint, not an SDK one. Hence
`PROFILE_PARAMS["idleTimeConst"] = 1200`.

---

## 5. Validation sequence (when the EVM is hooked up)

```bash
# 1. Sanity — no chip needed.
py -3.11-32 radar_dca/studio_bringup.py --sanity

# 2. Load DLL — no chip needed, no firmware download.
py -3.11-32 radar_dca/studio_bringup.py --load-only -v

# 3. Bring up chip + start streaming. Watch the DCA UDP port (4098)
#    with data_port.py for raw frames.
py -3.11-32 radar_dca/studio_bringup.py -v

# 4. Stop.
py -3.11-32 radar_dca/studio_bringup.py --stop
```

Note the **`-3.11-32`** flag — `RadarLinkDLL.dll` is 32-bit (machine
type `0x014C` / IMAGE_FILE_MACHINE_I386), so we need a 32-bit Python
interpreter to load it. See the Risks section below.
