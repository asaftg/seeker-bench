# Studio integration — what was added, and how to revert

This file enumerates every change made to integrate **mmW Studio** as the
radar firmware backend in Seeker. It exists so we can cleanly **back out
of Studio** when Phase 2 (pure-Python mmwavelink client, Jetson-portable)
lands and Studio becomes dead weight. None of the changes touch the
existing demoDDM code path — that stays intact and can be selected at
any time via `--radar-firmware demoDDM`.

> **The Studio dependency is contained in three new files + one new flag
> + a small branch in `composite_manager.py`. Removing all of it is a
> ~30-line revert.**

---

## 1. Files added (delete these to ditch Studio)

| Path | Purpose | Notes |
|---|---|---|
| `radar_dca/studio_launcher.py` | Spawns mmW Studio as a child process; sets the registry key that points Studio at our bring-up Lua; waits for first DCA1000 UDP packet on 4098 (proves chip is streaming); kills Studio on shutdown. | Imports nothing from the rest of Seeker except `common.logging_setup`. Self-contained — delete this file and remove its import line in `composite_manager.py` to back out. |
| `scripts/seeker_studio_bringup.lua` | Studio Lua script that does the full chip bring-up (SOPControl → Connect → DownloadBSSFw → DownloadMSSFw → ProfileConfig → ChirpConfig → FrameConfig → DCA1000 EthInit/Mode/PacketDelay → StartFrame). Studio runs this at launch via the registry-key-pointed startup script. | Encodes the chirp profile from `radar/cfg/awr2944P_unified.cfg`. **No `StartRecord_*` call** — DCA1000 forwards UDP without writing to disk. |
| `scripts/start_continuous.lua` | Helper for manual debugging. Stops any running frame/record, then issues `StartFrame` → `StartRecord_ContinuousStreamData`. Not used by Seeker; just for operator convenience when iterating in Studio's Lua Shell. | Safe to delete. |
| `docs/STUDIO_RUNBOOK.md` | Operator runbook for the Phase-1 Studio path. | |
| `docs/STUDIO_INTEGRATION_DIFF.md` | This file. | |

## 2. Files modified

### `main.py`

Two additions:

```python
# (parse_args)
p.add_argument(
    "--radar-firmware",
    choices=["demoDDM", "studio"],
    default="studio",
    help="...",
)
```

```python
# (radar bring-up section, replaces the old `radar.start()`)
if args.radar_firmware == "studio":
    log.info("Studio firmware mode: skipping RadarManager.start() ...")
else:
    radar.start()
```

```python
# CompositeRadarBackend constructor call gets the firmware flag:
composite = CompositeRadarBackend(
    ...,
    radar_firmware=args.radar_firmware,
)
```

**To revert:** remove the `--radar-firmware` flag definition, change the
`if args.radar_firmware == "studio":` block back to a bare `radar.start()`,
and drop the `radar_firmware=args.radar_firmware` kwarg.

### `radar/composite_manager.py`

Three additions, all gated on `self._radar_firmware == "studio"`:

1. **Constants + import.** New constants `RADAR_FIRMWARE_DEMODDM`,
   `RADAR_FIRMWARE_STUDIO`. New import of `StudioLauncher,
   StudioLauncherError` from `radar_dca.studio_launcher`.

2. **Constructor accepts `radar_firmware: str = RADAR_FIRMWARE_DEMODDM`.**
   When `studio`, the back-refs that the `dca_pipeline` needs for its
   LVDS-stall watchdog (`_radar_manager_ref`, `_dca_control_ref`) are NOT
   set — that watchdog is for the demoDDM halt-recovery dance, which is
   irrelevant in Studio mode (no halts).

3. **`start()` method dispatches.** New `_start_studio()` does the
   Studio-specific lifecycle (launch + first-packet wait + dca pipeline);
   the original lifecycle is preserved verbatim in `_start_demoDDM()`.
   `stop()` similarly branches: in Studio mode it kills the Studio
   child process; in demoDDM mode it stops `_radar` + `_dca_control`.

4. **`set_mode()` keeps `_publish_enabled = True` in Studio mode** for
   all three modes (stock/ag/aa) since there's no TLV path. In demoDDM
   it preserves the existing "publish only in AA" behavior.

5. **`diagnostics()` adds `firmware: <str>`** and marks
   `tlv.active_in_studio_mode: false` so the GUI / diagnostics consumers
   know the TLV path is offline.

**To revert:** remove the imports, constants, the constructor parameter,
and merge `_start_studio` / `_start_demoDDM` back into a single
`start()`. The old `start()` is still byte-for-byte intact inside
`_start_demoDDM()` — copy-paste it back.

## 3. Persistent system state created (clean these on full revert)

| What | Where | Why | How to clean |
|---|---|---|---|
| Studio startup-script registry key | `HKCU\Software\Texas Instruments\mmWave Studio\3.1.4.4\Settings\Startup Script` (`Use=TRUE`, `Path=<bringup.lua>`) | Studio reads this at launch and runs our Lua. Set by `StudioLauncher._set_startup_script()`. | `reg delete "HKCU\Software\Texas Instruments\mmWave Studio\3.1.4.4\Settings\Startup Script" /v Use /f` and `... /v Path /f`. Or just set `Use=FALSE`. |
| MATLAB Compiler Runtime v8.5.1 (32-bit) install | `C:\Program Files (x86)\MATLAB\MATLAB Runtime\v851\` | Required by Studio. Installed via `MCR_R2015aSP1_win32_installer.exe`. | Standard MATLAB Runtime uninstaller in Programs & Features. |
| MCR path on system PATH | Machine-level `PATH` env var | Added by MCR installer. | Cleared automatically by MCR uninstaller. |

## 4. Hardware state (no software change to revert)

The chip is in **SOP2 mode** (J17 closed, J18 closed, J20 OPEN) for
Studio to download firmware. To go back to demoDDM you'd flip jumpers to
SOP0 (J20 closed only) so the chip boots from QSPI flash with whatever
firmware is in there. That's a physical step, not a software revert.

## 5. Topic-level wire-format compatibility

**No breaking change.** Studio-mode raw-ADC processing publishes on
`Topic.RADAR_AA` exactly like demoDDM-mode AA. The GUI's `sensor_bridge`
already synthesizes a "connected" radar wire payload from `RADAR_AA`
alone when TLV is offline (it has been doing this since Phase 3, before
Studio). So all three GUI modes (Stock / A/G / A/A) keep working. The
only visible change in Studio mode: `tlv.frame_id` is `None` and
`tlv.active_in_studio_mode = false`.

## 6. Files NOT touched (reassurance)

The following stay byte-identical to pre-Studio Seeker:

* `radar/radar_manager.py` — RadarManager class unchanged. In Studio
  mode the instance exists but `.start()` is never called, so it's a
  passive holder for set_tuning() / set_extrinsic() / az_bias_deg etc.
  delegated to it from `composite_manager`.
* `radar/cfg_sender.py` — only used by RadarManager.start, which we skip.
* `radar/cfg/awr2944P_unified.cfg` — unchanged (we read it for chirp
  dimensions, but don't push it to the chip in Studio mode — the
  Lua bringup encodes the same params directly).
* `radar_dca/dca_control.py` — unchanged. Skipped in Studio mode (Studio
  programs the DCA1000 itself via mmwavelink).
* `radar_dca/data_port.py` — unchanged. Reads UDP exactly the same.
* `radar_dca/dca_pipeline.py` — unchanged. PMM AoA + cluster filters are
  pre-Studio work and apply identically.
* `radar_dca/pmm_detector.py` — unchanged.
* `radar_dca/bin_parser.py` — unchanged.
* `gui/sensor_bridge.py`, `gui/app.py`, `gui/static/*` — unchanged.
* `fusion/`, `eo/`, `thermal/`, `gimbal/`, `common/` — unchanged.

## 7. Phase-2 plan: what replaces Studio

`radar_dca/studio_bringup.py` (already scaffolded — needs the rlClientCbs_t
transport callbacks fleshed out — ~150 LOC). Once it can boot the chip
without Studio:

1. Add `--radar-firmware studio-py` choice to the existing flag.
2. New `_start_studio_py()` branch in `composite_manager.py` that calls
   `studio_bringup.bringup(dll, mss_bin, bss_bin)` instead of launching
   the GUI.
3. Delete `radar_dca/studio_launcher.py` + `scripts/seeker_studio_bringup.lua`
   + `docs/STUDIO_RUNBOOK.md` + this doc.
4. Hardware: chip stays in SOP2 mode but the Python client downloads
   the firmware over the J10 FTDI UART; same chip, same DCA1000, same
   waveform.

Reference implementations: `OpenRadar`, `pyRadar`, `mmwave-capture-std`
all have working Python mmwavelink clients for AWR2243 — porting the
struct sizes for AWR2944P is the only chip-specific delta (struct
sizes already verified in the existing scaffold's `--sanity` mode).
