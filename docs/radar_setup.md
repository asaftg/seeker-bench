# AWR2944P radar setup log

Ticket 5a. First hardware bring-up of the TI AWR2944P EVM on the Seeker-01
bench. Stock `mmw_demoDDM` firmware over UART; DCA1000 not yet connected.

Target outcome: live point cloud + on-chip Group Tracker target boxes on the
radar panel of the Seeker GUI.

## Status

- [x] Phase A1 — workspace + docs skeleton
- [x] Phase A2 — FTDI/TI USB-UART driver installed, 4 COM ports recognized (COM6–COM9)
- [x] Phase A3 — MMWAVE-MCUPLUS-SDK installed, firmware + .cfg located (tracker TLVs N/A — not compiled in)
- [x] Phase A4 — UniFlash installed
- [x] Phase B — mmw_demoDDM firmware flashed via XDS110 UART (J8), EVM boots in functional mode
- [x] Phase C — CLI (COM11) + data (COM10 @ 3,125,000 baud) ports identified, stock .cfg pushed, magic word verified
- [x] Phase D1 — common/frames.py extended (RadarDetection, RadarTarget)
- [x] Phase D2 — radar/tlv_parser.py written
- [x] Phase D3 — radar/radar_manager.py + cfg_sender.py written
- [x] Phase D4 — radar/clustering.py (DBSCAN fallback) written
- [x] Phase E1 — gui/sensor_bridge.py::radar_to_wire() serializes real frames
- [x] Phase E2 — gui/static/js/radar_view.js renders points + target boxes
- [x] Phase E3 — RadarManager wired into main.py
- [x] Phase F — end-to-end bench test: 30–36 pts/frame streaming, JSON wire layer valid (vehicle tests pending once radar is out of the garage)
- [x] Phase G — docs finalized, config filled, commit + push

## Driver

- Version: **2.12.36.20** (10/28/2024)
- Installer: `CDM2123620_Setup.exe` (from `CDM2123620_Setup.zip`, Asaf manually downloaded because ftdichip.com is Cloudflare-gated)
- Install method: manual click-through — `/S`, `/silent`, `/quiet` all ignored by this FTDI installer; GUI appeared each attempt and was dismissed by user
- Driver store INFs: `ftdibus.inf` (USB class), `ftdiport.inf` (Ports class)
- 4 COM port numbers bound after INF force-install: **COM6, COM7, COM8, COM9**
- Additional INFs used: `C:\ti\mmwave_mcuplus_sdk_04_07_02_01\tools\ftdi\ftdibus.inf` + `ftdiport.inf` (TI-modified, DriverVer 08/16/2017 v2.12.28). Needed because AR-DevPack-EVM-012 enumerates as VID=0451 / PID=FD03 (TI's VID, not generic FTDI), so generic CDM 2.12.36.20 ignored it. Force-installed via `pnputil /add-driver <inf> /install` (pnputil accepts an older-version driver when the newer one doesn't match the hardware IDs).

## SDK (MMWAVE-MCUPLUS-SDK)

- Version: **4.7.2.1**
- Install root: `C:\ti\mmwave_mcuplus_sdk_04_07_02_01`
- Installer filename: `mmwave_mcuplus_sdk_4.7.2.1-windows-x86-install.exe`
- Install method / flag used: InstallBuilder `--mode unattended --prefix C:\ti`
- AWR2944P demoDDM appimage: `C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\awr2x44P_mmw_demoDDM.appimage`
  - NOTE: the AWR2944P-specific binary lives under `awr2x44P/` (capital P). `awr2944/` (no P) is the non-P variant; wrong flash would brick-for-this-ticket the EVM until re-flashed.
- SBL bootloader: `C:\ti\mmwave_mcuplus_sdk_04_07_02_01\tools\awr2x44P\sbl_qspi.release.tiimage`
- Stock DDM `.cfg` path: `C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\profiles\awr2944P\profile_3d_3Azim_1ElevTx_DDMA_awr2944P_highRange.cfg`
  - trackingCfg present? **NO** — profile has no `trackingCfg` line; `guiMonitor -1 2 1 0 0 0 1` emits only detected-points + side-info TLVs.
- Firmware Group Tracker TLVs: **NOT EMITTED** by this build of `mmw_demoDDM` — gtrack library ships as source under SDK but is not linked into the demo binary. Consequence: target-box rendering on the GUI runs off DBSCAN (client side) as the PRIMARY path for this ticket, not a fallback. Update parser scope to points + side info only.

## UniFlash

- Version: **9.5.0.5651**
- Install root: `C:\ti\uniflash_9.5.0`
- `dslite.bat` path: `C:\ti\uniflash_9.5.0\dslite.bat`

## Firmware

- Variant: `awr2x44P_mmw_demoDDM`
- SBL path: `C:\ti\mmwave_mcuplus_sdk_04_07_02_01\tools\awr2x44P\sbl_qspi.release.tiimage`
- App path: `C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\awr2x44P_mmw_demoDDM.appimage`
- SBL offset: `0x0`
- App offset: `0xA0000`
- Jumper map (from **SPRUJ22C §2.10.2 Table 2-12** — supersedes ticket body):
  - **Flash mode (SOP5, SOP[2:0]=101):** J17 **closed**, J18 **open**, J20 **closed**
  - **Functional mode (SOP4, SOP[2:0]=001):** J17 **open**, J18 **open**, J20 **closed**
  - (Convention: jumper closed = SOP bit 1, open = SOP bit 0)
  - Ticket body's original "functional = J18 closed, J20 open" was wrong — would land at SOP2 (010) = QSPI-flash-only dev mode, not functional boot.
- Flash transport: **XDS110 virtual COM on J8** (not the FTDI on J10 — see Quirks). MSS_UARTA/SBL-UART is only routed to the XDS110.
- Flash result: **OK** — post-flash `version` banner on COM11 reports RF F/W 02.05.04.00.23.05.16, ProcChain DDM, SDK 04.07.02.01.

## Ports

Two USB cables go into the EVM:

- **J10 (FTDI_USB, left)** → COM6–COM9 — four auxiliary FTDI UARTs. Unused by Seeker.
- **J8 (XDS_USB, right)** → COM10/11 — XDS110 virtual UARTs. **This is the pair we use.**

`MSS_UARTA` (CLI + SBL) is routed only to the XDS110, not the FTDI. The FTDI quad-UART is for application-level UARTs none of which are the demo's CLI.

- **CLI port**: `COM11` (XDS110 Class Application/User UART), **115200 baud**. Used to push the `.cfg` and to send `sensorStop` on shutdown.
- **Data port**: `COM10` (XDS110 Class Auxiliary Data Port), **3,125,000 baud**. The TLV stream comes out of here.
  - The firmware's `queryDemoStatus` reports `Data port baud rate: 892857`, which is a misleading nominal value — actual on-wire baud was empirically verified by scanning 115200 / 921600 / 1,250,000 / 2,000,000 / **3,125,000** / 3,000,000 and looking for magic-word alignment at offset 0. Only 3,125,000 aligned.
- Version banner sampled from COM11:
  ```
  Platform                : AWR2X44P
  RF F/W Version          : 02.05.04.00.23.05.16
  RF F/W Patch            : 02.06.03.01.25.05.22
  mmWaveLink Version      : 02.04.06.17
  ProcChain               : DDM
  mmWave SDK Version      : 04.07.02.01
  ```
- Profile pushed: `profile_3d_3Azim_1ElevTx_DDMA_awr2944P_highRange.cfg` (stock SDK, unmodified).
- Magic word (`02 01 04 03 06 05 08 07`) confirmed on data stream at offset 0 of the first 32 bytes: **Y**.

## Config profile

- Source: SDK stock — `C:\ti\mmwave_mcuplus_sdk_04_07_02_01\ti\demo\awr2x44P\mmw_ddm\profiles\awr2944P\profile_3d_3Azim_1ElevTx_DDMA_awr2944P_highRange.cfg`. Not edited, not copied into `config/radar_profiles/` — the stock profile works as-is on this firmware build.
- `trackingCfg` line: **absent** (firmware build has no gtrack; adding the line would be a no-op).
- `staticBoundaryBox` line: absent (same reason).
- `guiMonitor` line: `guiMonitor -1 2 1 0 0 0 1` — emits detected-points (TLV 1), range profile (TLV 2), stats (TLV 6), temperature (TLV 9). Notably does **not** emit SideInfo (TLV 7) despite `detectedObjects = 2`; see Quirks.

## End-to-end observations

Test was run with the EVM pointed at the closed garage — noise floor only, no vehicles or pedestrians in FOV.

- Capture + parse: **82 packets in ~4 s** (~20 Hz, matches 50 ms `frameCfg` periodicity), ~33 points/frame avg.
- `radar_to_wire()` payload: valid JSON (~4 KB/frame), 35 points, no NaN/Inf after the sentinel-coerce fix.
- Sample point: `(x=0.10m, y=1.32m, z=0.01m, v=0.0 m/s, snr=0 dB)` — stationary reflector at 1.3 m, consistent with garage wall.
- Point cloud visible on walk-in: **pending** (radar still in garage)
- Target boxes visible on vehicles: **pending**
- Target box source: **always DBSCAN (dashed stroke)** on this firmware build — no firmware tracker available.
- Frames per second on radar panel: ~20 Hz expected based on capture-thread timing.
- Full-system launch regression check: deferred until vehicle test.

## Quirks encountered

_Running log of anything that didn't go by the book._

- **FTDI CDM download** — `ftdichip.com/wp-content/uploads/...` returns HTTP 403
  to every automated client tested (PowerShell Invoke-WebRequest, curl.exe, with
  and without full browser-spoofing headers + referer). The 403 body is a ~75 KB
  Cloudflare interactive challenge page. No JS-free bypass found. Pivoted to
  manual download by Asaf. Filed as "one more file to grab alongside the TI SDK".
- **TI SDK & UniFlash** — `dr-download.ti.com` does NOT require login for these
  two (ticket body anticipated a login wall that turns out to be specific to
  other TI products). Automated download works with a plain browser UA.
  UniFlash current version: **9.5.0.5651** (ticket body referenced 8.8.0 which
  is 404). SDK current: **4.7.2.1**.
- **XDS110 vs FTDI USB routing** — the EVM has two USB connectors (J10 = FTDI,
  J8 = XDS110). The MSS_UARTA (which carries both the SBL flash-mode prompt and
  the mmw_demo functional-mode CLI) is routed **only** to the XDS110. Without
  the J8 cable plugged in, the EVM appears dead on all four FTDI COM ports
  (silent, no 'C' XMODEM prompt) — we lost half a session chasing this before
  realizing the flash path was wrong.
- **Power-before-USB ordering** — plugging USB without the barrel jack gives
  nothing: the MCU is unpowered, the XDS110 won't enumerate. Plug the 5 V
  barrel jack first, wait for board LEDs, then plug USB.
- **Data port nominal baud vs actual** — `queryDemoStatus` reports the data
  port baud as **892857**, but the stream on the wire is at **3,125,000 baud**.
  We brute-forced common mmwave bauds until the magic word landed at offset 0
  — only 3,125,000 matched. The 892857 figure appears to be a firmware-internal
  nominal/divisor value, not what reaches the XDS110 USB endpoint.
- **`sensorStart` state machine** — TI's `mmw_demo` CLI only accepts the bare
  `sensorStart` (no args) when the chip is in state `INIT` (fresh boot). After
  any `sensorStop`, the chip enters state `STOPPED` and the only legal restart
  is `sensorStart 0` (reuse previously-loaded config). Every other combo
  returns `Error: Invalid Sensor Start.` which was the root cause of "chip
  stopped streaming after cfg push" — our manager used to always push the .cfg
  (which ends in `sensorStart`), and after the first boot cycle all subsequent
  starts were rejected. `RadarManager._push_profile` now queries
  `queryDemoStatus` first and sends `sensorStart 0` from any non-INIT state.
  To push a **new** .cfg, power-cycle the EVM so the chip boots back to INIT.
- **No SideInfo TLV in `mmw_demoDDM` build** — despite `guiMonitor`'s
  `detectedObjects = 2`, this DDM build emits TLVs {1, 2, 6, 9} only; TLV 7
  (SideInfo / per-point SNR + noise) is absent. Our parser marks SNR as `NaN`
  when SideInfo is missing; `RadarManager._process_and_publish` skips SNR
  gating for NaN points (the chip already CFAR-gated them). `sensor_bridge`
  coerces NaN → 0.0 for wire serialization (strict JSON forbids NaN).
  If we later rebuild the demo from source with SideInfo enabled, the NaN
  sentinel becomes a real dB value and the `snr_min_db` gate takes effect
  automatically — no manager change needed.
- **No firmware Group Tracker on this build** — the `awr2x44P_mmw_demoDDM`
  binary ships with `gtrack` source in the SDK but it isn't linked into the
  demo. No TLV 308/309, no on-chip target boxes. All target boxes on the GUI
  are DBSCAN-clustered client-side (dashed stroke in `radar_view.js`).

## First-light helper scripts (under `scripts/`)

- `radar_hello.py` — UART banner probe across 4 COM ports.
- `radar_send_cfg.py` — manual .cfg push for first-light sanity.
- `radar_cli_probe.py` — send a single CLI command and print the response.
- `radar_data_scan.py` — scan a port across a set of bauds, look for magic word.
- `radar_tlv_probe.py` — dump raw TLV types, point counts, SNR range.
- `radar_verify_stream.py` — end-to-end: push cfg, verify sensorStart ack, confirm stream.
- `radar_bus_probe.py` — run RadarManager, report FrameBus publishes.
- `radar_wire_probe.py` — run RadarManager, verify `radar_to_wire()` JSON.

---

_Setup started: 2026-04-23_
_Setup completed: 2026-04-23_
