# Studio firmware runbook (Phase 1) — production bring-up via mmW Studio

**Status:** validated 2026-05-02 — 5+ GB / multi-minute continuous LVDS streaming with no halts. Replaces the broken `mmw_demoDDM` path which halts after 30-90 s by chip design.

This runbook covers Phase 1: Studio is the bring-up tool, Seeker drives the data plane. Phase 2 (Python mmwavelink client, no Studio dependency) is the next step toward the Jetson production target.

---

## One-time setup (already done on this bench)

1. **mmW Studio 3.1.4.4** installed at `C:\ti\mmwave_studio_03_01_04_04`.
2. **MATLAB Compiler Runtime 8.5.1 (32-bit)** installed via `MCR_R2015aSP1_win32_installer.exe` from MathWorks. Required by Studio's startup or every `ar1.*` call returns `Object reference not set`.
3. **AWR2944PEVM jumpers** in **SOP2 mode**: J17 closed, J18 closed, J20 OPEN.
4. **Hardware connections:**
   - AWR J8 (XDS110) → host USB
   - AWR J10 (FTDI / "AR-DevPack") → host USB ← Studio talks via this port
   - DCA1000EVM J6 (RJ45) → host NIC (192.168.33.30 / 255.255.255.0)
   - DCA1000EVM J3 (60-pin) → AWR HD-60 ← LVDS data + UART pass-through
   - DCA1000EVM J1 (USB) → **UNPLUGGED** ← causes "more than one device detected"
   - 12 V to AWR, 5 V to DCA1000EVM
5. **Bring-up Lua script** at `scripts/seeker_studio_bringup.lua` — encodes the chirp profile from `radar/cfg/awr2944P_unified.cfg`. Studio runs it automatically when launched (registry key set by `radar_dca/studio_launcher.py`).

## Per-run bring-up (every time you want to run Seeker)

1. **Power on** AWR (12 V) and DCA1000 (5 V).
2. **Press AWR nRESET (S2)** — guarantees SOP2 is sampled fresh.
3. **No need to launch Studio manually** — Seeker spawns it as a child process via `radar_dca/studio_launcher.py`.
4. From the Seeker root:
   ```
   python main.py --no-eo --radar-firmware studio
   ```
   - `--radar-firmware studio` is the default; you can omit it.
   - `--no-eo` is needed until the EO deserializer issue is fixed.
5. Watch the log — expect:
   ```
   Composite[studio]: launching mmW Studio for chip bring-up ...
   Studio launched (PID=...)
   Studio bring-up complete: first packet at <X> s
   Composite[studio]: data plane up (Topic.RADAR_AA)
   ```
   First packet should land within ~30 s of Studio launch.

## Validation

`/api/radar/aa_diagnostics` should show:
- `firmware: "studio"`
- `udp.packets_total > 0` and rising
- `udp.bytes_per_s > 1_000_000` (well over 1 MB/s)
- `aa.frames_assembled` rising at ~20 fps
- `tlv.active_in_studio_mode: false` (no TLV in this mode)

GUI radar widget should show drone detections at real angles (PMM AoA fix is already in `dca_pipeline.py`). Switch modes (STOCK / AG / AA) — all three publish via the host-side raw-ADC pipeline since chip emits no TLV.

## Recovery

| Symptom | Action |
|---|---|
| `Studio bring-up failed` log + no UDP | Power-cycle AWR EVM (12 V off → on). Restart Seeker. |
| Studio crashes with `System.AccessViolationException at rlsGetNumofDevices` | The chip is in a confused state from a prior session. Power-cycle AWR EVM. Restart Seeker. |
| Studio Output says `Matlab Runtime Engine is not installed` | MCR uninstalled. Re-run `MCR_R2015aSP1_win32_installer.exe`. |
| GUI radar empty even with chip streaming (UDP non-zero) | Switch radar mode to AA in GUI — confirms PMM pipeline is live. |
| `Cannot bind UDP 0.0.0.0:4098` | Another Seeker / Studio session is still consuming. Kill via `Get-Process \| Where-Object Name -match "python\|mmWaveStudio" \| Stop-Process -Force`. |

## What NOT to do

- Don't open mmW Studio yourself before launching Seeker — Seeker manages Studio's lifecycle. If you have it open from a debugging session, close it first.
- Don't touch the **ContStream** tab in Studio — that's RF characterization mode, NOT radar imaging mode. Bring-up uses the regular sensor framing path.
- Don't click **DCA1000 ARM + StartRecord** in the GUI — that writes to disk at ~37 MB/s. We deliberately skip the disk-record path; DCA forwards UDP regardless.

## Phase 2 (Jetson-portable, in progress as a TODO)

Studio is Windows 32-bit only. For Jetson production we replace it with `radar_dca/studio_bringup.py` — a pure-Python mmwavelink client that re-implements the same command sequence (`SOPControl → Connect → DownloadBSSFw → DownloadMSSFw → PowerOn → ChannelConfig → AdcOutConfig → ProfileConfig → ChirpConfig → FrameConfig → DCA1000 setup → StartFrame`). Reference implementations: `OpenRadar` and `pyRadar`. Scaffold is already present (struct sizes verified); remaining work is the rlClientCbs_t transport callbacks (UART read/write + async events). Estimated 2-3 days of focused work.

Once Phase 2 lands, `--radar-firmware studio` is replaced with `--radar-firmware studio-py` (no Studio GUI dependency, runs on Jetson Linux ARM).
