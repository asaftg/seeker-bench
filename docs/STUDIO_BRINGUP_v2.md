# mmWave Studio bring-up runbook — AWR2944P + DCA1000EVM

**Goal:** continuous LVDS raw-ADC streaming for hours, no halts.
**Pivot from earlier plan:** SOP4 was wrong. mmWave Studio's "radar-only firmware" is actually downloaded over UART (RS232) at every boot, not flashed. The chip's QSPI flash content becomes irrelevant in this mode. SOP **2** (development mode) is correct. The DCA1000's 60-pin flat cable provides the SPI bridge — no AR-DevPack-EVM-012 needed.

This runbook is the **manual GUI validation** path. Once we confirm continuous streaming works on this exact hardware, we automate via Lua.

References:
- `C:\ti\mmwave_studio_03_01_04_04\docs\mmwave_studio_user_guide.pdf` (sections 6, 7, 8, 9, 11)
- `C:\ti\mmwave_studio_03_01_04_04\readme.txt` (confirms 3.01.04.04 supports AWR2944/AWR2x44P)

---

## Phase 0 — Hardware setup (one-time, ~5 min)

### Cables
- AWR2944PEVM **J8 (XDS110)** → host USB (existing)
- DCA1000EVM **J1 (Radar FTDI)** → host USB (existing)
- DCA1000EVM **J6 (Ethernet RJ45)** → host NIC (existing, 192.168.33.30/24)
- 60-pin HD ribbon: AWR HD-60 → DCA1000 J3 (existing — provides LVDS + SPI bridge)
- AWR 12V barrel power
- DCA1000 5V barrel power
- DCA1000 SW3 = DC JACK position (we power from external supply, not flat cable)

### SOP jumpers — change from current setup
**Current** (functional/QSPI boot, what we've been using): only J20 closed.
**Change to SOP2** (Studio development mode):
- **J17 closed** (SOP0 = 1)
- **J18 closed** (SOP1 = 1)
- **J20 OPEN**  (SOP2 = 0)

Per user guide page 22: *"Jumpers on SOP0 and SOP1 should be closed and SOP2 should be left open to set the device in SOP2 mode."*

After changing jumpers, **press the nRESET button** (S2) on the EVM, OR power-cycle.

### Host networking
- Host static IP = 192.168.33.30, mask 255.255.255.0 (existing)
- Firewall: allow Studio + DCA1000 ports (4096 Studio↔DCA1000, 4098 DCA1000→host raw ADC). Existing setup should already permit this.

---

## Phase 1 — Manual Studio GUI validation (~20 min)

This is the **prove-it-works** step. Do not skip — automation is meaningless if the manual path doesn't deliver continuous streaming.

### 1. Stop Seeker if running
Ctrl-C in the Seeker terminal so it releases COM10/COM11.

### 2. Launch Studio
```
C:\ti\mmwave_studio_03_01_04_04\mmWaveStudio\Runtime\mmWaveStudio.exe
```
(or use the Start Menu shortcut.)

The GUI opens. Default tab is **Connect**.

### 3. Connect tab
1. **Board Control → SOP Mode**: set to "Development Mode (SOP2)" → click **Set**. (No-op in our case since DCA1000 isn't reading SOP, but Studio expects this state.)
2. **RS232 Operations**: select COM port = the one labeled "**XDS110 Class Application/User UART**" (today this is **COM11**, but verify in Device Manager — Windows enumeration can shift).
3. Click **Connect**. Studio attempts 115200, then upshifts to 921600.
4. **Files → BSS** → browse to `C:\ti\mmwave_studio_03_01_04_04\rf_eval_firmware\radarss\xwr2x4xp_radarss_rprc.bin`
5. **Files → MSS** → browse to `C:\ti\mmwave_studio_03_01_04_04\rf_eval_firmware\masterss\awr2xxx_mmwave_full_mss_rprc.bin`
6. Click **Load** next to BSS firmware. Wait for "BSS firmware download complete" in the output window.
7. Click **Load** next to MSS firmware. Wait for "MSS firmware download complete".
8. Click **SPI Connect** (button label changes to "SPI Disconnect" on success).
9. Click **RF PowerUp**.

### 4. Static Config tab
Match our existing cfg's `channelCfg 15 15 0` and `adcCfg 2 0`:
- **Channel Config**: RX = 1+2+3+4 (all four), TX = 1+2+3 (first three). Cascading = 0.
  - **WAIT — see note below about lane count vs RX count.**
- **ADC Out Config**: ADC Bits = 16-bit, ADC Output Format = Complex 1x.
- **Low Power Mode**: Regular ADC mode.
- Click **RF Init**.

### 5. Data Config tab — CRITICAL
Per user guide page 25 NOTE:
> "Max LVDS lanes supported in xWR2x4xP are **2 only**. RX channel number and LVDS lane number should match. For 4 RX channels → 2 LVDS lanes."

So:
- **Lane Config**: 2 lanes
- **LVDS Clock Config**: select what GUI auto-detects for AWR2x4xP (typically **600 Mbps DDR**)
- **Data Format Config**: 4 RX, 16-bit complex
- Header: **disabled** (matches our `lvdsStreamCfg ... 0` which had enableHeader=0)

### 6. Sensor Config tab → Profile / Chirp / Frame
**Profile** (matching our cfg `profileCfg 0 77 12 7 20.81 0 0 8.883 0 384 30000 0 0 164`):
- Profile ID = 0
- Start Freq = 77 GHz
- Idle Time = 12 µs (or try 30 µs if first pass halts — in Studio mode it shouldn't, but it's a knob)
- ADC Start Time = 7 µs
- Ramp End Time = 20.81 µs
- Tx Output Power = 0
- Tx Phase Shifter = 0
- Freq Slope = 8.883 MHz/µs
- Tx Start Time = 0
- Num ADC Samples = 384
- Sample Rate = 30000 ksps
- HPF1 Corner = 0
- HPF2 Corner = 0
- Rx Gain = 164 (decode: 36 dB main + RF mode)

Click **Save** → **Activate**.

**Chirp** (matching `chirpCfg 0 5 0 0 0 0 0 15`):
- Chirp Start Idx = 0, End Idx = 5 (six chirps in DDM cycle)
- Profile ID = 0
- Vars (StartFreq/Slope/Idle/AdcStart) = 0
- TX Enable = 1+2+3 (mask = 7 — but our cfg says 15 = all 4 TX. Need to verify what TX count AWR2x4xP exposes in Studio — try 7 first; 15 may need cascading or be invalid for 3-TX silicon).

**Frame** (matching `frameCfg 0 5 128 0 384 50 1 0`):
- Chirp Start Idx = 0, End Idx = 5
- Num Loops = 128
- **Num Frames = 0** ← **infinite** (per user guide page 28 NOTE: "Number of frames is zero for infinite samples")
- Frame Periodicity = 50 ms
- Trigger Select = SW (= 1)
- Frame Trigger Delay = 0

### 7. Connection tab → Setup DCA1000 (if not already done)
- Set IP/mask: 192.168.33.30 / 255.255.255.0 (host)
- DCA1000 IP: 192.168.33.180
- Studio cmd port: 4096
- Data port: 4098
- Click **Setup DCA1000**.
- Verify "FPGA version" reads back (~2.8) and "all commands success" in output log.

### 8. Sensor Config tab → DCA1000 ARM → Trigger Frame
- Set output dump filename (e.g. `C:\Users\...\Desktop\studio_test.bin` — Studio writes its own file alongside our pipeline)
- Click **DCA1000 ARM**
- **Wait at least 2 seconds** (user guide warns: less = no LVDS data + 30s timeout)
- Click **Trigger Frame**

### 9. Validate streaming
- Studio's output log should show frame triggers, no errors.
- Check the dump file size growing on disk — should grow at ~5 MB/s sustained.
- **Run for at least 5 min** with no operator interaction. If it streams clean for 5 min, you've broken the wall.
- If it streams for 30+ min without halt → continuous streaming achieved. Move to Phase 2.

### 10. Stop Frame
Click **Stop Frame** to halt cleanly.

---

## Phase 2 — Seeker integration (after Phase 1 passes)

Two approaches, in order of complexity:

### A. Studio-as-bringup-only, Seeker reads UDP separately
- Studio brings up the chip via its GUI (Phase 1 manual flow OR a Lua startup script that does the same).
- Studio's `CaptureCardConfig_StartRecord_ContinuousStream(1)` mode tells DCA1000 to multicast UDP to 4098 — Studio doesn't consume those packets, our Seeker pipeline does.
- We modify `composite_manager.py` to skip the cfg-push to chip (chip is already configured by Studio) and start `DCAPipeline` directly.

### B. Full Lua automation
- Write `seeker_bringup.lua` that does Phase 1 steps 3-8 programmatically.
- Register the script as Studio's startup script (registry key `HKCU\Software\Texas Instruments\mmWave Studio\<ver>\Settings\Startup Script\Path`).
- Launch Studio from a Seeker startup hook, minimize the GUI window.
- Studio Lua exits after `Trigger Frame`; chip keeps streaming because frame trigger is sticky on the radarSS.

### Lua skeleton (for Phase 2B)
```lua
-- C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\scripts\seeker_studio_bringup.lua
ar1.SOPControl(2)
ar1.Connect(11, 921600, 1000)  -- COM11 - VERIFY ENUMERATION FIRST
ar1.frequencyBandSelection("77G")
ar1.SelectChipVersion("AR2944")  -- 'AR2x4xP' may also be valid; test both
ar1.DownloadBSSFw([[C:\ti\mmwave_studio_03_01_04_04\rf_eval_firmware\radarss\xwr2x4xp_radarss_rprc.bin]])
ar1.DownloadMSSFw([[C:\ti\mmwave_studio_03_01_04_04\rf_eval_firmware\masterss\awr2xxx_mmwave_full_mss_rprc.bin]])
ar1.SPIConnect()
ar1.PowerOn(0, 1000, 0, 0)
ar1.RfInit()

ar1.ChanNAdcConfig(1,1,1,1,1,1,1,2,1,0)  -- TX 1+2+3, RX 1+2+3+4, ADC 16-bit complex
ar1.LPModConfig(0,0)
ar1.RfInit()
ar1.DataPathConfig(513, 1216644097, 0)   -- LVDS, no header, RAW capture
ar1.LvdsClkConfig(1, 1)                  -- DDR, 600 Mbps (auto)
ar1.LaneConfig(1, 1, 0, 0, 0, 0, 0, 0)   -- 2 lanes for 4 RX

-- Profile values from awr2944P_unified.cfg:
ar1.ProfileConfig(0,            -- profileId
                  77,           -- startFreq GHz
                  12, 7, 20.81, -- idle, adcStart, rampEnd µs
                  0, 0,         -- tx out backoff, tx phase shift
                  0, 0, 0, 0,   -- vco/lo/hpc placeholders (verify against API)
                  8.883,        -- slope MHz/µs
                  0,            -- tx start time
                  384,          -- num adc samples
                  30000,        -- sample rate ksps
                  0, 0,         -- hpf corner freqs
                  164)          -- rx gain

ar1.ChirpConfig(0, 5, 0,         -- start, end, profileId
                0, 0, 0, 0,      -- vars
                7,               -- TX mask (1+2+3)
                0, 0, 0)         -- reserved

ar1.FrameConfig(0, 5,            -- chirp start, end
                0,               -- numFrames=0 → INFINITE
                128,             -- numLoops
                40,              -- framePeriodicity (ms × 200000? VERIFY)
                0, 0,            -- frame trigger delay, reserved
                1)               -- triggerSelect SW

ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180",
                              "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)        -- raw mode, LVDS
ar1.CaptureCardConfig_PacketDelay(25)
ar1.CaptureCardConfig_StartRecord_ContinuousStream(1) -- the magic: unbounded
ar1.StartFrame()
print("seeker_studio_bringup: chip is now streaming continuously")
```
**Several parameter values are guesses** — must be calibrated against the GUI flow's actual `CSV` files (Studio writes `ProfConfigData.csv`, `ChirpConfigData.csv` after each Save+Activate; those are ground truth).

---

## Risks & open questions

1. **TX count.** Our cfg says `txEnable=15` (4 TX). AWR2x4xP datasheet may only have 3 TX; 15 might silently fail. Manual GUI test resolves this.
2. **`framePeriodicity` LSB units.** mmwavelink doc says 5 ns LSB → 50 ms = 10000000. ar1.* Lua wrapper may pre-convert (40 = 40 × something). Verify by reading what GUI writes to the CSV after configuring 50 ms via the form.
3. **`SelectChipVersion`** — value strings: `AR2944` worked in older versions, `AR2x4xP` may be needed for AWR2944**P**. Try both.
4. **`StartRecord_ContinuousStream(1)`** — the (1) arg is "loop forever". Verify the data-rate math: 4.7 MB/frame × 20 fps = 94 MB/s. DCA1000 can sustain this only over jumbo frames + tuned host RX buffer (we have 64 MB SO_RCVBUF — should be enough).
5. **Studio's RS232 baud upshift to 921600** must work — if it stays at 115200 after firmware download, runtime API calls become slow. Check Studio output log.
6. **Going back to demoDDM:** revert SOP jumpers to your current setup (only J20 closed → SOP=001 functional) + power-cycle. Chip boots from QSPI flash with whatever firmware it has (currently v6). Studio path is non-destructive to QSPI.

---

## Decision tree after Phase 1

| Outcome of 5-min Studio GUI test | Next step |
|---|---|
| Streams continuously, no halts | Phase 2: write Lua + integrate. Customer POC path is alive. |
| Streams 5+ min then halts | Studio firmware has its OWN halt — escalate to TI E2E with the BSSEV log. We're worse off than mmw_demoDDM. |
| Won't even arm DCA1000 | DCA1000 wiring / IP / firmware. Independent of which chip firmware is in use. |
| RS232 connect fails | XDS110 USB driver / COM port enumeration / SOP jumpers wrong. |
| Profile/Chirp/Frame Activate fails | Wrong parameter ranges for AWR2x4xP — back-trace via the CSV that Studio writes to find the right values. |

---

**This runbook supersedes `STUDIO_BRINGUP.md` (which assumed SOP4 + Python ctypes — both wrong).** The Python `studio_bringup.py` scaffold can be deleted once Phase 2B Lua path is in place.
