# Seeker-01 Radar — Full Plan (v2)

**Owner:** Asaf
**Drafted:** 2026-05-03
**Constraint:** none. No bandaids. Architecturally correct only.
**End goal:** continuous, halt-free LVDS raw-ADC streaming on AWR2944P,
host-side Python pipeline, runs on Jetson Xavier AGX in production.
PMM drone detection working with realistic SNR.

**Asaf's decisions (locked):**
- No demoDDM stabilization. demoDDM is dead. PMM lives on raw-ADC, and
  demoDDM's halt + on-chip DSP collisions make it a wrong-tree.
- No time estimates / no "see something today" intermediate deliverable.
  Architecturally correct path only.
- Multi-agent execution OK; I plan and review before integrating.

---

## 1. The architecture (single answer)

```
   ┌────────────┐                ┌─────────────────┐
   │  Host      │                │   AWR2944P EVM  │
   │  (Win now, │                │   (SOP0 boot)   │
   │   Jetson   │                │                 │
   │   later)   │                │  ┌───────────┐  │
   │            │  UART          │  │  MSS      │  │ LVDS    ┌───────────┐
   │  Phase 2   ├────────────────┼──┤ Cortex-R5 ├──┼─────────┤ DCA1000   │
   │  Python    │  mmwavelink    │  │           │  │ 4-lane  │ FPGA      │
   │  mmwave-   │  (115200 then  │  │ mmw_      │  │ DDR     │           │
   │  link      │   handshake to │  │ studio_cli│  │         └─────┬─────┘
   │  client    │   921600)      │  │ (custom)  │  │               │ UDP
   │            │                │  └─────┬─────┘  │               │ 4098
   │  PMM       │                │        │        │               │
   │  + fusion  │                │  ┌─────┴─────┐  │     ┌─────────┴────┐
   │  pipeline  │                │  │  BSS      │  │     │  Host        │
   │            │                │  │  loaded   │  │     │  Pipeline    │
   │            │                │  │  from QSPI│  │     │  (already    │
   └────────────┘                │  │  at boot  │  │     │   exists)    │
                                 │  └───────────┘  │     └──────────────┘
                                 │                 │
                                 │  QSPI flash:    │
                                 │  - mmw_studio_  │
                                 │    cli.appimage │
                                 │  - BSS .rprc    │
                                 │  - MSS .rprc    │
                                 └─────────────────┘
```

**Three artifacts to deliver:**

1. **`mmw_studio_cli` MCU+SDK app** — runs on AWR2944P MSS Cortex-R5.
   On boot: loads BSS+MSS .rprc images from a fixed QSPI offset into
   the chip's BSS subsystem (replicating exactly what Studio's
   `ar1.DownloadBSSFw` / `DownloadMSSFw` do, but chip-side, no host
   needed). Then enters a passive mmwavelink command router on UART:
   every command from host is dispatched to the BSS or interpreted
   locally (channel cfg, profile cfg, frame cfg, sensor start, etc.).
   No on-chip DSP. No TLV emission. The chip's only job is BSS
   firmware host + LVDS data plane.

2. **Phase 2 Python mmwavelink client** — already exists. Just needs a
   chip that actually answers. Will speak to `mmw_studio_cli` on UART
   after SOP0 boot.

3. **Existing host pipeline** — `radar_dca/dca_pipeline.py`,
   `pmm_detector.py`, `dca_manager.py`, `composite_manager.py`. No
   changes needed beyond `--radar-firmware studio-py` becoming the
   only supported mode (demoDDM + studio modes get deprecated).

---

## 2. Why this and not anything else

| Alternative | Why not |
|---|---|
| Fix demoDDM halt | demoDDM runs on-chip DSP that produces TLV; LVDS framing in this mode is contaminated by the DSP path. Even halt-free demoDDM is the wrong data plane for PMM. |
| Fix Studio's BSS download | Studio's broken on this hardware (8+ failed attempts). Even if fixed, Windows-only, no Jetson path. Throwaway. |
| Phase 2 firmware download via UART | AWR2944P bootloader doesn't accept BSS firmware on UART — Studio uses SPI. Reverse-engineering Studio's SPI-MPSSE protocol is multi-week work AND it produces a Windows-only solution. |
| Stay in SOP2 + host firmware download | Forces every boot to re-download firmware from host. Fragile. Not a production architecture. |

**SOP0 + custom app + flash-resident BSS** is the only path that's both
production-clean and Jetson-portable.

---

## 3. The single workstream

### Stage 1 — Environment (Step 0 of everything)

| Item | Output | Validation |
|---|---|---|
| 1.1 Inventory MCU+SDK install: ti-arm-clang version, sysconfig version, makefile entry points | `docs/MCU_SDK_INVENTORY.md` with versions + paths | `which`-style outputs verified |
| 1.2 Build the reference `sbl_qspi` for AWR2944P | `.tiimage` produced under `examples/drivers/boot/sbl_qspi/awr2x44p-evm/` | clean build, no errors |
| 1.3 Build the reference `mmw_ddm` for AWR2944P (just to prove the toolchain works) | mmw_ddm `.tiimage` produced | clean build |

### Stage 2 — Read mmw_ddm and extract Studio-equivalent paths

| Item | Output | Validation |
|---|---|---|
| 2.1 Identify exactly how mmw_ddm calls `rlDeviceFileDownload` to push BSS firmware | Annotated source citations from `ti/demo/awr2x44P/mmw_ddm/` | Specific function names + line numbers |
| 2.2 Identify the mmwavelink command-router code path mmw_ddm uses (since mmw_ddm itself receives mmwavelink commands from its CLI parser) | Same | Same |
| 2.3 Identify the LVDS data path setup (rlDevHsiClk + rlDevHsiCfg + lane enable) | Same | Same |
| 2.4 Identify what mmw_ddm does on the on-chip DSP path that we need to STRIP for our app | Same | Same |
| 2.5 Document the call sequence as a target for our app | `docs/STUDIO_EQUIV_CALL_SEQUENCE.md` | Reviewed against mmw_ddm tests |

### Stage 3 — Build `mmw_studio_cli` app

| Item | Output | Validation |
|---|---|---|
| 3.1 New folder `firmware/mmw_studio_cli/` cloned from mmw_ddm structure | Project boilerplate (.syscfg, .projectspec, makefile) | Builds empty app with no functionality (entry point + idle loop) |
| 3.2 Implement boot-time BSS firmware load from QSPI offset 0x80000 | App reads BSS .rprc from QSPI, calls `rlDeviceFileDownload` on the local bus | BSSEV `BSS RUN_CALIB done` event observed on host |
| 3.3 Implement mmwavelink command router on UART (incoming bytes parsed, dispatched to BSS or handled locally) | UART RX thread + dispatcher | Phase 2 host's `get_version()` returns a real D2H frame (not echo) |
| 3.4 Strip all DSP / TLV emission. Implement only LVDS data plane (channel cfg, profile cfg, chirp cfg, frame cfg, sensor start) | Stripped app | LVDS streams to DCA1000 after Phase 2 sensor_start, UDP arrives on host:4098 |
| 3.5 Async event handler: BSS events get forwarded to host via mmwavelink async messages | UART TX path | Host sees BSSEV calibration messages (currently we see them via mmw_ddm CLI) |

### Stage 4 — Validate via SBL UART boot (no flash yet)

| Item | Output | Validation |
|---|---|---|
| 4.1 Set chip jumpers to SOP1 (UART boot) | physical jumper change, single jumper move | Device Manager re-enumerates, COM ports present |
| 4.2 Use `tools/boot/uart_uniflash.py --flash-writer` to send our app via UART boot (no flash write yet) | Chip boots our app from RAM | App's idle loop heartbeat visible on UART |
| 4.3 Run Phase 2 host bring-up against the live app | `bringup_phase2('COM9')` runs end-to-end, real D2H responses, sensor_start succeeds | UDP packets at 4098, no echo, no halt |
| 4.4 Soak test: 30+ min continuous LVDS, no halt, no DCA recovery events | Soak log + PMM detector output on empty scene | 0 false detections, no kick_lvds events |

### Stage 5 — Flash to QSPI for permanent SOP0 boot

| Item | Output | Validation |
|---|---|---|
| 5.1 Use `sbl_uart_uniflash` to write our app + BSS .rprc + MSS .rprc to QSPI at the right offsets | QSPI image programmed | Flash verify pass |
| 5.2 Move chip jumpers to SOP0 (J17 open, J18 open, J20 closed) | physical jumper change | already in place from yesterday's work |
| 5.3 Power cycle. Chip auto-boots our app from flash. | App alive on UART without any host firmware push | Phase 2 host runs `bringup_phase2()` and gets real responses immediately |
| 5.4 Soak test: same as 4.4 but from cold boot | Cold-boot soak log | clean, no halt, drone PMM detection works |

### Stage 6 — Wire into Seeker as default

| Item | Output | Validation |
|---|---|---|
| 6.1 Make `--radar-firmware studio-py` the default in `main.py` | Updated argparse default | `START_SEEKER.bat` launches with the new path |
| 6.2 Mark `studio` and `demoDDM` modes deprecated; remove them from the CLI choices in a follow-up | code annotation + deprecation log line | clean code path |
| 6.3 PMM detector threshold tightening so empty-scene = 0 detections, drone-in-scene = clean detections | Updated `radar_dca/pmm_detector.py` | 5-min empty-scene soak: drone_detections=0; drone-in-scene: realistic SNR |

### Stage 7 — Jetson port

| Item | Output | Validation |
|---|---|---|
| 7.1 Verify pyserial enumerates the AWR's FTDI as `/dev/ttyUSBN` on Jetson | Jetson smoke test | `bringup_phase2('/dev/ttyUSB...')` works |
| 7.2 Verify DCA1000 networking (host static IP 192.168.33.30, MTU 9000, jumbo frames enabled on Jetson) | Jetson config doc | 100% UDP delivery during soak |
| 7.3 Run full Seeker pipeline on Jetson with radar | Live pipeline run | Same outputs as Windows host |

---

## 4. Multi-agent execution

I run agents in parallel within a stage where the tasks are independent.
I review output before integrating. No agent does more than one stage at
a time.

### Stage 1 (parallel)

- **Agent E1** — Inventory MCU+SDK install (1.1) and produce
  `docs/MCU_SDK_INVENTORY.md`. Output: ti-arm-clang version, sysconfig
  version, paths, makefile entrypoint.
- **Agent E2** — Build `sbl_qspi` reference (1.2). Output: working
  `.tiimage`, build log.
- **Agent E3** — Build `mmw_ddm` reference (1.3). Output: working
  `.tiimage`, build log.

### Stage 2 (parallel after Stage 1)

- **Agent S1** — Read mmw_ddm BSS firmware download path (2.1, 2.3).
  Output: annotated source.
- **Agent S2** — Read mmw_ddm mmwavelink router + identify DSP path to
  strip (2.2, 2.4). Output: annotated source.
- After both: I integrate into `docs/STUDIO_EQUIV_CALL_SEQUENCE.md`.

### Stage 3 (sequential, single agent at a time, I review per item)

- **Agent F1** — 3.1 (boilerplate)
- **Agent F2** — 3.2 (BSS firmware boot-time load)
- **Agent F3** — 3.3 (mmwavelink command router)
- **Agent F4** — 3.4 (strip DSP, keep only LVDS data plane)
- **Agent F5** — 3.5 (async event forwarding)

### Stage 4-5 (I drive directly, hardware in loop, no agents)

- These stages need real hardware feedback. Single operator (me).

### Stage 6-7 (parallel)

- **Agent J1** — Jetson port (Stage 7)
- **Agent J2** — PMM threshold tightening (6.3)

---

## 5. Anti-bandaid commitments

- ❌ No demoDDM mode in any default. Removed from supported modes once
  studio-py works.
- ❌ No "30s burst" celebrations. The deliverable is "30+ min continuous
  with no halts" or it isn't done.
- ❌ No "PMM detection" claims without operator confirming ground truth
  (drone in scene vs not).
- ❌ No `xds110reset` or `UniFlash JTAG erase` cowboy moves. JTAG/SBL
  flashing only via the documented `sbl_uart_uniflash` path.
- ❌ No partial deliveries. Each Stage's exit criterion is binary.
- ❌ No time estimates in any status update.

---

## 6. Status reporting cadence

After each Stage exit-criterion is met (or fails), I write a single
status update with:
- What ran
- What the validation showed
- What the next stage is
- Any blockers (chip state, missing tool, missing TI doc, etc.)

No daily check-ins. Stage-by-stage.

---

## 7. Open questions for Asaf (before I start Stage 1)

1. **Hardware state right now:** chip in SOP0, jumpers set, both AWR
   USBs (J8 + J10) plugged, DCA on Ethernet only. Is that still the
   physical state, or has anything moved? *Confirms baseline.*

2. **Does the bench have an FT4232H breakout for SBL UART boot?**
   Stage 4.1 needs to put the chip in **SOP1 (UART boot)** to load our
   app via `sbl_uart_uniflash`. SOP1 = J17 closed, J18 open, J20 open.
   That's a physical jumper change you'd make once for stage 4-5, then
   change back to SOP0 once flashed.

3. **Cosmetic:** is `mmw_studio_cli` a fine name for the new app, or do
   you want a different name? It'll be visible in the firmware/ folder.

I'll wait for your answers on (1)-(2) before launching Stage 1 agents.
(3) is bikeshedding; default `mmw_studio_cli` unless you say otherwise.
