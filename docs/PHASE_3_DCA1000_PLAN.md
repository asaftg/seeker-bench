# Phase 3 — DCA1000 raw-ADC + advanced detection modes

## Context

Today's `mmw_demoDDM` firmware does ALL signal processing on the AWR2944P
chip and emits a TLV point-cloud over UART. That's why every chirp /
CFAR / Doppler tweak we want for small-RCS targets at long range is
locked behind firmware rebuilds.

The DCA1000EVM bypasses on-chip DSP and streams raw ADC over LVDS to
Ethernet at ~600+ Mbps, so every parameter becomes a host-side Python
knob. This unlocks mmHawkeye-style long-integration drone detection,
zero-Doppler retention for stationary targets, and propeller
micro-Doppler classification.

Current detection capabilities (humans, vehicles via DSP+CFAR+DBSCAN)
are NOT lost — they're re-implemented as a host-side Python profile
called "STOCK" that mirrors today's behaviour bit-for-bit. The
firmware change is a one-time plumbing change; everything switchable
from the GUI lives in host-side Python profiles operating on the
shared raw-ADC stream.

## Hardware (AWR2944PEVM + DCA1000EVM, independent boards)

Both boards run with independent power supplies and an independent
USB to the host. The 60-pin Samtec HD ribbon between them carries
LVDS data plus UART/I2C/reset/optional-power pass-through (verified
against the SPRUIJ4A connector pinout) — but for our setup the boards
do NOT share power because the AWR is 12 V and the DCA is 5 V.

**Total: 2 USBs + 1 Ethernet + 2 separate power supplies + 1 LVDS ribbon.**

| # | Cable | From | To | Notes |
|---|-------|------|-----|------|
| 1 | micro-USB | AWR2944PEVM **J8** (XDS110) | host | JTAG flash + UART CLI to AWR. |
| 2 | micro-USB | DCA1000EVM **J1** (Radar FTDI) | host | Per SPRUIJ4A p.9: "Allows access to the xWR1xxx EVM through UART, SPI, or I2C through the FTDI chip." So the AWR's UART CLI is reachable via this USB *too* (through the 60-pin pass-through), AND this USB is what `DCA1000EVM_CLI_Control` uses for capture config. |
| 3 | RJ45 | DCA1000EVM **J6** (Ethernet) | host NIC (direct, no switch) | Raw-ADC UDP stream. DCA default IP `192.168.33.180` (verify against doc); host static IP on same /24. Jumbo frames (MTU 9000), firewall whitelist for the UDP capture ports. |
| 4 | 12 V barrel | wall PSU **(SDI65-12-U-P5 tested)** | AWR2944PEVM | 12 V, ≥ 2.5 A, 2.1 mm center-positive, per **SPRUJ22C**. |
| 5 | 5 V barrel | wall PSU | DCA1000EVM DC jack | 5 V, 2.5 A. **DCA SW3 set to DC JACK position** — power-share through 60-pin only works with 5 V xWR EVMs and we have 12 V. |
| 6 | 60-pin HD ribbon | AWR1xxx HD-60 connector | DCA1000EVM **J3** | Per SPRUIJ4A p.12, carries: LVDS data (4 lanes + clock), UART RX/TX, I2C SDA/SCL, MCU_RSTn, GND, plus optional 5 V (unused in our 12 V setup). |

Not connected in normal operation:
- DCA1000EVM **J4** (FPGA JTAG micro-USB). Only used for updating the DCA's FPGA firmware itself, which is rare. Per SPRUIJ4A p.10. The board has 2 USB ports physically; only J1 is plugged for runtime.

Sources:
- **SPRUJ22C** — AWR2944EVM/AWR2944PEVM User's Guide (12 V, 2.5 A barrel; XDS110 USB at J8).
- **SPRUIJ4A** — DCA1000EVM Data Capture Card User's Guide Rev. A (5 V, 2.5 A; J1/J3/J4/J6 silkscreen labels and pinout).
- **SPRUIK7** — DCA1000EVM Quick Start Guide.

Sanity test after wiring (no software change yet):

```
ping 192.168.33.180   # DCA replies once J1 USB has powered the FT4232H
```

## Firmware decision

Reflash once over AWR XDS110, then all toggles live in Python.

- Replace `mmw_demoDDM` with `mmw_studio_cli` (or equivalent raw-ADC
  variant from the mmwave SDK). Driven by **UniFlash** over the
  AWR XDS110 USB.
- Old firmware stays a click away in UniFlash for rollback. Flow:
  1. Power both boards.
  2. UniFlash → select `xWR2944P` → choose
     `<sdk>/packages/ti/demo/awr2944P/mmw_studio_cli/mmw_studio.bin`.
  3. Power-cycle AWR.
  4. Verify with `DCA1000EVM_CLI_Control fpga` test capture.
  5. Rollback (if needed): same flow, swap the .bin path back to
     `mmw_demoDDM` build artifacts (keep them archived in
     `firmware_archive/` in the repo).
- After flashing: AWR streams raw ADC over LVDS only. The on-chip
  TLV UART output goes silent (or is bypassed). Everything that today
  comes out of TLV — point cloud, target list, classification — is
  reproduced host-side.

The GUI mode picker switches between **host-side Python profiles**:

| GUI option | Pipeline | Equivalent to |
|------------|----------|---------------|
| **STOCK** | Range-FFT → Doppler-FFT → CA-CFAR → DBSCAN → Kalman tracker | Today's on-chip behaviour (acceptance: ±5% same target list) |
| **LONG RANGE** | + 256→512 chirps coherent integration + OS-CFAR + Capon beamforming | mmHawkeye-style drone-at-distance |
| **STATIC** | + zero-Doppler retention + slow-mean clutter subtraction | Parked vehicles, standing humans |
| **PMM DRONE** | + slow-time STFT + propeller-sideband detector + Doppler unfolding | Drone classifier (vs. bird vs. clutter) |

Mode switch is instantaneous (no reflash, no AWR reboot — just swap
which Python pipeline consumes the UDP stream).

## Software architecture

Add new package; do not replace `radar/`.

```
radar/                     # existing stock TLV pipeline (kept for reference / fallback)
radar_dca/                 # new — host-side raw-ADC pipeline
  dca_capture.py           # UDP socket listener, ADC reassembly
  dca_config.py            # wraps DCA1000EVM_CLI_Control to start/stop captures
  range_doppler.py         # range-FFT + Doppler-FFT
  cfar.py                  # CA / OS / GO CFAR
  beamform.py              # virtual-array AoA, Capon beamformer
  micro_doppler.py         # STFT + propeller-signature features
  doppler_unfold.py        # staggered-PRF / cepstrum-based unfolding
  tracker.py               # 6D Kalman (existing radar/clustering.py is a candidate to lift)
  profiles/
    stock.py + stock.cfg
    long_range.py + long_range.cfg
    static.py + static.cfg
    pmm_drone.py + pmm_drone.cfg
  dca_manager.py           # thread that owns the pipeline,
                           # publishes RadarFrame on the existing bus topic
```

Critical: `dca_manager` emits the **same `RadarTarget` / `RadarFrame`
contract on the same bus topic** as today's `RadarManager`. So
`FusionManager`, GUI, top-5 list, all unchanged.

Lean on existing OSS rather than rolling our own DSP from scratch:

- **`OpenRadar`** (Stanford / pyOpenRadar) — Apache-2.0 Python
  implementation of the whole mmWave signal chain. Range/Doppler/CFAR/AoA
  all there.
- **TI `mmwave-readers`** — official DCA1000 packet parsers.
- `scipy.signal.stft` for micro-Doppler spectrograms.

## Detection profile details

### Profile A — STOCK
- Mirrors `awr2944P_highRange.cfg` chirp.
- CA-CFAR + DBSCAN + 6D Kalman.
- Acceptance: target list within ±5% of today's TLV output.
- Used as the bring-up baseline.

### Profile B — LONG RANGE (mmHawkeye-style)
Goal: 0.3 m quadrotor at 200 m, clean-sky background.

- 256 → 512 chirps per frame (frame rate drops to ~5–8 Hz).
  Coherent gain `+10 log₁₀(N)` ≈ +27 dB at 512.
- Narrow IF bandwidth → smaller range bins (4 cm) → lower noise
  floor.
- OS-CFAR instead of CA-CFAR (handles non-Gaussian clutter near
  foliage).
- Capon beamformer over the AWR2944P virtual array (~12-16 elements
  with DDM-MIMO) for ~+8-10 dB spatial gain.
- Velocity accuracy explicitly relaxed.

Estimated range for a 0.3 m drone with this profile: **150–250 m**,
depending on aspect angle, propeller phase, and clutter background.
Sky-cut scenarios closer to upper bound. Need PRD numbers to
validate.

### Profile C — STATIC
Goal: parked vehicles, standing humans.

- Retain zero-Doppler bin (stock pipeline drops it because that's
  where ground/wall clutter lives).
- Subtract a slow-running 30-frame mean of the zero-Doppler bin —
  real stationary targets persist, clutter cancels.
- Range-gate out the first 0.5 m of clutter ground bounce.
- In FusionManager: a static-radar return aligned with an EO bbox
  becomes a "parked vehicle" / "standing human" with high
  confidence (radar contributes range, EO contributes class).

### Profile D — PMM DRONE
Goal: drone-vs-bird-vs-clutter classifier from radar alone.

- For each confirmed range-Doppler hit, extract a 256-tap slow-time
  slice.
- STFT spectrogram.
- Look for **propeller sidebands**: DJI-class drones spin at 4-9 kHz
  blade rate; sidebands appear at ±BPF around the body Doppler.
- **Doppler unfolding** (PRF too low to directly resolve 5+ kHz):
  - Staggered PRF (alternate two PRFs frame-by-frame, CRT unfold).
  - HRRP cepstrum (mmHawkeye-style).
- Output: `RadarTarget.micro_doppler_class` ∈
  `{drone, bird, rotating_machinery, unknown}` + confidence.
- Fusion: a high-confidence radar-side `drone` class promotes the
  FusedTrack to `TargetClass.DRONE` *without* needing EO/thermal
  agreement. Lets us classify at 200 m where the camera can't
  resolve the target.

## Real-time vs offline

User wants real-time end-state. Iteration plan:

- **Bring-up + profile A/C**: target real-time directly (frame rate
  matches today's ~15 Hz easily).
- **Profile B (long-range)**: 5–8 Hz frame rate is acceptable for a
  long-range surveillance mode where targets aren't fast.
- **Profile D (PMM)**: develop offline against recorded `.bin`
  captures first, then move algorithm to real-time once validated.
  STFT + classifier on 256-tap slices is well within real-time
  budget once tuned.

Recording infra is cheap — `dca_capture.py` already needs to handle
UDP, dumping to `.bin` is one extra branch. We get offline iteration
for free.

## Milestones

1. **Day 1** — Wire all cables (XDS to AWR, FTDI to DCA, RJ45, two
   AWR=12 V & DCA=5 V supplies, HD-60 ribbon). Reflash AWR to `mmw_studio_cli` via
   UniFlash. Archive the old `mmw_demoDDM` .bin for rollback. Ping
   the DCA, run
   `DCA1000EVM_CLI_Control fpga ./scripts/dca1000_capture.json` to
   a 2-second `.bin` file. Pure cable + driver sanity.
2. **Day 2** — `radar_dca/` skeleton; bit-equivalent STOCK profile;
   side-by-side run against today's TLV stream; GUI mode picker
   wired.
3. **Day 3** — STATIC profile; bench test with parked vehicle.
4. **Day 4–5** — LONG RANGE profile; outdoor drone test progression
   50 m → 100 m → 200 m. Document achieved SNR vs. range.
5. **Week 2** — PMM DRONE profile; offline-first against recorded
   captures, then real-time once validated.

## Open items

- **PRD numbers** — user will paste relevant section into the chat
  (Drive isn't reachable from here). Plan currently uses an
  estimate of 0.3 m drone @ 200 m, RCS ≈ 0.005 m², clean-sky
  background.

## Out of scope (future)

- Multiple radars in a synced array (cooperative beamforming).
- Real-time DL micro-Doppler classifier (a CNN on the spectrogram).
  Start with handcrafted features, upgrade later.
