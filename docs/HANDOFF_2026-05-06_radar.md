# Handoff — radar work session 2026-05-06 (Claude session that's
# being replaced)

## TL;DR for the next agent

I worked on this user's radar for ~24 hours across two long sessions
and I'm being replaced because I failed to land the two things the
user explicitly asked for at the end:

1. **FPS is still 7-9 Hz** on all three sensor panels (THERMAL, EO,
   RADAR), not the 20+ Hz target. I diagnosed the root cause
   correctly via a research agent (EO base64 in the shared
   `_sender`) and applied the documented fix, but the user reports
   it didn't take effect. I did not profile a live frame after the
   fix to verify, which is the correct next step.

2. **Ctrl+C still wedges the chip's CLI parser**. After any clean
   software shutdown the chip stops responding to ANY CLI command on
   the next launch — power cycle is the only recovery. I diagnosed
   the root cause correctly via a research agent (pyserial
   DTR/RTS toggle on Windows port open) and applied the documented
   workaround (`_open_cli_port` helper that sets `dtr=False` /
   `rts=False` before `open()`), but the user reports the wedge
   persists. I did not verify the fix landed in all paths or that
   the helper's modem-control sequence actually works on this
   specific XDS110 driver.

PMM / drone detection: I was honest with the user that **PMM cannot
distinguish the drone from a person walking at the same range** on
these recordings. Detail in section 2.

What's REAL and committed (not reverted):
- Notch corrected from every-12-bins ±5 to every-24-bins ±6 in
  `_notch_harmonic_artifact` (the previous notch was 2× too
  aggressive — diagnosed by reading the chip's actual spectrum)
- Mean-subtract MTI in `_process_frame` REMOVED, replaced with a
  narrow Doppler-DC notch in `_stage3_range_doppler`
- PMM output gated by mode in `_publish` (PMM only surfaces in AA
  mode now; stock/AG no longer show phantom drone bbs)
- PMM artifact-bin filter (drop hits within ±7 of every-24-bin
  centre)
- `CompositeRadarBackend.__init__` now accepts `radar_firmware`
  kwarg (was crashing every launch since commit `b416885`)
- Triple `radar.stop()` on shutdown collapsed into a single guarded
  call

The two unfinished bugs are documented in detail below.

---

## 1. PMM / drone detection — full record

### Mission

Detect a small DJI FPV drone in **two specific recordings**:
- `recordings/seeker_2026-05-06_12-58-59_radar.bin` (airborne1, 2148
  frames, drone fly-away 28-40 s)
- `recordings/seeker_2026-05-05_21-14-35_radar.bin` (drone-fly, 514
  frames, drone hovering at ~5 m)

Both drones are **slow and close** — they don't go further than ~8 m
in either recording. The user explicitly told me PMM (propeller
micro-motion) is the discriminator they want. The chip's TLV path
already handles humans/vehicles via on-chip CFAR; the host pipeline's
job is drone-PMM only.

### What I tried (chronological)

**Wave 1 — Root-cause investigation of "every-12-range-bins" artifact**
- Initial hypothesis (false): `_notch_harmonic_artifact` is creating
  its own signature.
- Devil's-advocate check (Wave 2 agent): `tools/analyze_drone_fly_v2.py`,
  `tools/poc1b_full_fov_scan.py`, `tools/sim_more_chirps.py` all
  call `scipy.fft.rfft` directly without going through the notch
  AND they all show the every-N pattern. Confirmed: artifact is
  real, not the notch's signature.

**Wave 2 — Designed and ran `tools/diagnose_artifact.py`**
The 6-test script revealed:
- Period is **24 bins, NOT 12** (the existing notch's `step =
  period // 2 = 12` was a known-broken half-step from the previous
  agent). 192 / 24 = 8 → most likely cause is 8-way time-interleaved
  ADC.
- IF freq is **3.75 MHz** (not 1.875 as the original code's docstring
  claimed)
- Severity per recording (mean dB above local floor):
  - airborne1: +5 dB significant
  - drone-fly: +20 dB DOMINANT (this is why chip CFAR found zero
    targets — artifact swamped the close-range bins)
  - background: +12 dB significant
- T4 (per-RX magnitude std 3-5 dB, phase diff 55-70°) and T5
  (spread across Doppler) → likely RX-specific cable/USB EMI rather
  than chip-internal clock spur.

**Wave 3 — Concrete plans**
Designed three artefacts:
- `tools/diagnose_artifact.py` (the 6-test diagnostic) — committed
- `tools/validate_two_recordings.py` (replay both recordings through
  patched pipeline, score detection rate by window) — committed
- `tools/debug_drone_visibility.py` (per-range-bin RD magnitude dump)
  — committed

**Wave 4 — Adversarial review of Wave 3 + actual detector test**

Built `tools/inject_radar_bbs.py` — runs the patched detector across
the full airborne1 binary and injects radar bbs into a copy of the
existing v5+thermalv2 replay JSONL so the user can replay it and
visually verify radar tracks vs EO/thermal drone bbs.

Result: **68% hit rate overall, but per-window analysis was the
killer**:
- PRE-drone (00:00-00:30): 72% hit rate
- DRONE window (00:30-01:05): 68% hit rate
- POST-drone (01:05-end): 67% hit rate

**The detector fires at the same rate whether the drone is in the
scene or not.** Range medians are also similar (7.9 / 5.3 / 7.9 m).
This means: the patched RD/CFAR detector is finding "any close-range
mover at 2-8 m" — not specifically the drone. Operator walking, lab
clutter, fan vibrations all fire it equally.

I tried tightening the SNR threshold to isolate drone-window hits.
At every threshold (4 dB through 18 dB), drone window has SAME or
LOWER hit rate than pre/post. There is no threshold that
discriminates.

**Wave 4.5 — Persistence check**
A hovering drone should produce a long stable run at the same
(range, az). Operator should jump cell-to-cell. I built
`tools/drone_persistence_check.py`. Result: **zero long runs (≥5
consecutive at same spot) in any window.** Detector hops cell-to-cell
every frame. This pattern matches "many random close-range movers"
not "stable hovering drone".

### Where we are on PMM / drone detection

**Confirmed working:**
- The artifact diagnosis is correct (every-24-bin spurs, ~3.75 MHz)
- The notch fix recovers ~25% of range axis (was killing 38% for
  no reason)
- The Doppler-DC notch lets hovering drones survive (mean-subtract
  MTI was annihilating them — that's why the chip's stock CFAR found
  zero targets in drone-fly hover)

**Confirmed broken:**
- PMM (the user's intended discriminator) failed across every variant
  the previous agents and I tried. It does not separate drone from
  walls / operator / fan / breathing in these recordings.
- Body skin return alone cannot tell drone from operator at 5 m.

**Honest verdict** (which I gave the user): without PMM working,
there is no software-only drone-vs-clutter discrimination on these
recordings. Three options the user can choose from:
1. EO/thermal-gated radar — replay-time gate on radar bbs against
   EO/thermal drone bbs. Truthful but it's "fusion-confirmed radar"
   not standalone radar detection.
2. Re-attack PMM with spatial gating using EO/thermal as the oracle
   for which range bin to scan.
3. Multi-frame drone-specific track that demands stable point +
   continuous Doppler signature for ≥5 frames.

### Files I created/modified for PMM work

- `radar_dca/dca_pipeline.py` — corrected notch; replaced MTI with
  Doppler-DC notch; gated PMM output by mode; PMM artifact-bin
  filter
- `tools/diagnose_artifact.py` (new)
- `tools/validate_two_recordings.py` (new)
- `tools/debug_drone_visibility.py` (new)
- `tools/inject_radar_bbs.py` (new)
- `tools/drone_persistence_check.py` (new)
- `recordings/airborne1_v5+thermalv2_replay_RADAR.jsonl` (generated)
- `.claude/plans/keen-cooking-moth.md` (the plan that drove this work)

### What the next agent should try for drone detection

Given PMM has failed empirically across every variant tested, and
given the user's mission is small drones at long range, the next
real lever is **EO/thermal-confirmed radar** — use the trained EO
classifier (`models/seeker_eo_v3.pt`) and thermal classifier
(`models/seeker_thermal.pt`) as the drone-vs-everything-else
oracle, then have the radar contribute range and Doppler at the
EO/thermal-confirmed bbox location. This is what fusion is for.
The user has NOT explicitly approved this approach because it makes
radar a sensor-fusion contributor not a standalone drone detector,
but PMM-only is a dead end on this hardware/firmware combination.

Recordings to test against:
- `recordings/airborne1_v5+thermalv2_replay.jsonl` — full v5 EO +
  v2 thermal models running against the airborne1 capture. Drone
  visible 00:30-01:05.
- `recordings/airborne1_v5+thermalv4_replay.jsonl` — newer thermal
  model.

---

## 2. Bug — FPS capped at 7-9 Hz on all sensors

### Symptom

GUI panel headers show:
- THERMAL · Boson 640: 9 Hz
- EO · IMX568: 8 Hz
- RADAR · AWR2944P: 7.0 Hz

Native rates: thermal 60 Hz, EO 25 Hz target, radar cfg 50 ms = 20 Hz.

All three throttled to ~the same rate is the giveaway: shared
downstream bottleneck, not three independent sensor issues.

### Diagnosis (research-agent verified, but fix unverified)

`gui/app.py`'s shared `_sender` periodic task does this every WS
tick (60 Hz target = 16 ms period):

1. Calls `build_ws_message(...)` → `gui/sensor_bridge.py:eo_to_wire(ef, jpeg_quality, ...)`
2. `eo_to_wire` calls `base64.b64encode(ef.jpeg_bytes).decode("ascii")`
   on the cached ~460 KB EO JPEG. Cost: 10-15 ms per call.
3. Then `gui/app.py:595-596` (before my fix) did
   `payload["eo"]["jpeg_b64"] = None` — i.e. **threw away the result
   that just took 10-15 ms to compute**. EO bytes go via the binary
   `_eo_sender` fast-path; the shared sender's EO entry is supposed
   to be metadata-only. The post-build null-out was a band-aid that
   ran AFTER the cost was already paid.
4. Plus `json.dumps(payload, default=str)` re-walks the entire dict
   (includes thermal jpeg_b64 ~70 KB, radar points/targets, fused).
   Another 5-15 ms.
5. Plus thermal JPEG is base64-encoded synchronously inside `_sender`
   every tick at `gui/sensor_bridge.py:144-146` even though
   ThermalFrame may have a cached encode (no `jpeg_bytes` cache field
   on ThermalFrame like there is on EOFrame).

Combined: ~30-50 ms of synchronous CPU work on the asyncio event
loop per WS tick. At 16 ms target period, the loop runs as fast as
it can which lands at ~10 Hz. Every other coroutine (`_eo_sender`
binary path, `_receiver`) pays the cooperative-yield tax behind the
same `await ws.send_text(...)` write lock, so the EO fast-path also
drops to ~7-9 Hz.

The producer threads keep publishing at native rates (Boson at 60
Hz, EO at 25 Hz, radar at 14-20 Hz onto FrameBus). The GUI's
per-panel Hz counter measures WS arrivals, which is bottlenecked on
`_sender`.

### What I tried (and the user reports it didn't work)

Modified `gui/sensor_bridge.py:eo_to_wire` to accept `skip_jpeg=True`
and skip the base64 path entirely; modified `build_ws_message` to
pass `skip_jpeg=True`; removed the dead null-out from `gui/app.py`.

User restarted and FPS was still 7-9 Hz. **I did not profile a live
frame after the fix to verify the change is actually taking effect.**

### What the next agent should do

1. **Profile a live frame.** Add timing around `build_ws_message`,
   `json.dumps`, and `ws.send_text`. The user has it at 7 Hz right
   now so any timing instrumentation will be obvious.
2. **Verify my edit actually took effect** — check that
   `eo_to_wire` is being called with `skip_jpeg=True` from
   `build_ws_message`, and that the base64 branch is NOT executing.
   I might have a bug there.
3. **Top suspects after the EO fix** (in order):
   - Thermal JPEG re-encoded every tick. Add `jpeg_bytes` +
     `jpeg_quality` fields to `ThermalFrame` mirroring `EOFrame`,
     encode once on the thermal process thread, reuse in
     `thermal_to_wire`.
   - `json.dumps(default=str)` is slow on these large payloads.
     Replace with `orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)`.
     Saves another 3-8 ms per tick.
   - Move thermal to a binary fast-path identical to `_eo_sender` so
     the shared JSON only carries metadata.
4. **Stretch goal:** push thermal/EO frames out as binary WS only
   when they actually arrive on the bus (sensor-arrival cadence)
   instead of on the periodic 60 Hz schedule. The shared `_sender`
   is currently running at the periodic rate, not the
   "any-sensor-changed" rate. Drives more efficient per-panel rates.

Files I touched:
- `gui/sensor_bridge.py` — added `skip_jpeg=True` flag to
  `eo_to_wire` and used it in `build_ws_message`
- `gui/app.py` — removed the post-build null-out

### Hypothesis I didn't test

The fusion manager declares `rate=15.0Hz` on startup. If fusion is
synchronous on the WS path, that alone caps the GUI at 15 Hz. Worth
checking `fusion/fusion_manager.py` for whether its tick rate gates
anything on the WS broadcast.

---

## 3. Bug — Ctrl+C wedges chip CLI, requires power cycle

### Symptom

After a clean software shutdown (Ctrl+C → all sensors stop cleanly →
process exits), the **next launch's first CLI command (sensorStop)
gets ZERO response from the chip**. Every cfg line then times out at
the 2-second ack window: 25 lines × 2 s = 51 seconds of silence,
then the connect loop retries forever. Pulling 12 V from the AWR EVM
for 5 seconds is the only recovery.

User confirmed: cold start (after power cycle) works perfectly. ANY
soft restart wedges the chip.

### Diagnosis (research-agent verified, but fix unverified)

**pyserial on Windows toggles DTR (and on some FTDI/XDS110 driver
builds RTS) inside the underlying Win32 `CreateFile` /
`SetCommState` sequence when a Serial port is opened.** The toggle
happens BEFORE Python user code can change those line states.

On the AWR2944P EVM:
- The chip's CLI port (COM11) is a virtual UART on the XDS110 (J8
  USB).
- The XDS110's User UART CDC routes those modem-control lines
  (DTR/RTS) onto chip-side GPIOs that `mmw_demoDDM` firmware
  apparently uses as host-handshake signals.
- Each Ctrl+C → relaunch reopens COM11 → toggles those GPIOs.
- On this firmware build the chip's CLI parser then deadlocks
  inside `UART_writePolling`, which is shared by the BSS async-event
  print path. From that point on the chip CLI is silent until 12 V
  power cycle.

Confirming evidence:
- pyserial issue [#124](https://github.com/pyserial/pyserial/issues/124)
  and #488 — same DTR/RTS toggle behavior on Windows port open
- TI's mmWave SDK User Guide explicitly tells users to "close the
  CLI terminal and reopen" after power cycle — i.e. TI knows the
  CLI is sensitive to host-side line transitions
- TI mmwave-L-SDK `MMWAVE_DEMO` architecture: CLI task uses
  `UART_writePolling` and runs in the same context as the BSS
  async-event prints (the `Done\n` after sensorStop is a BSS event)
- TI E2E forum posts confirm dual-CDC virtual UART backpressure
  issues on XDS110 (TUSB3410-based)

### Earlier shutdown attempts that didn't fix it

I tried several reorderings before the agent diagnosis pointed at
DTR/RTS:

1. Single `sensorStop` on shutdown with response drain — wedged
2. Same as #1 but `sensorStop` BEFORE closing data port — wedged
3. Single-flag guard so `radar.stop()` runs exactly once per process
   exit (not 2-3 times across `_shutdown` / `atexit` /
   `finally:`) — wedged
4. CR/LF wake bytes on next-launch CLI open — chip stays silent

After all of those, the diagnosis pivoted to the DTR/RTS hypothesis.

### What I tried (and the user reports it still doesn't work)

Added `_open_cli_port()` helper in `radar/radar_manager.py`:

```python
@contextmanager
def _cm():
    ser = serial.Serial()           # un-opened constructor
    ser.port = self.cli_port
    ser.baudrate = self.cli_baud
    ser.timeout = 0.5
    ser.dtr = False                 # set BEFORE open
    ser.rts = False
    ser.open()
    try:
        yield ser
    finally:
        ser.close()
```

Replaced ALL 4 `with serial.Serial(self.cli_port, ...)` sites in
`radar/radar_manager.py` to use `with self._open_cli_port() as ser:`.

User restarted and the wedge persisted. **I did not verify the
DTR/RTS lines are actually staying low on this specific XDS110
driver.** It's possible the property setters don't fully suppress
the toggle on this Windows + XDS110 driver combination — the
behaviour is documented as driver-dependent.

### What the next agent should try, in order

1. **Verify the DTR/RTS suppression actually works.** Use a
   logic analyzer or a serial monitor (e.g. RealTerm with line
   monitoring, or `pyserial-miniterm --develop`) to watch the
   physical DTR/RTS lines on COM11 across a Ctrl+C → relaunch
   cycle. If the lines DO toggle, my fix didn't take. If they
   stay low and the chip still wedges, the cause is something else.

2. **Send a hardware UART break before the first cfg line.** Per TI
   E2E, a 250 ms break on the CLI UART forces the chip's UART RX
   state machine to reset. Add `ser.send_break(0.25)` after the
   DTR-safe open, before any other write. This is the chip-side
   equivalent of "close+reopen teraterm" that the SDK guide
   recommends.

3. **Drive nRESET via XDS110.** Ti ships `xds110reset.exe` with
   their CCS install. It pulses nRST on the AWR via XDS110 GPIO
   without touching 12 V. Wrap it in a Python helper:
   `subprocess.run(["xds110reset.exe"], ...)`. Call on launch IFF
   the first CLI command gets no response within 500 ms. Converts
   the manual power-cycle into an automatic recovery. Path
   typically: `C:\ti\ccs<version>\ccs\ccs_base\common\uscif\xds110\xds110reset.exe`

4. **Last resort:** the previous agent built (but never wired in)
   `radar_dca/ftdi_pin_holder.py` which uses pyftdi/MPSSE to hold
   FTDI modem-control lines stable across host-process restarts.
   That file is on disk, untracked. If 1-3 don't work, pull it in
   as a fallback layer.

### Hypothesis I didn't test

The user has TWO FTDI sets in `pyserial.tools.list_ports`:
- `FT9KSAL9*` → COM12-15
- `FTAYGX0X*` → COM6-9

The radar uses XDS110 not FTDI (per `config/app_config.yaml:502-503`
which the user explicitly confirmed). The XDS110 is on J8 USB. But
both FTDI cards are still plugged in — they're not the radar's
control path but they ARE on the same USB hub. **It's possible the
XDS110 is being affected by something on the FTDI cards' USB
descriptors at the hub level.** Not tested.

Files I touched:
- `radar/radar_manager.py` — added `_open_cli_port()` helper, used
  in 4 places (stop, push_profile, kick_lvds×2)
- `main.py` — added single-shot `_shutdown_done` flag guarding
  `_shutdown` / `_atexit_radar_stop` / `finally:` block so
  `radar.stop()` runs exactly once per process exit

---

## 4. Files I touched in this session

```
gui/app.py                 |  10 lines diff (FPS fix attempt)
gui/sensor_bridge.py       |  42 lines diff (skip_jpeg flag)
main.py                    |  63 lines diff (single-shot shutdown guard)
radar/composite_manager.py |   7 lines diff (radar_firmware kwarg)
radar/radar_manager.py     | 341 lines diff (DTR-safe open + shutdown reorders + diagnostics)
radar_dca/dca_pipeline.py  | 127 lines diff (notch fix + MTI replacement + PMM mode gating + artifact filter)
```

Plus new files in `tools/`:
- `diagnose_artifact.py`
- `validate_two_recordings.py`
- `debug_drone_visibility.py`
- `inject_radar_bbs.py`
- `drone_persistence_check.py`

Plan file (planning record from earlier in session):
- `C:\Users\asaf.ruf.BLUERIVERTECH\.claude\plans\keen-cooking-moth.md`

## 5. What I want the next agent to read first

In order:
1. This file
2. `docs/STATUS_FOR_RETURN.md` (May 3 status from a different prior
   agent — radar firmware context, NOT current state)
3. `radar/radar_manager.py:_open_cli_port` — the DTR-safe helper
4. `radar/radar_manager.py:stop` — the multiple shutdown reorders
   I tried and the comment block explaining why each failed
5. `radar_dca/dca_pipeline.py:_notch_harmonic_artifact` — the
   corrected every-24-bin notch with full empirical justification
6. `radar_dca/dca_pipeline.py:_publish` — PMM mode gating
7. `tools/diagnose_artifact.py` — the 6-test diagnostic that
   nailed the artifact period and severity

## 6. External review by Gemini (the user ran it, asked me to comment)

The user pasted a Gemini review of my work (Gemini did not have full
repo access — it was working from limited file fragments and made
several wrong assumptions). My honest assessment:

### Gemini's points that hold up

1. **Foil-on-body sanity test (NEW, not in any of my prior work).**
   Tape aluminum foil to the DJI FPV's body, hover at 10 m, watch
   the RD map. If invisible, the failure is upstream of any
   detection algorithm — data integrity, gain, or chirp timing.
   This is the single best concrete test in Gemini's review and
   I should have suggested it. Run this before any more algorithm
   tuning.

2. **DCA1000 packet sequence audit.** Verify
   `radar_dca/data_port.py` and `radar_dca/dca_pipeline.py` track
   dropped packets and log gaps in the DCA1000 sequence numbers.
   I never audited this. Even occasional drops break PMM phase
   continuity. ~10 minutes to check.

3. **Single-TX test mode for PMM (one-shot).** Push a cfg variant
   with only 1 TX active. That keeps the full PRF (30478 Hz)
   instead of dividing by N_TX=4 (giving 7619 Hz per VA after DDMA
   unfold). 4× more chirps in the slow-time spectrum at the cost
   of losing MIMO array gain and angle resolution. Worth testing
   even if not the production path — if PMM works in single-TX and
   not in DDMA, the discriminator is "we need higher PRF" and the
   answer is to redesign the cfg around that.

### Gemini's points that are wrong (didn't read the code carefully)

1. **"Implement static clutter removal"** — already there, refined.
   The mean-subtract MTI was REMOVED on 2026-05-06 because it was
   killing hovering drones (this is documented in the diff for
   `radar_dca/dca_pipeline.py:_process_frame`). Replaced with a
   narrow Doppler-DC notch (DC ± 1 bin) in `_stage3_range_doppler`
   that preserves prop-wash micro-Doppler.

2. **"Coherent integration over 4-8 frames will help"** — false on
   this hardware/target combination. We tested N=2, 4, 8 with
   `tools/sim_more_chirps.py`. Drone advantage **decreased** with
   N because RPM jitter spreads blade-pass energy across freq
   bins faster than coherent integration can recover it. The
   "more chirps = better" intuition only holds for ideal stable
   rotors; real autopilots constantly correct RPM.

3. **"You're using TDM-MIMO which lowers PRF"** — wrong. We use
   DDMA-MIMO (cfg has `ddmPhaseShiftAntOrder`). All 4 TX fire
   simultaneously with phase coding. Effective PRF per VA after
   unfold is PRF/N_TX = 7619 Hz, same number TDM would give but
   without losing TX time.

4. **"Velocity aliasing of blade tips at 150 m/s"** — Gemini
   conflated blade-tip velocity (which does alias massively) with
   blade-pass frequency (which is what PMM detects). Blade-pass
   for a 6000-22000 RPM 3-blade DJI FPV is 300-1100 Hz, well
   inside our 7619 Hz per-VA Nyquist. The blade-tip aliasing
   doesn't matter for PMM detection.

5. **"Reduce bandwidth to 1 GHz for coarser range bins"** —
   Gemini misread the cfg. Slope is 8.883 MHz/μs over ~6.4 μs
   ADC window → sampled BW ~57 MHz → range bin **already 2.6 m**.
   We're already on a coarse bin. There's no 5 GHz BW to reduce.

6. **"Verify ADC format"** — verified extensively. `adcCfg 2 0` =
   real ADC mode (NOT complex IQ). Cube reshape is RX-major
   non-interleaved. The debug script `tools/debug_drone_visibility.py`
   confirmed the drone signal IS at the expected close-range bins
   in the RD map.

### Gemini's framing that's mostly right but applies less to us

- "You can't code your way out of bad physics" — correct general
  principle. But we've actually done the bench-side physics work
  the previous agent didn't (artifact diagnosed at every-24-bin
  3.75 MHz, MTI replaced, range axis recovered). The remaining
  PMM-vs-clutter problem is genuinely a discrimination problem,
  not a sensitivity one — close-range drones ARE visible in the
  RD map, they just look identical to operators / fans / breathing.

### What I'd take from Gemini's review into the next session

In priority order:

1. **Run the foil-on-body test at 10 m** (5 minutes, no code).
   Gives you a clean Yes/No on whether the radar can see the
   drone's airframe at all without prop modulation help.
2. **Add packet-sequence-gap logging** in `radar_dca/data_port.py`
   if it isn't there. If gaps exist, PMM cannot work no matter
   what the algorithm is.
3. **Single-TX cfg variant** as a PMM-only test profile. If PMM
   becomes detectable with a single TX firing at full 30478 Hz
   PRF, the long-term answer is a chirp redesign.
4. Ignore the rest of Gemini's recommendations — they are based on
   misreadings of the cfg or didn't account for work already done.

## 7. Things I would tell the user about my own mistakes

- I told the user "5-10 s for cfg push" when in the wedged state it
  was 51 s. I knew the per-line timeout was 2 s but didn't multiply
  by the 25 cfg lines. Bad time estimate.
- I claimed "drone-fly: 96% detection at 2.6 m" without first
  checking the per-window hit rate against pre-drone clutter. The
  number was technically true but it was "any close-range mover at
  2.6 m" not "the drone". The user called this out and was right to.
- I twice applied a fix and asked the user to test without first
  verifying the fix actually took effect on a running system. Both
  times the user reported the fix didn't work and I had to backpedal.
- I told the user to "unplug J8 XDS110 USB" early in the session as
  the EMI test. The user's `config/app_config.yaml:502-506` is
  explicit that COM10/11 (the radar's control + data ports) are
  routed through J8 XDS110 in this build — I should have read the
  config first. The user's J8-unplug then took the radar offline,
  costing 30 minutes.
- I suggested launching multiple "waves" of investigation agents
  for the artifact root-cause when the user just wanted the
  artifact gone. I burned a lot of context on validation theatre.
