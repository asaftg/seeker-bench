# Session state — 2026-05-07 (radar agent #2)

User authorized fully autonomous, multi-day, infinite-budget operation.
Wake-ups are scheduled for 2026-05-08 01:20, 12:00, 2026-05-09 08:00,
and 2026-05-10 08:00 PDT.

## TL;DR

We solved drone_fly. The propeller signature is at **1374 Hz blade-pass**
at the drone's hover range (bins 0-1, 0-2.6 m). Detection rate 100%
across the recording with SNR 13.8-16.1 dB and peak-frequency stability
MAD 5 Hz.

Background: zero false alarms.

Airborne1: NOT detected at the FPV blade-pass band [300, 2200] Hz at
any range or time. There IS a 64.5 Hz constant tone at close range
that is not the FPV — possibly a different drone model, a fan, or
something else.

The path to detection went through finding and fixing **two layers
of bugs / mis-interpretations** that the prior agent (and the user's
Gemini chat) all missed:
1. The live pipeline reshape was wrong (wire is sample-major, not RX-major).
2. Only RX0 and RX1 carry data (firmware has 2 LVDS lanes wired).
3. The "every-24-bin chip artifact" notch was a band-aid for the
   reshape bug — there is no real chip artifact, just bytes getting
   scrambled.
4. The DDMA-fold spurs at PRF/12 and PRF/6 contaminate the slow-time
   spectrum at the chirp-rate axis, but slot-0 decimation (every 6th
   chirp) bypasses them cleanly.

## Validated detection (drone_fly)

```
=== Per-5s-window: hit>60%, pf_MAD<50Hz, snr_max>8dB ===

drone_fly (514 frames):
    t_s    bin   rng_m    hit%   pf_med     pf_mad   snr_max
    0      0     0.0      100    1374.1     5.0      15.84
    0      1     2.6      100    1371.7     7.5      13.81
    5      0     0.0      100    1374.1     5.0      15.77
    5      1     2.6      100    1374.1     5.0      14.59
    10     0     0.0      100    1374.1     5.0      16.06
    10     1     2.6      100    1374.1     9.9      14.21
    15     0     0.0      100    1374.1     39.6     15.62

background (996 frames): NO DETECTIONS
airborne1 (500 frames):  NO DETECTIONS
```

## Code added

```
tools/diagnose_pmm/__init__.py                     (empty)
tools/diagnose_pmm/_common.py                       (BUGGY — uses wrong reshape)
tools/diagnose_pmm/_common_fixed.py                 (CORRECT layout, RX0+RX1 only)
tools/diagnose_pmm/01_seq_audit.py                  (packet-drop audit; PASS)
tools/diagnose_pmm/04_spectrogram_grid.py           (uses BUGGY common)
radar_dca/herm_detector_v2.py                       (slot-0 blade-pass detector)
radar_dca/herm_replay_v2.py                         (offline replay over a recording)
runs/seq_audit/                                     (per-rec packet-drop CSV+PNG)
runs/correct_layout/static_scene/                   (static range scene PNGs)
runs/correct_layout/close_range_spectra/            (spectra at bins 1-5)
runs/correct_layout/slot0_decimated/                (slot-0 decimated spectra grid)
runs/herm_v2/drone_fly_6db/                         (validated detection results)
runs/herm_v2/drone_test_background_6db/
runs/herm_v2/drone_test_airborne_1_6db/
runs/herm_v2/drone_test_airborne_1_4db/             (relaxed threshold attempt)
docs/SESSION_2026-05-07_radar2.md                   (this file)
```

No live pipeline (`radar_dca/dca_pipeline.py`) code modified. No git
commits.

## The bug, explained

The chip dumps real-ADC int16 samples in **sample-major, RX-interleaved**
layout per chirp:
```
chirp_bytes = [I_RX0_s0, I_RX1_s0, 0, 0, I_RX0_s1, I_RX1_s1, 0, 0, ...]
```
where the trailing zeros are RX2/RX3 padding (only 2 LVDS lanes are
wired between AWR2944P and DCA1000, per
`firmware/mmw_demoDDM_patched/mss/mmw_lvds_stream.c:234` setting
`lvdsLaneEnable=0x3U`).

The live pipeline at `radar_dca/dca_pipeline.py:541-545` reshapes the
buffer as:
```python
real_cube = (
    raw.reshape(d.n_chirps, d.n_rx, d.n_samples)   # WRONG
       .transpose(0, 2, 1)
       .astype(np.float32)
)
```
This treats the wire as **RX-major** (`[RX0_s0, RX0_s1, ..., RX0_s191,
RX1_s0, ..., RX3_s191]`), which it isn't. The reshape produces
scrambled "RX" channels each containing time-domain slices from
different parts of the actual chirp, multiplexed with the all-zero
RX2/RX3 padding.

The correct reshape is:
```python
real_cube = (
    raw.reshape(d.n_chirps, d.n_samples, d.n_rx)
       [:, :, :2]                                  # only RX0 + RX1
       .astype(np.float32)
)
```

When the buggy reshape is FFT'd in range, the period-4 zero pattern
in the int16 buffer creates spurs at multiples of `n_samples/8 = 24`
range bins. The prior agent chased this spur and built a 24-bin
notch (radius ±6) at `dca_pipeline.py:_notch_harmonic_artifact:552`
to "kill the chip artifact." With correct parsing, no such artifact
exists and the notch should be removed entirely. Critically: the
prior notch was zeroing real data, including the drone's expected
range bins around 60-80 m and 130-145 m.

## Why slot-0 decimation works

The DDMA scheme uses 6 chirp slots per loop (`chirpCfg 0 5`) with all
4 TX active each slot (`txEnableMask=15`). The DDMA phase code
(`ddmPhaseShiftAntOrder 0 2 3 1`) applies different phase increments
per TX per slot to encode the 4 TX into separate Doppler regions.

A naive slow-time FFT over all 768 chirps mixes the 4 TX phase-coded
copies plus their fold replicas, contaminating the spectrum with
spurs at PRF/4, PRF/6, and PRF/2 (which I observed empirically at
~119-124 dB).

**Slot-0 decimation** (taking only chirps at slot 0 of each loop, i.e.
chirps `[0, 6, 12, ..., 762]`) gives 128 slow-time samples at
PRF/6 = 5080 Hz. At slot 0, all 4 TX are at phase 0 (per the DDMA
code), so the signal is a clean coherent sum of all 4 TX with NO
phase modulation. The DDMA fold spurs vanish. Per-VA Nyquist is
2540 Hz, comfortably above the FPV blade-pass band of [300, 1500] Hz.

This sidesteps the need for explicit DDMA un-mixing entirely, at the
cost of a 6× reduction in slow-time samples per frame. For HERM
detection we have plenty of resolution: 5080 / 128 = 39.7 Hz per
slow-FFT bin, refined to ~5 Hz per bin with 1024-point zero-pad.

## Why airborne1 is harder than drone_fly

Drone_fly: drone at 5 m hover, body return strong (range bin 1-2),
prop modulation rides on it, easy detection.

Airborne1: drone is reportedly at 30-70 m. RCS of an FPV airframe
at that range, with only 2 RX coherently summed, is below the slow-time
noise floor at our chirp PRF. The body return at bin 11-27 isn't
visible; without a body return, prop modulation has nothing to
modulate. Result: nothing for the detector to fire on.

The 64.5 Hz constant tone at airborne1 close-range bins (0-3) at
SNR 18-23 dB with MAD=0 is **not** the FPV — it's something
specific to that recording site (possibly a fan, possibly a different
slow-rotating object, possibly a chip artifact specific to airborne1's
hardware state). 64.5 Hz blade-pass = 1290 RPM × 3 blades = below
typical FPV idle. It is too low and too stable to be the FPV.

## Open questions for next sessions

1. **Why is airborne1 silent at the FPV blade-pass band?** Was the
   drone really there? At what range? Was it the same DJI FPV or a
   different drone? Check the JSONL live-detection log timestamps
   against operator notes if available.
2. **What is the 64.5 Hz tone in airborne1?** A fan, a vibrating
   structure, a non-FPV drone, or a chip artifact?
3. **Can we detect airborne1's drone with longer integration** (sum
   across 10 frames instead of 1, then look for the comb)?
4. **The live pipeline still uses the broken reshape.** Fix in
   `radar_dca/dca_pipeline.py:_stage1_range_fft` and remove the
   24-bin notch at `_notch_harmonic_artifact`. Document the change.

## Recommended fix to live pipeline (NOT applied)

```python
# In radar_dca/dca_pipeline.py:_stage1_range_fft
# OLD (lines 541-545, BUGGY):
real_cube = (
    raw.reshape(d.n_chirps, d.n_rx, d.n_samples)
       .transpose(0, 2, 1)
       .astype(np.float32)
)
# NEW (CORRECT):
real_cube = (
    raw.reshape(d.n_chirps, d.n_samples, d.n_rx)
       [:, :, :2]                # only RX0 + RX1; RX2/RX3 are zero-padding
       .astype(np.float32)
)

# And REMOVE the 24-bin notch at _notch_harmonic_artifact (line 552):
# def _notch_harmonic_artifact(self, range_cube): pass
# (the artifact does not exist with correct parsing)
```

The user must apply these fixes manually after reviewing this session.
The fix is a ONE-LINE change to the reshape plus a no-op to the notch.

## Wake-up policy update

The 2026-05-08 01:20 PDT wake-up was originally pointed at
"implement DDMA un-mix and run per-VA spectrogram." That's now obviated
— slot-0 decimation works. The wake-up prompts have been updated to
read SESSION_2026-05-07_radar2.md (this file) first and pivot to:
- Verify reproducibility of the drone_fly detection.
- Investigate the airborne1 negative result (longer integration,
  cross-recording differential, body-Doppler search).
- Write the final report.
