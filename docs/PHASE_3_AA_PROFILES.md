# Phase 3 — A/A profile catalog

The current `radar/cfg/awr2944P_aa.cfg` is profile **AA-1** — a
single-frame compromise between long-range integration gain and
fast-target observability. This doc lists candidate profiles we may
need depending on what the driveway test reveals.

## Constraint summary

- AWR2944P — 4 RX, 3 TX (DDM-MIMO virtual array of 12 elements).
- 77 GHz, ~4 GHz BW available; we use 30 MHz/μs slope × ~52 μs ramp.
- Range-Doppler ambiguities and the SDK chirp-table limits cap us
  at ~512 chirps per frame at this slope before the DSP overrun
  kicks in.
- Antenna main lobe (3 dB): ±30° az / ±15° el. Beyond that we eat
  sidelobes and false-AoA.
- DCA1000 bandwidth: ~600 Mbps sustained. With 4 RX × 384 samples ×
  4 bytes/sample × PRF, our budget allows PRF ≤ ~50 kHz at full
  cube. We're well under that.

## Profile AA-1 — single-frame balance (current default)

What `awr2944P_aa.cfg` ships today.

| param | value | notes |
|---|---|---|
| numLoops | 384 chirps | +4.8 dB coherent gain vs stock 128 |
| framePeriod | 200 ms | 5 Hz frame rate |
| ramp slope | 30 MHz/μs | ~4 cm range bin |
| range bins | 384 samples | up to ~250 m |
| max unambig velocity | ±9.6 m/s (DDM, after unfolding) | borderline for fast FPV |
| FoV | ±25° az / ±15° el | antenna main lobe |
| compressionCfg | 0 (off) | avoids the streaming-controller fault |
| lvdsStreamCfg | on | raw ADC to DCA |

Strengths: simplest, best for a Shahed-class slow target at long
range. The ~20 ms slow-time window resolves blade-pass frequencies
in the 100-500 Hz range cleanly.

Weaknesses: 5 Hz frame rate is marginal for tracking a 40 m/s FPV at
short range — between frames the drone moves 8 m, which is 200
range bins. PMM still fires per-frame, but the tracker handoff is
ugly.

## Profile AA-2 — fast-target single-frame

For when the FPV is closer than ~80 m and we care more about update
rate than peak SNR.

| param | value | notes |
|---|---|---|
| numLoops | 192 chirps | -3 dB gain vs AA-1, but |
| framePeriod | 80 ms | 12.5 Hz frame rate |
| max unambig velocity | wider | more headroom for fast FPV |
| everything else | as AA-1 |

Trades 3 dB of integration gain for 2.5× faster frames. PMM detector
still gets a 10 ms slow-time window which is enough for FPV blade
rates of 200-500 Hz (>= 2 cycles).

## Profile AA-3 — long-integration sky-cut

For when we know we're searching empty sky and want every dB of SNR.

| param | value | notes |
|---|---|---|
| numLoops | 512 chirps | +6 dB vs stock 128 |
| framePeriod | 333 ms | 3 Hz frame rate |
| ramp slope | reduce to 25 MHz/μs | trade range res for SNR (~5 cm bin) |
| FoV | ±15° az / ±10° el | only sky, narrow beam |
| zero-Doppler retention | OFF | we don't expect static targets in sky |

Best for the "is there a drone at 200 m?" question. Useless against
a fast-moving target — at 3 Hz the drone has moved 13 m between
frames at 40 m/s.

## Profile AA-4 — staggered PRF (Doppler unfolding)

For when AA-1 PMM hits show the body Doppler is aliased and we
can't tell the prop sidebands apart from the body return.

Cycling subframes:
- subframe 0: PRF1 = 10 kHz, 192 chirps
- subframe 1: PRF2 = 13 kHz, 192 chirps  (coprime ratio)

The CRT (Chinese Remainder Theorem) unfolds true Doppler from the
two PRF wrappings. Adds complexity in `dca_pipeline.py` but lets us
classify FPV at high body Doppler without ambiguity.

This is the profile we'd graduate to if the field test shows
"PMM fires but blade_freq is in the wrong band."

## Profile AA-5 — PMM-only, no CFAR

For pure-physics drone classification on every range bin, ignoring
the standard detection chain. Equivalent to running
`scan_range_bins` over the whole range-time matrix per frame.

Useful when the drone RCS is below CFAR threshold (small FPV at
long range) but the propeller signature is strong enough to detect
on its own. mmHawkeye operates close to this regime.

Implementation: same cfg as AA-1; the change is host-side
(`pipeline_profile=pmm_only` in the YAML). The pipeline skips the
CFAR-then-PMM gating and PMM-scans every range bin. Cost is CPU,
not chip-side.

## Decision tree for the driveway test

```
DJI FPV at <80 m, hovering or slow:
    AA-1 default. Should classify cleanly.

DJI FPV at <80 m, sprinting:
    AA-2 if AA-1 misses or shows blade_freq in the wrong band.

DJI FPV at >150 m:
    AA-3 if FPV is hovering or slow.
    AA-1 + maybe AA-5 if it's moving (frame rate matters more).

Shahed-class at any range:
    AA-1 is fine. Slow targets, long body, plenty of RCS.

Body-Doppler aliased / blade_freq in wrong band:
    AA-4 to unfold. Implement when this fault mode actually shows up.
```

## How profiles map to code

Each profile is a `radar/cfg/awr2944P_*.cfg` plus optional pipeline
overrides in `app_config.yaml`. To add AA-2 you'd:

1. Drop `radar/cfg/awr2944P_aa2.cfg` with the changes above.
2. Add to `radar.cfg_paths` in `app_config.yaml`.
3. Add `aa2` to `radar/backend_factory.py`'s `_BACKEND_OVERRIDES`.
4. The GUI mode picker exposes it as a fourth option.

We're not paying that cost up front — only profile AA-1 ships in v1
because that's the only one we can validate on the bench tomorrow.
The others get built when their fault mode is observed in the wild.
