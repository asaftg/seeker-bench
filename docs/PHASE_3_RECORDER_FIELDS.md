# Phase 3 — recorder field spec

What goes into a flight/bench recording so post-flight analysis
isn't guesswork.

A recording is a directory:

```
recordings/<timestamp>_<label>/
  meta.yaml                  # mode + cfg + chip params (this doc)
  dca_raw.bin                # the raw-ADC UDP byte stream as captured
  dca_index.csv              # per-packet index (row → byte offset, ts, seq)
  bus_radar.jsonl            # RadarFrame messages we published, one per line
  bus_eo.jsonl               # optional, when EO is recording in parallel
  bus_thermal.jsonl          # optional
  bus_fusion.jsonl           # optional
  truth.csv                  # optional ground-truth (drone GPS log, marker board, etc.)
  notes.md                   # free-form human notes
```

Everything except `dca_raw.bin` is < 1 MB; the binary stream is
~hundreds of MB per minute (PRF × n_rx × n_samples × 4 bytes ≈
60 MB/s for awr2944P_aa.cfg). Plan disk accordingly.

## meta.yaml — the canonical "what was this?" file

```yaml
# Phase-3 recording metadata. Written once at recording start.
schema_version: 1
recorded_at: "2026-04-29T14:33:21+03:00"
operator: "asaf"
label: "driveway_dji_fpv_50m"

mode: aa                          # stock | ag | aa
backend: dca_manager              # stock_radar_manager | dca_manager
profile_name: awr2944p_aa         # matches RadarFrame.profile

# Chip + cfg state captured at recording start
awr_cfg_path: "radar/cfg/awr2944P_aa.cfg"
awr_cfg_sha256: "ab12...."        # so post-flight knows the cfg was tampered with or not
awr_firmware:
  rf_fw: "02.05.04.00.23.05.16"
  rf_fw_patch: "02.06.03.01.25.05.22"
  mmwave_link: "02.04.06.17"
  mmwave_sdk: "04.07.02.01"
awr_cli_port: "COM10"
awr_cli_baud: 115200

dca:
  control_endpoint: "192.168.33.180:4096"
  data_endpoint:    "192.168.33.30:4098"
  fpga_version: "<DCA1000EVM_CLI_Control fpga output>"   # if obtainable

# Frame dimensions — these determine .bin parser shape
frame_dims:
  n_chirps: 2304        # chirpCfg 0 5 (6 burst chirps) × numLoops 384 = 2304 chirps/frame
  n_rx: 4
  n_samples: 384
  bytes_per_sample: 4   # I + Q, int16 LE each
  bytes_per_frame: 14155776     # = 2304 × 4 × 384 × 4
  prf_hz: 10000.0
  chirp_period_s: 100e-6
  range_resolution_m: 0.04
  max_range_m: 250.0

# Pipeline settings at recording time
pipeline:
  pmm_band_low_hz: 50.0
  pmm_band_high_hz: 500.0
  pmm_threshold_db: 6.0

# Network capture sanity (from the OS)
capture:
  host_nic: "Realtek PCIe GbE Family Controller"
  host_ip: "192.168.33.30"
  mtu: 9000
  socket_recv_buffer_bytes: 33554432   # 32 MB; needed to avoid drops at 600 Mbps

# Targets known to be in the scene (free text + structured)
scene:
  background: "open driveway, no foliage, light traffic ~30 m behind sensor"
  weather: "clear, ~15 °C, light wind from W"
  targets:
    - id: "DJI_FPV_1"
      type: "drone"
      model: "DJI FPV"
      planned_range_m: [10, 50, 100, 200]
      planned_speed_mps: [0, 8, 30, 40]
    - id: "human_1"
      type: "human"
      planned_range_m: [5, 20]
      planned_speed_mps: [0, 1.5]
```

## dca_index.csv — per-packet index

So we can seek into `dca_raw.bin` without re-parsing 60 MB/s from
the start.

| col | dtype | meaning |
|---|---|---|
| `ts_host_ns` | uint64 | host monotonic-ns time the packet was received |
| `byte_offset` | uint64 | offset in `dca_raw.bin` where this packet's payload begins |
| `payload_len` | uint32 | bytes in this packet's payload (post-10-byte header strip) |
| `seq_num` | uint32 | DCA's 4-byte sequence number from the packet header |
| `chunk_offset` | uint64 | the 6-byte "byte count" field from the DCA header |

One row per UDP packet. ~30k rows/s; OK as CSV up to a few minutes,
switch to parquet beyond that.

## bus_radar.jsonl — what we published

One `RadarFrame` per line, JSON-encoded via the existing
`common.frames.RadarFrame.to_json()` (or pickle, whichever the bus
recorder is already using). Lets us replay the exact target list
that fed the fusion + GUI without re-running the radar pipeline.

If we're running `aa` mode, look for:
- `connected: true` → pipeline alive
- `targets[*].source: "pmm"` → PMM detector hit
- `targets[*].confidence` ≥ 0.5 → above threshold
- `targets[*].pos_y_m` → range in meters (boresight assumed until Capon BF lands)

## truth.csv — optional ground-truth

When we have a drone with a GPS logger, dump it here:

| col | dtype | meaning |
|---|---|---|
| `ts_unix_ns` | uint64 | logger time, GPS-disciplined if possible |
| `lat_deg`, `lon_deg`, `alt_m` | float64 | drone position |
| `vel_n_mps`, `vel_e_mps`, `vel_d_mps` | float32 | NED velocity |
| `bearing_deg` | float32 | drone heading (for prop visibility analysis) |

Same idea applies to vehicle/human trials with a marker board or
total-station truth — just adapt the columns. Keeping the file name
constant makes the analysis scripts simpler.

## What goes in `notes.md`

Free-form. The kind of thing that's painful to remember a week
later: which prop you flew, where you stood, whether the gimbal was
locked, what failed and why we stopped, did you forget to plug the
12 V back in after a battery swap. Take 30 s to type it before you
move on.

## Recorder implementation notes

- `dca_raw.bin` is just a write-through of every UDP payload (after
  the 10-byte DCA header is stripped). Capture in a separate
  thread with a queue → file writer; do NOT block the listener
  (the listener thread must always be ready to recv).
- `dca_index.csv` rows are written in the same writer thread,
  one per packet, append-mode.
- `meta.yaml` is written once at recording start. If anything
  changes mid-recording (mode switch, cfg push), close the
  recording and start a new one. Don't try to handle mid-stream
  schema changes.
- `bus_*.jsonl` is written by the existing bus recorder; nothing
  Phase-3-specific here, just make sure it's enabled when DCA
  mode is on.

## Acceptance for the recorder

A recording is "good" if, given only the directory:
1. `meta.yaml` parses and tells us mode + chip + dims.
2. `dca_raw.bin` size matches `frame_count × bytes_per_frame ± 1 frame`.
3. `dca_index.csv` row count × payload_len mean ≈ `len(dca_raw.bin)`.
4. We can run `python -m radar_dca.replay <dir>` and see the same
   `RadarFrame` sequence the live run produced (within numerical
   tolerance for floating-point ops).

(`radar_dca/replay.py` doesn't exist yet — it's a one-day task once
real `.bin` files are in hand.)
