"""POC #1b: FULL-FOV BEAMFORMING SCAN using the 4-RX azimuth array.

This is the CORRECT version. The chip CFAR in this branch uses only the
4 RX channels for AoA — we'll do the same. Beamform by steering across a
range of azimuth angles, find the angle where the drone pops above noise.

For each beam direction:
  - Phase-shift each RX by 2*pi*n*0.5*sin(az) (standard ULA beamforming)
  - Sum the 4 RX with these phase corrections
  - Compute slow-time spectrum at every range bin
  - Find max signal across band per range bin

Drone should appear as a peak at its true azimuth (~0 deg, near boresight)
that is NOT present at other beam directions.

Then ALSO check: at the drone's actual azimuth, does the drone bin (rb=4-17)
pop above clutter range bins?
"""
import sys, io, json, importlib.util, numpy as np, scipy.fft as sfft, os

N_CHIRPS, N_RX, N_SAMPLES = 768, 4, 192
PRF_HZ = 30478.51264858275
RANGE_RES_M = 2.638
N_FFT = 1024
BYTES_PER_FRAME = N_CHIRPS*N_RX*N_SAMPLES*2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)
WIN_SLOW_FULL = np.hanning(N_CHIRPS).astype(np.float32)  # full N_CHIRPS, not divided by TX
DOP_BIN_HZ_FULL = PRF_HZ / N_FFT  # using full PRF since we're not splitting by TX

AIR = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-58-59_radar.bin"
JSONL = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\drone test airborne 1.jsonl"

def load_frame(path,idx):
    with open(path,"rb") as f: f.seek(idx*BYTES_PER_FRAME); buf=f.read(BYTES_PER_FRAME)
    raw=np.frombuffer(buf,dtype=np.int16)
    cube=(raw.reshape(N_CHIRPS,N_RX,N_SAMPLES).transpose(0,2,1).astype(np.float32))
    cube-=cube.mean(axis=1,keepdims=True); return cube

def beamformed_rd(path, idx, az_deg):
    """Range-Doppler at one beam direction using 4-RX phase steering.
    Returns (n_doppler, n_range) magnitude in dB.

    Phase per RX: 2*pi * n * 0.5 * sin(az_rad), n=0,1,2,3
    """
    cube = load_frame(path, idx)  # (n_chirps, n_samples, n_rx)
    rc = sfft.rfft(cube*HANN_FAST[None,:,None], axis=1, workers=2).astype(np.complex64)
    rc[:, 1:-1, :] *= 2.0
    rc -= rc.mean(axis=0, keepdims=True)  # MTI
    # rc shape: (n_chirps, n_range, n_rx)
    # Apply azimuth steering across RX: weight by conj(steering vector)
    az_rad = np.deg2rad(az_deg)
    n = np.arange(N_RX)
    sv = np.exp(1j * 2*np.pi * n * 0.5 * np.sin(az_rad)).astype(np.complex64)  # (4,)
    # Beamform: sum over RX with conj(sv)
    rc_bf = (rc * np.conj(sv)[None, None, :]).sum(axis=2)  # (n_chirps, n_range)
    # Slow-time FFT
    spec = np.fft.fft(rc_bf*WIN_SLOW_FULL[:,None], n=N_FFT, axis=0)
    spec_shifted = np.fft.fftshift(spec, axes=0)
    return 20*np.log10(np.maximum(np.abs(spec_shifted), 1e-30))

# Load drone position
base_ts = None; detections = []
with io.open(JSONL,"r",encoding="utf-8") as fh:
    for line in fh:
        try: r=json.loads(line)
        except: continue
        if r.get("channel") != "radar/frame": continue
        ts = r.get("ts_ns")
        if base_ts is None: base_ts = ts
        for t in (r.get("msg") or {}).get("targets") or []:
            if t.get("tid") != 47: continue
            t_rel = (ts - base_ts)/1e9
            x=t.get("x",0); y=t.get("y",0); z=t.get("z",0); rm=(x*x+y*y+z*z)**0.5
            az = np.degrees(np.arctan2(x, y))
            detections.append((t_rel, rm, az))

filtered = [d for d in detections if 28 <= d[0] <= 40 and d[1] >= 16]
sampled = []; last_t = -1
for d in filtered:
    if d[0] - last_t < 0.05: continue
    last_t = d[0]; sampled.append(d)
FPS_EST = 14.0

# Pick a few drone frames to test
test_frames = [(int(d[0] * FPS_EST), int(round(d[1]/RANGE_RES_M)), d[2])
               for d in sampled[::10]]  # every 10th drone detection
print("Testing %d drone frames" % len(test_frames))
print("\nFor each frame: scan beam from -50 to +50 deg in 10 deg steps.")
print("At each beam direction, compute: max_signal at the drone's range bin, plus max_signal at clean bins.")
print("\nExpected: at beam_az = drone_az, the drone bin shows a peak. At other beams, drone bin is dimmer.")

az_grid = np.arange(-50, 51, 10)
n_dop = N_FFT
freqs_hz = (np.arange(n_dop) - n_dop // 2) * DOP_BIN_HZ_FULL
# Skip DC and chip-spur freqs (chip spur is around the same Doppler in full PRF? need to check)
# Actually with full PRF (no DDMA), Doppler bin spacing is different. PRF=30478, N_FFT=1024 → 29.8 Hz/bin, Nyquist ±15239 Hz
# That's huge! Operator body Doppler ~1 m/s = 510 Hz, blade tip ~75 kHz = folded
# The "chip spur" position depends on what we're seeing here
print("\nWith full PRF (no DDMA split), Nyquist = +/-%.0f Hz, bin = %.1f Hz" % (
    n_dop/2 * DOP_BIN_HZ_FULL, DOP_BIN_HZ_FULL))

pos_band = (freqs_hz > 100) & (freqs_hz < 5000)  # exclude DC, focus on first 5kHz

for fr_idx, (frame_idx, drone_rb, drone_az) in enumerate(test_frames[:4]):
    print("\n=== Frame %d (t=%.1fs), drone at rb=%d (%.1fm), drone_az=%.1fdeg ===" % (
        frame_idx, frame_idx/FPS_EST, drone_rb, drone_rb*RANGE_RES_M, drone_az))
    print("  beam_az  drone_bin_max  rb14_max  rb40_max  rb70_max  drone-rb14_diff")

    for az in az_grid:
        rd = beamformed_rd(AIR, frame_idx, az)
        # Mask freqs
        rd_masked = rd.copy()
        rd_masked[~pos_band, :] = -1e9
        drone_max = rd_masked[:, drone_rb].max()
        rb14_max = rd_masked[:, 14].max()  # different range, not chip
        rb40_max = rd_masked[:, 40].max()
        rb70_max = rd_masked[:, 70].max()
        diff = drone_max - rb14_max
        marker = ""
        if abs(az - drone_az) <= 5: marker = " <-- AT DRONE AZ"
        print("    %+5.0f    %7.1f    %7.1f   %7.1f   %7.1f    %+7.1f%s" % (
            az, drone_max, rb14_max, rb40_max, rb70_max, diff, marker))
