"""POC #3: TRACK-BEFORE-DETECT.

Hypothesis: a weak drone signal at +6 dB excess per frame is hard to detect.
But if we sum the drone's signal along its trajectory over N=140 frames,
the cumulative signal is much stronger than any random walk.

Test:
  Use the within-recording median residuals from POC #2 (chip artifacts killed).
  For each candidate "linear radial track" (start_rb, velocity), sum the
  residual at (rb, frame) along the track.

  Compare:
    - The drone's TRUE track (chip-CFAR-derived rb_t over time)
    - Random clutter tracks (random start_rb + random velocity)
    - Stationary tracks (fixed rb across all frames)

Drone's true track should produce a track-sum significantly higher than
random or stationary tracks — IF the drone signal is consistently positive.
"""
import sys, io, json, importlib.util, numpy as np, scipy.fft as sfft, os

spec_d = importlib.util.spec_from_file_location("ddma", r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\radar_dca\__pycache__\ddma.cpython-311.pyc")
ddma_mod = importlib.util.module_from_spec(spec_d); spec_d.loader.exec_module(ddma_mod)
ddma_unfold = ddma_mod.ddma_unfold

N_CHIRPS, N_RX, N_SAMPLES = 768, 4, 192
PRF_HZ = 30478.51264858275; N_TX = 4; N_FFT = 1024; N_VA = 16
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS*N_RX*N_SAMPLES*2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)
WIN_SLOW = np.hanning(N_CHIRPS//N_TX).astype(np.float32)
DOP_BIN_HZ = (PRF_HZ/N_TX) / N_FFT

AIR = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-58-59_radar.bin"
JSONL = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\drone test airborne 1.jsonl"

def load_frame(path,idx):
    with open(path,"rb") as f: f.seek(idx*BYTES_PER_FRAME); buf=f.read(BYTES_PER_FRAME)
    raw=np.frombuffer(buf,dtype=np.int16)
    cube=(raw.reshape(N_CHIRPS,N_RX,N_SAMPLES).transpose(0,2,1).astype(np.float32))
    cube-=cube.mean(axis=1,keepdims=True); return cube

def coh_mag_db_full(path, idx):
    cube=load_frame(path,idx)
    rc=sfft.rfft(cube*HANN_FAST[None,:,None],axis=1,workers=2).astype(np.complex64)
    rc[:,1:-1,:]*=2.0; rc-=rc.mean(axis=0,keepdims=True)
    virtual=ddma_unfold(rc); n_per_tx=virtual.shape[0]
    slow=virtual.reshape(n_per_tx,virtual.shape[1],N_VA)
    spec=np.fft.fft(slow*WIN_SLOW[:,None,None],n=N_FFT,axis=0)
    coh = np.fft.fftshift(spec.sum(axis=2), axes=0)
    return 20*np.log10(np.maximum(np.abs(coh), 1e-30))

# Phase 1: build "max-Doppler" image — at each (frame, range_bin) cell,
# what's the max signal across all Doppler bins (after masking chip freqs)?
# This is the track-before-detect input.

n_total = os.path.getsize(AIR) // BYTES_PER_FRAME
print("Recording: %d frames" % n_total)

# Process every frame in 28-40s window (frames 392-560)
FR_LO, FR_HI = 392, 561
n_frames = FR_HI - FR_LO

# Need within-recording median to subtract first. Sample frames across recording.
print("Building global median across recording (sampling every 10th frame)...")
sample_frames = list(range(0, n_total, 10))
mag_stack = []
for i, fid in enumerate(sample_frames):
    if i % 30 == 0: print("  %d/%d" % (i, len(sample_frames)))
    try: mag_stack.append(coh_mag_db_full(AIR, fid))
    except: continue
mag_stack = np.array(mag_stack)
median_db = np.median(mag_stack, axis=0)  # (n_dop, n_range)
print("Median computed: shape", median_db.shape)
n_dop = median_db.shape[0]
n_range = median_db.shape[1]
freqs_hz = (np.arange(n_dop) - n_dop // 2) * DOP_BIN_HZ

# Mask out chip-spur freqs and DC
chip_mask = (np.abs(freqs_hz) > 2480) & (np.abs(freqs_hz) < 2600)
dc_mask = np.abs(freqs_hz) < 100
bad = chip_mask | dc_mask | (freqs_hz < 0)  # only positive Doppler band
good_mask = ~bad

print("\nBuilding track-before-detect image: max residual per (frame, range_bin)")
# For each frame in 392-560, compute residual = mag - median
# Then take max across positive-Doppler good freqs
tbd_image = np.zeros((n_frames, n_range))
for i, fid in enumerate(range(FR_LO, FR_HI)):
    try:
        mag = coh_mag_db_full(AIR, fid)
        residual = mag - median_db
        residual[~good_mask] = -1e9
        tbd_image[i, :] = residual.max(axis=0)
    except:
        tbd_image[i, :] = -1e9
    if i % 30 == 0: print("  frame %d (i=%d/%d)" % (fid, i, n_frames))

print("TBD image shape:", tbd_image.shape, "value range:", tbd_image.min(), "to", tbd_image.max())

# Phase 2: load drone's true track from chip CFAR
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
            detections.append((t_rel, rm))
filtered = [d for d in detections if 28 <= d[0] <= 40]
sampled_d = []; last_t = -1
for d in filtered:
    if d[0] - last_t < 0.05: continue
    last_t = d[0]; sampled_d.append(d)
FPS_EST = 14.0

# True drone track: list of (frame_offset_in_window, drone_rb)
true_track = []
for d in sampled_d:
    t_rel, rm = d
    bin_idx = int(round(t_rel * FPS_EST))
    fi = bin_idx - FR_LO
    if 0 <= fi < n_frames:
        drone_rb = max(2, min(int(round(rm / RANGE_RES_M)), 96))
        true_track.append((fi, drone_rb))

print("\nDrone true track: %d points" % len(true_track))
print("  starts: frame_off=%d, rb=%d" % (true_track[0][0], true_track[0][1]))
print("  ends:   frame_off=%d, rb=%d" % (true_track[-1][0], true_track[-1][1]))

def score_track(frames_rbs):
    """Sum the TBD image values along a list of (frame_offset, rb) pairs."""
    s = 0.0; n = 0
    for f, r in frames_rbs:
        if 0 <= f < n_frames and 0 <= r < n_range:
            v = tbd_image[f, r]
            if v > -1e8:
                s += v; n += 1
    return s, n, (s/n if n > 0 else 0)

# Phase 3: score the true drone track + many candidate tracks
print("\n=== TRACK-BEFORE-DETECT scoring ===")

# Score the TRUE drone track
drone_sum, drone_n, drone_avg = score_track(true_track)
print("\nTRUE DRONE TRACK: sum=%+.1f dB over %d frames, average=%+.2f dB/frame" % (drone_sum, drone_n, drone_avg))

# Score random "drone-like" tracks: linear radial motion at various velocities
# Drone velocity in fly-away ~3 m/s = ~1 bin per second = ~1 bin per 14 frames
# Try velocities: -2, -1, 0, +1, +2 bins per second
# Try start range bins: every 4 from rb=4 to rb=92
print("\nCandidate tracks (start_rb, vel_bins_per_sec):")
print("  TRACK params      |   sum_dB   n_frames   avg_dB/frame  rank")

candidate_results = []
true_velocity_per_sec = (true_track[-1][1] - true_track[0][1]) / ((true_track[-1][0] - true_track[0][0]) / FPS_EST)
print("  (true velocity = %.2f bins/sec)" % true_velocity_per_sec)

for start_rb in range(4, 93, 2):
    for vel_per_sec in [-3, -2, -1, 0, 1, 2, 3]:
        track = []
        for fi in range(0, n_frames, 1):
            rb = start_rb + int(round(fi / FPS_EST * vel_per_sec))
            if 2 <= rb < n_range:
                track.append((fi, rb))
        s, n, avg = score_track(track)
        candidate_results.append((start_rb, vel_per_sec, s, n, avg, track))

# Rank true track among candidates
candidate_results.sort(key=lambda x: -x[2])  # by sum_dB, descending
true_rank = 999
for i, (sr, v, s, n, avg, tk) in enumerate(candidate_results):
    # Check if this candidate matches true track
    if abs(sr - true_track[0][1]) <= 1 and abs(v - true_velocity_per_sec) <= 0.5:
        true_rank = i; break

print("\nTop 20 candidate linear tracks:")
print("  rank  start_rb  vel(bin/s)    sum_dB  n_frames  avg/frame   marker")
for i, (sr, v, s, n, avg, _) in enumerate(candidate_results[:20]):
    marker = ""
    if abs(sr - true_track[0][1]) <= 1 and abs(v - true_velocity_per_sec) <= 0.5:
        marker = " <-- MATCHES DRONE"
    print("   %3d      %3d        %+.1f     %+8.1f    %4d    %+.2f%s" % (
        i+1, sr, v, s, n, avg, marker))

print("\nDrone TRUE track rank: #%d out of %d candidates" % (true_rank+1, len(candidate_results)))
print("Drone track sum: %+.1f dB" % drone_sum)
print("Best candidate (rank #1): start_rb=%d, vel=%+.1f, sum=%+.1f dB" % (
    candidate_results[0][0], candidate_results[0][1], candidate_results[0][2]))

# Compare to STATIONARY tracks (same range bin, no movement)
print("\nStationary tracks (vel=0, fixed rb) for comparison:")
print("  rb    sum_dB   avg/frame  marker")
stationary_scores = [(sr, s, avg) for sr, v, s, n, avg, _ in candidate_results if v == 0]
stationary_scores.sort(key=lambda x: -x[1])
for rb, s, avg in stationary_scores[:10]:
    marker = ""
    if rb == true_track[0][1]: marker = " <-- TRUE DRONE START rb"
    print("   %3d   %+8.1f    %+.2f%s" % (rb, s, avg, marker))

print("\nVERDICT:")
if true_rank < 5:
    print("  PASS: drone's true track ranks #%d (top 5)" % (true_rank + 1))
elif true_rank < 20:
    print("  PARTIAL: drone's true track ranks #%d (top 20)" % (true_rank + 1))
else:
    print("  FAIL: drone's true track ranks #%d (poor)" % (true_rank + 1))
