"""POC #2: WITHIN-RECORDING TEMPORAL MEDIAN as background subtraction.

Hypothesis: at each (range bin, Doppler-freq) cell, take the median value
across many frames of the recording itself. The drone passes through any
single cell briefly (it moves) — so it doesn't dominate the median.
Stationary clutter and chip artifacts persist throughout the recording —
they DO dominate the median.

Subtracting the median should leave drone signal as positive residual,
while killing clutter and chip artifacts.

Test: for the drone fly-away frames (28-40s), compute residuals at the
drone bin vs clean bins.
"""
import sys, io, json, importlib.util, numpy as np, scipy.fft as sfft

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
    """Returns (n_dop, n_range) magnitude in dB after omni VA sum."""
    cube=load_frame(path,idx)
    rc=sfft.rfft(cube*HANN_FAST[None,:,None],axis=1,workers=2).astype(np.complex64)
    rc[:,1:-1,:]*=2.0; rc-=rc.mean(axis=0,keepdims=True)
    virtual=ddma_unfold(rc); n_per_tx=virtual.shape[0]
    slow=virtual.reshape(n_per_tx,virtual.shape[1],N_VA)
    spec=np.fft.fft(slow*WIN_SLOW[:,None,None],n=N_FFT,axis=0)
    coh = np.fft.fftshift(spec.sum(axis=2), axes=0)  # (n_dop, n_range)
    return 20*np.log10(np.maximum(np.abs(coh), 1e-30))

# Phase 1: build the WITHIN-RECORDING MEDIAN
# Sample frames across the recording (every 10th frame to manage memory)
import os
n_total_frames = os.path.getsize(AIR) // BYTES_PER_FRAME
print("Recording has %d frames" % n_total_frames)

SAMPLE_STEP = 10  # every 10 frames
sample_frames = list(range(0, n_total_frames, SAMPLE_STEP))
print("Sampling %d frames for median (every %dth)" % (len(sample_frames), SAMPLE_STEP))

# Build stack of mag_db for each sampled frame
print("Computing magnitudes for sampled frames (this takes ~2 min)...")
mag_stack = []
for i, fid in enumerate(sample_frames):
    if i % 20 == 0:
        print("  %d/%d (frame %d)" % (i, len(sample_frames), fid))
    try:
        mag_stack.append(coh_mag_db_full(AIR, fid))
    except Exception as e:
        print("  skip frame %d: %s" % (fid, e))
mag_stack = np.array(mag_stack)  # (n_samp, n_dop, n_range)
print("mag_stack shape:", mag_stack.shape)

# Compute median per (dop, range) cell
median_db = np.median(mag_stack, axis=0)  # (n_dop, n_range)
print("Median db computed. Range: %.1f to %.1f" % (median_db.min(), median_db.max()))

# Phase 2: At each drone frame, compute residual = mag - median, and check the drone bin
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

filtered = [d for d in detections if 28 <= d[0] <= 40 and d[1] >= 16]
sampled_d = []; last_t = -1
for d in filtered:
    if d[0] - last_t < 0.05: continue
    last_t = d[0]; sampled_d.append(d)
FPS_EST = 14.0

# Test on every 4th drone frame
print("\n=== POC #2: within-recording median BG subtraction ===")
print("For each drone frame: compute residual = current_mag - global_median")
print("Check if drone bin shows POSITIVE residual that clean bins don't\n")

n_dop = mag_stack.shape[1]
freqs_hz = (np.arange(n_dop) - n_dop // 2) * DOP_BIN_HZ
pos_band_mask = (freqs_hz > 100) & (freqs_hz < 3500)

print(" t_s   rb  | drn_max_resid drn_pk_Hz | clean_max_resid | rank_among_all_rbs")
print("-" * 80)

results = []
for d in sampled_d[::4]:
    t_rel, rm = d
    bin_idx = int(round(t_rel * FPS_EST))
    drone_rb = max(2, min(int(round(rm / RANGE_RES_M)), 96))
    clean_rb = (drone_rb + 5) if drone_rb < 80 else (drone_rb - 5)
    if clean_rb in (12,24,36,48,60,72,84,96): clean_rb += 2
    try:
        cur_mag = coh_mag_db_full(AIR, bin_idx)
    except: continue
    residual = cur_mag - median_db  # (n_dop, n_range)

    # Mask out chip-spur freqs and DC
    chip_freq_mask = ((np.abs(freqs_hz) > 2480) & (np.abs(freqs_hz) < 2600))
    dc_mask = np.abs(freqs_hz) < 100
    bad_freqs = chip_freq_mask | dc_mask | ~pos_band_mask
    residual_masked = residual.copy()
    residual_masked[bad_freqs, :] = -1e10

    drone_residual = residual_masked[:, drone_rb]
    clean_residual = residual_masked[:, clean_rb]
    drn_max = float(drone_residual.max())
    drn_pk_freq = float(freqs_hz[drone_residual.argmax()])
    cln_max = float(clean_residual.max())

    # Rank: among all non-chip range bins, how does drone_rb's max-residual rank?
    chip_rbs = set([12,24,36,48,60,72,84,96, 11,13,23,25,47,49,71,73,95])
    operator_rbs = set(range(0, 6))
    excluded = chip_rbs | operator_rbs
    rb_max_residuals = []
    for rb in range(2, residual_masked.shape[1]):
        if rb in excluded: continue
        rb_max_residuals.append((rb, float(residual_masked[:, rb].max())))
    rb_max_residuals.sort(key=lambda x: -x[1])
    drone_rank = next((i for i, (rb, _) in enumerate(rb_max_residuals) if rb == drone_rb), 999)
    winner_rb, winner_resid = rb_max_residuals[0]

    print(" %4.1f  %3d | %+12.1f  %+8.0f  | %+14.1f | rank=%2d (winner rb=%d at %+.1f dB)" % (
        t_rel, drone_rb, drn_max, drn_pk_freq, cln_max, drone_rank, winner_rb, winner_resid))
    results.append((drn_max, cln_max, drone_rank, winner_resid))

if results:
    arr = np.array([(d, c, r, w) for d, c, r, w in results])
    print("\n=== AVERAGED across %d frames ===" % len(results))
    print("Drone-bin max residual:    %+.1f dB" % arr[:,0].mean())
    print("Clean-bin max residual:    %+.1f dB" % arr[:,1].mean())
    print("Drone-vs-clean residual:   %+.1f dB" % (arr[:,0]-arr[:,1]).mean())
    print("Drone bin rank (among all non-chip):  median=%d, mean=%.1f" % (
        int(np.median(arr[:,2])), arr[:,2].mean()))
    n_top1 = int(np.sum(arr[:,2] == 0))
    n_top3 = int(np.sum(arr[:,2] < 3))
    n_top5 = int(np.sum(arr[:,2] < 5))
    print("Drone in top-1: %d/%d (%.0f%%)" % (n_top1, len(results), 100*n_top1/len(results)))
    print("Drone in top-3: %d/%d (%.0f%%)" % (n_top3, len(results), 100*n_top3/len(results)))
    print("Drone in top-5: %d/%d (%.0f%%)" % (n_top5, len(results), 100*n_top5/len(results)))

    if (arr[:,0]-arr[:,1]).mean() > 3.0:
        print("\nVERDICT: PASS — drone bin shows %+.1f dB more residual than clean" % (arr[:,0]-arr[:,1]).mean())
    elif arr[:,2].mean() < 5:
        print("\nVERDICT: PARTIAL — drone bin ranks well among all bins (mean rank %.1f)" % arr[:,2].mean())
    else:
        print("\nVERDICT: FAIL — drone bin not distinguishable")
