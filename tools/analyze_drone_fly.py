"""Comprehensive analysis of the 'drone fly' recording (yesterday).
Drone at ~5m for ~10 seconds — should have STRONG signal (+40 dB vs airborne1).

What I'll do:
  1. Inspect file: how many frames, what timestamps
  2. Look at radar/frame chip CFAR detections — where is the drone over time
  3. For each drone frame, extract slow-time spectrum at the drone's range bin
  4. Compute mmHawkeye fold curve at drone bin vs clean bins
  5. Plot the spectrum and fold for visual inspection
  6. Quantify: is drone signal CLEARLY above clutter at this short range?
"""
import sys, io, json, importlib.util, numpy as np, scipy.fft as sfft, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to load ddma; if not available, use raw 4-RX
try:
    spec_d = importlib.util.spec_from_file_location("ddma", r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\radar_dca\__pycache__\ddma.cpython-311.pyc")
    ddma_mod = importlib.util.module_from_spec(spec_d); spec_d.loader.exec_module(ddma_mod)
    ddma_unfold = ddma_mod.ddma_unfold
    HAS_DDMA = True
except:
    HAS_DDMA = False
    print("WARN: ddma not available, using raw 4-RX coherent sum")

N_CHIRPS, N_RX, N_SAMPLES = 768, 4, 192
PRF_HZ = 30478.51264858275
N_TX = 4
N_FFT = 1024
N_VA = 16
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS*N_RX*N_SAMPLES*2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)
WIN_SLOW_TX = np.hanning(N_CHIRPS//N_TX).astype(np.float32)
WIN_SLOW_FULL = np.hanning(N_CHIRPS).astype(np.float32)
DOP_BIN_HZ_TX = (PRF_HZ/N_TX) / N_FFT
DOP_BIN_HZ_FULL = PRF_HZ / N_FFT

DRONE_FLY_BIN = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin"
DRONE_FLY_JSONL = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\drone fly.jsonl"

n_total = os.path.getsize(DRONE_FLY_BIN) // BYTES_PER_FRAME
print("Drone fly recording: %d frames" % n_total)
print("At ~14 fps, that's %.1f seconds" % (n_total/14.0))

def load_frame(path,idx):
    with open(path,"rb") as f: f.seek(idx*BYTES_PER_FRAME); buf=f.read(BYTES_PER_FRAME)
    raw=np.frombuffer(buf,dtype=np.int16)
    cube=(raw.reshape(N_CHIRPS,N_RX,N_SAMPLES).transpose(0,2,1).astype(np.float32))
    cube-=cube.mean(axis=1,keepdims=True); return cube

def per_va_spec(path, idx):
    """Returns (n_dop, n_range) magnitude in dB after VA-coherent sum.
    Uses DDMA if available, else 4-RX coherent sum across full chirps."""
    cube=load_frame(path,idx)
    rc=sfft.rfft(cube*HANN_FAST[None,:,None],axis=1,workers=2).astype(np.complex64)
    rc[:,1:-1,:]*=2.0; rc-=rc.mean(axis=0,keepdims=True)
    if HAS_DDMA:
        virtual=ddma_unfold(rc); n_per_tx=virtual.shape[0]
        slow=virtual.reshape(n_per_tx,virtual.shape[1],N_VA)
        spec=np.fft.fft(slow*WIN_SLOW_TX[:,None,None],n=N_FFT,axis=0)
        coh = np.fft.fftshift(spec.sum(axis=2), axes=0)
    else:
        rc_sum = rc.sum(axis=2)  # (n_chirps, n_range)
        spec=np.fft.fft(rc_sum*WIN_SLOW_FULL[:,None],n=N_FFT,axis=0)
        coh = np.fft.fftshift(spec, axes=0)
    return 20*np.log10(np.maximum(np.abs(coh), 1e-30))

# Step 1: chip CFAR detections from drone-fly jsonl
print("\nStep 1: Loading chip CFAR detections from drone-fly jsonl...")
base_ts = None
chip_dets = []  # (t_rel, x, y, z, range, vel)
with io.open(DRONE_FLY_JSONL,"r",encoding="utf-8") as fh:
    for line in fh:
        try: r=json.loads(line)
        except: continue
        ch = r.get("channel"); ts = r.get("ts_ns")
        if base_ts is None: base_ts = ts
        if ch == "radar/frame":
            t_rel = (ts - base_ts)/1e9
            for t in (r.get("msg") or {}).get("targets") or []:
                x=t.get("x",0); y=t.get("y",0); z=t.get("z",0)
                rm=(x*x+y*y+z*z)**0.5
                vx=t.get("vx",0); vy=t.get("vy",0); vz=t.get("vz",0)
                v_rad = (vx*x+vy*y)/max(rm,0.01)
                chip_dets.append((t_rel, t.get("tid","?"), x, y, z, rm, v_rad))

print("Total chip CFAR detections in drone-fly: %d" % len(chip_dets))
if chip_dets:
    print("  t range: %.1fs to %.1fs" % (min(d[0] for d in chip_dets), max(d[0] for d in chip_dets)))
    # Group by tid
    from collections import defaultdict
    by_tid = defaultdict(list)
    for d in chip_dets: by_tid[d[1]].append(d)
    print("  tids:")
    for tid, dets in sorted(by_tid.items(), key=lambda x: -len(x[1]))[:5]:
        rs = [d[5] for d in dets]
        print("    tid=%s n=%d range=%.1fm to %.1fm t=%.1fs to %.1fs" % (
            tid, len(dets), min(rs), max(rs), dets[0][0], dets[-1][0]))

# Step 2: compute DRONE'S range bin track and key frames
# Use most-detected tid as "the drone"
if chip_dets:
    main_tid = max(by_tid.keys(), key=lambda k: len(by_tid[k]))
    drone_track = sorted(by_tid[main_tid])
    print("\nUsing tid=%s as drone (most detections, %d points)" % (main_tid, len(drone_track)))
    print("  Range: %.1f to %.1f m" % (min(d[5] for d in drone_track), max(d[5] for d in drone_track)))
    print("  Time span: %.1fs to %.1fs" % (drone_track[0][0], drone_track[-1][0]))
else:
    # If no chip CFAR detections, just use rb=2 (5m) as best guess
    print("\nNo chip CFAR detections — assuming drone at rb=2 (~5m)")
    drone_track = [(t, "?", 0, 5, 0, 5.0, 0) for t in np.arange(0, n_total/14.0, 0.07)]

FPS_EST = 14.0

# Step 3: Inspect a representative drone frame
mid_t = (drone_track[0][0] + drone_track[-1][0]) / 2
mid_d = min(drone_track, key=lambda d: abs(d[0] - mid_t))
mid_frame = int(round(mid_d[0] * FPS_EST))
mid_drone_rb = max(2, min(int(round(mid_d[5] / RANGE_RES_M)), 96))
print("\nStep 3: Mid-flight frame %d (t=%.1fs), drone at rb=%d (%.1fm)" % (
    mid_frame, mid_d[0], mid_drone_rb, mid_d[5]))

mid_rd = per_va_spec(DRONE_FLY_BIN, mid_frame)
n_dop = mid_rd.shape[0]
n_range = mid_rd.shape[1]
DOP_BIN_HZ = DOP_BIN_HZ_TX if HAS_DDMA else DOP_BIN_HZ_FULL
freqs_hz = (np.arange(n_dop) - n_dop // 2) * DOP_BIN_HZ
print("  Doppler bin width %.2f Hz, Nyquist +/- %.0f Hz" % (
    DOP_BIN_HZ, (n_dop/2) * DOP_BIN_HZ))

# Print top peaks at drone bin
def show_top_peaks(coh_2d, rb, label, n_top=15):
    s_db = coh_2d[:, rb]
    top = np.argsort(-s_db)[:n_top]
    print("\n  --- %s (rb=%d) top %d peaks ---" % (label, rb, n_top))
    print("    %8s  %8s" % ("freq_Hz", "mag_dB"))
    for i in sorted(top):
        if abs(freqs_hz[i]) < 50: continue  # skip DC region
        print("    %+8.0f  %8.1f" % (freqs_hz[i], s_db[i]))
    return s_db

drone_peaks = show_top_peaks(mid_rd, mid_drone_rb, "DRONE bin")
clean_rb = mid_drone_rb + 8 if mid_drone_rb < 80 else mid_drone_rb - 8
clean_peaks = show_top_peaks(mid_rd, clean_rb, "CLEAN bin")

# Step 4: mmHawkeye fold at drone bin and clean bin
def mmhawkeye_fold(spec_db, freqs_hz, f0_grid, k_max=6, tol_pct=0.05):
    spec_lin = 10**(spec_db/10)
    pos_mask = freqs_hz > 0
    pos_lin = spec_lin[pos_mask]
    pos_freqs = freqs_hz[pos_mask]
    fold_db = np.zeros(len(f0_grid))
    for i, f0 in enumerate(f0_grid):
        s = 0.0
        for k in range(1, k_max+1):
            kf0 = k * f0
            if kf0 > pos_freqs[-1]: break
            window_lo = kf0 * (1 - tol_pct)
            window_hi = kf0 * (1 + tol_pct)
            mask = (pos_freqs >= window_lo) & (pos_freqs <= window_hi)
            if mask.any():
                s += pos_lin[mask].max()
        fold_db[i] = 10*np.log10(max(s, 1e-30))
    return fold_db

f0_grid = np.arange(50, 2000, 5)
drone_fold = mmhawkeye_fold(drone_peaks, freqs_hz, f0_grid)
clean_fold = mmhawkeye_fold(clean_peaks, freqs_hz, f0_grid)

print("\nStep 4: mmHawkeye fold at single mid-flight frame")
print("  Drone bin fold: max=%.1f dB at f0=%d Hz" % (drone_fold.max(), f0_grid[drone_fold.argmax()]))
print("  Clean bin fold: max=%.1f dB at f0=%d Hz" % (clean_fold.max(), f0_grid[clean_fold.argmax()]))
print("  Drone fold ADVANTAGE over clean: %+.1f dB" % (drone_fold.max() - clean_fold.max()))

# Step 5: PLOT
print("\nStep 5: making plots...")
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Top-left: full spectrum at drone bin vs clean bin
ax = axes[0, 0]
zoom = (freqs_hz > 50) & (freqs_hz < 5000)
ax.plot(freqs_hz[zoom], drone_peaks[zoom], 'r-', label='DRONE rb=%d (%.1fm)' % (mid_drone_rb, mid_drone_rb*RANGE_RES_M), linewidth=1.5)
ax.plot(freqs_hz[zoom], clean_peaks[zoom], 'b-', label='CLEAN rb=%d (%.1fm)' % (clean_rb, clean_rb*RANGE_RES_M), linewidth=1.0, alpha=0.7)
ax.set_xlabel("Doppler frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Slow-time spectrum at t=%.1fs (mid-flight)" % mid_d[0])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Top-right: fold
ax = axes[0, 1]
ax.plot(f0_grid, drone_fold, 'r-', label='DRONE rb=%d' % mid_drone_rb, linewidth=1.5)
ax.plot(f0_grid, clean_fold, 'b-', label='CLEAN rb=%d' % clean_rb, linewidth=1.0, alpha=0.7)
ax.set_xlabel("Candidate blade-pass f0 (Hz)")
ax.set_ylabel("Fold strength (dB)")
ax.set_title("mmHawkeye fold curve (single frame)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Middle-left: per-range-bin energy at this frame
ax = axes[1, 0]
range_axis_m = np.arange(n_range) * RANGE_RES_M
# Max signal per range bin (in band 100-3000 Hz)
band = (freqs_hz > 100) & (freqs_hz < 3000)
max_per_rb = mid_rd[band, :].max(axis=0)
ax.plot(range_axis_m, max_per_rb, 'k-', linewidth=1.0)
ax.axvline(mid_drone_rb * RANGE_RES_M, color='r', linestyle='--', label='drone bin (%.1fm)' % (mid_drone_rb*RANGE_RES_M))
ax.set_xlabel("Range (m)")
ax.set_ylabel("Max magnitude in 100-3000 Hz band (dB)")
ax.set_title("Energy per range bin (mid-flight frame)")
ax.legend()
ax.grid(True, alpha=0.3)

# Middle-right: time series of drone-bin signal across all flight frames
print("Building time-series of drone bin signal across all frames...")
all_drone_dB = []
all_clean_dB = []
all_drone_fold = []
sample_step = 2  # every 2nd drone detection
for d in drone_track[::sample_step]:
    t_rel, _, x, y, z, rm, _ = d
    bin_idx = int(round(t_rel * FPS_EST))
    rb = max(2, min(int(round(rm / RANGE_RES_M)), 96))
    cln_rb = rb + 8 if rb < 80 else rb - 8
    try:
        rd_i = per_va_spec(DRONE_FLY_BIN, bin_idx)
    except: continue
    drn_max = rd_i[band, rb].max()
    cln_max = rd_i[band, cln_rb].max()
    drn_fold = mmhawkeye_fold(rd_i[:, rb], freqs_hz, f0_grid).max()
    all_drone_dB.append((t_rel, drn_max))
    all_clean_dB.append((t_rel, cln_max))
    all_drone_fold.append((t_rel, drn_fold))

ts = [x[0] for x in all_drone_dB]
ax = axes[1, 1]
ax.plot(ts, [x[1] for x in all_drone_dB], 'r-', label='DRONE bin max', linewidth=1.5)
ax.plot(ts, [x[1] for x in all_clean_dB], 'b-', label='CLEAN bin max', linewidth=1.0, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Max magnitude in 100-3000 Hz (dB)")
ax.set_title("Drone vs Clean bin signal over flight")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom-left: range-Doppler IMAGE at mid-flight
ax = axes[2, 0]
extent = [freqs_hz[0], freqs_hz[-1], 0, n_range * RANGE_RES_M]
im = ax.imshow(mid_rd.T, aspect='auto', origin='lower', extent=extent,
               cmap='turbo', vmin=mid_rd.max()-30, vmax=mid_rd.max())
ax.axhline(mid_drone_rb * RANGE_RES_M, color='r', linestyle='--', linewidth=0.8, label='drone range')
ax.set_xlabel("Doppler (Hz)")
ax.set_ylabel("Range (m)")
ax.set_title("Range-Doppler at t=%.1fs" % mid_d[0])
ax.set_xlim(-3000, 3000)
ax.set_ylim(0, 60)
ax.legend()
plt.colorbar(im, ax=ax, label='dB')

# Bottom-right: fold over time
ax = axes[2, 1]
ax.plot(ts, [x[1] for x in all_drone_fold], 'r-', linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Drone-bin fold strength (dB)")
ax.set_title("mmHawkeye fold strength at drone bin over time")
ax.grid(True, alpha=0.3)

plt.suptitle("DRONE FLY recording: %d frames, drone at ~5m for ~10s" % n_total, fontsize=12)
plt.tight_layout()
out_path = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\drone_fly_analysis.png"
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print("\nSaved:", out_path)

# SUMMARY
print("\n" + "="*70)
print("SUMMARY for drone-fly recording")
print("="*70)
mean_drone = np.mean([x[1] for x in all_drone_dB])
mean_clean = np.mean([x[1] for x in all_clean_dB])
print("Mean drone-bin max signal: %.1f dB" % mean_drone)
print("Mean clean-bin max signal: %.1f dB" % mean_clean)
print("Mean drone advantage:      %+.1f dB" % (mean_drone - mean_clean))
print("Max drone fold strength: %.1f dB" % max(x[1] for x in all_drone_fold))
print("Max single-frame drone-vs-clean: %+.1f dB" % max(
    (a[1]-c[1] for a,c in zip(all_drone_dB, all_clean_dB))))
