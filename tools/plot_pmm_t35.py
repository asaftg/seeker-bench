"""Make plots: slow-time spectrum + PMM fold curve at t=35s
for drone bin, chip bin, and clean bin. Save as PNG."""
import sys, importlib.util, numpy as np, scipy.fft as sfft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def load_frame(path,idx):
    with open(path,"rb") as f: f.seek(idx*BYTES_PER_FRAME); buf=f.read(BYTES_PER_FRAME)
    raw=np.frombuffer(buf,dtype=np.int16)
    cube=(raw.reshape(N_CHIRPS,N_RX,N_SAMPLES).transpose(0,2,1).astype(np.float32))
    cube-=cube.mean(axis=1,keepdims=True); return cube

def per_va_spec_full(path, idx):
    cube=load_frame(path,idx)
    rc=sfft.rfft(cube*HANN_FAST[None,:,None],axis=1,workers=2).astype(np.complex64)
    rc[:,1:-1,:]*=2.0; rc-=rc.mean(axis=0,keepdims=True)
    virtual=ddma_unfold(rc); n_per_tx=virtual.shape[0]
    slow=virtual.reshape(n_per_tx,virtual.shape[1],N_VA)
    spec=np.fft.fft(slow*WIN_SLOW[:,None,None],n=N_FFT,axis=0)
    coh = spec.sum(axis=2)
    return np.fft.fftshift(coh, axes=0)

def coh_mag_db(coh, rb):
    return 20*np.log10(np.maximum(np.abs(coh[:, rb]), 1e-30))

FRAME = 490
T_REL = FRAME / 14.0
DRONE_RB = 11
CHIP_RB = 24
CLEAN_RB = 18
CLEAN_RB_2 = 30

print("Loading frame", FRAME, "t=", T_REL)
coh = per_va_spec_full(AIR, FRAME)
n_dop = coh.shape[0]
freqs_hz = (np.arange(n_dop) - n_dop // 2) * DOP_BIN_HZ

drone_db = coh_mag_db(coh, DRONE_RB)
chip_db = coh_mag_db(coh, CHIP_RB)
clean_db = coh_mag_db(coh, CLEAN_RB)
clean2_db = coh_mag_db(coh, CLEAN_RB_2)

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

f0_grid = np.arange(50, 1500, 5)
drone_fold = mmhawkeye_fold(drone_db, freqs_hz, f0_grid)
chip_fold = mmhawkeye_fold(chip_db, freqs_hz, f0_grid)
clean_fold = mmhawkeye_fold(clean_db, freqs_hz, f0_grid)
clean2_fold = mmhawkeye_fold(clean2_db, freqs_hz, f0_grid)

def notch(spec_db, freqs_hz, lo, hi):
    out = spec_db.copy()
    mask = (np.abs(freqs_hz) >= lo) & (np.abs(freqs_hz) <= hi)
    out[mask] = -120
    return out

drone_notched = notch(drone_db, freqs_hz, 2400, 2700)
chip_notched = notch(chip_db, freqs_hz, 2400, 2700)
clean_notched = notch(clean_db, freqs_hz, 2400, 2700)

drone_fold_n = mmhawkeye_fold(drone_notched, freqs_hz, f0_grid)
chip_fold_n = mmhawkeye_fold(chip_notched, freqs_hz, f0_grid)
clean_fold_n = mmhawkeye_fold(clean_notched, freqs_hz, f0_grid)

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

pos_mask = freqs_hz > 50
ax = axes[0,0]
ax.plot(freqs_hz[pos_mask], drone_db[pos_mask], "r-", label="DRONE rb=" + str(DRONE_RB) + " (29m)", linewidth=1.5)
ax.plot(freqs_hz[pos_mask], chip_db[pos_mask], "k-", label="CHIP rb=" + str(CHIP_RB) + " (63m)", linewidth=1.0, alpha=0.7)
ax.plot(freqs_hz[pos_mask], clean_db[pos_mask], "b-", label="CLEAN rb=" + str(CLEAN_RB) + " (48m)", linewidth=1.0, alpha=0.7)
ax.plot(freqs_hz[pos_mask], clean2_db[pos_mask], "g-", label="CLEAN rb=" + str(CLEAN_RB_2) + " (79m)", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Doppler frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Slow-time spectrum at t=" + str(T_REL) + "s, frame " + str(FRAME))
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(50, 3810)
ax.axvspan(2400, 2700, alpha=0.15, color="gray")

ax = axes[0,1]
zoom = (freqs_hz > 200) & (freqs_hz < 1500)
ax.plot(freqs_hz[zoom], drone_db[zoom], "r-", label="DRONE rb=" + str(DRONE_RB), linewidth=1.5)
ax.plot(freqs_hz[zoom], chip_db[zoom], "k-", label="CHIP rb=" + str(CHIP_RB), linewidth=1.0, alpha=0.7)
ax.plot(freqs_hz[zoom], clean_db[zoom], "b-", label="CLEAN rb=" + str(CLEAN_RB), linewidth=1.0, alpha=0.7)
ax.plot(freqs_hz[zoom], clean2_db[zoom], "g-", label="CLEAN rb=" + str(CLEAN_RB_2), linewidth=0.8, alpha=0.5)
ax.set_xlabel("Doppler frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("ZOOMED 200-1500 Hz (DJI FPV blade-pass region)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1,0]
ax.plot(f0_grid, drone_fold, "r-", label="DRONE rb=" + str(DRONE_RB), linewidth=1.5)
ax.plot(f0_grid, chip_fold, "k-", label="CHIP rb=" + str(CHIP_RB), linewidth=1.0, alpha=0.7)
ax.plot(f0_grid, clean_fold, "b-", label="CLEAN rb=" + str(CLEAN_RB), linewidth=1.0, alpha=0.7)
ax.plot(f0_grid, clean2_fold, "g-", label="CLEAN rb=" + str(CLEAN_RB_2), linewidth=0.8, alpha=0.5)
ax.set_xlabel("Candidate blade-pass f0 (Hz)")
ax.set_ylabel("Fold strength (dB)")
ax.set_title("mmHawkeye fold curve (sum at k*f0 for k=1..6, +/-5% tolerance)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.plot(f0_grid, drone_fold_n, "r-", label="DRONE rb=" + str(DRONE_RB), linewidth=1.5)
ax.plot(f0_grid, chip_fold_n, "k-", label="CHIP rb=" + str(CHIP_RB), linewidth=1.0, alpha=0.7)
ax.plot(f0_grid, clean_fold_n, "b-", label="CLEAN rb=" + str(CLEAN_RB), linewidth=1.0, alpha=0.7)
ax.set_xlabel("Candidate blade-pass f0 (Hz)")
ax.set_ylabel("Fold strength (dB)")
ax.set_title("mmHawkeye fold AFTER notching 2400-2700 Hz chip region")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

print("Integrating across 30 frames...")
N_INT = 30
drone_fold_int = np.zeros(len(f0_grid))
chip_fold_int = np.zeros(len(f0_grid))
clean_fold_int = np.zeros(len(f0_grid))
for fi in range(FRAME - N_INT//2, FRAME + N_INT//2):
    coh_i = per_va_spec_full(AIR, fi)
    drone_fold_int += mmhawkeye_fold(coh_mag_db(coh_i, DRONE_RB), freqs_hz, f0_grid) / N_INT
    chip_fold_int += mmhawkeye_fold(coh_mag_db(coh_i, CHIP_RB), freqs_hz, f0_grid) / N_INT
    clean_fold_int += mmhawkeye_fold(coh_mag_db(coh_i, CLEAN_RB), freqs_hz, f0_grid) / N_INT

ax = axes[2,0]
ax.plot(f0_grid, drone_fold_int, "r-", label="DRONE rb=" + str(DRONE_RB), linewidth=1.5)
ax.plot(f0_grid, chip_fold_int, "k-", label="CHIP rb=" + str(CHIP_RB), linewidth=1.0, alpha=0.7)
ax.plot(f0_grid, clean_fold_int, "b-", label="CLEAN rb=" + str(CLEAN_RB), linewidth=1.0, alpha=0.7)
ax.set_xlabel("Candidate blade-pass f0 (Hz)")
ax.set_ylabel("Fold strength (dB)")
ax.set_title("mmHawkeye fold INTEGRATED over " + str(N_INT) + " frames")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2,1]
diff = drone_fold_int - clean_fold_int
ax.plot(f0_grid, diff, "r-", linewidth=1.5)
ax.set_xlabel("Candidate blade-pass f0 (Hz)")
ax.set_ylabel("Drone fold - Clean fold (dB)")
ax.set_title("Drone fold ADVANTAGE over clean bin (positive = drone-specific resonance)")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axhline(3, color="red", linewidth=0.5, linestyle="--", label="+3dB threshold")
ax.legend()

plt.suptitle("PMM analysis at t=" + str(T_REL) + "s (frame " + str(FRAME) + "). Drone at rb=" + str(DRONE_RB) + " (~29m).", fontsize=12)
plt.tight_layout()
out_path = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\pmm_analysis_t35.png"
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("Saved:", out_path)

print("\n=== Numbers at t=35s, frame", FRAME, "===")
print("DRONE rb=" + str(DRONE_RB) + " max spec dB:", round(drone_db[freqs_hz>0].max(), 1))
print("CHIP rb=" + str(CHIP_RB) + " max spec dB:", round(chip_db[freqs_hz>0].max(), 1))
print("CLEAN rb=" + str(CLEAN_RB) + " max spec dB:", round(clean_db[freqs_hz>0].max(), 1))
print("Drone fold max single frame:", round(drone_fold.max(), 1), "at f0=", f0_grid[drone_fold.argmax()])
print("Chip fold max single frame:", round(chip_fold.max(), 1), "at f0=", f0_grid[chip_fold.argmax()])
print("Clean fold max single frame:", round(clean_fold.max(), 1), "at f0=", f0_grid[clean_fold.argmax()])
print("After notch DRONE fold max:", round(drone_fold_n.max(), 1), "at f0=", f0_grid[drone_fold_n.argmax()])
print("After notch CHIP fold max:", round(chip_fold_n.max(), 1), "at f0=", f0_grid[chip_fold_n.argmax()])
print("After notch CLEAN fold max:", round(clean_fold_n.max(), 1), "at f0=", f0_grid[clean_fold_n.argmax()])
print("Integrated 30f DRONE fold max:", round(drone_fold_int.max(), 1), "at f0=", f0_grid[drone_fold_int.argmax()])
print("Integrated 30f CHIP fold max:", round(chip_fold_int.max(), 1), "at f0=", f0_grid[chip_fold_int.argmax()])
print("Integrated 30f CLEAN fold max:", round(clean_fold_int.max(), 1), "at f0=", f0_grid[clean_fold_int.argmax()])
print("Max drone-fold-advantage over clean:", round((drone_fold_int - clean_fold_int).max(), 1), "dB at f0=", f0_grid[(drone_fold_int - clean_fold_int).argmax()])
