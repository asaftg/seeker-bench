"""Drone-fly v2: focus on the BLADE-PASS BAND (200-1500 Hz) where real
propeller PMM signature should live. Time-resolved analysis to see if
signal pattern matches takeoff/landing.
"""
import sys, importlib.util, numpy as np, scipy.fft as sfft, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    spec_d = importlib.util.spec_from_file_location("ddma", r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\radar_dca\__pycache__\ddma.cpython-311.pyc")
    ddma_mod = importlib.util.module_from_spec(spec_d); spec_d.loader.exec_module(ddma_mod)
    ddma_unfold = ddma_mod.ddma_unfold
    HAS_DDMA = True
except Exception as e:
    print("DDMA not available:", e); HAS_DDMA = False

N_CHIRPS, N_RX, N_SAMPLES = 768, 4, 192
PRF_HZ = 30478.51264858275; N_TX = 4; N_FFT = 1024; N_VA = 16
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS*N_RX*N_SAMPLES*2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)
WIN_SLOW_TX = np.hanning(N_CHIRPS//N_TX).astype(np.float32)
DOP_BIN_HZ = (PRF_HZ/N_TX) / N_FFT

DRONE_FLY_BIN = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin"
n_total = os.path.getsize(DRONE_FLY_BIN) // BYTES_PER_FRAME
FPS_EST = 14.0
print("Recording: %d frames = %.1f sec" % (n_total, n_total/FPS_EST))

def load_frame(path,idx):
    with open(path,"rb") as f: f.seek(idx*BYTES_PER_FRAME); buf=f.read(BYTES_PER_FRAME)
    raw=np.frombuffer(buf,dtype=np.int16)
    cube=(raw.reshape(N_CHIRPS,N_RX,N_SAMPLES).transpose(0,2,1).astype(np.float32))
    cube-=cube.mean(axis=1,keepdims=True); return cube

def per_va_spec(path, idx):
    cube=load_frame(path,idx)
    rc=sfft.rfft(cube*HANN_FAST[None,:,None],axis=1,workers=2).astype(np.complex64)
    rc[:,1:-1,:]*=2.0; rc-=rc.mean(axis=0,keepdims=True)
    if HAS_DDMA:
        virtual=ddma_unfold(rc); n_per_tx=virtual.shape[0]
        slow=virtual.reshape(n_per_tx,virtual.shape[1],N_VA)
        spec=np.fft.fft(slow*WIN_SLOW_TX[:,None,None],n=N_FFT,axis=0)
        coh = np.fft.fftshift(spec.sum(axis=2), axes=0)
    return 20*np.log10(np.maximum(np.abs(coh), 1e-30))

n_dop = N_FFT
freqs_hz = (np.arange(n_dop) - n_dop // 2) * DOP_BIN_HZ

# Build a (frame, range_bin) image of MAX SIGNAL IN BLADE-PASS BAND (200-1500 Hz)
# and a separate image for LOW-FREQ BAND (50-200 Hz, body-Doppler region)
print("\nBuilding time-vs-range image (max-in-blade-pass-band per frame per range)...")
blade_band = (freqs_hz > 200) & (freqs_hz < 1500)
body_band = (freqs_hz > 50) & (freqs_hz < 200)
high_band = (freqs_hz > 1500) & (freqs_hz < 3500)

blade_img = np.zeros((n_total, 97))
body_img = np.zeros((n_total, 97))
high_img = np.zeros((n_total, 97))
for i in range(n_total):
    if i % 50 == 0: print("  %d/%d" % (i, n_total))
    try:
        rd = per_va_spec(DRONE_FLY_BIN, i)
        blade_img[i] = rd[blade_band, :].max(axis=0)
        body_img[i] = rd[body_band, :].max(axis=0)
        high_img[i] = rd[high_band, :].max(axis=0)
    except: pass

print("\n=== Per-range-bin TIME AVERAGES ===")
print(" rb  range_m  body(50-200Hz)  blade(200-1500Hz)  high(1500-3500Hz)")
for rb in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 36, 48]:
    if rb >= 97: continue
    print(" %2d  %5.1f       %6.1f         %6.1f             %6.1f" % (
        rb, rb*RANGE_RES_M, body_img[:, rb].mean(), blade_img[:, rb].mean(), high_img[:, rb].mean()))

# Find which range bin has STRONGEST BLADE-BAND signal across the recording
mean_blade = blade_img.mean(axis=0)
print("\nTop 10 range bins by mean BLADE-BAND signal (200-1500 Hz):")
print(" rank  rb  range_m  blade_dB  body_dB  high_dB")
top_blade = np.argsort(-mean_blade)[:10]
for rank, rb in enumerate(top_blade):
    print("  %2d   %2d  %5.1f    %6.1f   %6.1f   %6.1f" % (
        rank+1, rb, rb*RANGE_RES_M, mean_blade[rb], body_img[:, rb].mean(), high_img[:, rb].mean()))

# TIME SERIES at rb=2 (where we expect drone) and a few nearby + clean
print("\n=== TIME-RESOLVED signal across the recording ===")
print("Looking for takeoff/landing pattern (signal rises when drone airborne)")
print(" sec  rb=1   rb=2   rb=3   rb=4   rb=5   rb=8   rb=15  rb=24")
for fi in range(0, n_total, 20):
    t = fi / FPS_EST
    print("%4.1f  " % t + "  ".join("%5.1f" % blade_img[fi, rb] for rb in [1, 2, 3, 4, 5, 8, 15, 24]))

# DETAILED spectrum at the LOUDEST drone-frame time
loudest_rb = top_blade[0]
loudest_frame = int(np.argmax(blade_img[:, loudest_rb]))
print("\nLoudest blade-band signal: rb=%d at frame %d (t=%.1fs), value=%.1f dB" % (
    loudest_rb, loudest_frame, loudest_frame/FPS_EST, blade_img[loudest_frame, loudest_rb]))

# Print spectrum at this peak
peak_rd = per_va_spec(DRONE_FLY_BIN, loudest_frame)
print("\nFull spectrum at rb=%d, frame %d, top 25 peaks above 50 Hz:" % (loudest_rb, loudest_frame))
print(" freq_Hz   mag_dB")
peak_spec = peak_rd[:, loudest_rb]
top = np.argsort(-peak_spec)[:50]
shown = 0
for i in sorted(top):
    if abs(freqs_hz[i]) < 50: continue
    if shown >= 25: break
    print(" %+7.0f   %6.1f" % (freqs_hz[i], peak_spec[i]))
    shown += 1

# Compare to clean bin at same frame
clean_rb = loudest_rb + 8 if loudest_rb < 80 else loudest_rb - 8
clean_spec = peak_rd[:, clean_rb]
print("\nSAME FRAME, CLEAN bin rb=%d top 15 peaks:" % clean_rb)
top = np.argsort(-clean_spec)[:30]
shown = 0
for i in sorted(top):
    if abs(freqs_hz[i]) < 50: continue
    if shown >= 15: break
    print(" %+7.0f   %6.1f" % (freqs_hz[i], clean_spec[i]))
    shown += 1

# Drone signature check: are there peaks at 200-1500 Hz that AREN'T in clean bin?
print("\nDrone-vs-clean diff (top 15 freqs where drone is most above clean, 100-3000 Hz):")
band_mask = (freqs_hz > 100) & (freqs_hz < 3000)
diff = peak_spec - clean_spec
diff_band = np.where(band_mask, diff, -1e9)
top = np.argsort(-diff_band)[:15]
print(" freq_Hz   drone_dB  clean_dB  diff")
for i in sorted(top):
    print(" %+7.0f   %7.1f   %7.1f   %+5.1f" % (freqs_hz[i], peak_spec[i], clean_spec[i], diff[i]))

# PLOT: heatmap (time x range) of blade-band signal
fig, axes = plt.subplots(3, 1, figsize=(15, 11))

ax = axes[0]
im = ax.imshow(blade_img.T, aspect='auto', origin='lower',
               extent=[0, n_total/FPS_EST, 0, 97*RANGE_RES_M],
               cmap='turbo', vmin=blade_img.max()-25, vmax=blade_img.max())
ax.set_xlabel("Time (s)"); ax.set_ylabel("Range (m)")
ax.set_title("Max signal in BLADE-PASS band (200-1500 Hz) over time x range")
ax.set_ylim(0, 80)
plt.colorbar(im, ax=ax, label='dB')

ax = axes[1]
im = ax.imshow(body_img.T, aspect='auto', origin='lower',
               extent=[0, n_total/FPS_EST, 0, 97*RANGE_RES_M],
               cmap='turbo', vmin=body_img.max()-25, vmax=body_img.max())
ax.set_xlabel("Time (s)"); ax.set_ylabel("Range (m)")
ax.set_title("Max signal in BODY-DOPPLER band (50-200 Hz) over time x range")
ax.set_ylim(0, 30)
plt.colorbar(im, ax=ax, label='dB')

ax = axes[2]
ax.plot(np.arange(n_total)/FPS_EST, blade_img[:, 2], 'r-', label='rb=2 (5m) blade-band', linewidth=1.0)
ax.plot(np.arange(n_total)/FPS_EST, blade_img[:, 3], 'orange', label='rb=3 (8m) blade-band', linewidth=1.0)
ax.plot(np.arange(n_total)/FPS_EST, blade_img[:, 1], 'm-', label='rb=1 (2.6m) blade-band', linewidth=0.8)
ax.plot(np.arange(n_total)/FPS_EST, blade_img[:, 5], 'g-', label='rb=5 (13m) blade-band', linewidth=0.8)
ax.plot(np.arange(n_total)/FPS_EST, blade_img[:, 10], 'b-', label='rb=10 (26m) blade-band CLEAN', linewidth=0.8, alpha=0.6)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Max signal in 200-1500 Hz (dB)")
ax.set_title("Time-series at close-range bins. Drone airborne should = high blade-band signal.")
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("DRONE FLY: time-vs-range PMM analysis", fontsize=13)
plt.tight_layout()
out_path = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\drone_fly_v2_analysis.png"
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print("\nSaved plot:", out_path)
