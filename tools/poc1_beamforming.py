"""POC #1: BEAMFORMING at the drone's known direction.

Hypothesis: by phase-steering the 16 virtual antennas at the drone's
known (az, el) instead of summing them omnidirectionally, we'll suppress
clutter from other directions and the drone signal will gain SNR.

Test method:
  At each drone frame in t=28-40s, compute the slow-time spectrum at the
  drone's range bin THREE ways:
    (A) Omnidirectional sum (what I've been doing)
    (B) Beamformed AT the drone's chip-CFAR-estimated direction
    (C) Beamformed AT a wrong direction (sanity check — should weaken signal)

  Compare drone-bin signal in each of (A), (B), (C) vs a clean bin.

  Expected: (B) gives drone bin much higher signal than (A) or (C).
"""
import sys, io, json, importlib.util, numpy as np, scipy.fft as sfft

spec_d = importlib.util.spec_from_file_location("ddma", r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\radar_dca\__pycache__\ddma.cpython-311.pyc")
ddma_mod = importlib.util.module_from_spec(spec_d); spec_d.loader.exec_module(ddma_mod)
ddma_unfold = ddma_mod.ddma_unfold
DEFAULT_ANTENNA_ORDER = ddma_mod.DEFAULT_ANTENNA_ORDER

# Try to import the AoA constants from pmm_detector (existing module)
try:
    from radar_dca.pmm_detector import (
        _VA_COLS, _VA_ROWS, _COL_SPACING_LAMBDA, _ROW_SPACING_LAMBDA,
    )
    print("Loaded antenna geometry: VA_COLS=%s VA_ROWS=%s col_spacing=%s row_spacing=%s" % (
        _VA_COLS, _VA_ROWS, _COL_SPACING_LAMBDA, _ROW_SPACING_LAMBDA))
except Exception as e:
    # Fallback: typical AWR2944P DDMA layout — 4 azimuth columns × 4 row positions
    # Spacings in units of lambda/2
    print("WARN: pmm_detector import failed (%s), using default geometry" % e)
    _VA_COLS = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    _VA_ROWS = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    _COL_SPACING_LAMBDA = 0.5
    _ROW_SPACING_LAMBDA = 0.5

import numpy as np

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

def per_va_spec_full(path, idx):
    cube=load_frame(path,idx)
    rc=sfft.rfft(cube*HANN_FAST[None,:,None],axis=1,workers=2).astype(np.complex64)
    rc[:,1:-1,:]*=2.0; rc-=rc.mean(axis=0,keepdims=True)
    virtual=ddma_unfold(rc); n_per_tx=virtual.shape[0]
    slow=virtual.reshape(n_per_tx,virtual.shape[1],N_VA)
    spec=np.fft.fft(slow*WIN_SLOW[:,None,None],n=N_FFT,axis=0)
    return np.fft.fftshift(spec, axes=0)  # (n_dop, n_range, n_va) complex

def steering_vector(az_deg, el_deg):
    """Compute the 16-element steering vector for a target at (az, el)."""
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    cols = np.array(_VA_COLS, dtype=np.float64)
    rows = np.array(_VA_ROWS, dtype=np.float64)
    # Antenna positions (in units of lambda)
    x_pos = cols * _COL_SPACING_LAMBDA  # azimuth direction
    y_pos = rows * _ROW_SPACING_LAMBDA  # elevation direction
    # Phase = 2*pi * (x*sin(az)*cos(el) + y*sin(el))
    phase = 2 * np.pi * (x_pos * np.sin(az_rad) * np.cos(el_rad) +
                          y_pos * np.sin(el_rad))
    return np.exp(1j * phase).astype(np.complex64)  # (16,)

def beamform(spec_per_va, az_deg, el_deg):
    """Apply phase-steering and sum across VAs.
    spec_per_va: (n_dop, n_range, n_va) complex
    Returns: (n_dop, n_range) complex
    """
    sv = steering_vector(az_deg, el_deg)  # (16,)
    # Conjugate sv for matched-filter beamforming: weight = conj(sv) so that
    # actual signal phase * conj(sv) phase = 0 (constructive sum)
    return (spec_per_va * np.conj(sv)[None, None, :]).sum(axis=2)

def coh_mag_db(coh_2d, rb):
    return 20*np.log10(np.maximum(np.abs(coh_2d[:, rb]), 1e-30))

# Load drone positions
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
            az_deg = np.degrees(np.arctan2(x, y))
            el_deg = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
            detections.append((t_rel, x, y, z, rm, az_deg, el_deg))

filtered = [d for d in detections if 28 <= d[0] <= 40 and d[4] >= 16]
sampled = []; last_t = -1
for d in filtered:
    if d[0] - last_t < 0.05: continue
    last_t = d[0]; sampled.append(d)
print("Sampled %d drone-frame detections in t=28-40s, rb>=6" % len(sampled))
FPS_EST = 14.0

# At each frame, compute spectrum 3 ways at drone bin AND at clean bin
# Then compare drone signal in each case
n_dop = N_FFT
freqs_hz = (np.arange(n_dop) - n_dop // 2) * DOP_BIN_HZ
pos_band = (freqs_hz > 100) & (freqs_hz < 3500)

print("\nPer-frame: max signal in drone bin and clean bin under 3 beamforming modes")
print("MODES: (A) omni-sum  (B) beamform AT drone direction  (C) beamform AT WRONG direction (10 deg off)")
print("Higher = more energy. Drone-vs-clean diff = drone-specific signature lift.")
print()
print(" t_s   rb  az_drn el_drn |  A_drn  A_cln  A_dif |  B_drn  B_cln  B_dif |  C_drn  C_cln  C_dif")
print("-" * 108)

results = []
for d in sampled[::4]:  # every 4th drone frame for speed
    t_rel, x, y, z, rm, az_deg, el_deg = d
    bin_idx = int(round(t_rel * FPS_EST))
    drone_rb = max(2, min(int(round(rm / RANGE_RES_M)), 96))
    clean_rb = (drone_rb + 5) if drone_rb < 80 else (drone_rb - 5)
    if clean_rb in (12,24,36,48,60,72,84,96): clean_rb += 2

    spec = per_va_spec_full(AIR, bin_idx)

    # Mode A: omni sum (no phase correction)
    coh_omni = spec.sum(axis=2)
    # Mode B: beamform at drone's actual direction
    coh_B = beamform(spec, az_deg, el_deg)
    # Mode C: beamform at WRONG direction (10 deg off in az)
    coh_C = beamform(spec, az_deg + 10.0, el_deg)

    def max_in_band(coh_2d, rb):
        s = 20*np.log10(np.maximum(np.abs(coh_2d[:, rb]), 1e-30))
        return float(s[pos_band].max())

    A_drn = max_in_band(coh_omni, drone_rb)
    A_cln = max_in_band(coh_omni, clean_rb)
    B_drn = max_in_band(coh_B, drone_rb)
    B_cln = max_in_band(coh_B, clean_rb)
    C_drn = max_in_band(coh_C, drone_rb)
    C_cln = max_in_band(coh_C, clean_rb)

    print(" %4.1f  %3d  %+5.1f  %+5.1f | %6.1f %6.1f %+6.1f | %6.1f %6.1f %+6.1f | %6.1f %6.1f %+6.1f" % (
        t_rel, drone_rb, az_deg, el_deg,
        A_drn, A_cln, A_drn - A_cln,
        B_drn, B_cln, B_drn - B_cln,
        C_drn, C_cln, C_drn - C_cln,
    ))
    results.append((A_drn, A_cln, B_drn, B_cln, C_drn, C_cln))

if results:
    arr = np.array(results)
    print("\n=== AVERAGED across %d frames ===" % len(results))
    print("Mode A (omni sum)        : drone=%.1f dB, clean=%.1f dB, drone-vs-clean=%+.1f dB" % (
        arr[:,0].mean(), arr[:,1].mean(), (arr[:,0]-arr[:,1]).mean()))
    print("Mode B (beamform AT drone): drone=%.1f dB, clean=%.1f dB, drone-vs-clean=%+.1f dB" % (
        arr[:,2].mean(), arr[:,3].mean(), (arr[:,2]-arr[:,3]).mean()))
    print("Mode C (beamform OFF dir) : drone=%.1f dB, clean=%.1f dB, drone-vs-clean=%+.1f dB" % (
        arr[:,4].mean(), arr[:,5].mean(), (arr[:,4]-arr[:,5]).mean()))

    print("\nKEY METRIC: 'Drone-vs-clean diff' improvement from beamforming:")
    omni_diff = (arr[:,0]-arr[:,1]).mean()
    B_diff = (arr[:,2]-arr[:,3]).mean()
    C_diff = (arr[:,4]-arr[:,5]).mean()
    print("  Beamform improvement (B - A): %+.1f dB" % (B_diff - omni_diff))
    print("  Wrong direction (C - A):      %+.1f dB (should be NEGATIVE if beamforming works)" % (C_diff - omni_diff))
    print("\nVERDICT:")
    if B_diff - omni_diff > 3.0:
        print("  PASS: beamforming improves drone-vs-clean by %+.1f dB" % (B_diff - omni_diff))
    elif B_diff - omni_diff > 0.5:
        print("  MARGINAL: beamforming gives modest improvement %+.1f dB" % (B_diff - omni_diff))
    else:
        print("  FAIL: beamforming does not help (%+.1f dB)" % (B_diff - omni_diff))
