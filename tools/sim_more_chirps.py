"""Simulate 'more chirps per frame' by combining N consecutive frames'
chirps into one big coherent integration. Process drone-fly recording
this way and see if drone signal becomes detectable.

For each "super-frame" (combination of N=2,4,8 real frames):
  - Concatenate slow-time chirps across the N frames
  - Apply MTI (mean-subtract across the bigger slow-time)
  - Slow-time FFT (proportionally larger)
  - Compare drone-bin signal to clean-bin signal

Expected: each 2x increase in coherent integration = +3 dB SNR.
N=4 should give +6 dB; N=8 should give +9 dB.
"""
import sys, importlib.util, numpy as np, scipy.fft as sfft, os

try:
    spec_d = importlib.util.spec_from_file_location("ddma", r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\radar_dca\__pycache__\ddma.cpython-311.pyc")
    ddma_mod = importlib.util.module_from_spec(spec_d); spec_d.loader.exec_module(ddma_mod)
    ddma_unfold = ddma_mod.ddma_unfold
    HAS_DDMA = True
except Exception as e:
    HAS_DDMA = False; print("ddma not avail:", e)

N_CHIRPS, N_RX, N_SAMPLES = 768, 4, 192
PRF_HZ = 30478.51264858275; N_TX = 4
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS*N_RX*N_SAMPLES*2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

DRONE_FLY_BIN = r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin"
n_total = os.path.getsize(DRONE_FLY_BIN) // BYTES_PER_FRAME
print("Drone-fly: %d frames" % n_total)

def load_frame(path,idx):
    with open(path,"rb") as f: f.seek(idx*BYTES_PER_FRAME); buf=f.read(BYTES_PER_FRAME)
    raw=np.frombuffer(buf,dtype=np.int16)
    cube=(raw.reshape(N_CHIRPS,N_RX,N_SAMPLES).transpose(0,2,1).astype(np.float32))
    cube-=cube.mean(axis=1,keepdims=True); return cube

def super_frame_spec(start_idx, n_combine, drone_rb, clean_rbs):
    """Combine n_combine consecutive frames into one big slow-time stream.
    Returns dict with drone-bin and clean-bin spectra (in dB).
    """
    # Load all N frames and concatenate along chirp axis
    cubes = []
    for i in range(n_combine):
        try: cubes.append(load_frame(DRONE_FLY_BIN, start_idx + i))
        except: return None
    big_cube = np.concatenate(cubes, axis=0)  # shape: (n_combine*N_CHIRPS, N_SAMPLES, N_RX)
    n_big_chirps = big_cube.shape[0]

    # Range FFT
    rc = sfft.rfft(big_cube * HANN_FAST[None, :, None], axis=1, workers=2).astype(np.complex64)
    rc[:, 1:-1, :] *= 2.0
    # MTI on big stream
    rc -= rc.mean(axis=0, keepdims=True)

    # If DDMA available, unfold; else use full chirps
    if HAS_DDMA:
        # ddma_unfold expects (n_chirps, n_range, n_rx). It returns (n_per_tx, n_range, n_rx, n_tx).
        # n_per_tx = n_chirps // n_tx = (n_combine*768)//4 = n_combine*192
        virtual = ddma_unfold(rc)
        n_per_tx = virtual.shape[0]
        slow = virtual.reshape(n_per_tx, virtual.shape[1], 16)  # 16 VAs
        nfft = max(1024, 2 ** int(np.ceil(np.log2(n_per_tx))))
        win = np.hanning(n_per_tx).astype(np.float32)
        spec = np.fft.fft(slow * win[:, None, None], n=nfft, axis=0)
        coh = spec.sum(axis=2)  # (nfft, n_range)
        eff_prf = PRF_HZ / N_TX
    else:
        rc_sum = rc.sum(axis=2)  # (n_big_chirps, n_range)
        nfft = max(1024, 2 ** int(np.ceil(np.log2(n_big_chirps))))
        win = np.hanning(n_big_chirps).astype(np.float32)
        spec = np.fft.fft(rc_sum * win[:, None], n=nfft, axis=0)
        coh = spec
        eff_prf = PRF_HZ

    coh_shifted = np.fft.fftshift(coh, axes=0)
    mag_db = 20*np.log10(np.maximum(np.abs(coh_shifted), 1e-30))
    bin_hz = eff_prf / nfft
    freqs_hz = (np.arange(nfft) - nfft//2) * bin_hz

    # Pos-band masks
    blade_band = (freqs_hz > 200) & (freqs_hz < 1500)
    high_band = (freqs_hz > 1500) & (freqs_hz < 3500)

    # Avoid chip-spur freq band 2480-2600 (when binning fine, this band is narrow)
    chip_mask = (np.abs(freqs_hz) > 2480) & (np.abs(freqs_hz) < 2600)
    blade_band &= ~chip_mask
    high_band &= ~chip_mask

    drone_blade = mag_db[blade_band, drone_rb].max()
    drone_high = mag_db[high_band, drone_rb].max()
    drone_blade_freq = freqs_hz[blade_band][np.argmax(mag_db[blade_band, drone_rb])]

    clean_blades = [mag_db[blade_band, rb].max() for rb in clean_rbs]
    clean_highs = [mag_db[high_band, rb].max() for rb in clean_rbs]

    return {
        'drone_blade_max': drone_blade,
        'drone_blade_freq': drone_blade_freq,
        'drone_high_max': drone_high,
        'clean_blade_max_avg': np.mean(clean_blades),
        'clean_blade_max_med': np.median(clean_blades),
        'clean_high_max_avg': np.mean(clean_highs),
        'n_chirps': n_big_chirps,
        'eff_prf': eff_prf,
        'nfft': nfft,
        'bin_hz': bin_hz,
    }

# Test setup
DRONE_RB = 2  # drone at ~5m
CLEAN_RBS = [8, 10, 14, 18, 30, 40, 60]
START_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400]  # spread across recording

print("\n=== COMPARING SAME RECORDING WITH DIFFERENT 'effective chirps' ===")
print("Combining N consecutive frames simulates a chip cfg with N*768 chirps/frame")
print("Each 4x = +6 dB ideal coherent SNR boost")
print()

for n_combine in [1, 2, 4, 8]:
    print("--- N_COMBINE = %d (effective chirps = %d) ---" % (n_combine, n_combine * N_CHIRPS))
    drone_blades = []; clean_blades = []; advantages = []
    drone_freqs = []
    for start in START_FRAMES:
        if start + n_combine > n_total: continue
        r = super_frame_spec(start, n_combine, DRONE_RB, CLEAN_RBS)
        if r is None: continue
        adv = r['drone_blade_max'] - r['clean_blade_max_med']
        drone_blades.append(r['drone_blade_max'])
        clean_blades.append(r['clean_blade_max_med'])
        advantages.append(adv)
        drone_freqs.append(r['drone_blade_freq'])
        print("   frame %3d: drone_blade=%.1f dB at %+.0f Hz | clean_med=%.1f | adv=%+.1f" % (
            start, r['drone_blade_max'], r['drone_blade_freq'], r['clean_blade_max_med'], adv))
    if drone_blades:
        print("   ==> mean drone-blade=%.1f, mean clean-med=%.1f, mean ADVANTAGE=%+.1f dB" % (
            np.mean(drone_blades), np.mean(clean_blades), np.mean(advantages)))
        # Check if drone freq is consistent (real signal should pick consistent f)
        unique_close = sum(1 for f in drone_freqs if 600 <= abs(f) <= 1200)
        print("   ==> drone-blade peaks in 600-1200 Hz band: %d/%d frames" % (
            unique_close, len(drone_freqs)))
    print()

print("\n=== SUMMARY ===")
print("If drone PMM is real, advantage should INCREASE with more chirps (+3 dB per 2x)")
print("If drone PMM is below noise, advantage stays FLAT at noise level")
