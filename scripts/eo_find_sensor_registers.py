"""Find the IMX568 sensor registers that ACTUALLY control output brightness.

The Sony SMIA standard says exposure lives at 0x0202/0x0203 and analog gain
at 0x0204/0x0205. But Leopard's FX3 firmware on the LI-IMX568-GMSL2 returns
[0, 0] from reads of those registers despite the sensor producing usable
output, which means the firmware is keeping the live values somewhere else.

This script binary-searches the candidate address space:

  For each register R in CANDIDATES:
    1. Write a "dim" value (very small or zero)
    2. Open PyAV, capture 10 frames, measure mean → mean_dim
    3. Write a "bright" value (very large)
    4. Open PyAV, capture 10 frames, measure mean → mean_bright
    5. delta = mean_bright - mean_dim

Any register R where |delta| > 5 is a control knob worth investigating.

Cost: ~3-4 seconds per candidate (helper spawn + 2 PyAV opens). 30 candidates
≈ 2 minutes. Acceptable for a one-shot discovery sweep.

Output: scripts/eo_snapshots/diagnostic/register_sweep.json with per-register
deltas, sorted by |delta| descending. The top entries are the registers we
plug into the auto-calibrator next.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eo.imx568_capture import IMX568Capture  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
PY32 = REPO / "tools" / "python311-x86" / "python.exe"
OUT_DIR = REPO / "scripts" / "eo_snapshots" / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Candidate register addresses, grouped by likelihood:
#
#   Sony SMIA standard (0x0200..0x021F): exposure & gain block in every
#   Sony sensor. Even if the FX3 shadows them, writes here may still
#   poke through.
#
#   Sony manufacturer-specific (0x3000..0x3300): IMX5xx-family sensor
#   often puts true control here.
#
#   Sony 16-bit gain extension (0x0400..0x040F): seen on IMX477 / IMX568.
#
#   Common Sony global timing (0x0340..0x0344): frame_length / line_length.
#
# Each entry is (label, base_register). We test 16-bit pairs by writing
# 0x00 to base and 0xFF to base+1 (low value), then 0xFF to both (high).
SUBADDR = 0x34  # confirmed alive — chip-id read returned non-zero

CANDIDATES = [
    # SMIA standard exposure / gain
    ("smia.coarse_integ_time",    0x0202),
    ("smia.analog_gain",          0x0204),
    ("smia.digital_gain_global",  0x020E),
    ("smia.short_integ_time",     0x0218),
    ("smia.frame_length",         0x0340),
    ("smia.line_length",          0x0342),
    # Sony 16-bit gain (newer SMIA-CCS)
    ("ccs.gain_a",                0x0400),
    ("ccs.gain_b",                0x0402),
    ("ccs.coarse_int_a",          0x0404),
    ("ccs.coarse_int_b",          0x0406),
    # Sony manufacturer-specific (IMX5xx common addresses)
    ("mfg.exposure_a",            0x3014),
    ("mfg.gain_a",                0x301E),
    ("mfg.exposure_b",            0x3050),
    ("mfg.gain_b",                0x3060),
    ("mfg.exposure_c",            0x3164),
    ("mfg.0x3300",                0x3300),
    ("mfg.0x3500",                0x3500),
    ("mfg.0x3502",                0x3502),
    # Some Sony parts put gain at these
    ("alt.0x305A",                0x305A),
    ("alt.0x305C",                0x305C),
    ("alt.0x3060",                0x3060),
    ("alt.0x3066",                0x3066),
    # Black level / pedestal candidates
    ("black.0x3300",              0x3300),
    ("black.0x3308",              0x3308),
]


def write_pair(reg: int, hi: int, lo: int) -> dict:
    """Write hi→reg and lo→reg+1 in one helper call."""
    cmd = [str(PY32), str(HELPER),
           "--ae", "off", "--exposure-ext", "1000",
           "--i2c-write", f"0x{SUBADDR:02x}:0x{reg:04x}:0x{hi:02x}",
           "--i2c-write", f"0x{SUBADDR:02x}:0x{reg + 1:04x}:0x{lo:02x}",
           "--json"]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return json.loads(cp.stdout)
    except Exception:
        return {"err": cp.stderr[-300:], "raw": cp.stdout[-300:]}


def capture_mean(n_grab: int = 12, n_avg: int = 5) -> tuple[float, float, float]:
    cap = IMX568Capture(device_index="auto")
    cap.start()
    means = []
    for _ in range(n_grab):
        f = cap.grab()
        if f is None:
            continue
        y = f[..., 0] if f.ndim == 3 else f
        means.append(float(y.mean()))
    cap.stop()
    if not means:
        return -1.0, -1.0, -1.0
    last = np.array(means[-n_avg:])
    return float(last.min()), float(last.mean()), float(last.max())


def reset_to_baseline() -> None:
    """Re-baseline after each candidate: write ExposureExt only (no I2C
    pokes) so any latched register from the previous test loses precedence
    on the next iteration.
    """
    cmd = [str(PY32), str(HELPER), "--ae", "off",
           "--exposure-ext", "1000", "--json"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    time.sleep(0.4)


def baseline_mean() -> float:
    """Mean with no I2C pokes — the reference all deltas are measured against."""
    reset_to_baseline()
    _, m, _ = capture_mean()
    return m


def main() -> int:
    print(f"baseline mean (ExposureExt=1000, no I2C) ...")
    base_mean = baseline_mean()
    print(f"  baseline = {base_mean:.1f}\n")

    rows = []
    for label, reg in CANDIDATES:
        # Test 1: write 0x0000 (low end)
        write_pair(reg, 0x00, 0x00)
        time.sleep(0.4)
        _, m_lo, _ = capture_mean()

        # Test 2: write 0xFFFF (high end)
        write_pair(reg, 0xFF, 0xFF)
        time.sleep(0.4)
        _, m_hi, _ = capture_mean()

        delta = m_hi - m_lo
        delta_vs_base = max(abs(m_lo - base_mean), abs(m_hi - base_mean))
        moved = abs(delta) > 3.0 or delta_vs_base > 3.0
        flag = "  *** MOVES ***" if moved else ""
        print(f"  reg 0x{reg:04x} {label:<28s}  "
              f"lo={m_lo:5.1f}  hi={m_hi:5.1f}  "
              f"delta={delta:+6.1f}  vs_base={delta_vs_base:5.1f}{flag}")
        rows.append({
            "label": label, "reg": reg,
            "mean_lo": round(m_lo, 1),
            "mean_hi": round(m_hi, 1),
            "delta": round(delta, 1),
            "delta_vs_base": round(delta_vs_base, 1),
            "moves": moved,
        })

        # Reset before the next register so we don't compound effects.
        reset_to_baseline()

    rows.sort(key=lambda r: abs(r["delta"]) + r["delta_vs_base"], reverse=True)
    out = {
        "subaddr": SUBADDR,
        "baseline_mean": round(base_mean, 1),
        "results": rows,
    }
    out_path = OUT_DIR / "register_sweep.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {out_path}")

    movers = [r for r in rows if r["moves"]]
    if movers:
        print(f"\n=== {len(movers)} REGISTER(S) THAT MOVE THE OUTPUT ===")
        for r in movers:
            print(f"  0x{r['reg']:04x} {r['label']:<28s}  delta={r['delta']:+.1f}")
    else:
        print("\nNo registers in this sweep moved the output. Either the "
              "FX3 firmware locks the sensor's I2C control to its own "
              "internal AE/exposure pipeline, or the addresses we're hitting "
              "aren't where Leopard puts the live values. Will need the "
              "register map from Leopard support.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
