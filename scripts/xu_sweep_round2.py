"""Round 2 XU sweep: selectors I missed in round 1 + multi-step combos.

Round 1 selectors tested: 0x02, 0x06, 0x09, 0x0c, 0x0d
Round 2 adds: 0x01, 0x03, 0x04, 0x08, 0x0a, 0x0b, 0x0e
Plus a multi-step "set sensor_mode + reset stream" combo since the FX3
may only commit a content-mode change after a stream stop/start.
"""
from __future__ import annotations
import os, sys, time, subprocess
import numpy as np

sys.path.insert(0, "/home/asaftg/seeker-bench")
from eo.leopard_linux import LeopardLinux, _SIZES

_SIZES.update({
    0x01: 2, 0x02: 4, 0x03: 2, 0x04: 8, 0x06: 2, 0x07: 49, 0x08: 33,
    0x09: 2, 0x0a: 4, 0x0b: 2, 0x0c: 256, 0x0d: 8, 0x0e: 5, 0x0f: 2, 0x10: 262,
})

DEV = "/dev/video0"
W, H = 2472, 2064
RAW = "/tmp/xu_fp.raw"


def fingerprint() -> dict:
    if os.path.exists(RAW):
        os.remove(RAW)
    subprocess.run(
        ["v4l2-ctl", f"--device={DEV}",
         f"--set-fmt-video=width={W},height={H},pixelformat=YUYV"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["timeout", "4", "v4l2-ctl", f"--device={DEV}",
         "--stream-mmap=4", "--stream-count=4",
         f"--stream-to={RAW}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not os.path.exists(RAW) or os.path.getsize(RAW) < 1000:
        return {"err": "no frame"}
    b = np.fromfile(RAW, dtype=np.uint8)
    fb = W * H * 2
    n = len(b) // fb
    if n == 0:
        return {"err": "small"}
    frame = b[(n - 1) * fb : n * fb]
    even, odd = frame[::2], frame[1::2]
    u16 = np.frombuffer(frame.tobytes(), dtype="<u2")
    return {
        "even_mean": float(even.mean()), "even_std": float(even.std()),
        "odd_mean": float(odd.mean()), "odd_std": float(odd.std()),
        "u16_mean": float(u16.mean()), "u16_max": int(u16.max()),
        "u16_p99": float(np.percentile(u16, 99)),
        "byte_corr": float(np.corrcoef(even[:100000].astype(float),
                                       odd[:100000].astype(float))[0, 1]),
    }


def fmt(d: dict) -> str:
    if "err" in d:
        return f"ERR {d['err']}"
    return (f"even={d['even_mean']:6.1f}/{d['even_std']:5.1f}  "
            f"odd={d['odd_mean']:6.1f}/{d['odd_std']:5.1f}  "
            f"corr(e,o)={d['byte_corr']:+.3f}  "
            f"u16(p99={d['u16_p99']:6.0f})")


def is_raw12(d: dict, base: dict) -> bool:
    if "err" in d or "err" in base:
        return False
    # RAW12 packed: even/odd would have correlation > 0.5 (same sample's
    # high/low halves are correlated with scene), and odd_mean would lift
    # off ~0 baseline.
    return (d["odd_mean"] > 20 and d["odd_mean"] > 3 * base["odd_mean"]) or \
           (abs(d["byte_corr"]) > 0.5 and base["byte_corr"] < 0.1)


def write_xu(sel: int, value: bytes):
    c = LeopardLinux(DEV)
    try:
        c._xu_write(sel, value)
    finally:
        c.close()


def restore():
    c = LeopardLinux(DEV)
    try:
        c._xu_write(0x09, (0).to_bytes(2, "little"))
    finally:
        c.close()


def main() -> int:
    print("=== Baseline ===")
    base = fingerprint()
    print(f"  baseline: {fmt(base)}")
    if "err" in base:
        return 1

    print()
    print("=== Round 2 XU sweep (selectors 0x01, 0x03, 0x04, 0x08, 0x0a, 0x0b, 0x0e) ===")
    candidates = [
        # Sensor mode (0x01) — different modes may produce different output formats
        (0x01, b"\x01\x00", "0x01 sensor_mode=1"),
        (0x01, b"\x02\x00", "0x01 sensor_mode=2"),
        (0x01, b"\x03\x00", "0x01 sensor_mode=3"),
        (0x01, b"\x04\x00", "0x01 sensor_mode=4"),
        (0x01, b"\x05\x00", "0x01 sensor_mode=5"),
        (0x01, b"\x06\x00", "0x01 sensor_mode=6"),
        (0x01, b"\x07\x00", "0x01 sensor_mode=7"),
        # LED modes (0x03)
        (0x03, b"\x00\x00", "0x03 LED=0"),
        (0x03, b"\x01\x00", "0x03 LED=1"),
        # RGB gain (0x04, 8B)
        (0x04, b"\x00\x01\x00\x01\x00\x01\x00\x01", "0x04 RGB gain=256"),
        # 0x08 (firmware-extension, 33B) — was UUID-like
        (0x08, bytes([0x0c]).ljust(33, b"\x00"), "0x08 = bits-12 first-byte"),
        # Trigger delay (0x0a, 4B)
        (0x0a, b"\x00\x00\x00\x00", "0x0a trig_delay=0"),
        # Trigger mode (0x0b, 2B) — different from soft-trigger 0x09
        (0x0b, b"\x01\x00", "0x0b trig_mode=1"),
        (0x0b, b"\x02\x00", "0x0b trig_mode=2"),
        # Sensor reg RW (0x0e, 5B)
        (0x0e, b"\x01\x00\x16\x00\x0c", "0x0e write reg 0x0016=0x000c"),
        (0x0e, b"\x01\x30\x12\x00\x0c", "0x0e write reg 0x3012=0x000c"),
    ]

    hits = []
    for sel, value, label in candidates:
        try:
            write_xu(sel, value)
        except Exception as e:
            print(f"  {label:50s}  WRITE FAIL: {e}")
            continue
        time.sleep(0.4)
        d = fingerprint()
        flag = "  <-- RAW12 CANDIDATE" if is_raw12(d, base) else ""
        print(f"  {label:50s}  {fmt(d)}{flag}")
        if is_raw12(d, base):
            hits.append((sel, value, label, d))

    print()
    print("=== Results ===")
    if hits:
        print(f"FOUND {len(hits)} hits:")
        for sel, value, label, d in hits:
            print(f"  {label}")
    else:
        print("Round 2 negative.")
        print("Remaining theories:")
        print("  1. FX3 needs PROBE/COMMIT-time vendor extension data (not just XU)")
        print("  2. Mode change requires libusb-level alt-setting handshake")
        print("  3. Firmware was simply built without RAW12 path on Linux")

    print()
    print("=== Restoring XU 0x09 = 0 ===")
    restore()
    print("done.")
    return 0 if hits else 2


if __name__ == "__main__":
    sys.exit(main())
