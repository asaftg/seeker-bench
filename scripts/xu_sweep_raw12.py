"""XU sweep with even/odd-byte fingerprint to find the FX3 RAW12 toggle.

Hypothesis: the FX3 firmware ships RAW12 over the SAME UVC endpoint as YUYV
(same byte count: 2472 * 2064 * 2 = 10,204,416). Switching content mode is
done via a vendor XU write. We sweep candidate selectors with candidate
values; after each write we capture a frame and fingerprint:

    YUYV (mono camera):  even-byte mean ~ scene Y (~120),  odd-byte mean ~3
    RAW12 packed u16:    even/odd correlated, both spread, no Y/UV split

Looking for any candidate where odd-byte mean jumps from ~3 to ~scene-mean.
"""
from __future__ import annotations
import os, sys, time, subprocess
import numpy as np

sys.path.insert(0, "/home/asaftg/seeker-bench")
from eo.leopard_linux import LeopardLinux, _SIZES

# Make all candidate selectors writable
_SIZES.update({
    0x02: 4, 0x06: 2, 0x09: 2, 0x0c: 256, 0x0d: 8,
    0x10: 262, 0x11: 33, 0x12: 2, 0x14: 4, 0x18: 2, 0x1f: 256,
})

DEV = "/dev/video0"
W, H = 2472, 2064
RAW = "/tmp/xu_fp.raw"


def fingerprint(label: str = "") -> dict:
    """Capture a few frames + return even/odd byte stats."""
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
        return {"err": f"frame too small ({len(b)})"}
    # Take last frame
    frame = b[(n - 1) * fb : n * fb]
    even, odd = frame[::2], frame[1::2]
    # Also interpret as packed u16 LE (RAW12 hypothesis)
    u16 = np.frombuffer(frame.tobytes(), dtype="<u2")
    return {
        "even_mean": float(even.mean()), "even_std": float(even.std()),
        "odd_mean": float(odd.mean()), "odd_std": float(odd.std()),
        "u16_mean": float(u16.mean()), "u16_max": int(u16.max()),
        "u16_p99": float(np.percentile(u16, 99)),
        "frames": n,
    }


def fmt(d: dict) -> str:
    if "err" in d:
        return f"ERR {d['err']}"
    return (f"even={d['even_mean']:6.1f}/{d['even_std']:5.1f}  "
            f"odd={d['odd_mean']:6.1f}/{d['odd_std']:5.1f}  "
            f"u16(p99={d['u16_p99']:6.0f}/max={d['u16_max']})")


def is_raw12_signature(d: dict, baseline: dict) -> bool:
    """odd-byte mean should jump from baseline ~3 to scene-mean (>20)."""
    if "err" in d or "err" in baseline:
        return False
    return d["odd_mean"] > 20 and d["odd_mean"] > 3 * baseline["odd_mean"]


def write_xu(sel: int, value: bytes):
    c = LeopardLinux(DEV)
    try:
        c._xu_write(sel, value)
    finally:
        c.close()


def read_xu(sel: int) -> bytes:
    c = LeopardLinux(DEV)
    try:
        return c._xu_read(sel)
    finally:
        c.close()


def restore_known_good():
    """Put camera back into XU 0x09=0 free-running mode (the soft-trigger fix)."""
    c = LeopardLinux(DEV)
    try:
        c._xu_write(0x09, (0).to_bytes(2, "little"))
    finally:
        c.close()


def main() -> int:
    print("=== Baseline (current state) ===")
    baseline = fingerprint("baseline")
    print(f"  baseline: {fmt(baseline)}")

    if "err" in baseline:
        print("FAIL: cannot get baseline frame. Is /dev/video0 free?")
        return 1

    print()
    print("=== Snapshot of current XU state ===")
    for sel in [0x02, 0x06, 0x09, 0x0c, 0x0d]:
        try:
            v = read_xu(sel)
            print(f"  sel 0x{sel:02x} ({len(v)}B) = {v.hex()}")
        except Exception as e:
            print(f"  sel 0x{sel:02x}: ERR {e}")

    print()
    print("=== XU Sweep ===")
    candidates = [
        # (sel, value_bytes, label)
        # Selector 0x02 (4B): possibly bits/mode select as u32
        (0x02, b"\x0c\x00\x00\x00", "0x02 = bits=12 (u32)"),
        (0x02, b"\x10\x00\x00\x00", "0x02 = bits=16 (u32)"),
        (0x02, b"\x01\x00\x00\x00", "0x02 = mode=1"),
        (0x02, b"\x02\x00\x00\x00", "0x02 = mode=2"),
        # Selector 0x06 (2B): currently exposure_ext, but try bits semantics
        (0x06, b"\x0c\x00", "0x06 = 12"),
        (0x06, b"\x10\x00", "0x06 = 16"),
        # Selector 0x09 (2B): currently soft-trigger; try alt modes
        (0x09, b"\x01\x00", "0x09 = 1 (re-arm)"),
        (0x09, b"\x02\x00", "0x09 = 2"),
        # Selector 0x0c (256B): bulk register configuration — most likely candidate
        # Try opcode-style payloads
        (0x0c, bytes([0x0c]).ljust(256, b"\x00"), "0x0c = [12,0,0,...] bits=12"),
        (0x0c, bytes([0x10]).ljust(256, b"\x00"), "0x0c = [16,0,0,...] bits=16"),
        (0x0c, bytes([0x01, 0x0c]).ljust(256, b"\x00"), "0x0c = [01,0c,...] mode+bits"),
        (0x0c, bytes([0x02, 0x0c]).ljust(256, b"\x00"), "0x0c = [02,0c,...] alt op+bits"),
        (0x0c, bytes([0xa5, 0x0c, 0x00]).ljust(256, b"\x00"), "0x0c = [a5,0c,...] magic+bits"),
        # Selector 0x0d (8B): currently zeros, possibly mode field
        (0x0d, b"\x0c\x00\x00\x00\x00\x00\x00\x00", "0x0d = bits=12 in [0:2]"),
        (0x0d, b"\x00\x0c\x00\x00\x00\x00\x00\x00", "0x0d = bits=12 in [2:4]"),
        (0x0d, b"\x01\x00\x0c\x00\x00\x00\x00\x00", "0x0d = mode+bits"),
    ]

    hits = []
    for sel, value, label in candidates:
        try:
            write_xu(sel, value)
        except Exception as e:
            print(f"  {label:50s}  WRITE FAIL: {e}")
            continue
        time.sleep(0.4)
        d = fingerprint(label)
        flag = "  <-- RAW12 CANDIDATE" if is_raw12_signature(d, baseline) else ""
        print(f"  {label:50s}  {fmt(d)}{flag}")
        if is_raw12_signature(d, baseline):
            hits.append((sel, value, label, d))

    print()
    print("=== Results ===")
    if hits:
        print(f"FOUND {len(hits)} candidate(s):")
        for sel, value, label, d in hits:
            print(f"  {label}: odd_mean={d['odd_mean']:.1f} (baseline {baseline['odd_mean']:.1f})")
    else:
        print("No XU value flipped odd-byte mean above baseline*3.")
        print("Likely interpretations:")
        print("  1. RAW12 toggle is in a selector we did not test")
        print("  2. RAW12 requires SetParam(W,H,...) UVC PROBE/COMMIT (XU not enough)")
        print("  3. RAW12 requires a multi-step sequence (e.g. Run() before/after toggle)")

    print()
    print("=== Restoring XU 0x09 = 0 (free-running streaming) ===")
    restore_known_good()
    print("done.")
    return 0 if hits else 2


if __name__ == "__main__":
    sys.exit(main())
