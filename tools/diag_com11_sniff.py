"""Sniff COM11 for 4 seconds to determine if AWR is in ROM bootloader (pure 'C'
bytes every ~3 s) or running application firmware (BSSEV/banner text).

Run with chip already in suspected flash mode (J17 closed, power-cycled).
"""
import serial
import time

PORT = "COM11"
SECS = 4.0

s = serial.Serial(PORT, 115200, timeout=0.5)
time.sleep(0.2)
s.reset_input_buffer()
buf = bytearray()
deadline = time.monotonic() + SECS
while time.monotonic() < deadline:
    n = s.in_waiting
    if n:
        buf.extend(s.read(n))
    else:
        time.sleep(0.05)
s.close()

print(f"len={len(buf)}")
print(f"first256_hex={buf[:256].hex()}")
ascii_repr = buf[:256].decode("ascii", errors="replace")
print(f"first256_ascii={ascii_repr!r}")
nC = sum(1 for b in buf if b == 0x43)
nNonCNonWS = sum(1 for b in buf if b not in (0x43, 0x0a, 0x0d, 0x20, 0x00))
print(f"count_C={nC}  count_other_non_ws_non_null={nNonCNonWS}")
if nNonCNonWS == 0 and nC > 0:
    print("VERDICT: ROM bootloader (only 'C' handshake bytes)")
elif nNonCNonWS > 0:
    print("VERDICT: application firmware running (text bytes other than 'C' present)")
else:
    print("VERDICT: silent line (no bytes received)")
