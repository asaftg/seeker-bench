"""Sniff COM11 for ~5 seconds and print whatever the chip emits at boot.
Looks for the patched-firmware markers from the README."""
import serial
import time

s = serial.Serial("COM11", 115200, timeout=0.5)
time.sleep(0.2)
s.reset_input_buffer()
buf = bytearray()
deadline = time.monotonic() + 5.0
while time.monotonic() < deadline:
    n = s.in_waiting
    if n:
        buf.extend(s.read(n))
    else:
        time.sleep(0.05)
s.close()

text = buf.decode("ascii", errors="replace")
print(f"--- {len(buf)} bytes received ---")
print(text)
print("--- end ---")

markers = {
    "PAD_BYPASS": "PAD_BYPASS" in text,
    "Init Calibration Status": "Init Calibration" in text,
    "mmwDemo": ("mmwDemo" in text) or ("mmw_demo" in text),
    "SEEKER PATCH": "SEEKER" in text,
}
print()
for k, v in markers.items():
    print(f"  {k}: {'YES' if v else 'no'}")
