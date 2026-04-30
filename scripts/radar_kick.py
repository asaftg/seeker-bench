"""Kick the AWR's LVDS DMA back to life by issuing sensorStop + sensorStart 0
through COM11. Use when /aa_diagnostics shows udp.last_packet_age_s
climbing despite mode=aa (chip's LVDS halted, TLV is still alive).

Safe to run while Seeker is up — Seeker's RadarManager only holds
COM11 transiently during cfg pushes, so most of the time it's free.
We retry briefly if the port is busy.
"""
import time
import serial


def main() -> None:
    for attempt in range(5):
        try:
            with serial.Serial("COM11", 115200, timeout=0.5) as ser:
                ser.reset_input_buffer()
                for cmd, wait in [("sensorStop", 1.0), ("sensorStart 0", 2.0)]:
                    ser.write((cmd + "\n").encode("ascii"))
                    ser.flush()
                    deadline = time.monotonic() + wait
                    buf = bytearray()
                    while time.monotonic() < deadline:
                        if ser.in_waiting:
                            buf.extend(ser.read(ser.in_waiting))
                            if b"Done" in buf or b"Error" in buf:
                                break
                        else:
                            time.sleep(0.05)
                    print(f"  {cmd!r}: {buf.decode('ascii', errors='replace').strip()!r}")
                return
        except serial.SerialException as e:
            print(f"  attempt {attempt+1}/5: COM11 busy ({e}); retrying...")
            time.sleep(0.6)
    print("FAILED to acquire COM11 after 5 attempts.")


if __name__ == "__main__":
    main()
