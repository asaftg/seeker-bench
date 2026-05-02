"""Raw bus diagnostic — send a ping at multiple baud rates and dump
whatever bytes come back. Helps tell apart 'no power on the rail',
'wrong baud', and 'TX echo on a half-duplex line that the parser
threw away'.
"""
from __future__ import annotations

import os
import sys
import time

import serial

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def ping_packet(servo_id: int) -> bytes:
    body = bytes([servo_id, 0x02, 0x01])
    chk = (~sum(body)) & 0xFF
    return b"\xff\xff" + body + bytes([chk])


def try_baud(port: str, baud: int, ids=(1, 2)) -> None:
    print(f"\n-- baud={baud} ---------------------------")
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.2)
    except Exception as e:
        print(f"  open failed: {e}")
        return
    try:
        for sid in ids:
            ser.reset_input_buffer()
            pkt = ping_packet(sid)
            ser.write(pkt)
            time.sleep(0.10)            # let echo + reply settle
            buf = ser.read(64)
            # Split: first len(pkt) bytes are TX echo, rest is real reply
            echo = buf[:len(pkt)]
            reply = buf[len(pkt):]
            echo_match = "ECHO_OK" if echo == pkt else f"ECHO_MISMATCH(got={echo.hex(' ')})"
            reply_str = reply.hex(' ') if reply else '(no reply)'
            print(f"  id={sid}  sent={pkt.hex(' ')}  total({len(buf)})  {echo_match}  reply({len(reply)})={reply_str}")
    finally:
        ser.close()


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM17"
    print(f"Probing on {port}")
    for baud in (1_000_000, 500_000, 115200, 57600, 38400):
        try_baud(port, baud)
