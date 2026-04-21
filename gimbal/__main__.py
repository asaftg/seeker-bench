"""Standalone WASD gimbal CLI.

    python -m gimbal

Controls:
    W / S       tilt up / down (bigger step with shift)
    A / D       pan left / right
    H           return to home
    +/-         change step size (deg)
    Q / Ctrl-C  quit

No GUI, no fusion, no sensors. Just the gimbal manager so the user
can verify the hardware in isolation before wiring the full stack.
"""
from __future__ import annotations

import sys
import time

from common.logging_setup import configure, get_logger
from gimbal.gimbal_manager import GimbalManager


def _getch() -> str:
    """Read a single key press, cross-platform."""
    if sys.platform.startswith("win"):
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            # Arrow key prefix; read the next byte and tag it.
            ch2 = msvcrt.getch()
            return "ARROW_" + {b"H": "UP", b"P": "DOWN",
                               b"K": "LEFT", b"M": "RIGHT"}.get(ch2, "?")
        try:
            return ch.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    else:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> int:
    configure(level="INFO")
    log = get_logger(__name__)
    log.info("Starting gimbal WASD CLI")

    gm = GimbalManager()
    gm.start()

    step = 2.0
    print("=== Seeker-01 Gimbal WASD CLI ===")
    print("w/s: tilt   a/d: pan   h: home   +/-: step   q: quit")
    print(f"step = {step:.1f}°")

    try:
        while True:
            key = _getch()
            if key in ("q", "\x03"):   # q or Ctrl-C
                break
            if key == "w":           gm.set_manual_delta(0, +step)
            elif key == "s":         gm.set_manual_delta(0, -step)
            elif key == "a":         gm.set_manual_delta(-step, 0)
            elif key == "d":         gm.set_manual_delta(+step, 0)
            elif key == "ARROW_UP":    gm.set_manual_delta(0, +step)
            elif key == "ARROW_DOWN":  gm.set_manual_delta(0, -step)
            elif key == "ARROW_LEFT":  gm.set_manual_delta(-step, 0)
            elif key == "ARROW_RIGHT": gm.set_manual_delta(+step, 0)
            elif key == "h":         gm.set_home()
            elif key == "+":
                step = min(20.0, step + 1.0)
                print(f"step = {step:.1f}°")
            elif key == "-":
                step = max(0.2, step - 1.0)
                print(f"step = {step:.1f}°")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        gm.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
