"""Verify the seeker_studio_bringup.lua hypothesis: chip emits LVDS,
DCA1000 forwards UDP to host:4098, NO disk write.

Listens on UDP 4098 for 60 sec. Counts packets, prints rate. Also
watches the PostProc folder for any new .bin file (which would indicate
the script accidentally still triggered disk recording).

Pass:  >50 MB/s sustained UDP, zero new .bin files in PostProc.
Fail:  no UDP packets (chip not streaming OR Studio config wrong)
                 OR new .bin files appearing (disk record still on).
"""
from __future__ import annotations

import os
import socket
import time

UDP_BIND = ("0.0.0.0", 4098)  # DCA1000 sends here
POSTPROC = r"C:\ti\mmwave_studio_03_01_04_04\mmWaveStudio\PostProc"
DUR_S    = 60


def list_bin_files() -> set[str]:
    try:
        return {f for f in os.listdir(POSTPROC) if f.endswith(".bin")}
    except FileNotFoundError:
        return set()


def main() -> None:
    bin_before = list_bin_files()
    print(f"[verify] PostProc .bin files at start: {sorted(bin_before)}")

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
    s.bind(UDP_BIND)
    s.settimeout(2.0)

    print(f"[verify] Listening on UDP {UDP_BIND}...")
    print("[verify] Run seeker_studio_bringup.lua in Studio NOW.")
    print()

    t_start = time.time()
    bytes_total = 0
    packets_total = 0
    last_report = t_start

    while time.time() - t_start < DUR_S:
        try:
            pkt, _ = s.recvfrom(2048)
        except socket.timeout:
            continue
        bytes_total += len(pkt)
        packets_total += 1

        now = time.time()
        if now - last_report > 3:
            elapsed = now - t_start
            mbs = (bytes_total / 1e6) / elapsed
            new_bins = list_bin_files() - bin_before
            print(f"[verify] t={elapsed:5.1f}s  pkts={packets_total:>8} "
                  f"  bytes={bytes_total/1e6:>7.1f} MB  rate={mbs:>5.1f} MB/s "
                  f"  new_bins={len(new_bins)}")
            if new_bins:
                print(f"[verify]   ! disk write detected: {sorted(new_bins)}")
            last_report = now

    s.close()
    elapsed = time.time() - t_start
    bin_after = list_bin_files()
    new_bins = bin_after - bin_before
    mbs = (bytes_total / 1e6) / elapsed if elapsed > 0 else 0

    print()
    print("=" * 64)
    print(f"[verify] DONE after {elapsed:.1f} s")
    print(f"[verify] Total packets: {packets_total}")
    print(f"[verify] Total bytes:   {bytes_total/1e6:.1f} MB")
    print(f"[verify] Sustained rate: {mbs:.1f} MB/s")
    print(f"[verify] New .bin files: {len(new_bins)}")
    if new_bins:
        print(f"[verify]   FAIL — disk write happened: {sorted(new_bins)}")
    if packets_total == 0:
        print("[verify]   FAIL — no UDP packets received.")
    if not new_bins and packets_total > 0:
        print("[verify]   PASS — UDP only, no disk write.")
    print("=" * 64)


if __name__ == "__main__":
    main()
