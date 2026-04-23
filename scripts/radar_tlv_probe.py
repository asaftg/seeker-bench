"""Dump raw TLV-parser output — num_detected_obj, TLV IDs seen, SNR range.

Bypasses RadarManager/SNR-gate entirely so we can see what the chip
is actually emitting.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from collections import Counter

import serial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from radar.tlv_parser import MAGIC_WORD, TLVStream, _HEADER_FMT, _TLV_HDR_FMT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=3_125_000)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    tlv_seen: Counter = Counter()
    ndet_hist: Counter = Counter()
    snr_min, snr_max = 1e9, -1e9
    n_frames = 0
    n_pts = 0

    stream = TLVStream()
    with serial.Serial(args.port, args.baud, timeout=0.1) as s:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            chunk = s.read(8192)
            if not chunk:
                continue
            for pkt in stream.feed(chunk):
                n_frames += 1
                n_pts += len(pkt.detections)
                ndet_hist[pkt.num_detected_obj] += 1
                for d in pkt.detections:
                    snr_min = min(snr_min, d.snr_db)
                    snr_max = max(snr_max, d.snr_db)

    # Secondary raw scan: show TLV-type histogram by re-parsing last buffer
    # (best-effort; we don't keep raw bytes above). Instead, re-probe for a
    # second and dump TLV IDs directly.
    with serial.Serial(args.port, args.baud, timeout=0.1) as s:
        buf = bytearray()
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            n = s.in_waiting
            if n:
                buf.extend(s.read(n))
            else:
                time.sleep(0.01)
    # Find first magic word and walk TLVs manually.
    idx = bytes(buf).find(MAGIC_WORD)
    if idx >= 0:
        offset = idx + 8
        if len(buf) >= offset + 32:
            hdr = struct.unpack_from(_HEADER_FMT, buf, offset=offset)
            total_len = hdr[1]
            num_tlvs = hdr[6]
            cursor = offset + 32
            end = idx + total_len
            for _ in range(num_tlvs):
                if cursor + 8 > min(end, len(buf)):
                    break
                tlv_type, tlv_len = struct.unpack_from(_TLV_HDR_FMT, buf, offset=cursor)
                tlv_seen[tlv_type] = tlv_len
                cursor += 8 + tlv_len

    print(f"frames_parsed={n_frames}  pts_total={n_pts}")
    print(f"num_detected_obj histogram: {dict(ndet_hist)}")
    if n_pts:
        print(f"snr_db range: [{snr_min:.1f}, {snr_max:.1f}]")
    print(f"TLV types in one sampled frame (type: payload_len): {dict(tlv_seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
