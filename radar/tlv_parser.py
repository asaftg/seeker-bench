"""mmw_demo output packet parser for AWR2944P (DDM variant).

Ticket 5a: pure-stdlib parser for the TLV stream that ``mmw_demoDDM``
emits on the data UART at 921600 baud. No numpy, no third-party deps
— the parser is a byte-level critical-path thing and the struct
module is both faster and easier to reason about.

Packet format (little-endian, TI's mmwave demo spec):

    magic_word (8 bytes) = 02 01 04 03 06 05 08 07
    header (8 × uint32):
        version, totalPacketLen, platform, frameNumber,
        timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber
    repeat numTLVs times:
        tlv_type (uint32), tlv_length (uint32), payload (tlv_length bytes)

TLVs we care about for this build of mmw_demoDDM:

    TLV 1  — DetectedPoints:   numDetectedObj × (x, y, z, doppler) float32
    TLV 7  — SideInfo:         numDetectedObj × (snr, noise) int16 scaled ×10

The AWR2944P mmw_demoDDM appimage does NOT link the Group Tracker
(gtrack), so there is no Target-List / Target-Index TLV to parse on
this build. Targets are produced client-side by ``radar.clustering``.

Unknown TLV types are skipped. If a frame's totalPacketLen disagrees
with what we've consumed we resync on the next magic word — radar
streams get corrupted by baud-rate drift, USB hiccups and cable
bumps, and the alternative (crashing the manager) is worse.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

from common.frames import RadarDetection


# Magic word (little-endian uint64 = 0x0708050603040102, or the 8 bytes
# 02 01 04 03 06 05 08 07 as they appear on the wire).
MAGIC_WORD: bytes = b"\x02\x01\x04\x03\x06\x05\x08\x07"

# Frame header immediately following the magic word: 8 × uint32.
_HEADER_FMT: str = "<IIIIIIII"
_HEADER_SIZE: int = struct.calcsize(_HEADER_FMT)   # 32 bytes

# Full envelope size = magic + header.
_ENVELOPE_SIZE: int = len(MAGIC_WORD) + _HEADER_SIZE  # 40 bytes

# TLV header = type (u32) + length (u32). length = payload bytes only.
_TLV_HDR_FMT: str = "<II"
_TLV_HDR_SIZE: int = struct.calcsize(_TLV_HDR_FMT)   # 8 bytes

# TLV type constants — mmw_demo standard, stable across MCUPLUS SDK versions.
TLV_TYPE_DETECTED_POINTS: int = 1
TLV_TYPE_SIDE_INFO:       int = 7
# Tracker TLVs (not emitted by the AWR2944P build of mmw_demoDDM, but
# wired up so a future tracker-linked firmware drops these straight in
# without a parser rewrite).
TLV_TYPE_TARGET_LIST:   int = 308
TLV_TYPE_TARGET_INDEX:  int = 309

# Per-point record sizes.
_POINT_FMT: str = "<ffff"                # x, y, z, doppler (m, m, m, m/s)
_POINT_SIZE: int = struct.calcsize(_POINT_FMT)    # 16 bytes

_SIDEINFO_FMT: str = "<hh"              # snr, noise (int16 ×10 scaling)
_SIDEINFO_SIZE: int = struct.calcsize(_SIDEINFO_FMT)    # 4 bytes

# Protection against runaway / garbage frames. The demo frame header
# can in theory claim any packet length; a corrupted magic-word sync
# could report GBs. Real frames are under ~32 KB even at 1000 points.
_MAX_PACKET_LEN: int = 256 * 1024

# Buffer cap for the streaming parser. If we never find a magic word
# within this many bytes something's badly wrong — drop the head half
# and keep scanning rather than grow without bound.
_MAX_BUFFER_BYTES: int = 1 * 1024 * 1024


@dataclass
class RadarPacket:
    """A single fully-parsed mmw_demo frame."""
    frame_number: int
    time_cpu_cycles: int
    num_detected_obj: int
    num_tlvs: int
    sub_frame: int
    detections: List[RadarDetection]
    # Raw byte length of the source packet — exposed for diagnostics
    # (plot it vs. frame number to spot UART dropouts).
    total_packet_len: int


def parse_packet(buf: bytes) -> Optional[RadarPacket]:
    """Parse one complete mmw_demo packet starting with the magic word.

    Returns None if the buffer is too short, the magic word is missing
    or the contents fail basic sanity checks. Callers that hold a
    rolling buffer should use :class:`TLVStream` instead — this
    function is the stateless inner kernel.
    """
    if len(buf) < _ENVELOPE_SIZE:
        return None
    if buf[: len(MAGIC_WORD)] != MAGIC_WORD:
        return None

    (
        _version,
        total_packet_len,
        _platform,
        frame_number,
        time_cpu_cycles,
        num_detected_obj,
        num_tlvs,
        sub_frame,
    ) = struct.unpack_from(_HEADER_FMT, buf, offset=len(MAGIC_WORD))

    if total_packet_len < _ENVELOPE_SIZE or total_packet_len > _MAX_PACKET_LEN:
        return None
    if len(buf) < total_packet_len:
        return None

    # Walk the TLV list twice-passy: first collect DetectedPoints, then
    # apply SideInfo (per-point SNR) on top. Any order is legal in the
    # stream but detected points land before side info in practice.
    xyz_dop: List[Tuple[float, float, float, float]] = []
    side: List[Tuple[float, float]] = []

    cursor = _ENVELOPE_SIZE
    for _ in range(num_tlvs):
        if cursor + _TLV_HDR_SIZE > total_packet_len:
            break
        tlv_type, tlv_len = struct.unpack_from(_TLV_HDR_FMT, buf, offset=cursor)
        cursor += _TLV_HDR_SIZE
        if cursor + tlv_len > total_packet_len:
            break
        payload = buf[cursor : cursor + tlv_len]
        cursor += tlv_len

        if tlv_type == TLV_TYPE_DETECTED_POINTS:
            # payload = num_detected_obj × (x, y, z, doppler) float32
            n = tlv_len // _POINT_SIZE
            for i in range(n):
                x, y, z, d = struct.unpack_from(_POINT_FMT, payload, offset=i * _POINT_SIZE)
                xyz_dop.append((x, y, z, d))
        elif tlv_type == TLV_TYPE_SIDE_INFO:
            # payload = n × (snr, noise) int16 × 10 (units of 0.1 dB)
            n = tlv_len // _SIDEINFO_SIZE
            for i in range(n):
                snr10, noise10 = struct.unpack_from(
                    _SIDEINFO_FMT, payload, offset=i * _SIDEINFO_SIZE
                )
                side.append((snr10 * 0.1, noise10 * 0.1))
        # All other TLVs (stats, temperature, compressed point cloud,
        # track-list on firmwares that ship gtrack) are intentionally
        # skipped here. This build doesn't emit them.

    detections: List[RadarDetection] = []
    n_points = min(len(xyz_dop), max(num_detected_obj, len(xyz_dop)))
    # The DDM variant of mmw_demo built for this SDK (4.7.2.1) does NOT
    # emit the SideInfo TLV (type 7) even when guiMonitor's detectedObjects
    # arg is set to 2. If the TLV is missing we mark SNR as NaN (=unknown)
    # rather than 0 so the manager's SNR gate can tell "unknown" from
    # "measured at 0 dB" and pass the points through ungated.
    have_side = len(side) > 0
    for i in range(n_points):
        x, y, z, dop = xyz_dop[i]
        if i < len(side):
            snr_db, noise_db = side[i]
        elif have_side:
            snr_db, noise_db = 0.0, 0.0
        else:
            snr_db, noise_db = float("nan"), float("nan")

        r = math.sqrt(x * x + y * y + z * z)
        # az: angle from +y axis (boresight), right-positive.
        # Guard the singularity when a point lands behind the sensor
        # (shouldn't happen post-CFAR but clipped radars sometimes do).
        az = math.degrees(math.atan2(x, y)) if (x != 0.0 or y != 0.0) else 0.0
        # el: elevation above the horizontal plane, up-positive.
        horiz = math.sqrt(x * x + y * y)
        el = math.degrees(math.atan2(z, horiz)) if horiz > 1e-6 else 0.0

        detections.append(RadarDetection(
            x_m=float(x), y_m=float(y), z_m=float(z),
            doppler_mps=float(dop),
            snr_db=float(snr_db),
            noise_db=float(noise_db),
            range_m=float(r),
            az_deg=float(az),
            el_deg=float(el),
            # target_id left at its default (255 = unassigned) — the
            # RadarManager fills this in after DBSCAN clusters the cloud.
        ))

    return RadarPacket(
        frame_number=int(frame_number),
        time_cpu_cycles=int(time_cpu_cycles),
        num_detected_obj=int(num_detected_obj),
        num_tlvs=int(num_tlvs),
        sub_frame=int(sub_frame),
        detections=detections,
        total_packet_len=int(total_packet_len),
    )


class TLVStream:
    """Stateful wrapper that consumes raw bytes and yields packets.

    Usage from a serial reader:

        stream = TLVStream()
        while True:
            chunk = ser.read(4096)
            for pkt in stream.feed(chunk):
                ...  # got a full RadarPacket

    The stream resyncs on the magic word whenever the length field
    disagrees with what's buffered, so a corrupted frame costs at
    most one frame of detections, not the whole session.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> List[RadarPacket]:
        if not chunk:
            return []
        self._buf.extend(chunk)
        return self._drain()

    def _drain(self) -> List[RadarPacket]:
        out: List[RadarPacket] = []
        while True:
            # Find the next magic word.
            idx = self._buf.find(MAGIC_WORD)
            if idx < 0:
                # Keep the last 7 bytes in case a magic word straddles
                # the boundary with the next chunk — anything earlier
                # is definitely garbage.
                if len(self._buf) > 7:
                    del self._buf[: len(self._buf) - 7]
                # Runaway protection: never let the buffer grow without bound.
                if len(self._buf) > _MAX_BUFFER_BYTES:
                    del self._buf[: len(self._buf) // 2]
                return out

            # Discard anything before the magic word — it's either the
            # tail of a previous corrupted frame or pre-sync noise.
            if idx > 0:
                del self._buf[:idx]

            # Do we have enough to read the header?
            if len(self._buf) < _ENVELOPE_SIZE:
                return out

            total_packet_len = struct.unpack_from(
                "<I", self._buf, offset=len(MAGIC_WORD) + 4
            )[0]
            if total_packet_len < _ENVELOPE_SIZE or total_packet_len > _MAX_PACKET_LEN:
                # Length field is nonsense — skip this magic word and
                # scan for the next one. The head byte goes; find()
                # will relocate the next real magic word.
                del self._buf[:1]
                continue

            if len(self._buf) < total_packet_len:
                # Full packet hasn't arrived yet.
                return out

            pkt = parse_packet(bytes(self._buf[:total_packet_len]))
            if pkt is None:
                # Parse failed despite header passing. Advance one byte
                # and resync; caller's wallclock-timeout handles any
                # prolonged desync as "disconnected".
                del self._buf[:1]
                continue

            del self._buf[:total_packet_len]
            out.append(pkt)
