"""Radar capture process — TI AWR2944P over USB-serial.

Owns /dev/seeker_radar_data (FTDI @ 3.125 Mbaud) for the streaming
TLV data port. CLI control (start/stop) lives in the main process
(infrequent and human-driven).

This process:
  1. Reads the TLV byte stream
  2. Parses point-cloud detections per frame
  3. Runs cKDTree-based DBSCAN clustering (Phase 1 fix)
  4. Maintains per-cluster Kalman tracker
  5. Publishes frame-by-frame radar "targets" (already-tracked) via
     a multiprocessing.Queue to the fusion process

There is no shared-memory ring for radar — payloads are tiny (a few
hundred floats per frame max), so a Queue is the right abstraction.

Replaces v1 radar/{tlv_parser,clustering,radar_manager}.py for the
data-streaming path. Chip startup, sensorStart, and CLI commands are
still in the main process (radar_manager helper module).
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("seeker_v2.radar_capture")


@dataclass
class RadarCaptureConfig:
    data_port: str = "/dev/seeker_radar_data"
    baud: int = 3125000
    # DBSCAN
    eps_pos_m: float = 1.5
    eps_dop_mps: float = 2.5
    min_samples: int = 3
    # Tracker (Kalman per cluster)
    track_match_gate_m: float = 2.0
    coast_max_misses: int = 8
    coast_vel_halflife_s: float = 0.5


def run(cfg: RadarCaptureConfig, ctrl_q, targets_q, stats_q) -> int:
    """Process entry point.

    targets_q: multiprocessing.Queue of dicts:
        {"kind": "radar_frame", "frame_id": N, "ts": float,
         "targets": [{"tid", "x", "y", "z", "vx", "vy", "vz",
                       "az_deg", "el_deg", "range_m", "rcs",
                       "miss_count", "hits"}, ...]}
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _stop = {"flag": False}

    def _on_sig(*_):
        _stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    # Import the v1 TLV parser + clustering — these don't change in v2.
    # (Phase 1 already swapped DBSCAN to cKDTree, so this gets the win.)
    try:
        from radar.tlv_parser import open_tlv_stream, parse_packet
        from radar.clustering import RadarClusterer, ClusterParams
    except Exception as e:
        log.error("radar imports failed: %r", e)
        return 1

    clusterer = RadarClusterer(ClusterParams(
        eps_pos_m=cfg.eps_pos_m,
        eps_dop_mps=cfg.eps_dop_mps,
        min_samples=cfg.min_samples,
    ))

    try:
        stream = open_tlv_stream(cfg.data_port, cfg.baud)
    except Exception as e:
        log.error("failed to open TLV stream %s @ %d: %r", cfg.data_port, cfg.baud, e)
        return 1

    frame_id = 0
    last_stats_emit = time.monotonic()
    fps_window: list = []

    try:
        while not _stop["flag"]:
            try:
                while True:
                    cmd = ctrl_q.get_nowait()
                    if cmd is None or cmd[0] == "shutdown":
                        _stop["flag"] = True
                        break
            except Exception:
                pass

            if _stop["flag"]:
                break

            try:
                packet = stream.read_packet(timeout=0.2)
            except Exception as e:
                log.warning("TLV read error: %r", e)
                time.sleep(0.05)
                continue
            if packet is None:
                continue

            try:
                detections, header = parse_packet(packet)
            except Exception as e:
                log.warning("TLV parse error: %r", e)
                continue

            try:
                _filtered_dets, targets = clusterer.step(
                    detections, dt=0.05,
                )
            except Exception as e:
                log.warning("clusterer error: %r", e)
                continue

            # Build a serializable target list
            targets_serial = []
            for t in targets:
                targets_serial.append({
                    "tid": int(getattr(t, "tid", -1)),
                    "x": float(getattr(t, "x_m", 0.0)),
                    "y": float(getattr(t, "y_m", 0.0)),
                    "z": float(getattr(t, "z_m", 0.0)),
                    "vx": float(getattr(t, "vx_mps", 0.0)),
                    "vy": float(getattr(t, "vy_mps", 0.0)),
                    "vz": float(getattr(t, "vz_mps", 0.0)),
                    "miss_count": int(getattr(t, "miss_count", 0)),
                    "hits": int(getattr(t, "hits", 0)),
                })

            try:
                targets_q.put_nowait({
                    "kind": "radar_frame",
                    "frame_id": frame_id,
                    "ts": time.time(),
                    "targets": targets_serial,
                    "n_detections": len(detections),
                })
            except Exception:
                # Queue full — drop oldest, retry
                try:
                    targets_q.get_nowait()
                    targets_q.put_nowait({
                        "kind": "radar_frame",
                        "frame_id": frame_id,
                        "ts": time.time(),
                        "targets": targets_serial,
                        "n_detections": len(detections),
                    })
                except Exception:
                    pass

            frame_id += 1
            fps_window.append(time.monotonic())
            cutoff = time.monotonic() - 5.0
            while fps_window and fps_window[0] < cutoff:
                fps_window.pop(0)

            now = time.monotonic()
            if now - last_stats_emit >= 1.0:
                last_stats_emit = now
                fps = len(fps_window) / 5.0 if fps_window else 0.0
                try:
                    stats_q.put_nowait({
                        "kind": "radar_stats",
                        "frame_id": frame_id,
                        "fps_5s": fps,
                        "n_targets": len(targets_serial),
                        "n_dets": len(detections),
                    })
                except Exception:
                    pass

    except Exception:
        log.exception("Radar capture loop died")
        return 2
    finally:
        try:
            stream.close()
        except Exception:
            pass
        log.info("Radar capture: clean shutdown")

    return 0


def spawn(mp_ctx, cfg: RadarCaptureConfig):
    ctrl_q = mp_ctx.Queue(maxsize=32)
    targets_q = mp_ctx.Queue(maxsize=32)
    stats_q = mp_ctx.Queue(maxsize=32)
    proc = mp_ctx.Process(
        target=run, args=(cfg, ctrl_q, targets_q, stats_q),
        name="seeker_v2_radar_capture", daemon=False,
    )
    proc.start()
    return proc, ctrl_q, targets_q, stats_q
