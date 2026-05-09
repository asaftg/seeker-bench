"""seeker_v2.main — multi-process orchestrator + GUI/WS server.

Architecture:
  capture procs (EO, thermal, radar) → shm rings + queues
                                       ↓
                                  inference proc
                                       ↓
                                  fusion proc
                                       ↓
                              MAIN proc (this file)
                              - asyncio FastAPI WS server
                              - reads frame rings + fused_q
                              - serves GUI to browser

Each capture/inference/fusion is a separate OS process with its own
GIL. Main process runs only async I/O — no CPU-bound work — so the
event loop can sustain 30+ Hz WS pushes without backpressure cascades.

Lifecycle:
  1. Parse args, load config
  2. Set up multiprocessing context ("spawn" — works on aarch64 + Windows)
  3. Spawn child processes; each owns its sensor + shm ring
  4. Start FastAPI server (uvicorn) in same loop
  5. Aggregator coroutine reads frame rings + queues, builds WS payload
  6. SIGTERM handler: send shutdown to all children, wait, unlink shm
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("seeker_v2.main")


# ── Config loader (yaml or sane defaults) ──────────────────────────────
def load_config(path: Optional[str]) -> dict:
    if path and Path(path).exists():
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log.warning("config load failed (%s): %r", path, e)
    return {}


def _build_cfgs(cfg: dict):
    """Translate a v1-style app_config.yaml into v2 process configs."""
    from seeker_v2.processes.eo_capture import EOCaptureConfig
    from seeker_v2.processes.thermal_capture import ThermalCaptureConfig
    from seeker_v2.processes.radar_capture import RadarCaptureConfig
    from seeker_v2.processes.inference import InferenceConfig
    from seeker_v2.processes.fusion import FusionConfig

    eo_cfg = cfg.get("eo", {})
    thermal_cfg = cfg.get("thermal", {})
    classifier_cfg = cfg.get("classifier", {})

    eo = EOCaptureConfig(
        dev_path="/dev/video0",
        initial_exposure_ext=int(
            (eo_cfg.get("auto_exposure", {}) or {}).get("initial_exposure_ext", 1264)
        ),
    )
    thermal = ThermalCaptureConfig(
        dev_path="/dev/video2",
        target_fps=int(thermal_cfg.get("target_fps", 60)),
        heat_tophat_kernel=11,
        jpeg_quality=int(
            (cfg.get("gui", {}) or {}).get("thermal_jpeg_quality", 70)
        ),
    )
    radar = RadarCaptureConfig(
        data_port="/dev/seeker_radar_data",
    )
    inference = InferenceConfig(
        eo_engine_path=str(
            (cfg.get("eo", {}).get("classifier", {}) or {}).get(
                "model", "models/seeker_eo_v3.engine"
            )
        ),
        eo_classify_interval=2,
        thermal_engine_path=str(
            classifier_cfg.get(
                "classifier_hv_model", "models/seeker_thermal_hv.engine"
            )
        ),
        thermal_classify_interval=int(
            classifier_cfg.get("classify_interval_frames", 4) or 4
        ),
        thermal_conf=float(classifier_cfg.get("classifier_hv_conf", 0.55)),
    )
    fusion = FusionConfig(
        rate_hz=25.0,
        eo_hfov_deg=float(eo_cfg.get("hfov_deg", 11.1)),
        eo_vfov_deg=float(eo_cfg.get("vfov_deg", 9.23)),
        thermal_hfov_deg=float(thermal_cfg.get("hfov_deg", 75.0)),
        thermal_vfov_deg=float(thermal_cfg.get("vfov_deg", 60.0)),
    )
    return eo, thermal, radar, inference, fusion


# ── Process registry ──────────────────────────────────────────────────
class ProcReg:
    """Hold all spawned children + queues + shutdown logic."""

    def __init__(self):
        self.eo_proc = None
        self.eo_ctrl = None
        self.eo_stats = None
        self.eo_jpeg_q = None
        self.thermal_proc = None
        self.thermal_ctrl = None
        self.thermal_stats = None
        self.thermal_jpeg_q = None
        # Latest JPEGs cached here by the aggregator coroutine, served
        # by /api/snapshot/{eo,thermal}.jpg.
        self.last_eo_jpeg: bytes = b""
        self.last_eo_jpeg_id: int = -1
        self.last_thermal_jpeg: bytes = b""
        self.last_thermal_jpeg_id: int = -1
        self.last_thermal_heat: list = []
        self.radar_proc = None
        self.radar_ctrl = None
        self.radar_targets_q = None
        self.radar_stats = None
        self.inference_proc = None
        self.inference_ctrl = None
        self.eo_det_q = None
        self.thermal_det_q = None
        self.inference_stats = None
        self.fusion_proc = None
        self.fusion_ctrl = None
        self.fused_q = None
        self.fusion_stats = None

        # Frame rings (consumer-side, attached for reading)
        self.eo_ring = None
        self.thermal_ring = None

    def shutdown(self):
        log.info("ProcReg shutting down children...")
        # Send shutdown commands first
        for q in (self.eo_ctrl, self.thermal_ctrl, self.radar_ctrl,
                  self.inference_ctrl, self.fusion_ctrl):
            if q is not None:
                try:
                    q.put(("shutdown", None))
                except Exception:
                    pass

        # Close consumer rings
        for r in (self.eo_ring, self.thermal_ring):
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass

        # Join processes (with timeouts)
        for proc in (self.fusion_proc, self.inference_proc,
                     self.radar_proc, self.thermal_proc, self.eo_proc):
            if proc is None:
                continue
            try:
                proc.join(timeout=3.0)
                if proc.is_alive():
                    log.warning("force-terminating %s", proc.name)
                    proc.terminate()
                    proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass


# ── FastAPI app + WS sender ───────────────────────────────────────────
def _build_fastapi_app(reg: ProcReg, cfg: dict):
    """Build the FastAPI app. Imports lazily so the orchestrator
    doesn't depend on FastAPI at module-load time."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, Response, FileResponse
    from fastapi.staticfiles import StaticFiles

    try:
        import orjson  # type: ignore
        def _dumps(obj) -> bytes:
            return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)
    except Exception:
        import json
        def _dumps(obj) -> bytes:
            return json.dumps(obj, default=str).encode("utf-8")

    app = FastAPI(title="seeker_v2")

    # Health endpoint
    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "version": "2.0.0-alpha"}

    # ── Phase 2.5: drain capture JPEG queues into reg cache ───────────
    def _drain_jpeg_queues():
        """Pull the freshest JPEG from each capture's small queue.

        Capture processes push (frame_id, jpeg_bytes[, ...]) tuples
        into a maxsize=2 queue. We drain to the latest and stash on
        ProcReg. Called from the WS coroutine and the snapshot handlers.
        """
        if reg.eo_jpeg_q is not None:
            try:
                while True:
                    item = reg.eo_jpeg_q.get_nowait()
                    if not item:
                        break
                    reg.last_eo_jpeg_id, reg.last_eo_jpeg = item[0], item[1]
            except Exception:
                pass
        if reg.thermal_jpeg_q is not None:
            try:
                while True:
                    item = reg.thermal_jpeg_q.get_nowait()
                    if not item:
                        break
                    fid, jpeg = item[0], item[1]
                    heat = item[2] if len(item) > 2 else []
                    reg.last_thermal_jpeg_id, reg.last_thermal_jpeg = fid, jpeg
                    reg.last_thermal_heat = heat
            except Exception:
                pass

    # Snapshot endpoints — the GUI's <img> tag pulls these on RAF.
    # Cache-Control: no-store so the browser always re-fetches.
    @app.get("/api/snapshot/eo.jpg")
    async def snapshot_eo():
        _drain_jpeg_queues()
        if not reg.last_eo_jpeg:
            return Response(status_code=503, content=b"no frame yet")
        return Response(content=reg.last_eo_jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store",
                                 "X-Frame-Id": str(reg.last_eo_jpeg_id)})

    @app.get("/api/snapshot/thermal.jpg")
    async def snapshot_thermal():
        _drain_jpeg_queues()
        if not reg.last_thermal_jpeg:
            return Response(status_code=503, content=b"no frame yet")
        return Response(content=reg.last_thermal_jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store",
                                 "X-Frame-Id": str(reg.last_thermal_jpeg_id)})

    # Static asset mount: seeker_v2/gui/static/
    static_dir = Path(__file__).parent / "gui" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Status endpoint
    @app.get("/api/status")
    async def status():
        s = {}
        for label, q in [("eo_stats", reg.eo_stats),
                         ("thermal_stats", reg.thermal_stats),
                         ("radar_stats", reg.radar_stats),
                         ("inference_stats", reg.inference_stats),
                         ("fusion_stats", reg.fusion_stats)]:
            if q is None:
                continue
            try:
                last = None
                while True:
                    last = q.get_nowait()
                if last is not None:
                    s[label] = last
            except Exception:
                pass
        return s

    # ── WebSocket sender ───────────────────────────────────────────────
    @app.websocket("/ws/sensors")
    async def ws_sensors(ws: WebSocket):
        await ws.accept()
        log.info("WS client connected")
        period = 1.0 / float((cfg.get("gui", {}) or {}).get("ws_fps", 30))

        last_eo_seq = 0
        last_thermal_seq = 0
        last_fused = []
        try:
            while True:
                t_tick = time.monotonic()

                # ── Drain JPEG queues so /api/snapshot stays fresh ──
                _drain_jpeg_queues()

                # ── Build frame payload ─────────────────────────────
                payload = {"ts": time.time()}

                if reg.eo_ring is not None:
                    desc, seq = reg.eo_ring.latest()
                    if desc is not None and seq != last_eo_seq:
                        last_eo_seq = seq
                        payload["eo_meta"] = {
                            "frame_id": desc.frame_id,
                            "w": desc.width, "h": desc.height,
                        }

                if reg.thermal_ring is not None:
                    desc, seq = reg.thermal_ring.latest()
                    if desc is not None and seq != last_thermal_seq:
                        last_thermal_seq = seq
                        payload["thermal_meta"] = {
                            "frame_id": desc.frame_id,
                            "w": desc.width, "h": desc.height,
                            "heat_dets": (desc.meta.get("heat_dets")
                                          or reg.last_thermal_heat),
                        }
                # Tell the client which JPEG IDs are current — frontend
                # uses these to bust the snapshot cache without polling.
                payload["snapshots"] = {
                    "eo_id": reg.last_eo_jpeg_id,
                    "thermal_id": reg.last_thermal_jpeg_id,
                }

                # Drain fused tracks queue
                if reg.fused_q is not None:
                    try:
                        while True:
                            msg = reg.fused_q.get_nowait()
                            if msg is None:
                                break
                            last_fused = msg.get("tracks", [])
                    except Exception:
                        pass
                payload["fused"] = last_fused

                # Send
                try:
                    t_send = time.monotonic()
                    await ws.send_bytes(_dumps(payload))
                    send_dt = time.monotonic() - t_send
                    if send_dt > 0.080:
                        # Yield event loop on slow send (Phase 1 fix #3)
                        await asyncio.sleep(0.005)
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    log.warning("ws send failed: %r", e)
                    break

                elapsed = time.monotonic() - t_tick
                await asyncio.sleep(max(0.0, period - elapsed))
        except WebSocketDisconnect:
            log.info("WS client disconnected")
        except Exception:
            log.exception("ws loop died")

    # Index page — serves the gui/index.html if present, else placeholder.
    gui_index = Path(__file__).parent / "gui" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        if gui_index.exists():
            return HTMLResponse(gui_index.read_text(encoding="utf-8"))
        return """<!DOCTYPE html>
<html><head><title>seeker_v2</title></head>
<body style="background:#111;color:#0f0;font-family:monospace;padding:20px">
<h1>seeker_v2 alpha</h1>
<p>GUI not built (seeker_v2/gui/index.html missing).</p>
<p>WebSocket: <code>/ws/sensors</code> · Status: <a style="color:#0f0" href="/api/status">/api/status</a></p>
</body></html>"""

    return app


# ── Main entry point ───────────────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="seeker_v2 multi-process orchestrator")
    parser.add_argument("--config", default="config/app_config.yaml",
                        help="Path to v1-style YAML config (optional).")
    parser.add_argument("--host", default="0.0.0.0",
                        help="HTTP/WS bind address")
    parser.add_argument("--port", type=int, default=8081,
                        help="HTTP/WS port (default 8081 to avoid v1 on 8080)")
    parser.add_argument("--no-eo", action="store_true")
    parser.add_argument("--no-thermal", action="store_true")
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--no-fusion", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load config
    cfg = load_config(args.config)
    eo_cfg, thermal_cfg, radar_cfg, inference_cfg, fusion_cfg = _build_cfgs(cfg)
    log.info("config loaded; spawning processes...")

    # Use spawn context — works on Linux + macOS + Windows.
    mp_ctx = mp.get_context("spawn")

    reg = ProcReg()

    # ── Spawn capture processes ────────────────────────────────────
    if not args.no_eo:
        from seeker_v2.processes.eo_capture import spawn as eo_spawn
        (reg.eo_proc, reg.eo_ctrl, reg.eo_stats,
         reg.eo_jpeg_q) = eo_spawn(mp_ctx, eo_cfg)
        log.info("EO capture spawned (pid=%d)", reg.eo_proc.pid)

    if not args.no_thermal:
        from seeker_v2.processes.thermal_capture import spawn as t_spawn
        (reg.thermal_proc, reg.thermal_ctrl, reg.thermal_stats,
         reg.thermal_jpeg_q) = t_spawn(mp_ctx, thermal_cfg)
        log.info("thermal capture spawned (pid=%d)", reg.thermal_proc.pid)

    if not args.no_radar:
        from seeker_v2.processes.radar_capture import spawn as r_spawn
        (reg.radar_proc, reg.radar_ctrl, reg.radar_targets_q,
         reg.radar_stats) = r_spawn(mp_ctx, radar_cfg)
        log.info("radar capture spawned (pid=%d)", reg.radar_proc.pid)

    # Allow capture procs to create their shm rings before consumers
    time.sleep(0.5)

    # Attach consumer rings (main needs them to read frames into the WS payload)
    try:
        from seeker_v2.processes.ipc import FrameRing
        reg.eo_ring = FrameRing.attach(
            eo_cfg.shm_name, n_slots=eo_cfg.n_slots,
            frame_bytes=eo_cfg.width * eo_cfg.height * 3,
        )
        reg.thermal_ring = FrameRing.attach(
            thermal_cfg.shm_name, n_slots=thermal_cfg.n_slots,
            frame_bytes=thermal_cfg.width * thermal_cfg.height * 3,
        )
    except Exception as e:
        log.warning("ring attach failed: %r", e)

    if not args.no_inference:
        from seeker_v2.processes.inference import spawn as i_spawn
        (reg.inference_proc, reg.inference_ctrl, reg.eo_det_q,
         reg.thermal_det_q, reg.inference_stats) = i_spawn(
            mp_ctx, inference_cfg
        )
        log.info("inference spawned (pid=%d)", reg.inference_proc.pid)

    if not args.no_fusion:
        from seeker_v2.processes.fusion import spawn as f_spawn
        eo_dq = reg.eo_det_q if reg.eo_det_q is not None else mp_ctx.Queue()
        thermal_dq = reg.thermal_det_q if reg.thermal_det_q is not None else mp_ctx.Queue()
        radar_tq = reg.radar_targets_q if reg.radar_targets_q is not None else mp_ctx.Queue()
        (reg.fusion_proc, reg.fusion_ctrl, reg.fused_q,
         reg.fusion_stats) = f_spawn(
            mp_ctx, fusion_cfg, eo_dq, thermal_dq, radar_tq,
        )
        log.info("fusion spawned (pid=%d)", reg.fusion_proc.pid)

    # Build + run FastAPI WS server
    app = _build_fastapi_app(reg, cfg)
    try:
        import uvicorn
    except Exception:
        log.error("uvicorn not installed — run: pip install fastapi uvicorn[standard] orjson")
        reg.shutdown()
        return 1

    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    # Install signal handlers for clean shutdown
    def _on_sigint(*_):
        log.info("shutdown signal received")
        server.should_exit = True

    signal.signal(signal.SIGTERM, _on_sigint)
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        log.info("starting server on http://%s:%d/", args.host, args.port)
        server.run()
    finally:
        reg.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
