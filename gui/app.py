"""
FastAPI backend for the Seeker-01 dashboard.

Two endpoints:
    GET  /                  serves the static SPA (gui/static/index.html)
    WS   /ws/sensors        streams a JSON frame at ~ws_fps

Everything the browser needs lives on one WebSocket message.
The handler pulls the latest ThermalFrame from FrameBus, bridges
it through sensor_bridge, and sends it. No REST config endpoints
in Phase A — we'll add them when the Engineering tab needs them.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from common.config import load_config
from common.frame_bus import BUS
from common.frames import Topic
from common.logging_setup import get_logger
from gui.sensor_bridge import build_ws_message

log = get_logger(__name__)


def _static_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parent
    return base / "static"


def create_app(thermal_manager=None, eo_manager=None, gimbal_manager=None) -> FastAPI:
    """Create the FastAPI app.

    `thermal_manager` and `eo_manager` are optional — when provided, the
    runtime config endpoints in this module can mutate detector
    parameters live and swap capture devices without restarting.
    """
    app = FastAPI(title="Seeker-01 Bench Test", version="0.1.0")
    app.state.thermal_manager = thermal_manager
    app.state.eo_manager = eo_manager
    app.state.gimbal_manager = gimbal_manager

    static_dir = _static_dir()
    if static_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(static_dir)),
            name="static",
        )

    # Disable caching on dev assets so edits show up on reload.
    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/")
    def root():
        index = static_dir / "index.html"
        if not index.exists():
            return {"status": "ok", "message": "GUI static files not found", "path": str(index)}
        return FileResponse(str(index), headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config/heat_detector")
    def get_heat_detector_config():
        tm = app.state.thermal_manager
        if tm is None or tm._detector is None:
            return {"available": False}
        cfg = tm._detector.cfg
        return {
            "available": True,
            "threshold_k": cfg.threshold_k,
            "min_blob_area_px": cfg.min_blob_area_px,
            "max_blob_area_px": cfg.max_blob_area_px,
            "max_detections": cfg.max_detections,
        }

    @app.post("/api/config/thermal")
    async def set_thermal_config(request: Request):
        tm = app.state.thermal_manager
        if tm is None:
            return Response(status_code=503)
        body = await request.json()
        if "zoom_preset" in body:
            ok = tm.set_zoom_preset(str(body["zoom_preset"]))
            if not ok:
                return Response(status_code=400, content=f"unknown preset {body['zoom_preset']}")
        if "device_index" in body:
            raw = body["device_index"]
            try:
                idx: int | str = int(raw)
            except (TypeError, ValueError):
                idx = str(raw)
            tm.set_device(idx)
            log.info("Thermal device_index -> %s", idx)
        return {"zoom_preset": tm._zoom_preset, "device_index": tm.device_index}

    @app.get("/api/devices/cameras")
    def list_cameras():
        """Enumerate working cv2 camera indices for the GUI selectors.

        Uses the EO package's enumerate_cameras() because it does a safe
        open+release probe. Returns thermal/EO currently-active indices
        so the dropdowns can default to the right rows.
        """
        from eo.webcam_capture import enumerate_cameras
        cams = enumerate_cameras(max_index=6)
        tm = app.state.thermal_manager
        em = app.state.eo_manager
        thermal_idx = None
        eo_idx = None
        try:
            if tm is not None and tm._source is not None:
                thermal_idx = getattr(tm._source, "device_index", None)
        except Exception:
            pass
        try:
            if em is not None and em._source is not None:
                eo_idx = getattr(em._source, "device_index", None)
        except Exception:
            pass
        return {
            "cameras": cams,
            "thermal_active": thermal_idx,
            "eo_active": eo_idx,
        }

    @app.post("/api/config/eo")
    async def set_eo_config(request: Request):
        em = app.state.eo_manager
        if em is None:
            return Response(status_code=503, content="EO manager not running")
        body = await request.json()
        if "device_index" in body:
            raw = body["device_index"]
            try:
                idx: int | str = int(raw)
            except (TypeError, ValueError):
                idx = str(raw)
            em.set_device(idx)
            log.info("EO device_index -> %s", idx)
        return {"device_index": em.device_index}

    @app.post("/api/config/heat_detector")
    async def set_heat_detector_config(request: Request):
        tm = app.state.thermal_manager
        if tm is None or tm._detector is None:
            return Response(status_code=503)
        body = await request.json()
        cfg = tm._detector.cfg
        # Only accept known fields and clamp to sane ranges.
        if "threshold_k" in body:
            cfg.threshold_k = max(0.5, min(100.0, float(body["threshold_k"])))
        if "min_blob_area_px" in body:
            cfg.min_blob_area_px = max(1, min(50000, int(body["min_blob_area_px"])))
        if "max_detections" in body:
            cfg.max_detections = max(1, min(50, int(body["max_detections"])))
        log.info(
            "heat_detector config updated: k=%.1f min_area=%d max_det=%d",
            cfg.threshold_k, cfg.min_blob_area_px, cfg.max_detections,
        )
        return {
            "threshold_k": cfg.threshold_k,
            "min_blob_area_px": cfg.min_blob_area_px,
            "max_detections": cfg.max_detections,
        }

    @app.websocket("/ws/sensors")
    async def sensors(ws: WebSocket) -> None:
        await ws.accept()
        cfg = load_config()
        ws_fps = float(cfg.get("gui", {}).get("ws_fps", 20))
        jpeg_quality = int(cfg.get("gui", {}).get("thermal_jpeg_quality", 80))
        period = 1.0 / max(1e-3, ws_fps)

        # Per-connection mutable state.
        #   tracked_target_id: the fused-track ID the user pressed TRACK
        #     on in the targets list. None = gimbal is manual.
        #   nir_mode / gimbal_*: pass-through to the wire payload.
        state = {
            "nir_mode": "auto",
            "tracked_target_id": None,  # int | None (mirror of gimbal_manager lock)
        }
        gm = app.state.gimbal_manager

        log.info("WebSocket client connected")

        async def _sender() -> None:
            while True:
                try:
                    tf = BUS.get_latest(Topic.THERMAL)
                    ef = BUS.get_latest(Topic.EO)
                    fused = BUS.get_latest(Topic.FUSED)
                    gstate = BUS.get_latest(Topic.GIMBAL)
                    payload = build_ws_message(
                        tf=tf,
                        ef=ef,
                        fused=fused,
                        gstate=gstate,
                        jpeg_quality=jpeg_quality,
                        nir_mode=state["nir_mode"],
                        tracked_target_id=state["tracked_target_id"],
                    )
                    # `default=str` is a safety net for numpy scalars that
                    # slip through the dataclass contracts — better to ship
                    # a stringified value than kill the WS connection.
                    text = json.dumps(payload, default=str)
                except WebSocketDisconnect:
                    raise
                except Exception:
                    log.exception("build_ws_message failed — skipping frame")
                    await asyncio.sleep(period)
                    continue

                try:
                    await ws.send_text(text)
                except WebSocketDisconnect:
                    raise
                except RuntimeError as e:
                    # Starlette raises RuntimeError once the socket is closed.
                    # Treat it as a clean disconnect rather than an error.
                    log.info("WebSocket send after close: %s", e)
                    return
                await asyncio.sleep(period)

        async def _receiver() -> None:
            async for raw in ws.iter_text():
                try:
                    cmd = json.loads(raw)
                except Exception:
                    continue
                command = cmd.get("command")

                if command == "track":
                    # Toggle a TRACK lock on a specific fused-track ID.
                    # `track_id: null` clears the lock (back to manual).
                    raw = cmd.get("track_id", None)
                    if raw is None:
                        state["tracked_target_id"] = None
                        if gm is not None: gm.set_track_target(None)
                        log.info("Track lock cleared → manual gimbal")
                    else:
                        try:
                            tid = int(raw)
                            state["tracked_target_id"] = tid
                            if gm is not None: gm.set_track_target(tid)
                            log.info("Track lock → fused-id %d", tid)
                        except (TypeError, ValueError):
                            log.warning("Bad track_id payload: %r", raw)

                elif command == "gimbal_manual":
                    # Manual dpad always wins — releases any track lock.
                    if state["tracked_target_id"] is not None:
                        log.info("Manual gimbal input — releasing track lock")
                        state["tracked_target_id"] = None
                    dp = float(cmd.get("delta_pan", 0))
                    dt = float(cmd.get("delta_tilt", 0))
                    if gm is not None:
                        gm.set_manual_delta(dp, dt)
                    log.debug("Gimbal manual delta pan=%.1f tilt=%.1f", dp, dt)

                elif command == "gimbal_home":
                    if state["tracked_target_id"] is not None:
                        state["tracked_target_id"] = None
                    if gm is not None:
                        gm.set_home()
                    log.info("Gimbal home")

                elif command == "nir":
                    mode = str(cmd.get("mode", "auto")).lower()
                    if mode in ("auto", "on", "off"):
                        state["nir_mode"] = mode
                    log.info("NIR mode → %s (no hardware — display only)", mode)

                else:
                    log.warning("Unknown WS command: %s", command)

        try:
            await asyncio.gather(_sender(), _receiver())
        except WebSocketDisconnect:
            log.info("WebSocket client disconnected")
        except Exception as e:
            log.exception("WebSocket handler error: %s", e)

    return app


# Allow `uvicorn gui.app:app` style launches (no thermal manager wired in
# this mode; runtime config endpoints return 503).
app = create_app()
