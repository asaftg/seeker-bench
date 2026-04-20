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


def create_app(thermal_manager=None) -> FastAPI:
    """Create the FastAPI app.

    `thermal_manager` is optional — when provided, the runtime config
    endpoints in this module can mutate detector parameters live.
    """
    app = FastAPI(title="Seeker-01 Bench Test", version="0.1.0")
    app.state.thermal_manager = thermal_manager

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
        return {"zoom_preset": tm._zoom_preset}

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

        # Per-connection mutable state (Phase B controls)
        state = {
            "tracker_on": True,
            "nir_mode": "auto",
            "gimbal_pan": 0.0,
            "gimbal_tilt": 60.0,
        }

        log.info("WebSocket client connected")

        async def _sender() -> None:
            while True:
                tf = BUS.get_latest(Topic.THERMAL)
                payload = build_ws_message(
                    tf=tf,
                    jpeg_quality=jpeg_quality,
                    tracker_on=state["tracker_on"],
                    nir_mode=state["nir_mode"],
                    gimbal_pan=state["gimbal_pan"],
                    gimbal_tilt=state["gimbal_tilt"],
                )
                await ws.send_text(json.dumps(payload))
                await asyncio.sleep(period)

        async def _receiver() -> None:
            async for raw in ws.iter_text():
                try:
                    cmd = json.loads(raw)
                except Exception:
                    continue
                command = cmd.get("command")

                if command == "tracker":
                    new_state = str(cmd.get("state", "on")).lower() == "on"
                    state["tracker_on"] = new_state
                    # When tracker is turned off, disable auto-track on gimbal
                    tm = app.state.thermal_manager
                    if tm is not None and not new_state:
                        pass  # gimbal auto-track disabled in Ticket 6
                    log.info("Tracker toggled → %s", "ON" if new_state else "OFF")

                elif command == "gimbal_manual":
                    if not state["tracker_on"]:
                        dp = float(cmd.get("delta_pan", 0))
                        dt = float(cmd.get("delta_tilt", 0))
                        state["gimbal_pan"]  = round(state["gimbal_pan"]  + dp, 1)
                        state["gimbal_tilt"] = round(state["gimbal_tilt"] + dt, 1)
                    log.debug("Gimbal manual delta pan=%.1f tilt=%.1f", dp if not state["tracker_on"] else 0, dt if not state["tracker_on"] else 0)

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
