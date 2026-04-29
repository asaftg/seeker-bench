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
import struct
import sys
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from common.config import load_config
from common.events import emit as emit_event
from common.frame_bus import BUS
from common.frames import Topic
from common.logging_setup import get_logger
from gui.sensor_bridge import build_ws_message, eo_to_wire, eo_to_wire_split

log = get_logger(__name__)


def _static_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        base = Path(__file__).resolve().parent
    return base / "static"


def create_app(thermal_manager=None, eo_manager=None, gimbal_manager=None,
               radar_manager=None, fusion_manager=None,
               recorder=None, config_snapshot=None) -> FastAPI:
    """Create the FastAPI app.

    `thermal_manager` and `eo_manager` are optional — when provided, the
    runtime config endpoints in this module can mutate detector
    parameters live and swap capture devices without restarting.
    `radar_manager` likewise exposes a live-tune hook for the DEV-tab
    sensitivity sliders.

    `recorder` is an optional ``JSONLRecorder`` instance. The WS
    "record" command toggles it on/off and emits matching events on
    the events stream so the recording itself documents the bracket.

    `config_snapshot` is the dict from ``common.config.load_config()``;
    written to the JSONL header when recording starts so replay tools
    have access to FOVs, calibration, etc.
    """
    app = FastAPI(title="Seeker-01 Bench Test", version="0.1.0")
    app.state.thermal_manager = thermal_manager
    app.state.eo_manager = eo_manager
    app.state.gimbal_manager = gimbal_manager
    app.state.radar_manager = radar_manager
    app.state.fusion_manager = fusion_manager
    app.state.recorder = recorder
    app.state.config_snapshot = config_snapshot or {}

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
            try:
                emit_event("zoom_preset_changed",
                           {"sensor": "thermal", "preset": str(body["zoom_preset"])})
            except Exception:
                pass
        if "device_index" in body:
            raw = body["device_index"]
            try:
                idx: int | str = int(raw)
            except (TypeError, ValueError):
                idx = str(raw)
            tm.set_device(idx)
            log.info("Thermal device_index -> %s", idx)
            try:
                emit_event("device_changed", {"sensor": "thermal", "idx": idx})
            except Exception:
                pass
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
            try:
                emit_event("device_changed", {"sensor": "eo", "idx": idx})
            except Exception:
                pass
        return {"device_index": em.device_index}

    @app.post("/api/config/eo_exposure")
    async def set_eo_exposure(request: Request):
        """Flip the SDK stream backend between AE-on and a manual lock.

        Body shape (JSON):
            {"mode": "auto"}                  → bridge AE on
            {"mode": "manual", "value": 1264} → AE off + lock to 1264

        Echoes back the resulting state so the dev-tab UI can confirm.
        """
        em = app.state.eo_manager
        if em is None:
            return Response(status_code=503, content="EO manager not running")
        body = await request.json()
        mode = str(body.get("mode", "auto")).lower()
        if mode == "manual":
            try:
                value = int(body.get("value"))
            except (TypeError, ValueError):
                return Response(status_code=400,
                                content="manual mode requires int 'value'")
            # Sanity clamp. Floor used to be 50, but on bright outdoor
            # scenes through the 35mm NIR-pass lens the IMX568 saturates
            # hard at ExposureExt>=25 — see scripts/sdk_daylight_diagnostic.py
            # output 2026-04-25, where ExposureExt in [5,15] gave the
            # only properly-exposed cloudy-daylight frames. Drop the
            # floor to 1 so daylight + NIR-pass usage actually works.
            value = max(1, min(50000, value))
            res = em.set_exposure_ext(value)
            try:
                emit_event("eo_exposure_set", {"mode": "manual", "value": value})
            except Exception:
                pass
            return res
        res = em.set_exposure_ext(None)
        try:
            emit_event("eo_exposure_set", {"mode": "auto"})
        except Exception:
            pass
        return res

    @app.get("/api/config/eo_exposure")
    async def get_eo_exposure():
        em = app.state.eo_manager
        if em is None:
            return Response(status_code=503, content="EO manager not running")
        v = getattr(em, "_manual_exposure_ext", None)
        return {
            "exposure_ext": v,
            "mode": "auto" if v is None else "manual",
        }

    @app.get("/api/eo/ae_state")
    async def get_eo_ae_state():
        """Software-AE introspection for the engineering tab.

        Returns the current bracket, last stats, and chosen ExposureExt.
        Useful for confirming the AE has converged on a tricky scene
        without tailing logs.
        """
        em = app.state.eo_manager
        if em is None:
            return Response(status_code=503, content="EO manager not running")
        try:
            return em.get_ae_state()
        except AttributeError:
            return Response(status_code=503,
                            content="AE state not available on this build")

    @app.post("/api/config/eo_lowlight")
    async def set_eo_lowlight(request: Request):
        """Toggle the EO low-light display boost (AGC stretch + gamma).

        Body: {"enabled": true|false}. Live — no helper restart needed.
        """
        em = app.state.eo_manager
        if em is None:
            return Response(status_code=503, content="EO manager not running")
        body = await request.json()
        enabled = bool(body.get("enabled", False))
        res = em.set_lowlight_mode(enabled)
        try:
            emit_event("eo_lowlight_toggled", {"enabled": enabled})
        except Exception:
            pass
        return res

    @app.get("/api/config/eo_lowlight")
    async def get_eo_lowlight():
        em = app.state.eo_manager
        if em is None:
            return Response(status_code=503, content="EO manager not running")
        return em.get_lowlight_mode()

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
        try:
            emit_event("heat_detector_set", {
                "threshold_k": cfg.threshold_k,
                "min_blob_area_px": cfg.min_blob_area_px,
                "max_detections": cfg.max_detections,
            })
        except Exception:
            pass
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
        # Separate knob for EO — it's a 2K mono sensor with a real lens
        # and the thermal default (80) shows visible JPEG ringing on
        # foliage/brick. Default 92 ≈ visually lossless on mono.
        eo_jpeg_quality = int(cfg.get("gui", {}).get("eo_jpeg_quality", 92))
        period = 1.0 / max(1e-3, ws_fps)
        # EO publishes on its own task at sensor-arrival cadence (see
        # _eo_sender below). The shared periodic sender continues to
        # carry thermal/fused/radar/gimbal at `period`, but its EO
        # field is metadata-only (no jpeg_b64) so per-tick cost is no
        # longer dominated by the EO encode/serialize.
        #
        # `_eo_idle_poll_s` is the only sleep on the EO fast path —
        # 5 ms when no new frame is on the bus, so we don't burn CPU
        # busy-looping. There is intentionally NO post-send throttle:
        # an earlier 33 ms post-send sleep was added as a "30 Hz cap"
        # but at sensor publish ~50 ms we'd miss roughly a third of
        # frames (cycle 50+15 > 50, lossy), and the GUI saw 14 Hz
        # instead of 20. Removing it lets us deliver every frame the
        # process thread publishes.
        _eo_idle_poll_s = 0.005

        # Per-connection mutable state.
        #   tracked_target_id: the fused-track ID the user pressed TRACK
        #     on in the targets list. None = gimbal is manual.
        #   nir_mode / gimbal_*: pass-through to the wire payload.
        state = {
            "nir_mode": "auto",
            "tracked_target_id": None,  # int | None (fused-track lock)
            "tracked_heat_id": None,    # int | None (dev-mode raw heat-blob lock)
            # Sync latch: becomes True once the gimbal publishes the lock
            # back at us (proves the command took). Only then will a
            # subsequent gimbal-side clear unwind our client mirror.
            "_lock_confirmed": False,
            # Recording toggle — backed by the JSONL recorder when one
            # is wired in (see main.py). The echoed value below mirrors
            # `recorder.is_recording`, so the REC pill stays in sync
            # across reconnects and across the auto-record flag.
            "recording": False,
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
                    # Reconcile client-echo lock state with the authoritative
                    # gimbal state. The gimbal thread owns the real lock;
                    # we just mirror it here for the outgoing payload. If
                    # the gimbal grace-dropped a lock, its published
                    # `tracked_target_id` goes None — clear our mirror so
                    # the GUI banner stops lying.
                    #
                    # IMPORTANT: compare against the gimbal's published
                    # tracked_target_id, NOT against `mode`. Mode lags a
                    # tick behind `set_track_heat()`, so clearing on
                    # `mode == "manual"` would race with fresh TRACK
                    # clicks and wipe them before the gimbal tick
                    # promotes mode to "auto" (symptom: first click
                    # centers but never shows "locked"; second works).
                    if gstate is not None:
                        gm_lock = getattr(gstate, "tracked_target_id", None)
                        if gm_lock is None:
                            # Only clear if we had previously confirmed a
                            # lock was live end-to-end. A tick where the
                            # client set a lock but the gimbal hasn't
                            # picked it up yet ALSO shows gm_lock=None,
                            # so we wait one more tick before clearing.
                            if state.get("_lock_confirmed"):
                                state["tracked_target_id"] = None
                                state["tracked_heat_id"] = None
                                state["_lock_confirmed"] = False
                        else:
                            state["_lock_confirmed"] = True
                    # Radar/thermal extrinsic read lock-free — floats,
                    # so a torn read just lands between two slider
                    # ticks; harmless.
                    rm_for_bias = app.state.radar_manager
                    fm_for_bias = app.state.fusion_manager
                    r_az = float(rm_for_bias.az_bias_deg) if rm_for_bias is not None else 0.0
                    r_el = float(rm_for_bias.el_bias_deg) if rm_for_bias is not None else 0.0
                    t_az = float(fm_for_bias.thermal_az_bias_deg) if fm_for_bias is not None else 0.0
                    t_el = float(fm_for_bias.thermal_el_bias_deg) if fm_for_bias is not None else 0.0
                    payload = build_ws_message(
                        tf=tf,
                        ef=ef,
                        fused=fused,
                        gstate=gstate,
                        jpeg_quality=jpeg_quality,
                        eo_jpeg_quality=eo_jpeg_quality,
                        nir_mode=state["nir_mode"],
                        tracked_target_id=state["tracked_target_id"],
                        tracked_heat_id=state["tracked_heat_id"],
                        radar_az_bias_deg=r_az,
                        radar_el_bias_deg=r_el,
                        thermal_az_bias_deg=t_az,
                        thermal_el_bias_deg=t_el,
                    )
                    # `default=str` is a safety net for numpy scalars that
                    # slip through the dataclass contracts — better to ship
                    # a stringified value than kill the WS connection.
                    # Attach the recording-state echo so the REC pill can
                    # reconcile after reconnects. (No-op for downstream
                    # consumers of build_ws_message — they don't inspect
                    # this field.)
                    rec = app.state.recorder
                    if rec is not None:
                        state["recording"] = bool(rec.is_recording)
                    payload["recording"] = bool(state.get("recording", False))
                    # Attach current radar tuning so the DEV-tab sliders
                    # can load correct initial positions (first frame only
                    # — tiny cost, keeps the sender branchless).
                    rm = app.state.radar_manager
                    if rm is not None:
                        try:
                            payload["radar_tuning"] = rm.get_tuning()
                        except Exception:
                            pass
                    # Extrinsic calibration (software bias vs. EO) — feed
                    # the DEV-tab sliders so they hydrate with current
                    # values on the first frame.
                    ext = {}
                    if rm is not None:
                        ext["radar_az_bias_deg"] = float(rm.az_bias_deg)
                        ext["radar_el_bias_deg"] = float(rm.el_bias_deg)
                    fm = app.state.fusion_manager
                    if fm is not None:
                        try:
                            ext.update(fm.get_extrinsic())
                        except Exception:
                            pass
                    if ext:
                        payload["extrinsic"] = ext
                    # Tag the message type — JS demuxes on this. The
                    # shared "sensors" message carries everything BUT the
                    # EO image bytes; the EO image is delivered via the
                    # `_eo_sender` fast path below so the EO panel can
                    # render at sensor-arrival cadence rather than at
                    # `period`. EO metadata (size, fov, detections) stays
                    # in the shared message because fused-track bbox_eo
                    # projection on the EO panel uses it every tick.
                    payload["type"] = "sensors"
                    if isinstance(payload.get("eo"), dict):
                        payload["eo"]["jpeg_b64"] = None
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

        async def _eo_sender() -> None:
            """EO fast path — binary WS frames.

            For each new EOFrame on the bus, emits ONE binary message
            framed as:

                [4 bytes LE  = header length N]
                [N bytes     = UTF-8 JSON header  {type, ts, eo: {...}}]
                [rest        = raw JPEG bytes (already encoded by the
                               EO process thread)]

            Skipping the base64 + JSON-string-escape + UTF-8-encode of
            the entire JPEG (~620 KB after expansion at 464 KB JPEGs)
            cuts both wire bytes (~33%) and Python serialization cost
            on every frame. Atomic per-message — header and JPEG can
            never get out of order with concurrent shared-sender
            messages, no client-side pairing logic required.

            JS counterpart parses the framing in `gui/static/js/main.js`
            ws.onmessage and renders via URL.createObjectURL so the
            JPEG never round-trips through a base64 data: URL.
            """
            last_frame_id = -1
            while True:
                ef = BUS.get_latest(Topic.EO)
                fid = getattr(ef, "frame_id", None) if ef is not None else None
                if fid is None or fid == last_frame_id:
                    await asyncio.sleep(_eo_idle_poll_s)
                    continue
                last_frame_id = int(fid)
                try:
                    eo_hdr, jpeg_bytes = eo_to_wire_split(
                        ef, jpeg_quality=eo_jpeg_quality
                    )
                    hdr_obj = {"type": "eo_only", "ts": time.time(), "eo": eo_hdr}
                    hdr_bytes = json.dumps(hdr_obj, default=str).encode("utf-8")
                    payload = (
                        struct.pack("<I", len(hdr_bytes))
                        + hdr_bytes
                        + (jpeg_bytes if jpeg_bytes is not None else b"")
                    )
                except Exception:
                    log.exception("eo_sender encode failed — skipping frame")
                    await asyncio.sleep(_eo_idle_poll_s)
                    continue
                try:
                    await ws.send_bytes(payload)
                except WebSocketDisconnect:
                    raise
                except RuntimeError as e:
                    log.info("WebSocket send after close: %s", e)
                    return
                # Yield to the shared _sender / _receiver so a tight
                # eo_only loop doesn't monopolize the event loop.
                # Zero-duration sleep = cooperative yield, no rate cap.
                await asyncio.sleep(0)

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
                        log.info("Fused track lock cleared → manual gimbal")
                        emit_event("track_released", {})
                    else:
                        try:
                            tid = int(raw)
                            state["tracked_target_id"] = tid
                            # Fused lock supersedes any dev-mode heat lock
                            state["tracked_heat_id"] = None
                            if gm is not None: gm.set_track_target(tid)
                            log.info("Track lock → fused-id %d", tid)
                            emit_event("track_engaged", {"target_id": tid})
                        except (TypeError, ValueError):
                            log.warning("Bad track_id payload: %r", raw)

                elif command == "track_heat":
                    # Dev-mode: lock the gimbal onto a raw heat-blob ID.
                    # `heat_id: null` clears. Mutually exclusive with
                    # the fused track lock.
                    raw = cmd.get("heat_id", None)
                    if raw is None:
                        state["tracked_heat_id"] = None
                        if gm is not None: gm.set_track_heat(None)
                        log.info("Heat track lock cleared → manual gimbal")
                        emit_event("track_heat_released", {})
                    else:
                        try:
                            hid = int(raw)
                            state["tracked_heat_id"] = hid
                            state["tracked_target_id"] = None
                            if gm is not None: gm.set_track_heat(hid)
                            log.info("Track lock → heat-id H#%d", hid)
                            emit_event("track_heat_engaged", {"heat_id": hid})
                        except (TypeError, ValueError):
                            log.warning("Bad heat_id payload: %r", raw)

                elif command == "gimbal_manual":
                    # Manual dpad always wins — releases any track lock
                    # (both fused and heat).
                    if state["tracked_target_id"] is not None:
                        log.info("Manual gimbal input — releasing fused track lock")
                        state["tracked_target_id"] = None
                        emit_event("track_released", {"reason": "manual_input"})
                    if state["tracked_heat_id"] is not None:
                        log.info("Manual gimbal input — releasing heat track lock")
                        state["tracked_heat_id"] = None
                        emit_event("track_heat_released", {"reason": "manual_input"})
                    dp = float(cmd.get("delta_pan", 0))
                    dt = float(cmd.get("delta_tilt", 0))
                    if gm is not None:
                        gm.set_manual_delta(dp, dt)
                    log.debug("Gimbal manual delta pan=%.1f tilt=%.1f", dp, dt)
                    emit_event("gimbal_manual_input", {"dpan": dp, "dtilt": dt})

                elif command == "gimbal_absolute":
                    # Slider drag — sends an absolute setpoint angle
                    # rather than a delta. Same semantics as manual
                    # delta wrt track release: any drag releases the
                    # current track lock so the operator gets manual
                    # control immediately. The manager rate-limits the
                    # slew internally, so it's safe to fire this on
                    # every `input` event from the slider.
                    if state["tracked_target_id"] is not None:
                        log.info("Manual gimbal input (slider) — releasing fused track lock")
                        state["tracked_target_id"] = None
                        emit_event("track_released", {"reason": "slider_input"})
                    if state["tracked_heat_id"] is not None:
                        log.info("Manual gimbal input (slider) — releasing heat track lock")
                        state["tracked_heat_id"] = None
                        emit_event("track_heat_released", {"reason": "slider_input"})
                    pan_deg  = float(cmd.get("pan_deg", 0))
                    tilt_deg = float(cmd.get("tilt_deg", 0))
                    if gm is not None:
                        gm.set_manual_absolute(pan_deg, tilt_deg)
                    log.debug("Gimbal absolute pan=%.1f tilt=%.1f", pan_deg, tilt_deg)
                    emit_event("gimbal_absolute_input",
                               {"pan": pan_deg, "tilt": tilt_deg})

                elif command == "gimbal_home":
                    if state["tracked_target_id"] is not None:
                        state["tracked_target_id"] = None
                        emit_event("track_released", {"reason": "home"})
                    if state["tracked_heat_id"] is not None:
                        state["tracked_heat_id"] = None
                        emit_event("track_heat_released", {"reason": "home"})
                    if gm is not None:
                        gm.set_home()
                    log.info("Gimbal home")
                    emit_event("gimbal_home_pressed", {})

                elif cmd.get("type") == "synthetic_target" or command == "synthetic_target":
                    # User drew a bbox on the thermal panel — seed a
                    # synthetic OF-only track. Payload:
                    #   {type:"synthetic_target", bbox:[x,y,w,h]}
                    tm_ref = app.state.thermal_manager
                    bbox = cmd.get("bbox")
                    if tm_ref is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        log.warning("synthetic_target: bad payload / no thermal manager")
                    else:
                        try:
                            x, y, w, h = (int(v) for v in bbox)
                        except (TypeError, ValueError):
                            log.warning("synthetic_target: non-int bbox %r", bbox)
                            continue
                        tid = tm_ref.seed_synthetic_target(x, y, w, h)
                        # Auto-lock the gimbal onto the synthetic target.
                        # A user-drawn box is an explicit "track this" gesture
                        # — requiring a second click on a TRACK button would
                        # be redundant and, worse, the synthetic track may
                        # not even appear in the fused TARGETS list that
                        # owns the TRACK buttons. Routing through the heat-
                        # track path works because `_resolve_heat_track`
                        # reads from `tf.heat_tracks`, which already
                        # includes synthetic tracks.
                        if tid is not None and gm is not None:
                            state["tracked_heat_id"] = int(tid)
                            state["tracked_target_id"] = None
                            gm.set_track_heat(int(tid))
                            log.info("Gimbal auto-locked on synthetic target id=%d", tid)
                        emit_event("synthetic_target_drawn",
                                   {"bbox": [int(x), int(y), int(w), int(h)],
                                    "tid": int(tid) if tid is not None else None})

                elif cmd.get("type") == "clear_synthetic_target" or command == "clear_synthetic_target":
                    tm_ref = app.state.thermal_manager
                    if tm_ref is not None:
                        tm_ref.clear_synthetic_target()
                    # Drop any gimbal lock on the synthetic so we don't
                    # keep chasing a phantom ID after the user cleared it.
                    if state["tracked_heat_id"] is not None:
                        state["tracked_heat_id"] = None
                        if gm is not None:
                            gm.set_track_heat(None)
                    emit_event("synthetic_target_cleared", {})

                elif command == "nir":
                    # Retained for backwards-compat with older GUI builds
                    # that still ship the NIR toggle. The current GUI no
                    # longer sends this (NIR is a manual flashlight), but
                    # accepting it silently avoids noisy "unknown command"
                    # warnings during the transition.
                    mode = str(cmd.get("mode", "auto")).lower()
                    if mode in ("auto", "on", "off"):
                        state["nir_mode"] = mode

                elif command == "radar_tune":
                    # Live-update radar filter + cluster knobs from the
                    # DEV-tab sliders. Any subset may be present — the
                    # RadarManager.set_tuning contract ignores None.
                    rm = app.state.radar_manager
                    if rm is None:
                        pass  # radar disabled — ignore silently
                    else:
                        try:
                            rm.set_tuning(
                                snr_min_db=cmd.get("snr_min_db"),
                                max_range_m=cmd.get("max_range_m"),
                                az_half_deg=cmd.get("az_half_deg"),
                                speed_min_mps=cmd.get("speed_min_mps"),
                                range_min_m=cmd.get("range_min_m"),
                                cluster_eps_pos_m=cmd.get("cluster_eps_pos_m"),
                                cluster_eps_dop_mps=cmd.get("cluster_eps_dop_mps"),
                                cluster_min_samples=cmd.get("cluster_min_samples"),
                            )
                            # Emit one event per tune. Sliders fire on
                            # every input event so this can be busy;
                            # the recorder + replay tools collapse
                            # adjacent events on display anyway.
                            emit_event("radar_tune", {
                                k: v for k, v in cmd.items()
                                if k != "command" and v is not None
                            })
                        except Exception as e:
                            log.warning("radar_tune failed: %s", e)

                elif command == "extrinsic_tune":
                    # Live-update software extrinsic (az/el bias) used
                    # to align radar + thermal to EO (ground truth).
                    # Any subset of the four knobs may be present.
                    # Phase 2: radar bias has TWO consumers — RadarManager
                    # (projection-overlay path) and FusionManager (late-
                    # fusion observation path). Route to both so the
                    # cyan radar bbox and the green fused bbox shift
                    # together when the slider moves.
                    rm = app.state.radar_manager
                    fm = app.state.fusion_manager
                    r_az = cmd.get("radar_az_bias_deg")
                    r_el = cmd.get("radar_el_bias_deg")
                    if rm is not None and (r_az is not None or r_el is not None):
                        try:
                            rm.set_extrinsic(
                                az_bias_deg=r_az,
                                el_bias_deg=r_el,
                            )
                        except Exception as e:
                            log.warning("radar extrinsic_tune (rm) failed: %s", e)
                    if fm is not None:
                        t_az = cmd.get("thermal_az_bias_deg")
                        t_el = cmd.get("thermal_el_bias_deg")
                        if (t_az is not None or t_el is not None
                                or r_az is not None or r_el is not None):
                            try:
                                fm.set_extrinsic(
                                    thermal_az_bias_deg=t_az,
                                    thermal_el_bias_deg=t_el,
                                    radar_az_bias_deg=r_az,
                                    radar_el_bias_deg=r_el,
                                )
                            except Exception as e:
                                log.warning("extrinsic_tune (fm) failed: %s", e)
                    try:
                        emit_event("extrinsic_tune", {
                            k: cmd.get(k) for k in (
                                "radar_az_bias_deg", "radar_el_bias_deg",
                                "thermal_az_bias_deg", "thermal_el_bias_deg")
                            if cmd.get(k) is not None
                        })
                    except Exception:
                        pass

                elif command == "extrinsic_tune_done":
                    # Optional debounced "user finished dragging" marker
                    # the GUI may or may not emit; harmless if absent.
                    emit_event("extrinsic_tune_done", {
                        "radar_az_bias_deg": cmd.get("radar_az_bias_deg"),
                        "radar_el_bias_deg": cmd.get("radar_el_bias_deg"),
                        "thermal_az_bias_deg": cmd.get("thermal_az_bias_deg"),
                        "thermal_el_bias_deg": cmd.get("thermal_el_bias_deg"),
                    })

                elif command == "extrinsic_save":
                    # Persist current az/el biases for radar + thermal
                    # to config/calibration.json so they survive restart.
                    # Source of truth = the LIVE manager state (not the
                    # WS payload), because the user may have nudged
                    # past the last extrinsic_tune that the WS captured.
                    rm = app.state.radar_manager
                    fm = app.state.fusion_manager
                    payload: dict = {}
                    if rm is not None:
                        try:
                            r = rm.get_extrinsic()
                            payload["radar_az"] = float(r.get("az_bias_deg", 0.0))
                            payload["radar_el"] = float(r.get("el_bias_deg", 0.0))
                        except Exception as e:
                            log.warning("get_extrinsic(radar) failed: %s", e)
                    if fm is not None:
                        try:
                            t = fm.get_extrinsic()
                            payload["thermal_az"] = float(t.get("thermal_az_bias_deg", 0.0))
                            payload["thermal_el"] = float(t.get("thermal_el_bias_deg", 0.0))
                        except Exception as e:
                            log.warning("get_extrinsic(thermal) failed: %s", e)
                    try:
                        from common import calibration_store
                        path = calibration_store.save(**payload)
                        log.info("Extrinsic calibration saved to %s: %s",
                                 path, payload)
                        emit_event("extrinsic_saved", {"biases": payload,
                                                       "path": str(path)})
                        await ws.send_json({
                            "event": "extrinsic_saved",
                            "ok": True,
                            "path": str(path),
                            "values": payload,
                        })
                    except Exception as e:
                        log.exception("extrinsic_save failed: %s", e)
                        try:
                            await ws.send_json({
                                "event": "extrinsic_saved",
                                "ok": False,
                                "error": str(e),
                            })
                        except Exception:
                            pass

                elif command == "record":
                    # Drive the JSONL recorder. ``on=true`` opens a fresh
                    # file; ``on=false`` flushes and closes. Idempotent.
                    on = bool(cmd.get("on", False))
                    rec = app.state.recorder
                    if rec is None:
                        log.warning("record cmd ignored — no recorder wired")
                        state["recording"] = False
                    else:
                        try:
                            if on and not rec.is_recording:
                                # The "recording_started" event is emitted
                                # AFTER start() so it lands inside the new
                                # file. We pass the live config snapshot so
                                # replay tools see the full config used at
                                # capture time.
                                cfg_snap = app.state.config_snapshot or load_config()
                                path = rec.start(config_snapshot=cfg_snap)
                                state["recording"] = True
                                log.info("Recording → ON: %s", path)
                                emit_event("recording_started",
                                           {"path": str(path)})
                            elif (not on) and rec.is_recording:
                                # Emit the stopped event BEFORE closing the
                                # file so it gets written.
                                emit_event("recording_stopped", {})
                                path = rec.stop()
                                state["recording"] = False
                                log.info("Recording → OFF: %s", path)
                                # Optional rename: client may send
                                # `rename_to: "my_run_3"` along with
                                # the off command. Sanitize, append
                                # .jsonl if missing, and rename in
                                # the same recordings/ dir. The
                                # original timestamp filename is
                                # used as fallback on collision.
                                rename_to = cmd.get("rename_to")
                                if rename_to and path:
                                    try:
                                        import os, re
                                        safe = re.sub(
                                            r"[^A-Za-z0-9 _\-\.]", "_",
                                            str(rename_to))[:80].strip()
                                        if safe:
                                            if not safe.lower().endswith(".jsonl"):
                                                safe = safe + ".jsonl"
                                            new_path = os.path.join(
                                                os.path.dirname(path), safe)
                                            # Avoid overwriting an
                                            # existing file — append a
                                            # numeric suffix until free.
                                            base, ext = os.path.splitext(new_path)
                                            n = 1
                                            while os.path.exists(new_path):
                                                new_path = f"{base}_{n}{ext}"
                                                n += 1
                                            os.rename(path, new_path)
                                            log.info("Recording renamed -> %s",
                                                     new_path)
                                            try:
                                                await ws.send_json({
                                                    "event": "recording_renamed",
                                                    "ok": True,
                                                    "path": new_path,
                                                })
                                            except Exception:
                                                pass
                                    except Exception as e:
                                        log.warning("rename failed: %s", e)
                            else:
                                state["recording"] = bool(rec.is_recording)
                        except Exception:
                            log.exception("record toggle failed")
                            state["recording"] = bool(rec.is_recording)

                else:
                    log.warning("Unknown WS command: %s", command)

        try:
            await asyncio.gather(_sender(), _eo_sender(), _receiver())
        except WebSocketDisconnect:
            log.info("WebSocket client disconnected")
        except Exception as e:
            log.exception("WebSocket handler error: %s", e)

    return app


# Allow `uvicorn gui.app:app` style launches (no thermal manager wired in
# this mode; runtime config endpoints return 503).
app = create_app()
