// Thermal view: renders the latest JPEG frame and overlays detections.
// Phase B: uses the unified drawDetectionBox with class-aware colors.
// Ticket 5: if a fused green box subsumes a raw detection, the raw box
// is suppressed so we don't stack two boxes on the same target.

import {
  drawDetectionBox,
  drawFusedBox,
  drawProjectedBox,
  drawHeatTrackDebug,
  drawSyntheticTargetBox,
  drawRubberBand,
  drawRadarBox,
  isSubsumedByFused,
  fusedIdForDet,
} from "./overlays.js";

const PANEL_SENSOR = "thermal";

export class ThermalView {
  constructor(canvasId, disconnectOverlayId) {
    this.canvas = document.getElementById(canvasId);
    this.overlay = disconnectOverlayId ? document.getElementById(disconnectOverlayId) : null;
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.img = new Image();
    this._lastFrameW = 0;
    this._lastFrameH = 0;
    this._lastDetections = [];
    this._lastFused = [];
    this._mainTargetId = null;
    this._lastHeatTracks = [];
    this._lastRadarTargets = [];
    this._devMode = false;
    // Draw-a-bbox state: when _drawMode is true, the canvas shows a
    // crosshair and click-drag paints a rubber-band rect used to seed a
    // synthetic OF-only track on the backend. _drag holds the in-flight
    // drag in CANVAS-space pixel coords; it's cleared on commit/cancel.
    this._drawMode = false;
    this._drag = null;  // {x0,y0,x1,y1} in canvas pixels, or null
    // Called on successful drag commit with IMAGE-space bbox {x,y,w,h}.
    this._onCommit = null;
    // Black-hot vs white-hot: applied only to the drawImage call, not
    // to overlays (boxes/labels stay their original colours).
    this._invert = false;
    // Optional extra rendering knobs for a mini preview instance.
    this._hideOverlays = false;
    if (this.img) {
      this.img.onload = () => this._draw();
    }
    window.addEventListener("resize", () => this._fitCanvas());
    this._fitCanvas();

    if (this.canvas) {
      this.canvas.addEventListener("mousedown", (e) => this._handleMouseDown(e));
      // Listen on window for move/up so a drag that leaves the canvas
      // still completes cleanly (matches standard rubber-band UX).
      window.addEventListener("mousemove", (e) => this._handleMouseMove(e));
      window.addEventListener("mouseup",   (e) => this._handleMouseUp(e));
    }
  }

  // ── Draw-a-bbox synthetic target API ────────────────────────────────
  // Toggled by the DRAW TARGET button in the thermal panel subbar.
  // When active, mousedown+drag+mouseup on the thermal canvas produces
  // an image-space bbox which is forwarded via onCommit to the caller
  // (which sends the WS command). ESC cancels an in-flight drag.
  setInvert(on) {
    const v = !!on;
    if (v === this._invert) return;
    this._invert = v;
    this._draw();
  }
  setHideOverlays(on) {
    this._hideOverlays = !!on;
  }
  // Public re-fit — call after the parent element becomes visible
  // (e.g. tab switch) so the canvas resizes from its 1×1 initial state.
  refit() { this._fitCanvas(); }

  setDrawMode(on, onCommit = null) {
    this._drawMode = !!on;
    this._onCommit = onCommit;
    if (!this.canvas) return;
    this.canvas.classList.toggle("draw-mode", this._drawMode);
    if (!this._drawMode) {
      this._drag = null;
      this._draw();
    }
  }
  isDrawMode() { return this._drawMode; }
  cancelDrag() {
    if (this._drag) {
      this._drag = null;
      this._draw();
    }
  }

  // Translate a mouse event to canvas-space pixels (same units as
  // this.canvas.width/height, i.e. DPR-scaled).
  _evtToCanvasPx(evt) {
    if (!this.canvas) return null;
    const rect = this.canvas.getBoundingClientRect();
    const cssX = evt.clientX - rect.left;
    const cssY = evt.clientY - rect.top;
    const sx = this.canvas.width  / rect.width;
    const sy = this.canvas.height / rect.height;
    return { x: cssX * sx, y: cssY * sy, cssX, cssY };
  }

  // Convert a canvas-space rect into image-space bbox using the same
  // letterbox transform as _draw. Returns null if image coords are
  // unavailable OR the rect is off-canvas / too small (<10x10 image px).
  _canvasRectToImageBbox(x0c, y0c, x1c, y1c) {
    const fw = this._lastFrameW, fh = this._lastFrameH;
    if (!fw || !fh) return null;
    const cw = this.canvas.width, ch = this.canvas.height;
    const scale = Math.min(cw / fw, ch / fh);
    const dw = fw * scale, dh = fh * scale;
    const dx = (cw - dw) / 2, dy = (ch - dh) / 2;

    // Clip to the image region of the canvas.
    const lo = (a, b) => Math.min(a, b);
    const hi = (a, b) => Math.max(a, b);
    const xMin = hi(dx,        lo(x0c, x1c));
    const yMin = hi(dy,        lo(y0c, y1c));
    const xMax = lo(dx + dw,   hi(x0c, x1c));
    const yMax = lo(dy + dh,   hi(y0c, y1c));
    if (xMax <= xMin || yMax <= yMin) return null;

    const ix = Math.round((xMin - dx) / scale);
    const iy = Math.round((yMin - dy) / scale);
    const iw = Math.round((xMax - xMin) / scale);
    const ih = Math.round((yMax - yMin) / scale);
    if (iw < 10 || ih < 10) return null;  // spec: ignore tiny rects
    return { x: ix, y: iy, w: iw, h: ih };
  }

  _handleMouseDown(evt) {
    if (!this._drawMode) return;
    const p = this._evtToCanvasPx(evt);
    if (!p) return;
    this._drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    evt.preventDefault();
  }
  _handleMouseMove(evt) {
    if (!this._drawMode || !this._drag) return;
    const p = this._evtToCanvasPx(evt);
    if (!p) return;
    this._drag.x1 = p.x;
    this._drag.y1 = p.y;
    this._draw();
  }
  _handleMouseUp(evt) {
    if (!this._drawMode || !this._drag) return;
    const p = this._evtToCanvasPx(evt);
    if (p) { this._drag.x1 = p.x; this._drag.y1 = p.y; }
    const d = this._drag;
    this._drag = null;
    const bbox = this._canvasRectToImageBbox(d.x0, d.y0, d.x1, d.y1);
    this._draw();
    if (bbox && typeof this._onCommit === "function") {
      this._onCommit(bbox);
    }
  }

  _fitCanvas() {
    if (!this.canvas) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    if (this._lastFrameW > 0) this._draw();
  }

  update(thermal, mainTargetId = null, fused = [], devMode = false, radarTargets = []) {
    this._mainTargetId = mainTargetId;
    this._lastFused = fused || [];
    this._devMode = !!devMode;
    this._lastRadarTargets = radarTargets || [];
    if (!thermal || !thermal.connected) {
      if (this.overlay) this.overlay.classList.remove("hidden");
      this._clear();
      return;
    }
    if (this.overlay) this.overlay.classList.add("hidden");

    this._lastFrameW = thermal.width || 0;
    this._lastFrameH = thermal.height || 0;
    this._lastDetections = thermal.detections || [];
    this._lastHeatTracks = thermal.heat_tracks || [];

    if (thermal.jpeg_b64) {
      this.img.src = "data:image/jpeg;base64," + thermal.jpeg_b64;
    } else {
      this._draw();
    }
  }

  _clear() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  _draw() {
    if (!this.ctx || !this.canvas) return;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    this.ctx.clearRect(0, 0, cw, ch);

    const fw = this._lastFrameW || this.img.naturalWidth;
    const fh = this._lastFrameH || this.img.naturalHeight;
    if (!fw || !fh) return;

    const scale = Math.min(cw / fw, ch / fh);
    const dw = fw * scale;
    const dh = fh * scale;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;

    if (this.img.complete && this.img.naturalWidth > 0) {
      if (this._invert) {
        this.ctx.save();
        this.ctx.filter = "invert(1)";
        this.ctx.drawImage(this.img, dx, dy, dw, dh);
        this.ctx.restore();
      } else {
        this.ctx.drawImage(this.img, dx, dy, dw, dh);
      }
    }

    // Mini-preview mode: image only, no boxes/labels.
    if (this._hideOverlays) {
      if (this._drag) {
        drawRubberBand(this.ctx, this._drag.x0, this._drag.y0,
                       this._drag.x1, this._drag.y1);
      }
      return;
    }

    // Raw detections — suppressed only when a 2+ sensor fused box is
    // about to be drawn on top (handled inside isSubsumedByFused).
    // Synthetic user-seeded detections are peeled off first and drawn
    // with the dedicated magenta "USER TARGET" style; they never
    // participate in fused suppression.
    for (const det of this._lastDetections) {
      if (det && det.synthetic) {
        drawSyntheticTargetBox(this.ctx, det, scale, dx, dy);
        continue;
      }
      if (isSubsumedByFused(det.bbox, this._lastFused, "bbox_thermal")) continue;
      // Pass the raw det through so fusedIdForDet can match by
      // det.track_id ↔ FusedTrack.thermal_heat_id (Phase B1, mirror
      // of the EO panel's eo_track_id link).
      const fusedId = fusedIdForDet(det.bbox, this._lastFused, "bbox_thermal", 0.20, det);
      const isMain = this._mainTargetId != null &&
                     fusedId != null &&
                     String(fusedId) === String(this._mainTargetId);
      drawDetectionBox(this.ctx, det, scale, dx, dy, isMain, fusedId, "T");
    }

    // Fused overlay rules:
    //   >=2 sensors → solid green box on every panel (the "confirmed" lock).
    //   =1 sensor   → nothing on the detecting panel (raw already drawn),
    //                 dashed class-colored box on the OTHER panel showing
    //                 the projected location ("another sensor says so").
    for (const trk of this._lastFused) {
      const nSensors = (trk.sensors || []).length;
      const bbox = trk.bbox_thermal;
      if (!bbox) continue;
      if (nSensors >= 2) {
        drawFusedBox(this.ctx, bbox, trk, scale, dx, dy);
      } else {
        const detectedHere = (trk.sensors || []).includes(PANEL_SENSOR);
        if (!detectedHere) {
          drawProjectedBox(this.ctx, bbox, trk, scale, dx, dy);
        }
      }
    }

    // (Radar overlay removed — radar-only targets are now FusedTrack
    //  entries with sensors=["radar"] and get drawn by the fused loop
    //  above in their class colour. Drawing both was duplicating every
    //  box on the thermal/EO panels.)

    // Developer overlay: every heat-blob tracker entry (incl. pending
    // + coasting). Drawn AFTER production boxes so the magenta lines
    // sit on top and are unambiguously the debug view.
    if (this._devMode) {
      for (const ht of this._lastHeatTracks) {
        // Skip synthetic tracks here — they're rendered via the
        // detection path above with their own style.
        if (ht && ht.synthetic) continue;
        drawHeatTrackDebug(this.ctx, ht, scale, dx, dy);
      }
    }

    // Rubber-band rectangle while the user is dragging in draw mode.
    // Drawn last so it sits on top of every overlay.
    if (this._drag) {
      drawRubberBand(this.ctx, this._drag.x0, this._drag.y0,
                     this._drag.x1, this._drag.y1);
    }
  }
}
