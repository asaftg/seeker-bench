// Thermal view: renders the latest JPEG frame and overlays detections.
// Phase B: uses the unified drawDetectionBox with class-aware colors.
// Ticket 5: if a fused green box subsumes a raw detection, the raw box
// is suppressed so we don't stack two boxes on the same target.

import {
  drawDetectionBox,
  drawFusedBox,
  drawProjectedBox,
  drawHeatTrackDebug,
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
    this._devMode = false;
    if (this.img) {
      this.img.onload = () => this._draw();
    }
    window.addEventListener("resize", () => this._fitCanvas());
    this._fitCanvas();
  }

  _fitCanvas() {
    if (!this.canvas) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    if (this._lastFrameW > 0) this._draw();
  }

  update(thermal, mainTargetId = null, fused = [], devMode = false) {
    this._mainTargetId = mainTargetId;
    this._lastFused = fused || [];
    this._devMode = !!devMode;
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
      this.ctx.drawImage(this.img, dx, dy, dw, dh);
    }

    // Raw detections — suppressed only when a 2+ sensor fused box is
    // about to be drawn on top (handled inside isSubsumedByFused).
    for (const det of this._lastDetections) {
      if (isSubsumedByFused(det.bbox, this._lastFused, "bbox_thermal")) continue;
      const fusedId = fusedIdForDet(det.bbox, this._lastFused, "bbox_thermal");
      const isMain = this._mainTargetId != null &&
                     fusedId != null &&
                     String(fusedId) === String(this._mainTargetId);
      drawDetectionBox(this.ctx, det, scale, dx, dy, isMain, fusedId);
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

    // Developer overlay: every heat-blob tracker entry (incl. pending
    // + coasting). Drawn AFTER production boxes so the magenta lines
    // sit on top and are unambiguously the debug view.
    if (this._devMode) {
      for (const ht of this._lastHeatTracks) {
        drawHeatTrackDebug(this.ctx, ht, scale, dx, dy);
      }
    }
  }
}
