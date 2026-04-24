// EO view: renders the latest JPEG frame from the EO pipeline and overlays
// person/vehicle detections. Same fusion-aware rendering as ThermalView:
// raw detections subsumed by a fused track are suppressed so the green
// fused box is the single box on that target.

import {
  drawDetectionBox,
  drawFusedBox,
  drawProjectedBox,
  drawRadarBox,
  isSubsumedByFused,
  fusedIdForDet,
} from "./overlays.js";

const PANEL_SENSOR = "eo";

export class EOView {
  constructor(canvasId, disconnectOverlayId) {
    this.canvas = document.getElementById(canvasId);
    this.overlay = disconnectOverlayId ? document.getElementById(disconnectOverlayId) : null;
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.img = new Image();
    this._lastFrameW = 0;
    this._lastFrameH = 0;
    this._lastDetections = [];
    this._lastFused = [];
    this._lastRadarTargets = [];
    this._mainTargetId = null;
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

  update(eo, mainTargetId = null, fused = [], radarTargets = []) {
    this._mainTargetId = mainTargetId;
    this._lastFused = fused || [];
    this._lastRadarTargets = radarTargets || [];
    if (!eo || !eo.connected) {
      if (this.overlay) this.overlay.classList.remove("hidden");
      this._clear();
      return;
    }
    if (this.overlay) this.overlay.classList.add("hidden");

    this._lastFrameW = eo.width || 0;
    this._lastFrameH = eo.height || 0;
    this._lastDetections = eo.detections || [];

    if (eo.jpeg_b64) {
      this.img.src = "data:image/jpeg;base64," + eo.jpeg_b64;
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

    // Raw detections — suppressed only by 2+ sensor fused overlays.
    for (const det of this._lastDetections) {
      if (isSubsumedByFused(det.bbox, this._lastFused, "bbox_eo")) continue;
      const fusedId = fusedIdForDet(det.bbox, this._lastFused, "bbox_eo");
      const isMain = this._mainTargetId != null &&
                     fusedId != null &&
                     String(fusedId) === String(this._mainTargetId);
      drawDetectionBox(this.ctx, det, scale, dx, dy, isMain, fusedId);
    }

    // Fused overlay rules — mirror of thermal_view.
    for (const trk of this._lastFused) {
      const nSensors = (trk.sensors || []).length;
      const bbox = trk.bbox_eo;
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

    // Radar overlay (projection-only, pre-fusion). Bbox is pre-projected
    // into EO pixel space server-side. Coasting tracks render dimmer.
    for (const rt of this._lastRadarTargets) {
      const bbox = rt.bbox_eo;
      if (!bbox) continue;
      const x = dx + bbox.x * scale;
      const y = dy + bbox.y * scale;
      const w = bbox.w * scale;
      const h = bbox.h * scale;
      const label = `R#${rt.tid}${rt.coasting ? " · coast" : ""}`;
      this.ctx.save();
      if (rt.coasting) this.ctx.globalAlpha = 0.55;
      drawRadarBox(this.ctx, x, y, w, h, label);
      this.ctx.restore();
    }
  }
}
