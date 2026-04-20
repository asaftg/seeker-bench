// EO view: renders the latest JPEG frame from the EO pipeline and overlays
// person/vehicle detections. Same fusion-aware rendering as ThermalView:
// raw detections subsumed by a fused track are suppressed so the green
// fused box is the single box on that target.

import { drawDetectionBox, drawFusedBox, isSubsumedByFused } from "./overlays.js";

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

  update(eo, mainTargetId = null, fused = []) {
    this._mainTargetId = mainTargetId;
    this._lastFused = fused || [];
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

    const fusedBoxes = this._lastFused
      .map(t => t.bbox_eo)
      .filter(Boolean);

    for (const det of this._lastDetections) {
      if (isSubsumedByFused(det.bbox, fusedBoxes)) continue;
      const isMain = this._mainTargetId != null &&
                     det.track_id != null &&
                     String(det.track_id) === String(this._mainTargetId);
      drawDetectionBox(this.ctx, det, scale, dx, dy, isMain);
    }

    for (const trk of this._lastFused) {
      drawFusedBox(this.ctx, trk.bbox_eo, trk, scale, dx, dy);
    }
  }
}
