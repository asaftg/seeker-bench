// Thermal view: renders the latest JPEG frame and overlays detections.

import { drawThermalBox, drawHandBox, drawDroneBox } from "./overlays.js";

// Zoom is applied on the backend now — the incoming JPEG is already
// the cropped-and-upscaled view, and detections arrive in that same
// coordinate space. The frontend just renders whatever it's given.

export class ThermalView {
  constructor(canvasId, disconnectOverlayId) {
    this.canvas = document.getElementById(canvasId);
    this.overlay = disconnectOverlayId ? document.getElementById(disconnectOverlayId) : null;
    this.ctx = this.canvas.getContext("2d");
    this.img = new Image();
    this._pending = null;
    this._lastFrameW = 0;
    this._lastFrameH = 0;
    this._lastDetections = [];
    this.img.onload = () => this._draw();
    window.addEventListener("resize", () => this._fitCanvas());
    this._fitCanvas();
  }

  _fitCanvas() {
    const r = this.canvas.getBoundingClientRect();
    // Use device pixels for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    if (this._lastFrameW > 0) this._draw();
  }

  update(thermal) {
    if (!thermal || !thermal.connected) {
      if (this.overlay) this.overlay.classList.remove("hidden");
      this._clear();
      return;
    }
    if (this.overlay) this.overlay.classList.add("hidden");

    this._lastFrameW = thermal.width || 0;
    this._lastFrameH = thermal.height || 0;
    this._lastDetections = thermal.detections || [];

    if (thermal.jpeg_b64) {
      this.img.src = "data:image/jpeg;base64," + thermal.jpeg_b64;
    } else {
      this._draw(); // no image, still draw overlays if any
    }
  }

  _clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  _draw() {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    this.ctx.clearRect(0, 0, cw, ch);

    const fw = this._lastFrameW || this.img.naturalWidth;
    const fh = this._lastFrameH || this.img.naturalHeight;
    if (!fw || !fh) return;

    // Letterbox-fit the (already-zoomed) frame into the canvas.
    const scale = Math.min(cw / fw, ch / fh);
    const dw = fw * scale;
    const dh = fh * scale;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;

    if (this.img.complete && this.img.naturalWidth > 0) {
      this.ctx.drawImage(this.img, dx, dy, dw, dh);
    }

    // Draw detections mapped into the same destination space.
    for (const det of this._lastDetections) {
      const b = det.bbox;
      const x = dx + b.x * scale;
      const y = dy + b.y * scale;
      const w = b.w * scale;
      const h = b.h * scale;
      const cls = det.classification && det.classification.target_class;
      const conf = det.classification && det.classification.confidence;
      if (cls === "hand") {
        drawHandBox(this.ctx, x, y, w, h, `HAND ${(conf*100|0)}%`);
      } else if (cls === "drone") {
        drawDroneBox(this.ctx, x, y, w, h, `DRONE ${(conf*100|0)}%`);
      } else {
        drawThermalBox(this.ctx, x, y, w, h, `HEAT Δ${det.contrast}`);
      }
    }
  }
}
