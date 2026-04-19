// Radar view (Phase A): draws an empty range-ring backdrop.
// Real radar rendering lands in Phase B.

export class RadarView {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    window.addEventListener("resize", () => this._fit());
    this._fit();
  }
  _fit() {
    if (!this.canvas) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    this._draw();
  }
  _draw() {
    if (!this.ctx) return;
    const w = this.canvas.width, h = this.canvas.height;
    const cx = w / 2, cy = h - 10;
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.strokeStyle = "rgba(0, 212, 255, 0.15)";
    this.ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      const rad = (Math.min(w, h * 2) / 2) * (i / 4);
      this.ctx.beginPath();
      this.ctx.arc(cx, cy, rad, Math.PI, 2 * Math.PI);
      this.ctx.stroke();
    }
    this.ctx.strokeStyle = "rgba(0, 212, 255, 0.2)";
    this.ctx.beginPath();
    this.ctx.moveTo(cx, cy);
    this.ctx.lineTo(cx, 0);
    this.ctx.stroke();
  }
  update(_radar) { /* no-op in Phase A */ }
}
