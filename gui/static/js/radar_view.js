// Radar view — renders the AWR2944P point cloud + DBSCAN target boxes
// on the half-circle polar canvas. See Ticket 5a.
//
// Wire shape (from sensor_bridge.radar_to_wire):
//   {
//     connected, frame_id, timestamp, profile, max_range_m,
//     num_points, num_targets,
//     points:  [{x, y, z, v, snr, r, az, el, tid}, ...],
//     targets: [{tid, x, y, z, vx, vy, vz, sx, sy, sz, conf, src, np, class}, ...],
//     detections: [],  // reserved for fusion-labeled output
//   }
//
// All world coords are metres in the sensor frame: +x = right, +y = forward
// (boresight), +z = up. Canvas is a top-down projection (x, y); z is ignored
// for drawing but would matter for a future side-view.

const TARGET_COLORS = [
  "#ff4d6d", "#40c4ff", "#ffd54f", "#81c784",
  "#ba68c8", "#ff8a65", "#4dd0e1", "#dce775",
];
const STATIC_DOPPLER_MPS = 0.2;

export class RadarView {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.disconnectEl = document.getElementById("radar-disconnected");

    // Cache the most recent payload so _fit()'s redraw after a resize
    // repaints what's actually there instead of just the backdrop.
    this._lastRadar = null;

    // Rolling FPS estimator — last 15 frame-id / timestamp pairs.
    this._hzSamples = [];   // [[frame_id, ts_ms], ...]

    // Range scale used by _worldToCanvas — updated from max_range_m.
    this._maxRangeM = 50;

    window.addEventListener("resize", () => this._fit());
    this._fit();
  }

  _fit() {
    if (!this.canvas) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    this._redraw();
  }

  // Top-down polar: origin at bottom-centre, +y (forward) points up the screen,
  // +x (right) points right. Return canvas (px, py) for world (x_m, y_m).
  _worldToCanvas(xm, ym) {
    const w = this.canvas.width, h = this.canvas.height;
    const cx = w / 2, cy = h - 10;
    // Fit the full semicircle inside the panel.
    const maxR = Math.min(w / 2, h - 20);
    const pxPerM = maxR / Math.max(this._maxRangeM, 1);
    return { px: cx + xm * pxPerM, py: cy - ym * pxPerM, pxPerM };
  }

  _drawBackdrop() {
    const ctx = this.ctx;
    const w = this.canvas.width, h = this.canvas.height;
    const cx = w / 2, cy = h - 10;
    const maxR = Math.min(w / 2, h - 20);

    ctx.clearRect(0, 0, w, h);

    // Range rings — 4 rings at 1/4 increments of max_range_m.
    ctx.strokeStyle = "rgba(0, 212, 255, 0.15)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath();
      ctx.arc(cx, cy, maxR * (i / 4), Math.PI, 2 * Math.PI);
      ctx.stroke();
    }
    // Boresight tick.
    ctx.strokeStyle = "rgba(0, 212, 255, 0.20)";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx, cy - maxR);
    ctx.stroke();

    // Range labels.
    ctx.fillStyle = "rgba(180, 220, 240, 0.45)";
    ctx.font = `${Math.round(11 * (window.devicePixelRatio || 1))}px sans-serif`;
    ctx.textAlign = "left";
    for (let i = 1; i <= 4; i++) {
      const r = (this._maxRangeM * i) / 4;
      ctx.fillText(`${r.toFixed(0)} m`, cx + 4, cy - maxR * (i / 4));
    }
  }

  // Colour a point by its doppler: red = approaching, blue = receding,
  // dim cyan = effectively static. Opacity scaled by SNR (12-40 dB band).
  _pointStyle(v, snr) {
    const s = Math.max(0.3, Math.min(1.0, (snr - 12) / 28));
    if (Math.abs(v) < STATIC_DOPPLER_MPS) {
      return `rgba(0, 212, 255, ${s * 0.55})`;
    }
    if (v > 0) return `rgba(255, 85, 85, ${s})`;     // approaching (+doppler)
    return `rgba(80, 170, 255, ${s})`;               // receding
  }

  _drawPoints(points) {
    if (!points || !points.length) return;
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const rad = 2 * dpr;
    for (const p of points) {
      const { px, py } = this._worldToCanvas(p.x, p.y);
      ctx.fillStyle = this._pointStyle(p.v, p.snr);
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  _drawTargets(targets) {
    if (!targets || !targets.length) return;
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    ctx.lineWidth = 1.5 * dpr;
    ctx.font = `${Math.round(11 * dpr)}px sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";

    for (const t of targets) {
      const colour = TARGET_COLORS[((t.tid % TARGET_COLORS.length) + TARGET_COLORS.length) % TARGET_COLORS.length];
      const { px, py, pxPerM } = this._worldToCanvas(t.x, t.y);
      const wpx = Math.max(6, 2 * t.sx * pxPerM);
      const hpx = Math.max(6, 2 * t.sy * pxPerM);
      // Dashed stroke for DBSCAN-sourced targets — reserved solid for
      // a future firmware Group-Tracker-sourced target (not emitted by
      // this build of mmw_demoDDM).
      ctx.setLineDash(t.src === "dbscan" ? [5 * dpr, 4 * dpr] : []);
      ctx.strokeStyle = colour;
      ctx.strokeRect(px - wpx / 2, py - hpx / 2, wpx, hpx);
      ctx.setLineDash([]);

      // Velocity arrow — 1 s of motion at current (vx, vy).
      const { px: tpx, py: tpy } = this._worldToCanvas(t.x + t.vx, t.y + t.vy);
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(tpx, tpy);
      ctx.stroke();
      // Arrow head — two short tick lines at the tip.
      const ang = Math.atan2(tpy - py, tpx - px);
      const hlen = 5 * dpr;
      ctx.beginPath();
      ctx.moveTo(tpx, tpy);
      ctx.lineTo(tpx - hlen * Math.cos(ang - 0.4), tpy - hlen * Math.sin(ang - 0.4));
      ctx.moveTo(tpx, tpy);
      ctx.lineTo(tpx - hlen * Math.cos(ang + 0.4), tpy - hlen * Math.sin(ang + 0.4));
      ctx.stroke();

      // Label: #tid + speed. Class is always "radar_detection" per the
      // Ticket 5a design (classification lives in fusion, not here) so
      // we omit it from the label to keep it readable.
      const speed = Math.hypot(t.vx, t.vy);
      const label = `#${t.tid}  ${speed.toFixed(1)} m/s`;
      ctx.fillStyle = colour;
      ctx.fillText(label, px - wpx / 2 + 2 * dpr, py - hpx / 2 - 2 * dpr);
    }
  }

  _updateHz(radar) {
    const tsMs = (radar.timestamp || 0) * 1000;
    if (!tsMs || radar.frame_id == null) return null;
    this._hzSamples.push([radar.frame_id, tsMs]);
    if (this._hzSamples.length > 30) this._hzSamples.shift();
    if (this._hzSamples.length < 2) return null;
    const [f0, t0] = this._hzSamples[0];
    const [f1, t1] = this._hzSamples[this._hzSamples.length - 1];
    if (t1 <= t0) return null;
    return ((f1 - f0) * 1000) / (t1 - t0);
  }

  _paintHzLabel(hz) {
    // Some layouts have a header <span id="radar-hz">; ignore when absent.
    const el = document.getElementById("radar-hz");
    if (!el) return;
    el.textContent = hz != null ? `${hz.toFixed(1)} Hz` : "— Hz";
  }

  _setDisconnected(disc) {
    if (!this.disconnectEl) return;
    this.disconnectEl.style.display = disc ? "" : "none";
  }

  _redraw() {
    this._drawBackdrop();
    const r = this._lastRadar;
    if (!r || !r.connected) return;
    this._drawPoints(r.points);
    this._drawTargets(r.targets);
  }

  update(radar) {
    if (!this.ctx) return;
    this._lastRadar = radar || null;

    if (!radar || !radar.connected) {
      this._setDisconnected(true);
      this._drawBackdrop();
      this._hzSamples.length = 0;
      this._paintHzLabel(null);
      return;
    }

    this._setDisconnected(false);
    if (radar.max_range_m && radar.max_range_m > 0) this._maxRangeM = radar.max_range_m;

    const hz = this._updateHz(radar);
    this._paintHzLabel(hz);
    this._redraw();
  }
}
