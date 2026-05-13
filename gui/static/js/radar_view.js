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
//
// The drawing scale `_viewRangeM` *breathes*: starts at DEFAULT_VIEW_M
// (100 m) and expands to fit the farthest returned target/point, with a
// small margin. When the scene empties out it shrinks back to the default
// — with a short hysteresis so a lone flicker doesn't yank the scale.

import { fusedIdForRadarTarget } from "./overlays.js";

const TARGET_COLORS = [
  "#ff4d6d", "#40c4ff", "#ffd54f", "#81c784",
  "#ba68c8", "#ff8a65", "#4dd0e1", "#dce775",
];
const STATIC_DOPPLER_MPS = 0.2;

// Two-state view: parked at NEAR_VIEW_M (100 m) unless a confirmed
// target sits beyond it, at which point we jump to FAR_VIEW_M (250 m).
// Shrink only after SHRINK_FRAMES quiet frames so a flickering long-
// range return doesn't yank the scale in and out.
//
// The NEAR_VIEW_M ring is drawn in a warm highlight colour in *both*
// states so the operator always has a "where is the 100 m mark"
// reference — the whole reason we jump views instead of hugging the
// farthest target is to keep that mark meaningful.
const NEAR_VIEW_M   = 100;
const FAR_VIEW_M    = 500;
const SHRINK_FRAMES = 30;
// Small dead-zone above 100 m so a bbox straddling the boundary
// doesn't chatter between views.
const FAR_TRIGGER_M = 105;

// Per-track trajectory trail — how long a history point lives on screen
// before it's pruned (ms), and the max samples kept per track. 3 s gives
// a readable tail at radar's ~10 Hz without drowning the plot; cap keeps
// memory bounded if a track lingers for a long time. Trails are rebuilt
// from the WS stream (no backend state), so tracks that die in the
// clusterer's reaper naturally age out of the map on the client too.
const TRAIL_MAX_AGE_MS = 10000;
const TRAIL_MAX_POINTS = 200;
// Client-side EMA smoothing applied as trail points are ingested —
// independent of the server-side Kalman. Lower = smoother trail at the
// cost of a small visual lag behind the bbox. The bbox itself always
// draws at the raw KF position so targeting accuracy is unaffected.
const TRAIL_EMA_ALPHA = 0.35;

// AWR2944P useful azimuth ≈ ±60° off boresight. Beyond this the
// antenna pattern collapses and range estimates get unreliable.
// This is a *fallback* — the actual gate travels on the wire as
// radar.fov_half_deg (settable live from the DEV-tab FOV slider).
const FOV_HALF_DEG_DEFAULT = 60;

export class RadarView {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.disconnectEl = document.getElementById("radar-disconnected");

    // Cache the most recent payload so _fit()'s redraw after a resize
    // repaints what's actually there instead of just the backdrop.
    this._lastRadar = null;

    // Rolling FPS estimator — last 30 frame-id / timestamp pairs.
    this._hzSamples = [];

    // Sensor-reported max range (used for ring label ceiling, not view).
    this._maxRangeM = 50;
    // Sensor-reported FOV half-angle — synced from radar.fov_half_deg
    // so the drawn wedge matches the actual host-side azimuth gate.
    this._fovHalfDeg = FOV_HALF_DEG_DEFAULT;

    // Dynamic view range (what _worldToCanvas actually divides by).
    this._viewRangeM = NEAR_VIEW_M;
    this._shrinkCounter = 0;

    // Per-tid trajectory history: tid → Array<{x, y, t_ms}>.
    this._trails = new Map();

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

  // Public re-fit for views whose host element was hidden at construct
  // time (DEV-tab mini preview). Call on tab activation.
  refit() { this._fit(); }

  // Geometry: origin at bottom-centre, +y (forward) up the screen.
  // Use the full canvas height for the semicircle radius so the plot
  // fills the panel vertically. Points at extreme ±azimuth may clip
  // horizontally — acceptable trade since AWR2944P's useful FOV is
  // roughly ±60° anyway.
  _radarGeom() {
    const w = this.canvas.width, h = this.canvas.height;
    const cx = w / 2;
    const cy = h - 8;                // hug the bottom
    const maxR = Math.max(20, h - 14);
    return { w, h, cx, cy, maxR };
  }

  _worldToCanvas(xm, ym) {
    const { cx, cy, maxR } = this._radarGeom();
    const pxPerM = maxR / Math.max(this._viewRangeM, 1);
    return { px: cx + xm * pxPerM, py: cy - ym * pxPerM, pxPerM };
  }

  // Two-state view: 100 m by default, 250 m only when a confirmed
  // (non-coasting) target reaches past 100 m. Points alone are *not*
  // enough to trigger the jump — a lone far clutter return shouldn't
  // dump the operator out of the close-in view.
  _updateViewRange(radar) {
    let farTrigger = false;
    const targets = radar.targets || [];
    for (const t of targets) {
      if (t.coasting) continue;           // only solid boxes widen the view
      const r = Math.hypot(t.x || 0, t.y || 0);
      if (r > FAR_TRIGGER_M) { farTrigger = true; break; }
    }

    if (farTrigger) {
      this._viewRangeM = FAR_VIEW_M;
      this._shrinkCounter = 0;
    } else if (this._viewRangeM > NEAR_VIEW_M) {
      this._shrinkCounter++;
      if (this._shrinkCounter >= SHRINK_FRAMES) {
        this._viewRangeM = NEAR_VIEW_M;
        this._shrinkCounter = 0;
      }
    } else {
      this._shrinkCounter = 0;
    }
  }

  _drawBackdrop() {
    const ctx = this.ctx;
    const { w, h, cx, cy, maxR } = this._radarGeom();

    ctx.clearRect(0, 0, w, h);

    // Range rings — 4 rings at 1/4 increments of _viewRangeM.
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      const ringM = (this._viewRangeM * i) / 4;
      // Highlight the 100 m reference ring in warm amber. In 100 m view
      // that's the outermost ring; in 250 m view it's the second-
      // innermost ring — same physical distance either way, so the
      // operator has a stable landmark telling them "where is 100 m".
      if (Math.abs(ringM - NEAR_VIEW_M) < 0.5) {
        ctx.strokeStyle = "rgba(255, 170, 60, 0.55)";
        ctx.lineWidth = 1.4;
      } else {
        ctx.strokeStyle = "rgba(0, 212, 255, 0.15)";
        ctx.lineWidth = 1;
      }
      ctx.beginPath();
      ctx.arc(cx, cy, maxR * (i / 4), Math.PI, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.lineWidth = 1;
    // ±FOV sector lines — mark the radar's host-side azimuth gate.
    // Now drawn ROTATED by the gimbal pan so the wedge follows where
    // the radar is actually pointing in world frame. Screen-up =
    // bench-forward (world +y); positive angle goes right (world +x).
    const fovHalf = this._fovHalfDeg;
    const fovRad  = (fovHalf * Math.PI) / 180;
    const panRad  = this._gimbalPanRad || 0;
    const panDeg  = this._gimbalPanDeg || 0;
    ctx.strokeStyle = "rgba(0, 212, 255, 0.22)";
    ctx.setLineDash([4, 4]);
    for (const sign of [-1, 1]) {
      const a = panRad + sign * fovRad;
      const ex = cx + Math.sin(a) * maxR;
      const ey = cy - Math.cos(a) * maxR;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Translucent fill inside the wedge so the eye lands on
    // "where the radar is currently looking" without effort.
    ctx.fillStyle = "rgba(0, 212, 255, 0.07)";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, maxR,
            -Math.PI / 2 + (panRad - fovRad),
            -Math.PI / 2 + (panRad + fovRad),
            false);
    ctx.closePath();
    ctx.fill();

    // Boresight tick (solid, stronger) at the gimbal pan direction.
    ctx.strokeStyle = "rgba(0, 212, 255, 0.45)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.sin(panRad) * maxR,
               cy - Math.cos(panRad) * maxR);
    ctx.stroke();
    ctx.lineWidth = 1;

    // Bench north tick (faint, always points to screen-up).
    // World +y = bench forward = where pan=0 looks. Operator uses
    // this as a stable reference for "which way is the bench facing"
    // even as the gimbal swings.
    ctx.strokeStyle = "rgba(180, 220, 240, 0.20)";
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx, cy - maxR);
    ctx.stroke();
    ctx.setLineDash([]);

    // Pan readout near origin
    ctx.fillStyle = "rgba(0, 212, 255, 0.65)";
    ctx.font = `${Math.round(10 * (window.devicePixelRatio || 1))}px sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(`PAN ${panDeg.toFixed(1)}°`, cx, cy + 14);

    // FOV edge labels (rotated with the wedge).
    ctx.fillStyle = "rgba(180, 220, 240, 0.45)";
    ctx.font = `${Math.round(10 * (window.devicePixelRatio || 1))}px sans-serif`;
    ctx.textAlign = "center";
    const aL = panRad - fovRad;
    const aR = panRad + fovRad;
    const lx = cx + Math.sin(aL) * maxR * 0.98;
    const ly = cy - Math.cos(aL) * maxR * 0.98;
    const rx = cx + Math.sin(aR) * maxR * 0.98;
    const ry = cy - Math.cos(aR) * maxR * 0.98;
    ctx.fillText(`${(panDeg - fovHalf).toFixed(0)}°`, lx, ly);
    ctx.fillText(`${(panDeg + fovHalf).toFixed(0)}°`, rx, ry);

    // Range labels — match the 100 m ring's warm highlight.
    ctx.font = `${Math.round(11 * (window.devicePixelRatio || 1))}px sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    for (let i = 1; i <= 4; i++) {
      const r = (this._viewRangeM * i) / 4;
      ctx.fillStyle = (Math.abs(r - NEAR_VIEW_M) < 0.5)
        ? "rgba(255, 190, 90, 0.9)"
        : "rgba(180, 220, 240, 0.55)";
      ctx.fillText(`${r.toFixed(0)} m`, cx + 4, cy - maxR * (i / 4));
    }
  }

  // Colour a point by its doppler: red = approaching, blue = receding,
  // dim cyan = effectively static. Opacity scaled by SNR (12-40 dB band).
  _pointStyle(v, snr) {
    const safeSnr = (typeof snr === "number" && isFinite(snr)) ? snr : 20;
    const s = Math.max(0.3, Math.min(1.0, (safeSnr - 12) / 28));
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
      // Solid stroke for Kalman tracks hit this frame; dashed + dim for
      // coasting (dead-reckoned) tracks so the operator can tell live
      // measurements from predicted state at a glance.
      const coasting = !!t.coasting;
      if (coasting) {
        ctx.setLineDash([5 * dpr, 4 * dpr]);
        ctx.globalAlpha = 0.55;
      } else {
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
      }
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

      // Label: id · speed · range. Class is always "radar_detection"
      // per the Ticket 5a design (classification lives in fusion, not
      // here) so we omit it from the label. Range is the distance from
      // the sensor to the target centroid — useful when the view has
      // breathed out to 250 m and a single box could be anywhere.
      // ID prefix follows the global namespace convention:
      //   #N  — fused track id when fusion has linked this radar tid
      //   R#N — raw radar Kalman id (per-sensor, transient)
      // Backend stamps t.fused_id directly; fusedIdForRadarTarget is
      // the legacy fallback for older recordings.
      const fusedId = (t.fused_id != null)
        ? t.fused_id
        : fusedIdForRadarTarget(t.tid, this._lastFused);
      const idPrefix = (fusedId != null) ? `#${fusedId}` : `R#${t.tid}`;
      const speed = Math.hypot(t.vx, t.vy);
      const range = Math.hypot(t.x, t.y);
      const suffix = coasting ? " · coast" : "";
      const label = `${idPrefix}  ${speed.toFixed(1)} m/s · ${range.toFixed(0)} m${suffix}`;
      ctx.fillStyle = colour;
      ctx.fillText(label, px - wpx / 2 + 2 * dpr, py - hpx / 2 - 2 * dpr);
      ctx.globalAlpha = 1.0;
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

  // Append the current target positions to their per-tid trails and
  // drop entries older than TRAIL_MAX_AGE_MS. Called from update()
  // once per WS frame — cheap (length bounded by #targets × cap).
  _ingestTrails(targets) {
    const nowMs = performance.now();
    if (targets && targets.length) {
      for (const t of targets) {
        if (t.tid == null) continue;
        let hist = this._trails.get(t.tid);
        if (!hist) { hist = []; this._trails.set(t.tid, hist); }
        // EMA-smooth each stored trail position against its predecessor.
        // First sample seeds the filter; subsequent samples blend with
        // TRAIL_EMA_ALPHA of raw + (1-α) of previous smoothed — classic
        // single-pole low-pass, suppresses KF output jitter visible on
        // the trail without biasing any single frame.
        let sx = t.x, sy = t.y;
        if (hist.length > 0) {
          const prev = hist[hist.length - 1];
          sx = TRAIL_EMA_ALPHA * t.x + (1 - TRAIL_EMA_ALPHA) * prev.x;
          sy = TRAIL_EMA_ALPHA * t.y + (1 - TRAIL_EMA_ALPHA) * prev.y;
        }
        hist.push({ x: sx, y: sy, t: nowMs });
        if (hist.length > TRAIL_MAX_POINTS) {
          hist.splice(0, hist.length - TRAIL_MAX_POINTS);
        }
      }
    }
    // Age-prune every trail, drop empties.
    for (const [tid, hist] of this._trails) {
      while (hist.length && (nowMs - hist[0].t) > TRAIL_MAX_AGE_MS) {
        hist.shift();
      }
      if (hist.length === 0) this._trails.delete(tid);
    }
  }

  _drawTrails() {
    if (!this._trails.size) return;
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const nowMs = performance.now();
    ctx.lineWidth = 1.4 * dpr;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const [tid, hist] of this._trails) {
      if (hist.length < 2) continue;
      const colourIdx = ((tid % TARGET_COLORS.length) + TARGET_COLORS.length) % TARGET_COLORS.length;
      const colour = TARGET_COLORS[colourIdx];

      // Project once so we aren't re-projecting the same point on both
      // sides of a segment boundary.
      const pts = hist.map(h => {
        const p = this._worldToCanvas(h.x, h.y);
        return { px: p.px, py: p.py, t: h.t };
      });

      // Smooth via quadratic-bezier through midpoints — classic canvas
      // trick. Each anchor is a raw sample's projected (px,py); each
      // control point is sample i, the curve arcs smoothly from the
      // midpoint(i-1,i) to the midpoint(i,i+1) bending through i.
      // Segments are drawn individually so each carries its own age
      // alpha (canvas has no per-vertex alpha on a single path).
      for (let i = 1; i < pts.length - 1; i++) {
        const ageMs = nowMs - pts[i].t;
        const alpha = Math.max(0.0, 1.0 - ageMs / TRAIL_MAX_AGE_MS) * 0.8;
        if (alpha <= 0.02) continue;
        const prev = pts[i - 1];
        const cur  = pts[i];
        const next = pts[i + 1];
        const mx0 = (prev.px + cur.px) * 0.5;
        const my0 = (prev.py + cur.py) * 0.5;
        const mx1 = (cur.px + next.px) * 0.5;
        const my1 = (cur.py + next.py) * 0.5;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = colour;
        ctx.beginPath();
        ctx.moveTo(mx0, my0);
        ctx.quadraticCurveTo(cur.px, cur.py, mx1, my1);
        ctx.stroke();
      }
      // Final straight stub from the second-to-last midpoint to the
      // latest sample so the trail reaches the current bbox exactly,
      // not just to a midpoint halfway back.
      if (pts.length >= 2) {
        const last = pts[pts.length - 1];
        const prev = pts[pts.length - 2];
        const ageMs = nowMs - last.t;
        const alpha = Math.max(0.0, 1.0 - ageMs / TRAIL_MAX_AGE_MS) * 0.8;
        if (alpha > 0.02) {
          const mx = (prev.px + last.px) * 0.5;
          const my = (prev.py + last.py) * 0.5;
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = colour;
          ctx.beginPath();
          ctx.moveTo(mx, my);
          ctx.lineTo(last.px, last.py);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1.0;
    ctx.lineCap = "butt";
    ctx.lineJoin = "miter";
  }

  _redraw() {
    this._drawBackdrop();
    const r = this._lastRadar;
    if (!r || !r.connected) return;
    this._drawPoints(r.points);
    this._drawTrails();
    this._drawTargets(r.targets);
  }

  // Radar (x,y) -> world correction for TI tracker overcompensation.
  //
  // 2026-04-27: Full R(+/-pan) rotation was too aggressive -- the TI
  // AWR2944P Kalman tracker already compensates for gimbal rotation
  // internally. However, it OVERCOMPENSATES by ~13%, causing targets
  // to drift in the pan direction.
  //
  // 2026-05-13: Quantified using "radar drift 1" and "radar drift 2"
  // recordings. For stationary targets (tid=317 at 77m, tid=320 at
  // 82m), measured pan-vs-x correlation. Optimal correction factor
  // eps=0.13 reduces drift from 3.96m to 1.00m (tid=317 over 21 deg
  // of pan sweep). The correction is R(+eps * pan): a small rotation
  // in the pan direction to undo the tracker overcompensation.
  //
  // History:
  //   2026-04-26  R(-p)      targets swing 10x expected (wrong sign)
  //   2026-04-27  R(+p)      targets fly off canvas (full rotation)
  //   2026-04-27  no-op      best of 3, but residual drift remains
  //   2026-05-13  R(+0.13*p) empirical correction, validated on
  //               2 recordings, 5 tracks. Minimises pan-x correlation
  //               for stationary targets without degrading moving ones.
  _rotateToWorld(x, y) {
    const EPS = 0.13;
    const a = EPS * this._gimbalPanRad;
    const c = Math.cos(a), s = Math.sin(a);
    return { x: x * c - y * s, y: x * s + y * c };
  }

  // Apply world-frame rotation in-place to one wire-shape entry that
  // carries radar-local (x, y, vx, vy). Returns a SHALLOW COPY so the
  // upstream payload isn't mutated (downstream consumers may still
  // need radar-local coords).
  _toWorldEntry(e) {
    const p = this._rotateToWorld(e.x || 0, e.y || 0);
    let vx = e.vx, vy = e.vy;
    if (vx != null && vy != null) {
      const v = this._rotateToWorld(vx, vy);
      vx = v.x; vy = v.y;
    }
    return { ...e, x: p.x, y: p.y, vx, vy };
  }

  update(radar, gimbalPanDeg = 0, fusedTracks = null) {
    if (!this.ctx) return;
    // Stash the fused-track list so _drawTargets can look up the
    // fused id for each radar target (Phase B2 — symmetric with the
    // EO/thermal panels' raw-det → fused-id matching).
    this._lastFused = fusedTracks || [];
    // Defensive re-fit: if the view was constructed before CSS layout
    // settled, the constructor's _fit() sized the canvas to 1×1. Check
    // on each update and re-fit if the bounding rect has grown.
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const wantW = Math.max(1, Math.floor(r.width * dpr));
    const wantH = Math.max(1, Math.floor(r.height * dpr));
    if (this.canvas.width !== wantW || this.canvas.height !== wantH) {
      this.canvas.width = wantW;
      this.canvas.height = wantH;
    }

    // Cache gimbal pose for the world-frame rotation. Stored in radians
    // for fast cos/sin in the hot loop.
    this._gimbalPanDeg = Number(gimbalPanDeg) || 0;
    this._gimbalPanRad = this._gimbalPanDeg * Math.PI / 180;

    // Pre-rotate the radar payload into world frame so all downstream
    // drawing, view-range fitting, and trail accumulation operate in
    // a fixed reference. Targets stay anchored on the panel; only the
    // radar's pointing wedge swings as the gimbal moves.
    let rWorld = radar || null;
    if (radar && radar.connected) {
      rWorld = {
        ...radar,
        points:  (radar.points  || []).map(p => this._toWorldEntry(p)),
        targets: (radar.targets || []).map(t => this._toWorldEntry(t)),
      };
    }
    this._lastRadar = rWorld;

    if (!rWorld || !rWorld.connected) {
      this._setDisconnected(true);
      // Reset to default when the sensor drops — avoids showing a stale
      // breathed-out scale on reconnect.
      this._viewRangeM = NEAR_VIEW_M;
      this._shrinkCounter = 0;
      this._trails.clear();
      this._drawBackdrop();
      this._hzSamples.length = 0;
      this._paintHzLabel(null);
      return;
    }

    this._setDisconnected(false);
    if (rWorld.max_range_m && rWorld.max_range_m > 0) this._maxRangeM = rWorld.max_range_m;
    if (typeof rWorld.fov_half_deg === "number" && rWorld.fov_half_deg > 0) {
      this._fovHalfDeg = rWorld.fov_half_deg;
    }
    this._ingestTrails(rWorld.targets);
    this._updateViewRange(rWorld);

    const hz = this._updateHz(rWorld);
    this._paintHzLabel(hz);
    this._redraw();
  }
}
