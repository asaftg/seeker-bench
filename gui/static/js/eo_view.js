// EO view: renders the latest JPEG frame from the EO pipeline and overlays
// person/vehicle detections. Same fusion-aware rendering as ThermalView:
// raw detections subsumed by a fused track are suppressed so the green
// fused box is the single box on that target.
//
// Phase B+ extension: an operator-driven "MEASURE" mode lets the user
// drag a bbox around a target of known real-world size and we compute
// distance via known-size triangulation — a stand-in for radar range
// until the radar half of the rig comes online.

import {
  drawDetectionBox,
  drawFusedBox,
  drawProjectedBox,
  drawRadarBox,
  drawRubberBand,
  drawLockBrackets,
  drawLockLabel,
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
    // Offscreen canvas holds the LAST FULLY-DECODED image. _draw blits
    // from this canvas, never directly from `this.img`. This decouples
    // canvas redraws from image-load state: when a new blob URL is
    // mid-decode, the offscreen still has the previous decoded frame,
    // so a _draw() triggered by the shared sensors path (between
    // binary frames) still gets a real picture to paint instead of
    // flashing the cleared background. Once img.onload fires, we
    // copy the new image into the offscreen and trigger a redraw.
    this._offscreen = document.createElement("canvas");
    this._offscreenCtx = this._offscreen.getContext("2d");
    this._haveImage = false;
    this._lastFrameW = 0;
    this._lastFrameH = 0;
    this._lastDetections = [];
    this._lastFused = [];
    this._lastRadarTargets = [];
    this._mainTargetId = null;
    this._lastHfovDeg = 11.05;  // updated from each WS payload

    // Measure-mode state — mirrors ThermalView's draw-target plumbing
    // but stays purely client-side: the bbox + class are turned into a
    // distance estimate in JS and rendered as a sticky overlay until
    // the user CLEARs it. No WS roundtrip needed.
    this._measureMode = false;
    this._drag = null;          // {x0,y0,x1,y1} canvas-space pixels
    this._onCommit = null;      // (imageBbox) => void on successful drag
    this._measurement = null;   // {bbox:{x,y,w,h}, label:"≈ 73 m · CAR"}

    // On image load: copy the decoded image into the offscreen
    // backbuffer ATOMICALLY (single drawImage = full bitmap or
    // nothing — the IMG is `complete` by the time onload fires),
    // then trigger _draw to blit from the offscreen onto the visible
    // canvas. After the offscreen has the new bitmap we can safely
    // revoke the previous blob URL — the bitmap is independent of
    // the URL once it's in the canvas.
    this.img.onload = () => {
      const w = this.img.naturalWidth;
      const h = this.img.naturalHeight;
      if (w > 0 && h > 0) {
        if (this._offscreen.width !== w || this._offscreen.height !== h) {
          this._offscreen.width = w;
          this._offscreen.height = h;
        }
        this._offscreenCtx.drawImage(this.img, 0, 0);
        this._haveImage = true;
      }
      this._draw();
      if (this._urlToRevoke) {
        try { URL.revokeObjectURL(this._urlToRevoke); } catch (_) {}
        this._urlToRevoke = null;
      }
    };
    // If a JPEG fails to decode, leave the offscreen alone (last
    // good frame stays visible) and revoke the bad URL.
    this.img.onerror = () => {
      if (this._urlToRevoke) {
        try { URL.revokeObjectURL(this._urlToRevoke); } catch (_) {}
        this._urlToRevoke = null;
      }
    };
    this._urlToRevoke = null;
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

  _fitCanvas() {
    if (!this.canvas) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    if (this._lastFrameW > 0) this._draw();
  }

  // Public re-fit: callers force this when the canvas's hidden tab
  // becomes visible (DEV tab in our case). The constructor's initial
  // _fitCanvas() ran while the tab was display:none → rect was 0×0 →
  // canvas backed at 1×1 → all subsequent draws looked like a
  // postage-stamp solid colour. Calling refit() after layout settles
  // gets us the real pixel size.
  refit() { this._fitCanvas(); }

  // ── Measure-mode API ────────────────────────────────────────────────
  // Toggled by the MEASURE button in the EO panel subbar. While active,
  // mousedown+drag+mouseup paints a rubber-band rect; on release we
  // hand the IMAGE-space bbox to onCommit (which knows the chosen
  // class + dimension and computes/persists the distance label).
  setMeasureMode(on, onCommit = null) {
    this._measureMode = !!on;
    this._onCommit = onCommit;
    if (!this.canvas) return;
    this.canvas.classList.toggle("draw-mode", this._measureMode);
    if (!this._measureMode) {
      this._drag = null;
      this._draw();
    }
  }
  isMeasureMode() { return this._measureMode; }
  cancelDrag() {
    if (this._drag) {
      this._drag = null;
      this._draw();
    }
  }
  // Persisted measurement to render every frame until CLEAR. Set to
  // null to remove. bbox is in IMAGE-space; label is whatever the
  // caller wants (typically "≈ 73 m · CAR · 1.8 m wide").
  setMeasurement(measurement) {
    this._measurement = measurement || null;
    this._draw();
  }
  clearMeasurement() {
    this._measurement = null;
    this._draw();
  }

  // ── Mouse → image-space helpers (same letterbox math as _draw) ──────
  _evtToCanvasPx(evt) {
    if (!this.canvas) return null;
    const rect = this.canvas.getBoundingClientRect();
    const cssX = evt.clientX - rect.left;
    const cssY = evt.clientY - rect.top;
    const sx = this.canvas.width  / rect.width;
    const sy = this.canvas.height / rect.height;
    return { x: cssX * sx, y: cssY * sy };
  }
  _canvasRectToImageBbox(x0c, y0c, x1c, y1c) {
    const fw = this._lastFrameW, fh = this._lastFrameH;
    if (!fw || !fh) return null;
    const cw = this.canvas.width, ch = this.canvas.height;
    const scale = Math.min(cw / fw, ch / fh);
    const dw = fw * scale, dh = fh * scale;
    const dx = (cw - dw) / 2, dy = (ch - dh) / 2;
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
    if (iw < 6 || ih < 6) return null;  // ignore micro-rects (misclick)
    return { x: ix, y: iy, w: iw, h: ih };
  }

  _handleMouseDown(evt) {
    if (!this._measureMode) return;
    const p = this._evtToCanvasPx(evt);
    if (!p) return;
    this._drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
    evt.preventDefault();
  }
  _handleMouseMove(evt) {
    if (!this._measureMode || !this._drag) return;
    const p = this._evtToCanvasPx(evt);
    if (!p) return;
    this._drag.x1 = p.x;
    this._drag.y1 = p.y;
    this._draw();
  }
  _handleMouseUp(evt) {
    if (!this._measureMode || !this._drag) return;
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

  update(eo, mainTargetId = null, fused = [], radarTargets = [], lock = null) {
    this._mainTargetId = mainTargetId;
    this._lastFused = fused || [];
    this._lastRadarTargets = radarTargets || [];
    // Lock-mode: persistent operator-engaged tracker output. Rendered
    // with priority over the projected fused-track bbox (so transient
    // YOLO/heat/fusion dropouts don't blink the box). null when lock
    // mode is off or no engagement is active.
    this._lock = (lock && lock.bbox_eo) ? lock : null;
    // Three states for the overlay scrim:
    //   1. Hard disconnect (camera unplugged / open failed) → red
    //      "EO · DISCONNECTED" — alarming on purpose.
    //   2. Initializing (software AE bracketing toward usable exposure,
    //      OR a planned source-helper restart while exposure is changed)
    //      → amber "EO · INITIALIZING…" — communicates "we're working on
    //      it" so the operator doesn't think the camera is broken during
    //      the ~30-55s daylight AE convergence.
    //   3. Connected with frames → hide scrim, draw image.
    // Frames during AE convergence DO arrive but may be black/saturated;
    // we still show the initializing scrim until the AE sets
    // initializing=false.
    const label = document.getElementById("eo-disconnected-label");
    if (!eo || !eo.connected || eo.initializing) {
      if (this.overlay) {
        this.overlay.classList.remove("hidden");
        // amber for initializing, default styling for hard disconnect
        if (eo && eo.initializing) {
          this.overlay.classList.add("initializing");
          if (label) label.textContent = "EO · INITIALIZING…";
        } else {
          this.overlay.classList.remove("initializing");
          if (label) label.textContent = "EO · DISCONNECTED";
        }
      }
      // For the initializing case, keep the (potentially dark) frame on
      // canvas so the user can see the AE working visually rather than
      // a uniform black panel.
      if (!eo || !eo.connected) {
        this._clear();
        return;
      }
      // Fall through to draw the partial frame under the amber scrim.
    } else if (this.overlay) {
      this.overlay.classList.add("hidden");
      this.overlay.classList.remove("initializing");
      if (label) label.textContent = "EO · DISCONNECTED";
    }

    this._lastFrameW = eo.width || 0;
    this._lastFrameH = eo.height || 0;
    this._lastDetections = eo.detections || [];
    if (eo.hfov_deg != null) this._lastHfovDeg = Number(eo.hfov_deg);

    if (eo._blobUrl) {
      // Binary WS path. Set img.src to the new blob URL; once it
      // decodes the constructor-bound onload paints it into the
      // offscreen backbuffer and revokes the previous URL. Until
      // then, _draw() called from any other path keeps blitting the
      // PREVIOUS frame from offscreen — no flicker.
      // If a previous URL is still pending revocation (e.g. the new
      // frame arrived before the old image even finished decoding),
      // revoke it now so we don't leak its blob.
      if (this._urlToRevoke) {
        try { URL.revokeObjectURL(this._urlToRevoke); } catch (_) {}
      }
      this._urlToRevoke = this._lastBlobUrl || null;
      this._lastBlobUrl = eo._blobUrl;
      this.img.src = eo._blobUrl;
    } else if (eo.jpeg_b64) {
      // Legacy / replay path. Data URLs decode synchronously so the
      // race that motivated the offscreen backbuffer doesn't apply,
      // but the code still flows through onload → offscreen → _draw
      // for uniformity.
      this.img.src = "data:image/jpeg;base64," + eo.jpeg_b64;
    } else {
      this._draw();
    }
  }

  // Read accessors so main.js can compute distances using the same
  // intrinsics the panel is currently rendering with.
  getHfovDeg()  { return this._lastHfovDeg; }
  getFrameWidth()  { return this._lastFrameW; }
  getFrameHeight() { return this._lastFrameH; }

  _clear() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  _draw() {
    if (!this.ctx || !this.canvas) return;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    this.ctx.clearRect(0, 0, cw, ch);

    // Use the offscreen backbuffer's dimensions when available — that
    // tracks the most recently DECODED frame. Falls back to img
    // naturalWidth/height for the very first frame before the
    // offscreen has been populated, and to _lastFrameW/H if the wire
    // payload was metadata-only.
    const fw = (this._haveImage ? this._offscreen.width : 0)
               || this._lastFrameW
               || this.img.naturalWidth;
    const fh = (this._haveImage ? this._offscreen.height : 0)
               || this._lastFrameH
               || this.img.naturalHeight;
    if (!fw || !fh) return;

    const scale = Math.min(cw / fw, ch / fh);
    const dw = fw * scale;
    const dh = fh * scale;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;

    // Blit from the offscreen backbuffer — guaranteed to be a fully
    // decoded image even if `this.img` is mid-load on the next blob
    // URL (the only path that touches img.src is onload, which paints
    // to offscreen ATOMICALLY before any _draw can race).
    if (this._haveImage) {
      this.ctx.drawImage(this._offscreen, dx, dy, dw, dh);
    } else if (this.img.complete && this.img.naturalWidth > 0) {
      this.ctx.drawImage(this.img, dx, dy, dw, dh);
    }

    // Solo render gate. When the operator has engaged a target AND
    // gimbal.lock_mode.solo_mode is on, we hide every non-engaged
    // bbox: red detections that don't map to the engaged fused id,
    // and fused tracks that aren't the engaged one. Only the
    // engaged target shows. Per operator request 2026-05-05:
    // "when I pick a target and 'lock' on it, we can have all
    // other targets bbs disappear."
    const soloEngaged = (this._lock && this._lock.solo_mode
                          && this._lock.engaged_id != null)
        ? this._lock.engaged_id : null;

    // Raw detections — suppressed only by 2+ sensor fused overlays.
    for (const det of this._lastDetections) {
      if (isSubsumedByFused(det.bbox, this._lastFused, "bbox_eo")) continue;
      // Backend stamps det.fused_id directly on the wire — we just
      // read it. No client-side matcher, no IoU drift, no E#vs#
      // desync. fusedIdForDet stays around as the bbox-IoU fallback
      // for older recordings without the fused_id field.
      const fusedId = (det.fused_id != null)
        ? det.fused_id
        : fusedIdForDet(det.bbox, this._lastFused, "bbox_eo", 0.20, det);
      // Solo: hide detections that don't belong to the engaged target.
      if (soloEngaged != null
          && (fusedId == null || Number(fusedId) !== soloEngaged)) {
        continue;
      }
      // Lock-bracket suppression: when a lock bbox is rendered for this
      // target on this panel, skip the raw YOLO/heat detection too so
      // the operator sees ONE engagement box (the lock corner brackets)
      // not the brackets stacked on top of the red classifier rectangle.
      if (soloEngaged != null
          && this._lock && this._lock.bbox_eo
          && fusedId != null && Number(fusedId) === soloEngaged) {
        continue;
      }
      const isMain = this._mainTargetId != null &&
                     fusedId != null &&
                     String(fusedId) === String(this._mainTargetId);
      drawDetectionBox(this.ctx, det, scale, dx, dy, isMain, fusedId, "E");
    }

    // Fused overlay rules — mirror of thermal_view.
    // v2: when lock mode is showing a lock_bbox_eo for the engaged
    // track, SUPPRESS the projected/fused green box for that exact
    // track id so the operator sees ONE green box (the lock), not
    // two. Other fused tracks render normally — UNLESS solo mode
    // is on, in which case all non-engaged fused tracks are hidden.
    const lockedId = (this._lock && this._lock.bbox_eo
                       && this._lock.target_id != null)
        ? this._lock.target_id : null;
    for (const trk of this._lastFused) {
      if (lockedId != null && trk.id === lockedId) continue;
      // Solo: hide non-engaged fused tracks entirely.
      if (soloEngaged != null && trk.id !== soloEngaged) continue;
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

    // (Radar overlay removed — radar-only targets are now FusedTrack
    //  entries with sensors=["radar"] and get drawn by the fused loop
    //  above in their class colour. Drawing both was duplicating every
    //  box on the EO panel.)

    // Sticky measurement box — magenta dashed, with a filled label
    // panel showing the distance estimate. Rendered AFTER detections
    // so the operator's hand-drawn rect always sits visibly on top.
    if (this._measurement && this._measurement.bbox) {
      const b = this._measurement.bbox;
      const x = dx + b.x * scale;
      const y = dy + b.y * scale;
      const w = b.w * scale;
      const h = b.h * scale;
      this.ctx.save();
      this.ctx.strokeStyle = "#ff5cd5";  // magenta — matches DEV color
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([7, 4]);
      this.ctx.strokeRect(x, y, w, h);
      this.ctx.setLineDash([]);
      // Crosshair at centroid for the operator's aim reference.
      const cx = x + w / 2, cy = y + h / 2;
      this.ctx.beginPath();
      this.ctx.moveTo(cx - 7, cy); this.ctx.lineTo(cx + 7, cy);
      this.ctx.moveTo(cx, cy - 7); this.ctx.lineTo(cx, cy + 7);
      this.ctx.stroke();
      // Label panel — opaque dark fill, magenta text. Anchor below the
      // box if there's no headroom above (top-line clipping is the
      // only way this can be illegible against varied scenes).
      const label = this._measurement.label || "—";
      this.ctx.font = "12px ui-monospace, monospace";
      const padX = 6, padY = 3;
      const tw = this.ctx.measureText(label).width;
      const labelH = 16;
      let lx = x;
      let ly = y - labelH - 2;
      if (ly < dy) ly = y + h + 2;
      this.ctx.fillStyle = "rgba(0,0,0,0.78)";
      this.ctx.fillRect(lx, ly, tw + padX * 2, labelH);
      this.ctx.fillStyle = "#ff5cd5";
      this.ctx.fillText(label, lx + padX, ly + labelH - padY - 1);
      this.ctx.restore();
    }

    // Lock-mode bbox (v2): corner brackets + center crosshair +
    // "LOCK #ID" label. Visually distinct from the simple rectangles
    // used for fused/projected tracks, so the operator can always
    // tell at a glance which box is the engaged lock vs which is a
    // raw classifier output. Solid green for ACTIVE, dashed amber
    // for COASTING. Drawn last so it sits over all overlays.
    if (this._lock && this._lock.bbox_eo) {
      const bb = this._lock.bbox_eo;
      const px = dx + bb.x * scale;
      const py = dy + bb.y * scale;
      const pw = bb.w * scale;
      const ph = bb.h * scale;
      const coasting = this._lock.state === "coasting";
      const color = coasting ? "#ffb000" : "#00e676";
      drawLockBrackets(this.ctx, px, py, pw, ph, color, coasting);
      const labelText = coasting
        ? `LOCK · COAST #${this._lock.target_id ?? ""}`
        : `LOCK #${this._lock.target_id ?? ""}`;
      drawLockLabel(this.ctx, px, py, color, labelText);
    }

    // Rubber-band rectangle while the user is dragging in measure mode.
    // Drawn last so it sits on top of every overlay.
    if (this._drag) {
      drawRubberBand(this.ctx, this._drag.x0, this._drag.y0,
                     this._drag.x1, this._drag.y1);
    }
  }
}
