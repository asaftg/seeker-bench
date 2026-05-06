// Shared bbox drawing primitives.
// Phase B color scheme per Ticket 2 spec:
//   orange  = heat / unclassified EO
//   blue    = drone
//   purple  = human / person
//   red     = vehicle
//   cyan    = radar-projected track (dashed)
//   green   = main target (overrides class color, 3px thick)

export const COLORS = {
  heat:    "#ff6b35",   // orange
  drone:   "#378ADD",   // blue
  person:  "#AFA9EC",   // purple
  vehicle: "#E24B4A",   // red
  hand:    "#ffa502",   // amber (legacy indoor test)
  radar:   "#00d4ff",   // cyan dashed
  main:    "#00e88f",   // green thick — main target override
  dev:     "#ff4ccc",   // magenta — developer overlays (tracker debug)
};

// ---------------------------------------------------------------------------
// Classify-aware box: picks color by class, allows main-target override.
// `sourceTag` is the per-sensor namespace prefix used when no fused id
// is known yet — "E" (EO ByteTrack), "T" (thermal heat tracker), "R"
// (radar Kalman tracker). Defaults to "E" for backwards compat with
// callers that haven't been updated yet.
// ---------------------------------------------------------------------------
export function drawDetectionBox(ctx, det, scale, dx, dy,
                                 isMainTarget = false, fusedId = null,
                                 sourceTag = "E") {
  const b = det.bbox;
  const x = dx + b.x * scale;
  const y = dy + b.y * scale;
  const w = b.w * scale;
  const h = b.h * scale;

  const cls  = det.classification && det.classification.target_class;
  const conf = det.classification && det.classification.confidence;

  let color, lineWidth, label;
  if (isMainTarget) {
    color = COLORS.main;
    lineWidth = 3;
  } else if (cls === "drone") {
    color = COLORS.drone;
    lineWidth = 1.5;
  } else if (cls === "person") {
    color = COLORS.person;
    lineWidth = 1.5;
  } else if (cls === "vehicle") {
    color = COLORS.vehicle;
    lineWidth = 1.5;
  } else if (cls === "hand") {
    color = COLORS.hand;
    lineWidth = 2;
  } else {
    color = COLORS.heat;
    lineWidth = 2;
  }

  // Prefix with the most-authoritative id we have. Order:
  //   1. fused track id  — `#N` (cross-sensor, lock target)
  //   2. per-sensor id   — `<sourceTag>#N` (transient, sensor-local)
  //   3. no prefix       — single-frame detection, no tracker
  // sourceTag is "E"/"T"/"R" picked by the calling panel.
  let idPrefix = "";
  if (fusedId != null) {
    idPrefix = `#${fusedId} `;
  } else if (det.track_id != null) {
    idPrefix = `${sourceTag}#${det.track_id} `;
  }
  if (conf != null && cls && cls !== "unknown") {
    const clsLabel = cls === "person" ? "HUMAN" : cls.toUpperCase();
    label = `${idPrefix}${clsLabel} ${(conf * 100) | 0}%`;
  } else {
    label = `${idPrefix}HEAT Δ${det.contrast}`;
  }

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash([]);
  ctx.strokeRect(x, y, w, h);
  if (label) _drawLabel(ctx, label, x, y, color);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Fused track: green box representing a target confirmed by 2+ sensors.
// Shown identically on every panel — the centroid is the world angle the
// drone would fly toward. `track.bbox_{thermal,eo}` is pre-projected into
// the panel's pixel space by the backend; we only need scale + offset.
// ---------------------------------------------------------------------------
export function drawFusedBox(ctx, bbox, track, scale, dx, dy) {
  if (!bbox) return;
  const x = dx + bbox.x * scale;
  const y = dy + bbox.y * scale;
  const w = bbox.w * scale;
  const h = bbox.h * scale;

  const cls = track.target_class;
  const clsLabel = cls === "person" ? "HUMAN" : (cls || "TARGET").toUpperCase();
  const conf = Math.round((track.confidence || 0) * 100);
  const nSensors = (track.sensors || []).length;
  const tag = nSensors >= 2 ? `${nSensors}×` : "";
  const label = `${tag}#${track.id} ${clsLabel} ${conf}%`;

  ctx.save();
  ctx.strokeStyle = COLORS.main;
  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  ctx.strokeRect(x, y, w, h);
  // Crosshair at centroid — this is the gimbal aim-point.
  const cx = x + w / 2, cy = y + h / 2;
  ctx.beginPath();
  ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy);
  ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
  ctx.stroke();
  _drawLabel(ctx, label, x, y, COLORS.main);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Projected (cross-sensor, unconfirmed) box.
// Used when a fused track has only 1 sensor and we're rendering the OTHER
// panel: shows the user "something is over here per the other sensor" but
// makes it visually obvious the local sensor hasn't confirmed yet.
// Dashed class-colored box (not green — green is reserved for 2+ sensors).
// ---------------------------------------------------------------------------
export function drawProjectedBox(ctx, bbox, track, scale, dx, dy) {
  if (!bbox) return;
  const x = dx + bbox.x * scale;
  const y = dy + bbox.y * scale;
  const w = bbox.w * scale;
  const h = bbox.h * scale;

  const cls = track.target_class;
  const color =
    cls === "drone"   ? COLORS.drone   :
    cls === "person"  ? COLORS.person  :
    cls === "vehicle" ? COLORS.vehicle : COLORS.heat;
  const clsLabel = cls === "person" ? "HUMAN" : (cls || "TARGET").toUpperCase();
  const srcSensor = (track.sensors && track.sensors[0]) || "?";
  const label = `#${track.id} ${clsLabel} ← ${srcSensor.toUpperCase()}`;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x, y, w, h);
  _drawLabel(ctx, label, x, y, color);
  ctx.restore();
}

// Returns true if a raw per-sensor detection is subsumed by any track
// in ``fusedTracks`` with >= 2 sensors (green-box tracks only — single-
// sensor tracks never have a green box, so they must never suppress the
// raw detection underneath). ``sideKey`` is "bbox_thermal" or "bbox_eo".
export function isSubsumedByFused(rawBBox, fusedTracks, sideKey, iouMin = 0.15) {
  if (!rawBBox || !fusedTracks || !fusedTracks.length) return false;
  for (const t of fusedTracks) {
    if (!t) continue;
    if (!t.sensors || t.sensors.length < 2) continue;
    const fb = t[sideKey];
    if (!fb) continue;
    if (_iou(rawBBox, fb) >= iouMin) return true;
  }
  return false;
}

// Per-sensor tracker-id field on a FusedTrack, keyed by sideKey.
// Single source of truth for the link map; adding a new sensor =
// one entry here.
const _LINK_FIELD_BY_SIDE = {
  "bbox_eo":      "eo_track_id",
  "bbox_thermal": "thermal_heat_id",
  // Radar isn't a per-bbox panel (polar plot), so it's matched
  // separately via fusedIdForRadarTarget below.
};

// Best fused ID for a raw detection on a given side, or null. Used to
// stamp raw detection labels with the same fusion ID shown in the
// targets list and the cross-sensor projection.
//
// Two-stage match:
//   1. Direct id (det.track_id ↔ FusedTrack[link_field]). Robust
//      under EMA smoothing of the fused track's stored angles
//      where bbox-IoU would drift below threshold despite being
//      the same physical target. Same code path for EO and thermal
//      via _LINK_FIELD_BY_SIDE — adding a sensor is one line.
//   2. Greedy IoU fallback on the projected bbox — for raw dets
//      that don't carry a per-sensor track_id, or for fused tracks
//      that haven't recorded a link yet (e.g. just-born tracks).
export function fusedIdForDet(rawBBox, fusedTracks, sideKey, iouMin = 0.20, det = null) {
  if (!fusedTracks || !fusedTracks.length) return null;

  // Stage 1: direct id match.
  const linkField = _LINK_FIELD_BY_SIDE[sideKey];
  if (linkField && det && det.track_id != null) {
    const tid = Number(det.track_id);
    if (tid >= 0) {
      for (const t of fusedTracks) {
        if (!t) continue;
        const trkLinkId = t[linkField];
        if (trkLinkId != null && Number(trkLinkId) === tid) {
          return t.id;
        }
      }
    }
  }

  // Stage 2: bbox-IoU fallback.
  if (!rawBBox) return null;
  let bestId = null;
  let bestIou = iouMin;
  for (const t of fusedTracks) {
    if (!t) continue;
    const fb = t[sideKey];
    if (!fb) continue;
    const iou = _iou(rawBBox, fb);
    if (iou >= bestIou) {
      bestIou = iou;
      bestId  = t.id;
    }
  }
  return bestId;
}

// Radar-specific lookup: given a RadarTarget.tid, find the fused id
// that includes it (or null). Used by radar_view.js to label radar
// boxes with the fused id when fusion has promoted the target.
export function fusedIdForRadarTarget(radarTid, fusedTracks) {
  if (radarTid == null || !fusedTracks || !fusedTracks.length) return null;
  const want = Number(radarTid);
  for (const t of fusedTracks) {
    if (!t) continue;
    if (t.radar_tid != null && Number(t.radar_tid) === want) {
      return t.id;
    }
  }
  return null;
}

function _iou(a, b) {
  const ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
  const bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

// ---------------------------------------------------------------------------
// Developer mode — heat-blob tracker overlay.
// Draws EVERY tracker entry (including unconfirmed + coasting) in magenta,
// with the internal track ID and hits/misses next to each box. Separate
// from the production heat/drone/fused boxes so turning dev-mode on/off
// never disturbs what a non-developer user sees.
//
//   confirmed + matched this tick → solid thin magenta
//   confirmed + coasting (miss)   → dashed magenta
//   unconfirmed (hits<min)        → dotted magenta
// ---------------------------------------------------------------------------
export function drawHeatTrackDebug(ctx, track, scale, dx, dy) {
  const b = track.bbox;
  if (!b) return;
  const x = dx + b.x * scale;
  const y = dy + b.y * scale;
  const w = b.w * scale;
  const h = b.h * scale;

  ctx.save();
  ctx.strokeStyle = COLORS.dev;
  ctx.lineWidth = 1;
  if (!track.confirmed) {
    ctx.setLineDash([2, 3]);         // dotted = not yet confirmed
  } else if (track.coasting) {
    ctx.setLineDash([6, 4]);         // dashed = coasting on last position
  } else {
    ctx.setLineDash([]);             // solid  = matched this tick
  }
  ctx.strokeRect(x, y, w, h);

  // Tiny corner tick at the bbox centroid so overlapping boxes are
  // still distinguishable.
  const cx = x + w / 2, cy = y + h / 2;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(cx - 2, cy); ctx.lineTo(cx + 2, cy);
  ctx.moveTo(cx, cy - 2); ctx.lineTo(cx, cy + 2);
  ctx.stroke();

  const tag = track.coasting ? "coast" : (track.confirmed ? "ok" : "pend");
  const label = `H#${track.id} ${track.hits}/${track.misses} ${tag}`;
  _drawLabel(ctx, label, x, y + h + 12, COLORS.dev);  // label BELOW the box
                                                       // so it doesn't overlap
                                                       // the production label
                                                       // drawn on a same-spot
                                                       // production detection.
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Synthetic "USER TARGET" box — magenta dashed. Drawn for any detection
// carrying synthetic=true (user "Draw Target" seed, propagated by the
// tracker's OF bridge). Independent of heat detector / classifier.
// ---------------------------------------------------------------------------
export function drawSyntheticTargetBox(ctx, det, scale, dx, dy) {
  const b = det && det.bbox;
  if (!b) return;
  const x = dx + b.x * scale;
  const y = dy + b.y * scale;
  const w = b.w * scale;
  const h = b.h * scale;

  ctx.save();
  ctx.strokeStyle = COLORS.dev;
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 4]);
  ctx.strokeRect(x, y, w, h);
  // Crosshair at centroid for the operator's aim reference.
  const cx = x + w / 2, cy = y + h / 2;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
  ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
  ctx.stroke();

  // The tracker assigns the synthetic track its own internal ID but
  // we don't carry it out through the wire as a distinct field —
  // reuse the area/contrast label slot for a stable "USER TARGET" tag.
  const idTag = (det.synthetic_id != null) ? `#${det.synthetic_id} ` : "";
  _drawLabel(ctx, `${idTag}USER TARGET`, x, y, COLORS.dev);
  ctx.restore();
}

// Transient rubber-band rectangle while the user is dragging in draw mode.
// Takes CANVAS-space coords (already scaled) — the caller is in the same
// coordinate frame as the mouse event.
export function drawRubberBand(ctx, x0, y0, x1, y1) {
  const x = Math.min(x0, x1), y = Math.min(y0, y1);
  const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0);
  ctx.save();
  ctx.strokeStyle = COLORS.dev;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Lock-mode v2 render helpers
// ---------------------------------------------------------------------------

/**
 * Draw four corner brackets + a center crosshair at (x, y, w, h).
 * Visually distinct from the solid/dashed rectangles used for fused
 * and projected tracks so the operator can always tell at a glance
 * which box is the engaged lock vs which is a regular detection.
 *   color = "#00e676" (green) → ACTIVE
 *   color = "#ffb000" (amber) → COASTING
 *   dashed = true             → COASTING line style
 */
export function drawLockBrackets(ctx, x, y, w, h, color, dashed) {
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  if (dashed) ctx.setLineDash([8, 4]);
  // Bracket length: 1/4 of the shorter side (or 18 px floor).
  const bl = Math.max(18, Math.min(w, h) * 0.25);
  // Top-left
  ctx.beginPath();
  ctx.moveTo(x, y + bl); ctx.lineTo(x, y); ctx.lineTo(x + bl, y);
  ctx.stroke();
  // Top-right
  ctx.beginPath();
  ctx.moveTo(x + w - bl, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + bl);
  ctx.stroke();
  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(x + w, y + h - bl); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - bl, y + h);
  ctx.stroke();
  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(x + bl, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - bl);
  ctx.stroke();
  // Center crosshair — solid lines (always solid even when COASTING).
  const cx = x + w / 2, cy = y + h / 2;
  const ch = Math.max(8, Math.min(w, h) * 0.10);
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(cx - ch, cy); ctx.lineTo(cx + ch, cy);
  ctx.moveTo(cx, cy - ch); ctx.lineTo(cx, cy + ch);
  ctx.stroke();
  ctx.restore();
}

/**
 * Filled-background label drawn just above (x, y) with the lock
 * color. Mirrors _drawLabel but takes the color as an argument.
 */
export function drawLockLabel(ctx, x, y, color, text) {
  ctx.save();
  ctx.font = "bold 12px 'JetBrains Mono', ui-monospace, monospace";
  const padX = 6, padY = 3;
  const tw = ctx.measureText(text).width + padX * 2;
  const th = 18;
  const ly = Math.max(0, y - th - 1);
  ctx.fillStyle = color;
  ctx.fillRect(x, ly, tw, th);
  ctx.fillStyle = "#000";
  ctx.fillText(text, x + padX, ly + th - padY - 2);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Radar projected track (cyan dashed)
// ---------------------------------------------------------------------------
export function drawRadarBox(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.strokeStyle = COLORS.radar;
  ctx.lineWidth = 1.2;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(x, y, w, h);
  if (label) _drawLabel(ctx, label, x, y, COLORS.radar);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Legacy helpers (kept for backward compat, delegate to new scheme)
// ---------------------------------------------------------------------------
export function drawThermalBox(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.strokeStyle = COLORS.heat;
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.strokeRect(x, y, w, h);
  if (label) _drawLabel(ctx, label, x, y, COLORS.heat);
  ctx.restore();
}

export function drawHandBox(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.strokeStyle = COLORS.hand;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  if (label) _drawLabel(ctx, label, x, y, COLORS.hand);
  ctx.restore();
}

export function drawDroneBox(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.strokeStyle = COLORS.drone;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  if (label) _drawLabel(ctx, label, x, y, COLORS.drone);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------
function _drawLabel(ctx, text, x, y, color) {
  ctx.font = "10px 'JetBrains Mono', monospace";
  const padX = 4, padY = 2;
  const metrics = ctx.measureText(text);
  const tw = metrics.width + padX * 2;
  const th = 14;
  const ly = Math.max(0, y - th - 1);
  ctx.fillStyle = "rgba(0,0,0,0.78)";
  ctx.fillRect(x, ly, tw, th);
  ctx.fillStyle = color;
  ctx.fillText(text, x + padX, ly + th - padY - 2);
}
