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
};

// ---------------------------------------------------------------------------
// Classify-aware box: picks color by class, allows main-target override
// ---------------------------------------------------------------------------
export function drawDetectionBox(ctx, det, scale, dx, dy, isMainTarget = false) {
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

  if (conf != null && cls && cls !== "unknown") {
    const clsLabel = cls === "person" ? "HUMAN" : cls.toUpperCase();
    label = `${clsLabel} ${(conf * 100) | 0}%`;
  } else {
    label = `HEAT Δ${det.contrast}`;
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

// Returns true if a raw per-sensor detection is subsumed by any fused
// bbox in the provided list. The GUI skips raw detections that match so
// we don't stack a colored box underneath the fused green box.
export function isSubsumedByFused(rawBBox, fusedBBoxes, iouMin = 0.30) {
  if (!rawBBox || !fusedBBoxes || !fusedBBoxes.length) return false;
  for (const fb of fusedBBoxes) {
    if (!fb) continue;
    const iou = _iou(rawBBox, fb);
    if (iou >= iouMin) return true;
  }
  return false;
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
