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
