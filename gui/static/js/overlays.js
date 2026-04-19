// Shared bbox drawing primitives. Canvas context coordinates.

export const COLORS = {
  thermal: "#ff6b35",   // orange
  radar:   "#00d4ff",   // cyan dashed
  fused:   "#00e88f",   // green thick
  hand:    "#ffa502",   // yellow
};

export function drawThermalBox(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.strokeStyle = COLORS.thermal;
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.strokeRect(x, y, w, h);
  if (label) drawLabel(ctx, label, x, y, COLORS.thermal);
  ctx.restore();
}

export function drawHandBox(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.strokeStyle = COLORS.hand;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  if (label) drawLabel(ctx, label, x, y, COLORS.hand);
  ctx.restore();
}

export function drawDroneBox(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.strokeStyle = COLORS.fused;
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  if (label) drawLabel(ctx, label, x, y, COLORS.fused);
  ctx.restore();
}

function drawLabel(ctx, text, x, y, color) {
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
