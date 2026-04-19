// WebSocket client, tab router, top-level state updates.

import { ThermalView } from "./thermal_view.js";
import { RadarView }   from "./radar_view.js";

const $ = (id) => document.getElementById(id);

// ───────── Tab router ─────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.dataset.page === name));
    // Canvases need a resize poke when they become visible
    window.dispatchEvent(new Event("resize"));
  });
});

// ───────── Views ─────────
const thermalMain = new ThermalView("thermal-canvas", "thermal-disconnected");
const thermalBig  = new ThermalView("thermal-canvas-big", "thermal-disconnected-big");
const radarView   = new RadarView("radar-canvas");

// ───────── Banner state ─────────
function setBanner(state, text) {
  const b = $("banner");
  b.className = "detection-banner " + state;
  $("banner-text").textContent = text;
}

function computeBanner(thermal) {
  if (!thermal || !thermal.connected) return ["banner-scanning", "THERMAL DISCONNECTED"];
  const dets = thermal.detections || [];
  if (dets.length === 0) return ["banner-scanning", "SCANNING — NO HEAT DETECTED"];

  // Prioritize: drone > hand > heat
  let hasDrone = false, hasHand = false;
  for (const d of dets) {
    const c = d.classification && d.classification.target_class;
    if (c === "drone") hasDrone = true;
    else if (c === "hand") hasHand = true;
  }
  if (hasDrone) return ["banner-drone", `DRONE DETECTED — ${dets.length} target${dets.length>1?"s":""}`];
  if (hasHand)  return ["banner-hand",  `HAND DETECTED — ${dets.length} target${dets.length>1?"s":""}`];
  return ["banner-heat", `HEAT DETECTED — ${dets.length} blob${dets.length>1?"s":""}`];
}

// ───────── Pills ─────────
function setPill(id, connected, label) {
  const pill = $(id);
  pill.className = "pill " + (connected ? "pill-connected" : "pill-disconnected");
  pill.innerHTML = `<span class="pill-dot"></span>${label}`;
}

// ───────── Detection table ─────────
function updateDetTable(thermal) {
  const tbody = $("det-tbody");
  if (!tbody) return;
  const dets = (thermal && thermal.detections) || [];
  let html = "";
  dets.forEach((d, i) => {
    const b = d.bbox;
    const cls = d.classification ? d.classification.target_class : "—";
    const conf = d.classification ? (d.classification.confidence*100).toFixed(0) + "%" : "—";
    html += `<tr>
      <td>${i+1}</td>
      <td>${b.x},${b.y} ${b.w}×${b.h}</td>
      <td>${d.area_px}</td>
      <td>${d.contrast}</td>
      <td>${cls}</td>
      <td>${conf}</td>
    </tr>`;
  });
  tbody.innerHTML = html || `<tr><td colspan="6" style="color:var(--text-muted);padding:10px 8px">no detections</td></tr>`;
}

// ───────── Engineering page ─────────
let _fpsCounter = { t0: performance.now(), frames: 0, current: 0 };
function tickFps() {
  _fpsCounter.frames += 1;
  const now = performance.now();
  const dt = now - _fpsCounter.t0;
  if (dt >= 1000) {
    _fpsCounter.current = (_fpsCounter.frames * 1000 / dt).toFixed(1);
    _fpsCounter.t0 = now;
    _fpsCounter.frames = 0;
  }
}

let _lastFrameTs = null;
function updateEngineering(msg) {
  const t = msg.thermal || {};
  $("eng-ws-fps").textContent    = _fpsCounter.current + " fps";
  $("eng-frame-id").textContent  = t.frame_id ?? "—";
  $("eng-size").textContent      = t.width && t.height ? `${t.width}×${t.height}` : "—";
  $("eng-fov").textContent       = (t.hfov_deg != null) ? `${t.hfov_deg}° / ${t.vfov_deg}°` : "—";
  $("eng-zoom").textContent      = t.zoom_preset ?? "—";
  $("eng-det").textContent       = t.detections ? t.detections.length : 0;
  const firstCls = t.detections && t.detections.find(d => d.classification);
  $("eng-classifier").textContent = firstCls ? firstCls.classification.classifier_used : "—";
  _lastFrameTs = performance.now();
}

// ───────── Status bar ─────────
function updateStatusBar(msg) {
  const t = msg.thermal || {};
  $("stat-thermal-fps").textContent = _fpsCounter.current;
  $("stat-det-count").textContent   = t.detections ? t.detections.length : 0;
  $("stat-radar").textContent       = (msg.radar && msg.radar.connected) ? "ONLINE" : "OFFLINE";
  $("stat-fusion").textContent      = (msg.fusion && msg.fusion.active)   ? "ACTIVE" : "INACTIVE";
}

// ───────── WebSocket ─────────
const wsUrl = `ws://${location.host}/ws/sensors`;
$("eng-ws-url").textContent = wsUrl;
let ws = null;

function connect() {
  $("ws-status").textContent = "WS: connecting…";
  $("eng-ws-state").textContent = "connecting";
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    $("ws-status").textContent = "WS: connected";
    $("eng-ws-state").textContent = "open";
  };
  ws.onclose = () => {
    $("ws-status").textContent = "WS: disconnected — retrying";
    $("eng-ws-state").textContent = "closed";
    setPill("pill-thermal", false, "THERMAL");
    setPill("pill-radar",   false, "RADAR");
    setPill("pill-fusion",  false, "FUSION");
    setBanner("banner-scanning", "DISCONNECTED");
    setTimeout(connect, 1500);
  };
  ws.onerror = () => { try { ws.close(); } catch(e){} };
  ws.onmessage = (ev) => {
    tickFps();
    let msg;
    try { msg = JSON.parse(ev.data); } catch(e){ return; }

    thermalMain.update(msg.thermal);
    thermalBig.update(msg.thermal);
    radarView.update(msg.radar);
    syncZoomButtons(msg.thermal && msg.thermal.zoom_preset);

    const connectedThermal = msg.thermal && msg.thermal.connected;
    setPill("pill-thermal", connectedThermal, "THERMAL");
    setPill("pill-radar",   msg.radar && msg.radar.connected, "RADAR");
    setPill("pill-fusion",  msg.fusion && msg.fusion.active,  "FUSION");

    const [state, text] = computeBanner(msg.thermal);
    setBanner(state, text);

    updateStatusBar(msg);
    updateDetTable(msg.thermal);
    updateEngineering(msg);
  };
}

// Frame-age updater
setInterval(() => {
  if (_lastFrameTs == null) return;
  const age = (performance.now() - _lastFrameTs) / 1000;
  $("eng-age").textContent = age.toFixed(2) + " s";
}, 250);

// Zoom buttons — send the preset to the backend. The ThermalManager
// crops raw16 + display BEFORE running the heat detector, so blobs
// outside the zoomed FOV are never computed, counted, or rendered.
// The "active" class is NOT set here — it's reconciled from the next
// WS frame so the UI always reflects real backend state.
document.querySelectorAll(".panel-btn[data-zoom]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = btn.dataset.zoom;
    fetch("/api/config/thermal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zoom_preset: preset }),
    }).catch(() => {});
  });
});

function syncZoomButtons(preset) {
  if (!preset) return;
  document.querySelectorAll(".panel-btn[data-zoom]").forEach((b) => {
    b.classList.toggle("active", b.dataset.zoom === preset);
  });
}

// ───────── Heat detector config sliders ─────────
// Both sliders are INVERTED: pushing left = stricter, pushing right =
// more sensitive. The slider's raw HTML value is a UI position; the
// real value sent to the backend is (min + max - value). Full-left on
// both sliders should suppress virtually everything.
function sliderReal(el) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const raw = parseFloat(el.value);
  return el.dataset.invert ? (min + max - raw) : raw;
}
function sliderSetReal(el, real) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const raw = el.dataset.invert ? (min + max - real) : real;
  el.value = Math.max(min, Math.min(max, raw));
}

async function loadDetectorConfig() {
  try {
    const r = await fetch("/api/config/heat_detector");
    const j = await r.json();
    if (j.available) {
      sliderSetReal($("thr-slider"), j.threshold_k);
      $("thr-val").textContent = Number(j.threshold_k).toFixed(1);
      sliderSetReal($("minarea-slider"), j.min_blob_area_px);
      $("minarea-val").textContent = j.min_blob_area_px;
    }
  } catch (e) { /* backend may not have thermal_manager wired */ }
}

let _cfgTimer = null;
function postDetectorConfig(patch) {
  clearTimeout(_cfgTimer);
  _cfgTimer = setTimeout(() => {
    fetch("/api/config/heat_detector", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).catch(() => {});
  }, 120);
}

$("thr-slider").addEventListener("input", (e) => {
  const v = sliderReal(e.target);
  $("thr-val").textContent = v.toFixed(1);
  postDetectorConfig({ threshold_k: v });
});
$("minarea-slider").addEventListener("input", (e) => {
  const v = Math.round(sliderReal(e.target));
  $("minarea-val").textContent = v;
  postDetectorConfig({ min_blob_area_px: v });
});

loadDetectorConfig();
connect();
