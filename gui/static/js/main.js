// SEEKER-01 Phase B — main WebSocket client + UI controller.
// No framework, no browser storage APIs, no position:fixed.

import { ThermalView } from "./thermal_view.js";
import { RadarView }   from "./radar_view.js";

const $ = (id) => document.getElementById(id);

// ─────────────────────────────────────────────────────────────────────────
// Views
// ─────────────────────────────────────────────────────────────────────────
const thermalView = new ThermalView("thermal-canvas", "thermal-disconnected");
// EO and radar panels have their own disconnected overlays; canvas drawing
// will be wired in Ticket 3 (EO) and Ticket 4 (Radar). For now they show DISCONNECTED.

// ─────────────────────────────────────────────────────────────────────────
// UI state (local mirror; reconciled from WS on each frame)
// ─────────────────────────────────────────────────────────────────────────
let _trackerOn  = true;
let _nirMode    = "auto";   // "auto" | "on" | "off"
let _gimbalPan  = null;
let _gimbalTilt = null;

// ─────────────────────────────────────────────────────────────────────────
// Connection pills
// ─────────────────────────────────────────────────────────────────────────
function setPill(id, state, label) {
  // state: "on" | "warn" | "off"
  const el = $(id);
  if (!el) return;
  el.className = "pill pill-" + state;
  el.innerHTML = `<span class="dot"></span>${label}`;
}

// ─────────────────────────────────────────────────────────────────────────
// Main-target card
// ─────────────────────────────────────────────────────────────────────────
function updateMainTarget(msg) {
  const id   = msg.main_target_id;
  const row  = $("mt-row");
  const cls  = $("mt-class");
  if (!row || !cls) return;

  if (!id) {
    cls.textContent = "NO LOCK";
    cls.style.color = "var(--text-3)";
    // Clear extra cells
    row.innerHTML = `<span class="mt-class" id="mt-class" style="color:var(--text-3);">NO LOCK</span>`;
    return;
  }

  // Find the track that matches main_target_id
  const tracks = msg.tracks || [];
  const t = tracks.find(t => String(t.id) === String(id));

  cls.textContent = t ? (t.class || "UNKNOWN").toUpperCase() : "UNKNOWN";
  cls.style.color = "var(--text)";

  let extra = "";
  if (t) {
    if (t.id    != null) extra += `<span class="mt-val">ID ${t.id}</span>`;
    if (t.range != null) extra += `<span class="mt-val">${t.range} m</span>`;
    if (t.speed != null) extra += `<span class="mt-val">${t.speed} m/s</span>`;
    if (t.confidence != null) extra += `<span class="mt-val">conf ${(t.confidence*100|0)}%</span>`;
    const sensors = t.sensors || 1;
    extra += `<span class="mt-badge">${sensors} sensor${sensors>1?"s":""}</span>`;
  }
  row.innerHTML = `<span class="mt-class" id="mt-class">${cls.textContent}</span>${extra}`;
}

// ─────────────────────────────────────────────────────────────────────────
// Tracker button
// ─────────────────────────────────────────────────────────────────────────
const trackerBtn = $("tracker-btn");
if (trackerBtn) {
  trackerBtn.addEventListener("click", () => {
    _trackerOn = !_trackerOn;
    syncTrackerUI();
    wsSend({ command: "tracker", state: _trackerOn ? "on" : "off" });
  });
}

function syncTrackerUI() {
  if (!trackerBtn) return;
  const stateEl = $("tracker-state");
  trackerBtn.classList.toggle("off", !_trackerOn);
  if (stateEl) stateEl.textContent = _trackerOn ? "● ON" : "○ OFF";

  // Enable / disable gimbal dpad
  document.querySelectorAll(".dpad-btn").forEach(b => {
    b.disabled = _trackerOn;   // manual only when tracker is OFF
  });
  const hint = $("gimbal-hint");
  if (hint) hint.textContent = _trackerOn ? "manual disabled · tracker ON" : "manual enabled · use arrows";
}

// ─────────────────────────────────────────────────────────────────────────
// NIR toggle
// ─────────────────────────────────────────────────────────────────────────
document.querySelectorAll(".nir-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.nir;
    _nirMode = mode;
    syncNirUI(mode);
    wsSend({ command: "nir", mode });
  });
});

function syncNirUI(mode) {
  document.querySelectorAll(".nir-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.nir === mode);
  });
  const pill = $("pill-nir");
  if (pill) {
    if (mode === "off") {
      setPill("pill-nir", "off", "NIR OFF");
    } else if (mode === "on") {
      setPill("pill-nir", "on", "NIR ON");
    } else {
      setPill("pill-nir", "warn", "NIR AUTO");
    }
  }
  const hint = $("nir-hint");
  if (hint) {
    hint.textContent = mode === "auto" ? "pulsed 20% duty (safe)" :
                       mode === "on"   ? "continuous — thermal caution!" :
                                         "illuminator off";
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Gimbal dpad
// ─────────────────────────────────────────────────────────────────────────
document.querySelectorAll(".dpad-btn[data-dp]").forEach(btn => {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    const dp = parseFloat(btn.dataset.dp || 0);
    const dt = parseFloat(btn.dataset.dt || 0);
    wsSend({ command: "gimbal_manual", delta_pan: dp, delta_tilt: dt });
  });
});

const homeBtn = $("dpad-home");
if (homeBtn) {
  homeBtn.addEventListener("click", () => {
    if (homeBtn.disabled) return;
    wsSend({ command: "gimbal_manual", delta_pan: -(_gimbalPan || 0), delta_tilt: 60 - (_gimbalTilt || 60) });
  });
}

function updateGimbalUI(gimbal) {
  if (!gimbal) return;
  _gimbalPan  = gimbal.pan;
  _gimbalTilt = gimbal.tilt;
  const panEl  = $("gimbal-pan");
  const tiltEl = $("gimbal-tilt");
  if (panEl)  panEl.textContent  = (_gimbalPan  != null) ? _gimbalPan.toFixed(1)  + "°" : "—";
  if (tiltEl) tiltEl.textContent = (_gimbalTilt != null) ? _gimbalTilt.toFixed(1) + "°" : "—";
}

// ─────────────────────────────────────────────────────────────────────────
// Zoom buttons
// ─────────────────────────────────────────────────────────────────────────
document.querySelectorAll(".zoom-btn[data-zoom]").forEach(btn => {
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
  document.querySelectorAll(".zoom-btn[data-zoom]").forEach(b => {
    b.classList.toggle("active", b.dataset.zoom === preset);
  });
  const fovMap = { full: "75°", wide: "37°", mid: "18°", narrow: "12°" };
  const fovEl = $("thermal-fov");
  if (fovEl && fovMap[preset]) fovEl.textContent = fovMap[preset] + " HFOV";
}

// ─────────────────────────────────────────────────────────────────────────
// Thermal tuning sliders
// ─────────────────────────────────────────────────────────────────────────
function sliderReal(el) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  return el.dataset.invert ? (min + max - parseFloat(el.value)) : parseFloat(el.value);
}
function sliderSetReal(el, real) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  el.value = Math.max(min, Math.min(max, el.dataset.invert ? (min + max - real) : real));
}

async function loadDetectorConfig() {
  try {
    const r = await fetch("/api/config/heat_detector");
    const j = await r.json();
    if (!j.available) return;
    const thr = $("thr-slider"), thrVal = $("thr-val");
    const area = $("minarea-slider"), areaVal = $("minarea-val");
    if (thr && j.threshold_k != null) {
      sliderSetReal(thr, j.threshold_k);
      if (thrVal) thrVal.textContent = Number(j.threshold_k).toFixed(1) + " ×";
    }
    if (area && j.min_blob_area_px != null) {
      sliderSetReal(area, j.min_blob_area_px);
      if (areaVal) areaVal.textContent = j.min_blob_area_px + " px";
    }
  } catch (_) {}
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

const thrSlider = $("thr-slider");
const thrVal    = $("thr-val");
if (thrSlider) {
  thrSlider.addEventListener("input", () => {
    const v = sliderReal(thrSlider);
    if (thrVal) thrVal.textContent = v.toFixed(1) + " ×";
    postDetectorConfig({ threshold_k: v });
  });
}
const areaSlider = $("minarea-slider");
const areaVal    = $("minarea-val");
if (areaSlider) {
  areaSlider.addEventListener("input", () => {
    const v = Math.round(sliderReal(areaSlider));
    if (areaVal) areaVal.textContent = v + " px";
    postDetectorConfig({ min_blob_area_px: v });
  });
}

// ─────────────────────────────────────────────────────────────────────────
// FPS counter
// ─────────────────────────────────────────────────────────────────────────
let _fps = { t0: performance.now(), frames: 0, current: "—" };
function tickFps() {
  _fps.frames++;
  const now = performance.now();
  const dt  = now - _fps.t0;
  if (dt >= 1000) {
    _fps.current = (_fps.frames * 1000 / dt).toFixed(0);
    _fps.t0      = now;
    _fps.frames  = 0;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Status bar
// ─────────────────────────────────────────────────────────────────────────
function updateStatusBar(msg) {
  const t = msg.thermal || {};
  const det = t.detections ? t.detections.length : 0;
  const tracks = (msg.tracks || []).length;
  const mainId = msg.main_target_id || "—";

  const left  = $("stat-left");
  const mid   = $("stat-mid");
  if (left) left.textContent =
    `ws ${_fps.current} Hz · thermal ${_fps.current} Hz · eo — · radar —`;
  if (mid) mid.textContent =
    `${det} det · ${tracks} tracks · main ${mainId} · tracker ${_trackerOn ? "ON" : "OFF"} · rec OFF`;
}

// ─────────────────────────────────────────────────────────────────────────
// WebSocket
// ─────────────────────────────────────────────────────────────────────────
const wsUrl = `ws://${location.host}/ws/sensors`;
let ws = null;

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

function connect() {
  const wsStatus = $("ws-status");
  if (wsStatus) wsStatus.textContent = "WS: connecting…";
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    if (wsStatus) wsStatus.textContent = "WS: connected";
    loadDetectorConfig();
  };

  ws.onclose = () => {
    if (wsStatus) wsStatus.textContent = "WS: disconnected — retrying";
    setPill("pill-thermal", "off", "THERMAL");
    setPill("pill-eo",      "off", "EO");
    setPill("pill-radar",   "off", "RADAR");
    setPill("pill-gimbal",  "off", "GIMBAL");
    setTimeout(connect, 1500);
  };

  ws.onerror = () => { try { ws.close(); } catch(_) {} };

  ws.onmessage = (ev) => {
    tickFps();
    let msg;
    try { msg = JSON.parse(ev.data); } catch(_) { return; }

    // ── Thermal panel ──
    const thermal = msg.thermal || {};
    thermalView.update(thermal, msg.main_target_id || null);
    syncZoomButtons(thermal.zoom_preset);

    const thermHz = $("thermal-hz");
    if (thermHz) thermHz.textContent = _fps.current + " Hz";

    setPill("pill-thermal",
            thermal.connected ? "on" : "off",
            "THERMAL");

    // ── EO panel ── (disconnected until Ticket 3)
    const eo = msg.eo || {};
    const eoDisc = $("eo-disconnected");
    if (eoDisc) eoDisc.classList.toggle("hidden", !!eo.connected);
    setPill("pill-eo", eo.connected ? "on" : "off", "EO");

    // ── Radar panel ── (disconnected until Ticket 4)
    const radar = msg.radar || {};
    const radarDisc = $("radar-disconnected");
    if (radarDisc) radarDisc.classList.toggle("hidden", !!radar.connected);
    setPill("pill-radar", radar.connected ? "on" : "off", "RADAR");

    // ── Illuminator pill ──
    const illum = msg.illuminator;
    if (illum) {
      // Sync NIR state from backend (handles reconnect)
      if (illum.state && illum.state !== _nirMode) {
        _nirMode = illum.state;
        syncNirUI(_nirMode);
      }
    }

    // ── Gimbal pill + display ──
    const gimbal = msg.gimbal;
    if (gimbal) {
      updateGimbalUI(gimbal);
      const mode = gimbal.mode || "manual";
      setPill("pill-gimbal",
              mode === "auto" ? "warn" : "off",
              mode === "auto" ? "GIMBAL AUTO" : "GIMBAL");
    }

    // ── Tracker state from backend ──
    if (msg.tracker_on !== undefined && msg.tracker_on !== _trackerOn) {
      _trackerOn = msg.tracker_on;
      syncTrackerUI();
    }

    // ── Main target card ──
    updateMainTarget(msg);

    // ── Status bar ──
    updateStatusBar(msg);
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────
syncTrackerUI();
syncNirUI(_nirMode);
connect();
