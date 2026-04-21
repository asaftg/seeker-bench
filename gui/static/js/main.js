// SEEKER-01 Phase B — main WebSocket client + UI controller.
// No framework, no browser storage APIs, no position:fixed.

import { ThermalView } from "./thermal_view.js";
import { EOView }      from "./eo_view.js";
import { RadarView }   from "./radar_view.js";

const $ = (id) => document.getElementById(id);

// ─────────────────────────────────────────────────────────────────────────
// Views
// ─────────────────────────────────────────────────────────────────────────
const thermalView = new ThermalView("thermal-canvas", "thermal-disconnected");
const eoView      = new EOView("eo-canvas", "eo-disconnected");

// Delegated TRACK button handler — bound ONCE on the list container.
// The list's innerHTML gets rewritten every WS frame (~20Hz), so any
// per-button listener would race the rewrite and lose its click. The
// container itself is permanent, so delegation is reliable.
(() => {
  const list = document.getElementById("targets-list");
  if (!list) { console.warn("[track] targets-list not found at load"); return; }
  console.log("[track] delegated click listener installed on #targets-list");
  list.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".tr-btn");
    if (!btn || !list.contains(btn)) return;
    ev.stopPropagation();
    const raw = btn.dataset.trackId;
    const id = raw != null ? Number(raw) : null;
    const isCurrent = (_trackedTargetId != null) && (_trackedTargetId === id);
    const nextId = isCurrent ? null : id;
    _trackedTargetId = nextId;
    console.log("[track] click id=", id, "→ send", nextId);
    wsSend({ command: "track", track_id: nextId });
  });
})();
// Radar panel canvas drawing will be wired in Ticket 4. For now it shows DISCONNECTED.

// ─────────────────────────────────────────────────────────────────────────
// UI state (local mirror; reconciled from WS on each frame)
// ─────────────────────────────────────────────────────────────────────────
let _nirMode          = "auto";   // "auto" | "on" | "off"
let _gimbalPan        = null;
let _gimbalTilt       = null;
let _trackedTargetId  = null;     // null = manual; int = user pressed TRACK

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
// Targets list (top 5) — each row has a TRACK toggle button.
// Pressing TRACK on a row sends {command:"track", track_id:N} to the
// backend. Pressing it again (or clicking another row's TRACK) clears
// and replaces the lock. When no row is tracked, gimbal stays manual.
// ─────────────────────────────────────────────────────────────────────────
const _CLASS_COLORS = {
  drone:   "#378ADD",
  person:  "#AFA9EC",
  vehicle: "#E24B4A",
  unknown: "#ff6b35",
};

function _clsLabel(cls) {
  if (!cls) return "TARGET";
  if (cls === "person") return "HUMAN";
  return cls.toUpperCase();
}

function renderTargets(msg) {
  const list = $("targets-list");
  const lockState = $("targets-lock-state");
  if (!list) return;

  const top = msg.top_targets || [];
  // Reconcile local tracked id with backend truth (it drops the lock
  // if the ID disappears from the fused list).
  const backendTracked = (msg.tracked_target_id != null)
    ? Number(msg.tracked_target_id) : null;
  if (backendTracked !== _trackedTargetId) {
    _trackedTargetId = backendTracked;
  }

  if (lockState) {
    if (_trackedTargetId != null) {
      lockState.textContent = `lock · #${_trackedTargetId} · gimbal AUTO`;
      lockState.style.color = "var(--fused-green, #00e88f)";
    } else {
      lockState.textContent = "no lock · gimbal manual";
      lockState.style.color = "var(--text-3)";
    }
  }

  if (!top.length) {
    list.innerHTML = `<div class="targets-empty">no fused targets</div>`;
    return;
  }

  const rows = top.map(t => {
    const cls = t.target_class || "unknown";
    const label = _clsLabel(cls);
    const color = _CLASS_COLORS[cls] || _CLASS_COLORS.unknown;
    const nSensors = (t.sensors || []).length;
    const sensorsTxt = (t.sensors || []).map(s => s.toUpperCase()).join("+") || "—";
    const conf = (t.confidence != null) ? `${(t.confidence * 100) | 0}%` : "—";
    const az = (t.az_deg != null) ? `${t.az_deg.toFixed(1)}°` : "—";
    const el = (t.el_deg != null) ? `${t.el_deg.toFixed(1)}°` : "—";
    const tracked = (_trackedTargetId != null) && (Number(t.id) === _trackedTargetId);
    const confirmed = nSensors >= 2;

    return `
      <div class="target-row ${tracked ? "tracked" : ""} ${confirmed ? "confirmed" : "single"}"
           data-id="${t.id}">
        <span class="tr-id">#${t.id}</span>
        <span class="tr-cls" style="color:${color}">${label}</span>
        <span class="tr-sensors ${confirmed ? "multi" : "solo"}">${nSensors}× ${sensorsTxt}</span>
        <span class="tr-conf">${conf}</span>
        <span class="tr-angle mono">${az}, ${el}</span>
        <button class="tr-btn ${tracked ? "active" : ""}" data-track-id="${t.id}">
          ${tracked ? "TRACKING" : "TRACK"}
        </button>
      </div>
    `;
  }).join("");

  list.innerHTML = rows;
  // NOTE: TRACK button clicks are handled by a single delegated
  // listener bound ONCE on the list container (see below). Re-binding
  // per-button here would race the 20Hz innerHTML rewrite — the button
  // the user clicked on often gets destroyed before its handler fires.
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
// Dpad is always live now — backend releases any active track lock
// as soon as the user nudges manually.
document.querySelectorAll(".dpad-btn[data-dp]").forEach(btn => {
  btn.addEventListener("click", () => {
    const dp = parseFloat(btn.dataset.dp || 0);
    const dt = parseFloat(btn.dataset.dt || 0);
    wsSend({ command: "gimbal_manual", delta_pan: dp, delta_tilt: dt });
  });
});

const homeBtn = $("dpad-home");
if (homeBtn) {
  homeBtn.addEventListener("click", () => {
    // Backend knows the configured home pose — don't compute it client-side.
    wsSend({ command: "gimbal_home" });
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
  const lock = (_trackedTargetId != null) ? `#${_trackedTargetId} AUTO` : "manual";
  if (mid) mid.textContent =
    `${det} det · ${tracks} tracks · main ${mainId} · gimbal ${lock} · rec OFF`;
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

    // Fusion: single green bbox on every panel for confirmed targets.
    const fused = msg.fused || [];

    // ── Thermal panel ──
    const thermal = msg.thermal || {};
    thermalView.update(thermal, msg.main_target_id || null, fused);
    syncZoomButtons(thermal.zoom_preset);

    const thermHz = $("thermal-hz");
    if (thermHz) thermHz.textContent = _fps.current + " Hz";

    setPill("pill-thermal",
            thermal.connected ? "on" : "off",
            "THERMAL");

    // ── EO panel ──
    const eo = msg.eo || {};
    eoView.update(eo, msg.main_target_id || null, fused);
    setPill("pill-eo", eo.connected ? "on" : "off", "EO");
    const eoHz = $("eo-hz");
    if (eoHz) eoHz.textContent = eo.connected ? (_fps.current + " Hz") : "— Hz";
    const eoFovEl = $("eo-fov");
    if (eoFovEl && eo.hfov_deg != null) {
      eoFovEl.textContent = Number(eo.hfov_deg).toFixed(1) + "° HFOV";
    }
    // Keep the panel title honest: the hardcoded "IMX568 35mm 11°"
    // becomes "TEST WEBCAM 67°" when running on a test webcam so the
    // displayed FOV matches what fusion is actually using.
    const eoTitle = $("eo-title");
    if (eoTitle && eo.hfov_deg != null) {
      const hfov = Number(eo.hfov_deg);
      if (hfov > 20) {
        eoTitle.textContent = `EO · TEST WEBCAM · ${hfov.toFixed(0)}°`;
      } else {
        eoTitle.textContent = `EO · IMX568 · 35mm NIR · ${hfov.toFixed(1)}°`;
      }
    }

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

    // ── Targets list (top 5 with TRACK buttons) ──
    renderTargets(msg);

    // ── Status bar ──
    updateStatusBar(msg);
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────
syncNirUI(_nirMode);
connect();
