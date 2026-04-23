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

    // Heat-blob row (dev mode) uses data-heat-id; fused row uses data-track-id.
    if (btn.dataset.heatId) {
      const raw = btn.dataset.heatId;
      const id = (raw === "" || raw == null) ? null : Number(raw);
      const isCurrent = (_trackedHeatId != null) && (_trackedHeatId === id);
      const nextId = isCurrent ? null : id;
      _trackedHeatId = nextId;
      // A heat-lock supersedes the fused lock — clear the mirror so
      // the UI doesn't briefly show two active rows before the server
      // echoes back the new state.
      if (nextId != null) _trackedTargetId = null;
      console.log("[track] heat click id=", id, "→ send", nextId);
      wsSend({ command: "track_heat", heat_id: nextId });
      return;
    }

    const raw = btn.dataset.trackId;
    const id = raw != null ? Number(raw) : null;
    const isCurrent = (_trackedTargetId != null) && (_trackedTargetId === id);
    const nextId = isCurrent ? null : id;
    _trackedTargetId = nextId;
    if (nextId != null) _trackedHeatId = null;
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
let _trackedHeatId    = null;     // dev-mode: raw heat-blob tracker ID we asked gimbal to follow
let _devMode          = false;    // developer overlays: heat-blob tracker debug, etc.

// Developer-mode toggle — flips a client-only flag that views consult
// when drawing. No round-trip: the backend always sends the debug
// payload, the client decides whether to paint it.
(() => {
  const btn = document.getElementById("dev-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    _devMode = !_devMode;
    btn.classList.toggle("active", _devMode);
    btn.title = _devMode
      ? "Developer overlays ON — click to hide heat-blob tracks"
      : "Developer overlays (heat-blob tracker debug)";
  });
})();

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

// Fixed 5-slot list: build rows ONCE, then mutate text/classes in place.
// Rewriting innerHTML at 20Hz would destroy the TRACK button the user
// is hovering/clicking every 50ms — that's what caused the flicker +
// missed clicks. Stable DOM nodes = stable hover, stable clicks.
const SLOTS = 5;
const HEAT_SLOTS = 5;
let _rowNodes = null;       // fused-track rows
let _heatHeader = null;     // divider shown above heat rows in dev mode
let _heatRowNodes = null;   // raw heat-blob rows (dev-mode only)

function _buildRowNodes() {
  const list = $("targets-list");
  if (!list) return null;
  list.innerHTML = "";
  const nodes = [];
  for (let i = 0; i < SLOTS; i++) {
    const row = document.createElement("div");
    row.className = "target-row placeholder";
    row.innerHTML = `
      <span class="tr-id">—</span>
      <span class="tr-cls">—</span>
      <span class="tr-sensors">—</span>
      <span class="tr-conf">—</span>
      <span class="tr-angle mono">—</span>
      <button class="tr-btn" data-track-id="">TRACK</button>
    `;
    list.appendChild(row);
    nodes.push({
      row,
      id:      row.children[0],
      cls:     row.children[1],
      sensors: row.children[2],
      conf:    row.children[3],
      angle:   row.children[4],
      btn:     row.children[5],
    });
  }
  return nodes;
}

function _buildHeatRowNodes() {
  const list = $("targets-list");
  if (!list) return null;

  // Section divider — hidden unless dev mode is on AND there's data.
  const header = document.createElement("div");
  header.className = "target-heat-header hidden";
  header.textContent = "HEAT · DEV";
  list.appendChild(header);
  _heatHeader = header;

  const nodes = [];
  for (let i = 0; i < HEAT_SLOTS; i++) {
    const row = document.createElement("div");
    row.className = "target-row target-row-heat placeholder hidden";
    row.innerHTML = `
      <span class="tr-id">—</span>
      <span class="tr-cls">—</span>
      <span class="tr-sensors">—</span>
      <span class="tr-conf">—</span>
      <span class="tr-angle mono">—</span>
      <button class="tr-btn" data-heat-id="">TRACK</button>
    `;
    list.appendChild(row);
    nodes.push({
      row,
      id:      row.children[0],
      cls:     row.children[1],
      sensors: row.children[2],
      conf:    row.children[3],
      angle:   row.children[4],
      btn:     row.children[5],
    });
  }
  return nodes;
}

function renderTargets(msg) {
  const lockState = $("targets-lock-state");
  if (!_rowNodes) _rowNodes = _buildRowNodes();
  if (!_heatRowNodes) _heatRowNodes = _buildHeatRowNodes();
  if (!_rowNodes) return;

  const top = msg.top_targets || [];
  const backendTracked = (msg.tracked_target_id != null)
    ? Number(msg.tracked_target_id) : null;
  if (backendTracked !== _trackedTargetId) {
    _trackedTargetId = backendTracked;
  }
  const backendHeat = (msg.tracked_heat_id != null)
    ? Number(msg.tracked_heat_id) : null;
  if (backendHeat !== _trackedHeatId) {
    _trackedHeatId = backendHeat;
  }

  if (lockState) {
    if (_trackedTargetId != null) {
      lockState.textContent = `lock · #${_trackedTargetId} · gimbal AUTO`;
      lockState.style.color = "var(--fused-green, #00e88f)";
    } else if (_trackedHeatId != null) {
      lockState.textContent = `lock · H#${_trackedHeatId} · gimbal AUTO (dev)`;
      lockState.style.color = "#ff7be2";
    } else {
      lockState.textContent = "no lock · gimbal manual";
      lockState.style.color = "var(--text-3)";
    }
  }

  _renderHeatRows(msg);

  for (let i = 0; i < SLOTS; i++) {
    const t = top[i] || null;
    const n = _rowNodes[i];

    if (t == null) {
      // Placeholder — keep row height, hide button, dim.
      if (n.row.className !== "target-row placeholder") {
        n.row.className = "target-row placeholder";
      }
      n.id.textContent      = "—";
      n.cls.textContent     = "—";
      n.cls.style.color     = "";
      n.sensors.textContent = "—";
      n.sensors.className   = "tr-sensors";
      n.conf.textContent    = "—";
      n.angle.textContent   = "—";
      n.btn.dataset.trackId = "";
      // Don't touch btn text/class — visibility:hidden handles it via CSS.
      continue;
    }

    const cls = t.target_class || "unknown";
    const label = _clsLabel(cls);
    const color = _CLASS_COLORS[cls] || _CLASS_COLORS.unknown;
    const nSensors = (t.sensors || []).length;
    const sensorsTxt = (t.sensors || []).map(s => s.toUpperCase()).join("+") || "—";
    const conf = (t.confidence != null) ? `${(t.confidence * 100) | 0}%` : "—";
    const az = (t.az_deg != null) ? `${t.az_deg.toFixed(1)}°` : "—";
    const el = (t.el_deg != null) ? `${t.el_deg.toFixed(1)}°` : "—";
    const tracked   = (_trackedTargetId != null) && (Number(t.id) === _trackedTargetId);
    const confirmed = nSensors >= 2;

    const rowCls =
      "target-row" +
      (tracked   ? " tracked"   : "") +
      (confirmed ? " confirmed" : " single");
    if (n.row.className !== rowCls) n.row.className = rowCls;

    n.id.textContent      = `#${t.id}`;
    n.cls.textContent     = label;
    n.cls.style.color     = color;
    n.sensors.textContent = `${nSensors}× ${sensorsTxt}`;
    const senCls = "tr-sensors " + (confirmed ? "multi" : "solo");
    if (n.sensors.className !== senCls) n.sensors.className = senCls;
    n.conf.textContent    = conf;
    n.angle.textContent   = `${az}, ${el}`;

    // Only update btn state when it actually changed — avoids any
    // attribute churn while the user is hovering.
    const wantId   = String(t.id);
    const wantText = tracked ? "TRACKING" : "TRACK";
    const wantCls  = "tr-btn" + (tracked ? " active" : "");
    if (n.btn.dataset.trackId !== wantId) n.btn.dataset.trackId = wantId;
    if (n.btn.textContent !== wantText)   n.btn.textContent     = wantText;
    if (n.btn.className   !== wantCls)    n.btn.className       = wantCls;
  }
}

// Dev-mode only: render raw heat-blob tracker rows below the fused list.
// Each row has a TRACK button that locks the gimbal onto the blob's
// current bbox center (converted to az/el on the backend, per frame).
// We only surface CONFIRMED tracks — pending/coasting entries would
// just thrash the list and a click on them would often miss.
function _renderHeatRows(msg) {
  if (!_heatRowNodes || !_heatHeader) return;

  const heatTracks = ((msg.thermal && msg.thermal.heat_tracks) || [])
    .filter(h => h && h.confirmed);
  // Sort by id ascending so rows are stable (same sort every tick).
  heatTracks.sort((a, b) => a.id - b.id);
  const show = _devMode && heatTracks.length > 0;

  if (_heatHeader.classList.contains("hidden") === show) {
    _heatHeader.classList.toggle("hidden", !show);
  }

  for (let i = 0; i < HEAT_SLOTS; i++) {
    const n = _heatRowNodes[i];
    const h = show ? (heatTracks[i] || null) : null;

    if (h == null) {
      if (!n.row.classList.contains("hidden")) {
        n.row.classList.add("hidden");
      }
      // Nothing else to do; button is inside a hidden row.
      continue;
    }

    if (n.row.classList.contains("hidden")) {
      n.row.classList.remove("hidden");
    }

    const tracked = (_trackedHeatId != null) && (h.id === _trackedHeatId);
    const rowCls =
      "target-row target-row-heat" +
      (tracked ? " tracked" : "");
    if (n.row.className !== rowCls) n.row.className = rowCls;

    const state = h.coasting ? "coast" : "ok";
    n.id.textContent      = `H#${h.id}`;
    n.cls.textContent     = "HEAT";
    n.cls.style.color     = "#ff7be2";   // dev magenta
    n.sensors.textContent = `thermal · ${state}`;
    n.sensors.className   = "tr-sensors";
    n.conf.textContent    = `${h.hits}/${h.misses}`;
    const b = h.bbox || {};
    n.angle.textContent   = `${b.w|0}×${b.h|0}`;

    const wantId   = String(h.id);
    const wantText = tracked ? "TRACKING" : "TRACK";
    const wantCls  = "tr-btn" + (tracked ? " active" : "");
    if (n.btn.dataset.heatId !== wantId) n.btn.dataset.heatId = wantId;
    if (n.btn.textContent    !== wantText) n.btn.textContent    = wantText;
    if (n.btn.className      !== wantCls)  n.btn.className      = wantCls;
  }
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
    thermalView.update(thermal, msg.main_target_id || null, fused, _devMode);
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
