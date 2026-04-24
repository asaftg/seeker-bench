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
const radarView   = new RadarView("radar-canvas");

// Mini preview in the DEVELOPERS tab — shows the live thermal image
// (no boxes/labels) next to the sensitivity sliders so the user sees
// blob-bb effects immediately while tuning. Optional; disappears if
// the DEV canvas isn't in the DOM.
// Mini preview SHOWS overlays — that's the whole point: let the user
// watch bounding boxes grow/shrink as they drag the sensitivity sliders.
const thermalMini = document.getElementById("thermal-mini-canvas")
  ? new ThermalView("thermal-mini-canvas", null)
  : null;
const radarMini = document.getElementById("radar-mini-canvas")
  ? new RadarView("radar-mini-canvas")
  : null;

// Black-hot ↔ white-hot toggle. Drives BOTH views so tuning preview
// and main panel stay consistent.
(() => {
  const cb = document.getElementById("thermal-invert");
  if (!cb) return;
  const apply = () => {
    thermalView.setInvert(cb.checked);
    if (thermalMini) thermalMini.setInvert(cb.checked);
  };
  cb.addEventListener("change", apply);
  apply();
})();

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
let _gimbalPan        = null;
let _gimbalTilt       = null;
let _trackedTargetId  = null;     // null = manual; int = user pressed TRACK
let _trackedHeatId    = null;     // dev-mode: raw heat-blob tracker ID we asked gimbal to follow
let _devMode          = false;    // developer overlays: heat-blob tracker debug, etc.
let _recOn            = false;    // REC pill toggle — stubbed recording (backend logs but writes nothing yet)

// Cross-sensor overlay gating — source-centric. Each flag controls
// whether that sensor's tracks project onto the OTHER panels:
//   radar:   draw radar bboxes (cyan dashed) on thermal + EO.
//   thermal: thermal's contribution (fused 2-sensor boxes projected
//            into EO pixel space, single-sensor-thermal projections
//            into EO) appears on EO panel.
//   eo:      EO's contribution appears on thermal panel.
// Raw native detections on a panel's OWN sensor are unaffected.
const _overlay = {
  radar:   true,
  thermal: true,
  eo:      true,
};

// ─────────────────────────────────────────────────────────────────────────
// Draw-a-bbox synthetic target — thermal-panel debug/demo tool.
// Toggle DRAW TARGET, drag a rectangle on the thermal feed, release to
// seed an OF-only track on the backend. CLEAR removes it. ESC cancels
// an in-flight drag (and also deactivates draw mode for quick exit).
// ─────────────────────────────────────────────────────────────────────────
(() => {
  const drawBtn = document.getElementById("draw-target-btn");
  const clearBtn = document.getElementById("clear-target-btn");
  if (!drawBtn && !clearBtn) return;

  const setDraw = (on) => {
    if (drawBtn) drawBtn.classList.toggle("active", !!on);
    thermalView.setDrawMode(!!on, (bbox) => {
      // bbox already clamped + min-size-checked by ThermalView.
      wsSend({
        type: "synthetic_target",
        bbox: [bbox.x, bbox.y, bbox.w, bbox.h],
      });
      // Auto-exit draw mode after a successful commit so the user can
      // immediately interact with the rest of the UI.
      if (drawBtn) drawBtn.classList.remove("active");
      thermalView.setDrawMode(false);
    });
  };

  if (drawBtn) {
    drawBtn.addEventListener("click", () => {
      setDraw(!thermalView.isDrawMode());
    });
  }
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      wsSend({ type: "clear_synthetic_target" });
    });
  }

  window.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && thermalView.isDrawMode()) {
      thermalView.cancelDrag();
      setDraw(false);
    }
  });
})();

// Developer-mode toggle — flips a client-only flag that views consult
// when drawing. Relocated to the DEVELOPERS tab; the button now shows
// its state as text ("ON"/"OFF") since it's a standalone control rather
// than a topbar pill.
(() => {
  const btn = document.getElementById("dev-toggle");
  if (!btn) return;
  const sync = () => {
    btn.classList.toggle("active", _devMode);
    btn.textContent = _devMode ? "ON" : "OFF";
  };
  sync();
  btn.addEventListener("click", () => {
    _devMode = !_devMode;
    sync();
  });
})();

// Tab switcher — MAIN vs DEVELOPERS. Pure DOM toggle; no state persists.
// The mini-preview canvases are hidden (display:none) at page load so
// their bounding rect is 0×0 — their views pick up a 1×1 back-buffer.
// On first activation of the DEV tab we nudge them to refit, then they
// stay in sync via the resize listener and per-update rect check.
(() => {
  const btns = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-content");
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      const want = btn.dataset.tab;
      btns.forEach(b => b.classList.toggle("active", b === btn));
      panels.forEach(p => p.classList.toggle("hidden", p.dataset.tab !== want));
      if (want === "dev") {
        // Two RAFs: first paint reveals the panel (giving canvases a
        // non-zero rect), second RAF lets layout settle before refit.
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (thermalMini && typeof thermalMini.refit === "function") thermalMini.refit();
          if (radarMini   && typeof radarMini.refit   === "function") radarMini.refit();
        }));
      }
    });
  });
})();

// Overlay-screen toggles — one checkbox per sensor; turning a sensor
// off hides its tracks from every OTHER panel.
(() => {
  const radar   = document.getElementById("overlay-radar");
  const thermal = document.getElementById("overlay-thermal");
  const eo      = document.getElementById("overlay-eo");
  radar?.addEventListener("change",   () => { _overlay.radar   = radar.checked; });
  thermal?.addEventListener("change", () => { _overlay.thermal = thermal.checked; });
  eo?.addEventListener("change",      () => { _overlay.eo      = eo.checked; });
})();

// REC pill toggle — stubbed until HDF5 recording ships. We mirror state
// visually so the user sees a red pulse while "recording", and send a
// WS command so the backend can start logging (even if it writes nothing
// to disk yet). The status bar reflects state from the WS echo.
(() => {
  const pill = document.getElementById("pill-rec");
  if (!pill) return;
  pill.addEventListener("click", () => {
    _recOn = !_recOn;
    pill.classList.toggle("pill-rec-on", _recOn);
    wsSend({ command: "record", on: _recOn });
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

// NIR illuminator: removed from GUI (manual flashlight, no host control).
// Any legacy `illuminator` field on the WS payload is silently ignored
// below in the message handler.

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
// Radar live tuning — DEV-tab sliders push via WS so the RadarManager's
// SNR / azimuth gate / DBSCAN params update on the next processed frame.
// ─────────────────────────────────────────────────────────────────────────
(() => {
  const rows = [
    { id: "radar-snr",    key: "snr_min_db",        fmt: v => v.toFixed(1) + " dB",  round: v => v },
    { id: "radar-az",     key: "az_half_deg",       fmt: v => "±" + v.toFixed(0) + "°", round: v => Math.round(v) },
    { id: "radar-speed",  key: "speed_min_mps",     fmt: v => v.toFixed(1) + " m/s", round: v => v },
    { id: "radar-range-min", key: "range_min_m",    fmt: v => v.toFixed(1) + " m",   round: v => v },
    { id: "radar-eps",    key: "cluster_eps_pos_m", fmt: v => v.toFixed(1) + " m",   round: v => v },
    { id: "radar-minpts", key: "cluster_min_samples", fmt: v => String(v),           round: v => Math.round(v), int: true },
  ];
  let _pending = {};
  let _timer = null;
  function flush() {
    _timer = null;
    const patch = _pending; _pending = {};
    if (Object.keys(patch).length > 0) {
      wsSend(Object.assign({ command: "radar_tune" }, patch));
    }
  }
  function queue(key, val) {
    _pending[key] = val;
    if (_timer == null) _timer = setTimeout(flush, 80);
  }
  for (const row of rows) {
    const sl = $(row.id);
    const lbl = $(row.id + "-val");
    if (!sl) continue;
    const paint = () => {
      const raw = parseFloat(sl.value);
      const v = row.int ? row.round(raw) : raw;
      if (lbl) lbl.textContent = row.fmt(v);
      queue(row.key, v);
    };
    sl.addEventListener("input", paint);
    paint();  // sync initial label
  }

  // Adopt server-side values on first message so manual edits in YAML
  // don't fight with the slider defaults hard-coded in HTML.
  let _hydrated = false;
  window.__hydrateRadarTuning = (rt) => {
    if (_hydrated || !rt) return;
    _hydrated = true;
    const map = {
      "radar-snr":    rt.snr_min_db,
      "radar-az":     rt.az_half_deg,
      "radar-speed":  rt.speed_min_mps,
      "radar-range-min": rt.range_min_m,
      "radar-eps":    rt.cluster_eps_pos_m,
      "radar-minpts": rt.cluster_min_samples,
    };
    for (const [id, v] of Object.entries(map)) {
      const sl = $(id); const lbl = $(id + "-val");
      if (!sl || v == null) continue;
      sl.value = String(v);
      const row = rows.find(r => r.id === id);
      if (lbl && row) lbl.textContent = row.fmt(row.int ? Math.round(v) : v);
    }
  };
})();

// ─────────────────────────────────────────────────────────────────────────
// Extrinsic calibration sliders — radar + thermal az/el bias vs. EO
// (ground truth). Debounced `extrinsic_tune` WS command; hydrates from
// the first server payload so YAML defaults win over HTML defaults.
// ─────────────────────────────────────────────────────────────────────────
(() => {
  const rows = [
    { id: "ext-radar-az",   key: "radar_az_bias_deg"   },
    { id: "ext-radar-el",   key: "radar_el_bias_deg"   },
    { id: "ext-thermal-az", key: "thermal_az_bias_deg" },
    { id: "ext-thermal-el", key: "thermal_el_bias_deg" },
  ];
  const fmt = v => (v >= 0 ? "+" : "") + v.toFixed(1) + "°";
  let _pending = {};
  let _timer = null;
  function flush() {
    _timer = null;
    const patch = _pending; _pending = {};
    if (Object.keys(patch).length > 0) {
      wsSend(Object.assign({ command: "extrinsic_tune" }, patch));
    }
  }
  function queue(key, val) {
    _pending[key] = val;
    if (_timer == null) _timer = setTimeout(flush, 80);
  }
  for (const row of rows) {
    const sl = $(row.id);
    const lbl = $(row.id + "-val");
    if (!sl) continue;
    const paint = () => {
      const v = parseFloat(sl.value);
      if (lbl) lbl.textContent = fmt(v);
      queue(row.key, v);
    };
    sl.addEventListener("input", paint);
    paint();
  }

  let _hydrated = false;
  window.__hydrateExtrinsic = (ext) => {
    if (_hydrated || !ext) return;
    _hydrated = true;
    const map = {
      "ext-radar-az":   ext.radar_az_bias_deg,
      "ext-radar-el":   ext.radar_el_bias_deg,
      "ext-thermal-az": ext.thermal_az_bias_deg,
      "ext-thermal-el": ext.thermal_el_bias_deg,
    };
    for (const [id, v] of Object.entries(map)) {
      const sl = $(id); const lbl = $(id + "-val");
      if (!sl || v == null) continue;
      sl.value = String(v);
      if (lbl) lbl.textContent = fmt(Number(v));
    }
  };
})();

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

    // Source-centric overlay gating (DEV tab → OVERLAY SCREEN). A
    // fused track appears on panel P iff at least one of its contributing
    // sensors other than P has its overlay toggle on — turning off a
    // sensor hides its contribution from the OTHER panels. Radar targets
    // (projection-only, no late fusion yet) are gated by `_overlay.radar`.
    const fusedAll = msg.fused || [];
    const fusedForPanel = (panel) => fusedAll.filter(t => {
      const sensors = t.sensors || [];
      return sensors.some(s => s !== panel && _overlay[s]);
    });
    const fusedThermal = fusedForPanel("thermal");
    const fusedEO      = fusedForPanel("eo");

    const radarTargetsAll = (msg.radar && msg.radar.connected && msg.radar.targets) || [];
    const radarForThermal = _overlay.radar ? radarTargetsAll : [];
    const radarForEO      = _overlay.radar ? radarTargetsAll : [];

    // ── Thermal panel ──
    const thermal = msg.thermal || {};
    thermalView.update(thermal, msg.main_target_id || null, fusedThermal, _devMode, radarForThermal);
    if (thermalMini) thermalMini.update(thermal, msg.main_target_id || null, fusedThermal, _devMode, radarForThermal);
    syncZoomButtons(thermal.zoom_preset);

    const thermHz = $("thermal-hz");
    if (thermHz) thermHz.textContent = _fps.current + " Hz";

    setPill("pill-thermal",
            thermal.connected ? "on" : "off",
            "THERMAL");

    // ── EO panel ──
    const eo = msg.eo || {};
    eoView.update(eo, msg.main_target_id || null, fusedEO, radarForEO);
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

    // ── Radar panel ── (Ticket 5a: live AWR2944P point cloud)
    const radar = msg.radar || {};
    const radarDisc = $("radar-disconnected");
    if (radarDisc) radarDisc.classList.toggle("hidden", !!radar.connected);
    setPill("pill-radar", radar.connected ? "on" : "off", "RADAR");
    radarView.update(radar);
    if (radarMini) radarMini.update(radar);

    // Hydrate DEV-tab radar sliders from the server-reported tuning on
    // first tick so they reflect YAML defaults, not HTML-hard-coded ones.
    if (msg.radar_tuning && typeof window.__hydrateRadarTuning === "function") {
      window.__hydrateRadarTuning(msg.radar_tuning);
    }
    if (msg.extrinsic && typeof window.__hydrateExtrinsic === "function") {
      window.__hydrateExtrinsic(msg.extrinsic);
    }

    // ── Recording pill echo ──
    if (msg.recording != null) {
      const want = !!msg.recording;
      if (want !== _recOn) {
        _recOn = want;
        const pill = $("pill-rec");
        if (pill) pill.classList.toggle("pill-rec-on", _recOn);
      }
    }

    // ── Gimbal pill + display ──
    // Green when the servos are online (connected + responding). Amber
    // when auto-tracking a target. Off only when the gimbal manager
    // failed to start or the controller is disconnected.
    const gimbal = msg.gimbal;
    if (gimbal) {
      updateGimbalUI(gimbal);
      const mode = gimbal.mode || "manual";
      const online = gimbal.connected !== false;   // treat missing field as online (older payloads)
      if (!online) {
        setPill("pill-gimbal", "off",  "GIMBAL");
      } else if (mode === "auto") {
        setPill("pill-gimbal", "warn", "GIMBAL AUTO");
      } else {
        setPill("pill-gimbal", "on",   "GIMBAL");
      }
    } else {
      setPill("pill-gimbal", "off", "GIMBAL");
    }

    // ── Targets list (top 5 with TRACK buttons) ──
    renderTargets(msg);
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────
connect();
