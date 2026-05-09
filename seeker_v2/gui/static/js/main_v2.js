/* seeker_v2 GUI — robust, RAF-driven. */
(() => {
    'use strict';

    // ── State ─────────────────────────────────────────────────────────
    const state = {
        ws: null, wsConnected: false,
        latest: null, stats: {},
        eoFetchedId: -1, thermalFetchedId: -1,
        // Hz from frame_id deltas in /api/status (works even when images
        // don't load).
        eoHz:    new RateMeter('eo_stats',      'frame_id'),
        thermalHz: new RateMeter('thermal_stats', 'frame_id'),
        radarHz: new RateMeter('radar_stats',   'frame_id'),
    };
    function RateMeter(statKey, field) {
        const m = { statKey, field, samples: [] };
        m.update = (stats) => {
            const s = stats && stats[statKey];
            if (!s || s[field] == null) return;
            const now = performance.now() / 1000;
            m.samples.push({ t: now, v: s[field] });
            const cut = now - 4.0;
            while (m.samples.length > 0 && m.samples[0].t < cut) m.samples.shift();
        };
        m.value = () => {
            if (m.samples.length < 2) return null;
            const a = m.samples[0], b = m.samples[m.samples.length - 1];
            const dt = b.t - a.t;
            const dv = b.v - a.v;
            if (dt <= 0) return null;
            return dv / dt;
        };
        return m;
    }

    // ── DOM ───────────────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);
    const dom = {
        wsState:        $('ws-state'),
        eoImg:          $('eo-img'),
        thermalImg:     $('thermal-img'),
        eoOverlay:      $('eo-overlay'),
        thermalOverlay: $('thermal-overlay'),
        eoHz:           $('eo-hz'),
        thermalHz:      $('thermal-hz'),
        radarHz:        $('radar-hz'),
        radarPolar:     $('radar-polar'),
        targets:        document.querySelectorAll('.targets-row .target-cell'),
        panRead:  $('pan-readout'),  tiltRead: $('tilt-readout'),
        panSlider: $('pan-slider'),  tiltSlider: $('tilt-slider'),
        homeBtn:  $('home-btn'),
        devEo: $('dev-eo'), devThermal: $('dev-thermal'),
        devRadar: $('dev-radar'), devInference: $('dev-inference'),
        devFusion: $('dev-fusion'), devWs: $('dev-ws'),
        pills: {
            thermal: document.querySelector('.pill[data-pill="thermal"]'),
            eo:      document.querySelector('.pill[data-pill="eo"]'),
            radar:   document.querySelector('.pill[data-pill="radar"]'),
            gimbal:  document.querySelector('.pill[data-pill="gimbal"]'),
            rec:     document.querySelector('.pill[data-pill="rec"]'),
        },
    };

    // ── Tabs ──────────────────────────────────────────────────────────
    document.querySelectorAll('.tab').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
            document.querySelectorAll('.tab-page').forEach(p => p.classList.toggle('active', p.dataset.page === target));
            console.log('[tabs] switched to', target);
        });
    });

    // Sliders update readout (UI-only for now; control hookup in next iter)
    if (dom.panSlider)  dom.panSlider.addEventListener('input',  () => dom.panRead.textContent  = parseFloat(dom.panSlider.value).toFixed(1) + '°');
    if (dom.tiltSlider) dom.tiltSlider.addEventListener('input', () => dom.tiltRead.textContent = parseFloat(dom.tiltSlider.value).toFixed(1) + '°');
    if (dom.homeBtn) dom.homeBtn.addEventListener('click', () => {
        dom.panSlider.value  = 0; dom.tiltSlider.value = 0;
        dom.panRead.textContent = '0.0°'; dom.tiltRead.textContent = '0.0°';
    });

    // ── WebSocket ─────────────────────────────────────────────────────
    function connectWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${proto}//${location.host}/ws/sensors`);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
            state.wsConnected = true;
            dom.wsState.textContent = 'online'; dom.wsState.className = 'online';
        };
        ws.onclose = () => {
            state.wsConnected = false;
            dom.wsState.textContent = 'offline'; dom.wsState.className = 'offline';
            setTimeout(connectWS, 1000);
        };
        ws.onerror = (e) => console.warn('[ws] error', e);
        ws.onmessage = ev => {
            try {
                const txt = (typeof ev.data === 'string')
                    ? ev.data : new TextDecoder().decode(ev.data);
                state.latest = JSON.parse(txt);
            } catch (e) { console.warn('[ws] parse', e); }
        };
        state.ws = ws;
    }

    // ── Snapshot fetcher ──────────────────────────────────────────────
    function maybeFetch(sensor, imgEl, key) {
        const snaps = state.latest && state.latest.snapshots;
        if (!snaps) return;
        const id = snaps[key] ?? -1;
        const stKey = sensor + 'FetchedId';
        if (id < 0 || id === state[stKey]) return;
        state[stKey] = id;
        imgEl.src = `/api/snapshot/${sensor}.jpg?id=${id}`;
    }

    // ── Overlay drawing ───────────────────────────────────────────────
    function drawOverlay(canvas, w, h, fn) {
        const dpr = window.devicePixelRatio || 1;
        const cw = canvas.clientWidth, ch = canvas.clientHeight;
        if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
            canvas.width = cw * dpr; canvas.height = ch * dpr;
        }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cw, ch);
        const sx = cw / w, sy = ch / h;
        fn(ctx, sx, sy);
    }
    function drawEoOverlay() {
        const m = state.latest && state.latest.eo_meta;
        if (!m) return;
        drawOverlay(dom.eoOverlay, m.w, m.h, (ctx, sx, sy) => {
            const fused = state.latest.fused || [];
            ctx.lineWidth = 2; ctx.font = '12px ui-monospace, monospace';
            for (const t of fused) {
                if (!t.eo_box) continue;
                const [x, y, w, h] = t.eo_box;
                ctx.strokeStyle = t.cls === 'vehicle' ? '#f6c560' : '#6ce085';
                ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
                ctx.fillStyle = ctx.strokeStyle;
                ctx.fillText(`#${t.tid} ${t.cls || ''}`, x * sx + 2, y * sy - 4);
            }
        });
    }
    function drawThermalOverlay() {
        const m = state.latest && state.latest.thermal_meta;
        if (!m) return;
        drawOverlay(dom.thermalOverlay, m.w, m.h, (ctx, sx, sy) => {
            ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffcd56';
            for (const d of (m.heat_dets || [])) {
                ctx.strokeRect(d.x * sx, d.y * sy, d.w * sx, d.h * sy);
            }
            const fused = state.latest.fused || [];
            ctx.lineWidth = 2;
            for (const t of fused) {
                if (!t.thermal_box) continue;
                const [x, y, w, h] = t.thermal_box;
                ctx.strokeStyle = t.cls === 'vehicle' ? '#f6c560' : '#6ce085';
                ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
            }
        });
    }

    // ── Radar polar ───────────────────────────────────────────────────
    function buildRadarPolar() {
        const svg = dom.radarPolar;
        if (!svg || svg.dataset.built) return;
        svg.dataset.built = '1';
        const NS = 'http://www.w3.org/2000/svg';
        for (const r of [25, 50, 75, 100]) {
            const ring = document.createElementNS(NS, 'path');
            ring.setAttribute('d', `M ${-r} 0 A ${r} ${r} 0 0 1 ${r} 0`);
            ring.setAttribute('class', 'ring'); svg.appendChild(ring);
            const lbl = document.createElementNS(NS, 'text');
            lbl.setAttribute('x', '0'); lbl.setAttribute('y', String(-r + 1));
            lbl.setAttribute('text-anchor', 'middle');
            lbl.setAttribute('class', 'axis-label');
            lbl.textContent = `${r} m`; svg.appendChild(lbl);
        }
        for (const a of [-60, -30, 0, 30, 60]) {
            const rad = a * Math.PI / 180;
            const x = 100 * Math.sin(rad), y = -100 * Math.cos(rad);
            const ln = document.createElementNS(NS, 'line');
            ln.setAttribute('x1', '0'); ln.setAttribute('y1', '0');
            ln.setAttribute('x2', String(x)); ln.setAttribute('y2', String(y));
            ln.setAttribute('class', 'axis'); svg.appendChild(ln);
        }
    }
    function drawRadarPolar() {
        if (!dom.radarPolar) return;
        buildRadarPolar();
        dom.radarPolar.querySelectorAll('.target').forEach(el => el.remove());
        const tgts = (state.stats && state.stats.radar_stats &&
                      state.stats.radar_stats.targets) || [];
        const NS = 'http://www.w3.org/2000/svg';
        for (const t of tgts.slice(0, 30)) {
            const az = (t.az_deg ?? 0) * Math.PI / 180;
            const rng = t.range_m ?? Math.hypot(t.x ?? 0, t.y ?? 0);
            const c = document.createElementNS(NS, 'circle');
            c.setAttribute('cx', String(rng * Math.sin(az)));
            c.setAttribute('cy', String(-rng * Math.cos(az)));
            c.setAttribute('r', '1.2'); c.setAttribute('class', 'target');
            dom.radarPolar.appendChild(c);
        }
    }

    // ── Targets row ───────────────────────────────────────────────────
    function renderTargets() {
        const fused = (state.latest && state.latest.fused) || [];
        for (let i = 0; i < dom.targets.length; i++) {
            const cell = dom.targets[i]; const t = fused[i];
            if (!t) {
                cell.className = 'target-cell empty';
                cell.textContent = '--';
            } else {
                cell.className = `target-cell cls-${t.cls || 'unknown'}`;
                cell.innerHTML =
                    `<span class="tid">#${t.tid}</span><br>` +
                    `${t.cls || '?'}<br>` +
                    `az ${(t.az_deg ?? 0).toFixed(1)}°<br>` +
                    `el ${(t.el_deg ?? 0).toFixed(1)}°`;
            }
        }
    }

    // ── Hz + pills + dev ──────────────────────────────────────────────
    function fmtHz(v) { return v == null ? '— Hz' : v.toFixed(1) + ' Hz'; }
    function updateMeters() {
        state.eoHz.update(state.stats);
        state.thermalHz.update(state.stats);
        state.radarHz.update(state.stats);
        const e = state.eoHz.value(), t = state.thermalHz.value(), r = state.radarHz.value();
        dom.eoHz.textContent      = fmtHz(e);
        dom.thermalHz.textContent = fmtHz(t);
        dom.radarHz.textContent   = fmtHz(r);
        // pills
        dom.pills.eo     .classList.toggle('online', e != null && e > 0.5);
        dom.pills.thermal.classList.toggle('online', t != null && t > 0.5);
        dom.pills.radar  .classList.toggle('online', r != null && r > 0.5);
        dom.pills.gimbal .classList.toggle('online', state.wsConnected);
    }
    function updateDev() {
        const fmt = o => o ? JSON.stringify(o, null, 2) : 'no data';
        const s = state.stats || {};
        dom.devEo.textContent        = fmt(s.eo_stats);
        dom.devThermal.textContent   = fmt(s.thermal_stats);
        dom.devRadar.textContent     = fmt(s.radar_stats);
        dom.devInference.textContent = fmt(s.inference_stats);
        dom.devFusion.textContent    = fmt(s.fusion_stats);
        dom.devWs.textContent        = fmt(state.latest);
    }

    // ── RAF loop ──────────────────────────────────────────────────────
    function rafTick() {
        try {
            maybeFetch('eo', dom.eoImg, 'eo_id');
            maybeFetch('thermal', dom.thermalImg, 'thermal_id');
            drawEoOverlay();
            drawThermalOverlay();
            drawRadarPolar();
            renderTargets();
            updateMeters();
            updateDev();
        } catch (e) { console.error('[raf]', e); }
        requestAnimationFrame(rafTick);
    }

    // ── Status poller (1.5 s) ─────────────────────────────────────────
    async function pollStatus() {
        try {
            const r = await fetch('/api/status', { cache: 'no-store' });
            if (r.ok) state.stats = await r.json();
        } catch (_) { /* ignore */ }
        setTimeout(pollStatus, 1500);
    }

    // ── Boot ──────────────────────────────────────────────────────────
    connectWS();
    pollStatus();
    requestAnimationFrame(rafTick);
    console.log('[seeker_v2] GUI booted');
})();
