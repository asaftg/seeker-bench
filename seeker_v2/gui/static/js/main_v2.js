/* seeker_v2 GUI — RAF-driven render loop, v1-style layout. */
(() => {
    'use strict';

    const state = {
        ws: null,
        wsConnected: false,
        latest: null,
        eoFetchedId: -1,
        thermalFetchedId: -1,
        fps: { eo: rfps(), thermal: rfps() },
        stats: {},
    };
    function rfps(win = 2.0) {
        const o = { win, t: [] };
        o.tick = () => {
            const n = performance.now() / 1000; o.t.push(n);
            const cut = n - o.win;
            while (o.t.length && o.t[0] < cut) o.t.shift();
        };
        o.value = () => o.t.length / o.win;
        return o;
    }

    const dom = {};
    [
        'eo-img', 'thermal-img', 'eo-overlay', 'thermal-overlay',
        'tracks-list', 'eo-hz', 'thermal-hz', 'radar-hz', 'ws-state',
        'pill-thermal', 'pill-eo', 'pill-radar', 'pill-gimbal', 'pill-rec',
        'pan-val', 'tilt-val', 'pan-slider', 'tilt-slider', 'home-btn',
        'radar-polar',
        'dev-eo', 'dev-thermal', 'dev-radar', 'dev-inference', 'dev-fusion', 'dev-ws',
    ].forEach(id => dom[id.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = document.getElementById(id));

    // ── Tabs ────────────────────────────────────────────────────────
    document.querySelectorAll('.tab').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
            document.querySelectorAll('.page').forEach(p => p.classList.toggle('hidden', p.dataset.page !== target));
        });
    });

    // ── WebSocket ───────────────────────────────────────────────────
    function connectWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${proto}//${location.host}/ws/sensors`);
        ws.binaryType = 'arraybuffer';
        ws.onopen = () => {
            state.wsConnected = true;
            dom.wsState.textContent = 'online';
            dom.wsState.className = 'online';
        };
        ws.onclose = () => {
            state.wsConnected = false;
            dom.wsState.textContent = 'offline';
            dom.wsState.className = 'offline';
            setTimeout(connectWS, 1000);
        };
        ws.onerror = e => console.warn('[ws] error', e);
        ws.onmessage = ev => {
            try {
                const txt = (typeof ev.data === 'string')
                    ? ev.data : new TextDecoder().decode(ev.data);
                state.latest = JSON.parse(txt);
            } catch (e) { console.warn('[ws] parse', e); }
        };
        state.ws = ws;
    }

    // ── Image streaming ──────────────────────────────────────────────
    function maybeFetch(sensor, imgEl, key) {
        const snaps = state.latest && state.latest.snapshots;
        if (!snaps) return;
        const id = snaps[key] ?? -1;
        const stKey = sensor + 'FetchedId';
        if (id < 0 || id === state[stKey]) return;
        state[stKey] = id;
        imgEl.src = `/api/snapshot/${sensor}.jpg?id=${id}`;
    }
    dom.eoImg.addEventListener('load',      () => state.fps.eo.tick());
    dom.thermalImg.addEventListener('load', () => state.fps.thermal.tick());

    // ── Overlay drawing ─────────────────────────────────────────────
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
                const [x, y, ww, hh] = t.eo_box;
                ctx.strokeStyle = t.cls === 'vehicle' ? '#f6c560' : '#6ce085';
                ctx.strokeRect(x * sx, y * sy, ww * sx, hh * sy);
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
            ctx.lineWidth = 2; ctx.font = '11px ui-monospace, monospace';
            for (const t of fused) {
                if (!t.thermal_box) continue;
                const [x, y, ww, hh] = t.thermal_box;
                ctx.strokeStyle = t.cls === 'vehicle' ? '#f6c560' : '#6ce085';
                ctx.strokeRect(x * sx, y * sy, ww * sx, hh * sy);
            }
        });
    }

    // ── Radar polar ─────────────────────────────────────────────────
    function buildRadarPolar() {
        const svg = dom.radarPolar;
        if (svg.dataset.built) return;
        svg.dataset.built = '1';
        const NS = 'http://www.w3.org/2000/svg';
        // semicircles at 25 50 75 100 m
        for (const r of [25, 50, 75, 100]) {
            const c = document.createElementNS(NS, 'path');
            c.setAttribute('d', `M ${-r} 0 A ${r} ${r} 0 0 1 ${r} 0`);
            c.setAttribute('class', 'ring');
            svg.appendChild(c);
            const lbl = document.createElementNS(NS, 'text');
            lbl.setAttribute('x', '0'); lbl.setAttribute('y', String(-r + 1));
            lbl.setAttribute('text-anchor', 'middle');
            lbl.setAttribute('class', 'axis-label');
            lbl.textContent = `${r} m`;
            svg.appendChild(lbl);
        }
        // axes at -60, -30, 0, 30, 60°
        for (const a of [-60, -30, 0, 30, 60]) {
            const rad = a * Math.PI / 180;
            const x = 100 * Math.sin(rad), y = -100 * Math.cos(rad);
            const ln = document.createElementNS(NS, 'line');
            ln.setAttribute('x1', '0'); ln.setAttribute('y1', '0');
            ln.setAttribute('x2', String(x)); ln.setAttribute('y2', String(y));
            ln.setAttribute('class', 'axis');
            svg.appendChild(ln);
        }
    }
    function drawRadarPolar() {
        buildRadarPolar();
        const svg = dom.radarPolar;
        // remove old targets
        svg.querySelectorAll('.target,.target-label').forEach(el => el.remove());
        const tgts = (state.stats && state.stats.radar_stats &&
                      state.stats.radar_stats.targets) || [];
        const NS = 'http://www.w3.org/2000/svg';
        for (const t of tgts.slice(0, 20)) {
            const az = (t.az_deg ?? 0) * Math.PI / 180;
            const rng = t.range_m ?? Math.hypot(t.x ?? 0, t.y ?? 0);
            const x = rng * Math.sin(az), y = -rng * Math.cos(az);
            const c = document.createElementNS(NS, 'circle');
            c.setAttribute('cx', String(x)); c.setAttribute('cy', String(y));
            c.setAttribute('r', '1.2');
            c.setAttribute('class', 'target');
            svg.appendChild(c);
        }
    }

    // ── Tracks list ─────────────────────────────────────────────────
    function renderTracks() {
        const fused = (state.latest && state.latest.fused) || [];
        const cells = dom.tracksList.children;
        for (let i = 0; i < 5; i++) {
            const t = fused[i];
            if (!t) {
                cells[i].textContent = '--';
                cells[i].className = 'target-cell';
            } else {
                cells[i].className = `target-cell cls-${t.cls || 'unknown'}`;
                cells[i].innerHTML =
                    `<b>#${t.tid}</b><br>${t.cls || '?'}<br>` +
                    `az ${(t.az_deg||0).toFixed(1)}°<br>el ${(t.el_deg||0).toFixed(1)}°`;
            }
        }
    }

    // ── Header pills ────────────────────────────────────────────────
    function updatePills() {
        const s = state.stats || {};
        dom.pillEo.classList.toggle('online',
            !!(s.eo_stats && (s.eo_stats.frame_id || 0) > 0));
        dom.pillThermal.classList.toggle('online',
            !!(s.thermal_stats && (s.thermal_stats.fps_5s || 0) > 0.1));
        dom.pillRadar.classList.toggle('online',
            !!(s.radar_stats && (s.radar_stats.fps_5s || 0) > 0.1));
        dom.pillGimbal.classList.toggle('online', state.wsConnected);
        // REC pill stays muted in v2 alpha (recording wired in next iter)
    }

    function updateMetrics() {
        dom.eoHz.textContent = state.fps.eo.value().toFixed(1) + ' Hz';
        dom.thermalHz.textContent = state.fps.thermal.value().toFixed(1) + ' Hz';
        const s = state.stats || {};
        const r = s.radar_stats && s.radar_stats.fps_5s;
        dom.radarHz.textContent = (r != null ? r.toFixed(1) : '--') + ' Hz';
    }

    function updateDevTab() {
        const fmt = o => o ? JSON.stringify(o, null, 2) : 'no data';
        const s = state.stats || {};
        dom.devEo.textContent = fmt(s.eo_stats);
        dom.devThermal.textContent = fmt(s.thermal_stats);
        dom.devRadar.textContent = fmt(s.radar_stats);
        dom.devInference.textContent = fmt(s.inference_stats);
        dom.devFusion.textContent = fmt(s.fusion_stats);
        dom.devWs.textContent = fmt(state.latest);
    }

    // ── RAF tick ────────────────────────────────────────────────────
    function rafTick() {
        try {
            maybeFetch('eo', dom.eoImg, 'eo_id');
            maybeFetch('thermal', dom.thermalImg, 'thermal_id');
            drawEoOverlay();
            drawThermalOverlay();
            drawRadarPolar();
            renderTracks();
            updatePills();
            updateMetrics();
            updateDevTab();
        } catch (e) { console.error('[raf]', e); }
        requestAnimationFrame(rafTick);
    }

    async function pollStatus() {
        try {
            const r = await fetch('/api/status', { cache: 'no-store' });
            if (r.ok) state.stats = await r.json();
        } catch (_) {}
        setTimeout(pollStatus, 1500);
    }

    // ── Boot ─────────────────────────────────────────────────────────
    connectWS();
    pollStatus();
    requestAnimationFrame(rafTick);
})();
