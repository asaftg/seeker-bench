/*
 * seeker_v2 GUI — RAF-driven render loop
 * =======================================================================
 *
 * Phase 2.5 of the rewrite. Goal: decouple WebSocket message arrival
 * from canvas painting so a momentary WS burst doesn't pile up paints
 * inside the requestAnimationFrame queue. v1's frontend repainted on
 * every onmessage which caused per-tab CPU spikes whenever the backend
 * batch-emitted, and worse, dropped the GPU into low-power state when
 * the browser fell behind.
 *
 * Architecture:
 *
 *   onmessage (WS) ──► state.latest = parsed_payload  (constant time)
 *   img.onload      ──► state.eoBitmap / thermalBitmap = new ImageBitmap
 *
 *                      state          (single source of truth)
 *                        ▲
 *                        │
 *   requestAnimationFrame  ──► reads state, repaints DOM/canvas
 *
 * The image fetch path uses one in-flight Image() per stream. When the
 * WS tells us there's a new snapshot id, we kick off a new fetch; the
 * onload swap is atomic. If the fetch is slower than the WS, we just
 * skip ids — the latest always wins.
 */

(() => {
    'use strict';

    // ── State ─────────────────────────────────────────────────────────
    const state = {
        ws: null,
        wsConnected: false,
        latest: null,        // last parsed WS payload
        // snapshot ids we've requested (so we don't refetch the same id)
        eoFetchedId: -1,
        thermalFetchedId: -1,
        // FPS rolling counters (per stream)
        fps: {
            eo: new RollingFps(),
            thermal: new RollingFps(),
            radar: new RollingFps(),
            inf: new RollingFps(),
        },
        // Stats from /api/status (refreshed every 2s)
        stats: {},
    };

    function RollingFps(windowSec = 2.0) {
        this.windowSec = windowSec;
        this.t = [];
    }
    RollingFps.prototype.tick = function () {
        const now = performance.now() / 1000;
        this.t.push(now);
        const cutoff = now - this.windowSec;
        while (this.t.length && this.t[0] < cutoff) this.t.shift();
    };
    RollingFps.prototype.value = function () {
        return this.t.length / this.windowSec;
    };

    // ── DOM refs ──────────────────────────────────────────────────────
    const dom = {
        eoImg: document.getElementById('eo-img'),
        thermalImg: document.getElementById('thermal-img'),
        eoCanvas: document.getElementById('eo-overlay'),
        thermalCanvas: document.getElementById('thermal-overlay'),
        tracksList: document.getElementById('tracks-list'),
        eoFps: document.getElementById('eo-fps'),
        thermalFps: document.getElementById('thermal-fps'),
        radarFps: document.getElementById('radar-fps'),
        infFps: document.getElementById('inf-fps'),
        wsState: document.getElementById('ws-state'),
    };

    // ── WebSocket ─────────────────────────────────────────────────────
    function connectWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws/sensors`;
        console.log('[ws] connecting', url);
        const ws = new WebSocket(url);
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
            // Reconnect with backoff.
            setTimeout(connectWS, 1000);
        };
        ws.onerror = (e) => console.warn('[ws] error', e);

        ws.onmessage = (ev) => {
            // Hot path: parse payload, stash, return. NO painting.
            try {
                let payload;
                if (typeof ev.data === 'string') {
                    payload = JSON.parse(ev.data);
                } else {
                    payload = JSON.parse(new TextDecoder().decode(ev.data));
                }
                state.latest = payload;
            } catch (e) {
                console.warn('[ws] parse failed', e);
            }
        };

        state.ws = ws;
    }

    // ── Image streaming ────────────────────────────────────────────────
    // Two-buffer pattern: img.src points at /api/snapshot/<sensor>.jpg?
    // ID bust. We let the browser network stack handle the actual GET;
    // RAF only checks whether to start a new request.
    function maybeFetchEo() {
        if (!state.latest) return;
        const snaps = state.latest.snapshots;
        if (!snaps) return;
        const id = snaps.eo_id ?? -1;
        if (id < 0 || id === state.eoFetchedId) return;
        state.eoFetchedId = id;
        // Cache-bust on the id, not on a timestamp — browsers will
        // dedupe identical URLs and skip refetching.
        dom.eoImg.src = `/api/snapshot/eo.jpg?id=${id}`;
    }
    function maybeFetchThermal() {
        if (!state.latest) return;
        const snaps = state.latest.snapshots;
        if (!snaps) return;
        const id = snaps.thermal_id ?? -1;
        if (id < 0 || id === state.thermalFetchedId) return;
        state.thermalFetchedId = id;
        dom.thermalImg.src = `/api/snapshot/thermal.jpg?id=${id}`;
    }

    // Track image load ticks for FPS (img.onload fires once per fetch)
    dom.eoImg.addEventListener('load', () => state.fps.eo.tick());
    dom.thermalImg.addEventListener('load', () => state.fps.thermal.tick());

    // ── Overlay drawing ──────────────────────────────────────────────
    function drawOverlay(canvas, w, h, drawFn) {
        // Resize to backing-store ratio for crispness on HiDPI
        const dpr = window.devicePixelRatio || 1;
        const cw = canvas.clientWidth, ch = canvas.clientHeight;
        if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
            canvas.width = cw * dpr;
            canvas.height = ch * dpr;
        }
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cw, ch);
        // Scale source-pixel coords (w x h) into the canvas display.
        const sx = cw / w, sy = ch / h;
        drawFn(ctx, sx, sy);
    }

    function drawEoOverlay() {
        const m = state.latest && state.latest.eo_meta;
        if (!m) {
            const ctx = dom.eoCanvas.getContext('2d');
            ctx.clearRect(0, 0, dom.eoCanvas.width, dom.eoCanvas.height);
            return;
        }
        drawOverlay(dom.eoCanvas, m.w, m.h, (ctx, sx, sy) => {
            const fused = state.latest.fused || [];
            ctx.lineWidth = 2;
            ctx.font = '12px ui-monospace, monospace';
            for (const t of fused) {
                if (!t.eo_box) continue;
                const [x, y, ww, hh] = t.eo_box;
                ctx.strokeStyle = t.cls === 'vehicle' ? '#f6c560' : '#6ce085';
                ctx.strokeRect(x * sx, y * sy, ww * sx, hh * sy);
                ctx.fillStyle = ctx.strokeStyle;
                ctx.fillText(`#${t.tid} ${t.cls || ''}`,
                             x * sx + 2, y * sy - 4);
            }
        });
    }

    function drawThermalOverlay() {
        const m = state.latest && state.latest.thermal_meta;
        if (!m) {
            const ctx = dom.thermalCanvas.getContext('2d');
            ctx.clearRect(0, 0, dom.thermalCanvas.width, dom.thermalCanvas.height);
            return;
        }
        drawOverlay(dom.thermalCanvas, m.w, m.h, (ctx, sx, sy) => {
            // Heat dets — yellow boxes
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = '#ffcd56';
            for (const d of (m.heat_dets || [])) {
                ctx.strokeRect(d.x * sx, d.y * sy, d.w * sx, d.h * sy);
            }
            // Fused track boxes (thermal projection)
            const fused = state.latest.fused || [];
            ctx.lineWidth = 2;
            ctx.font = '11px ui-monospace, monospace';
            for (const t of fused) {
                if (!t.thermal_box) continue;
                const [x, y, ww, hh] = t.thermal_box;
                ctx.strokeStyle = t.cls === 'vehicle' ? '#f6c560' : '#6ce085';
                ctx.strokeRect(x * sx, y * sy, ww * sx, hh * sy);
                ctx.fillStyle = ctx.strokeStyle;
                ctx.fillText(`#${t.tid}`, x * sx + 2, y * sy - 3);
            }
        });
    }

    function renderTracks() {
        const fused = (state.latest && state.latest.fused) || [];
        // Update DOM only if content changed; avoids layout thrash.
        const sig = JSON.stringify(fused.map(t =>
            [t.tid, t.cls, Math.round(t.az_deg ?? 0), Math.round(t.el_deg ?? 0)]
        ));
        if (renderTracks._lastSig === sig) return;
        renderTracks._lastSig = sig;

        const frag = document.createDocumentFragment();
        for (const t of fused) {
            const div = document.createElement('div');
            div.className = `track cls-${t.cls || 'unknown'}`;
            div.innerHTML = `
                <span class="tid">#${t.tid}</span>
                <span class="cls">${t.cls || '?'}</span>
                <span class="age">${(t.age_ms || 0)|0} ms</span>
            `;
            frag.appendChild(div);
        }
        dom.tracksList.replaceChildren(frag);
    }

    function updateMetrics() {
        dom.eoFps.textContent = state.fps.eo.value().toFixed(1);
        dom.thermalFps.textContent = state.fps.thermal.value().toFixed(1);
        const s = state.stats || {};
        const rfps = s.radar_stats && s.radar_stats.fps_5s;
        const ifps = s.inference_stats &&
                     ((s.inference_stats.eo_inferences || 0) +
                      (s.inference_stats.thermal_inferences || 0));
        dom.radarFps.textContent = rfps != null ? rfps.toFixed(1) : '--';
        dom.infFps.textContent = ifps != null ? ifps.toFixed(0) : '--';
    }

    // ── RAF loop ──────────────────────────────────────────────────────
    // The single point where we touch the DOM. Browser caps this at
    // monitor refresh rate so we never repaint faster than visible.
    function rafTick() {
        try {
            maybeFetchEo();
            maybeFetchThermal();
            drawEoOverlay();
            drawThermalOverlay();
            renderTracks();
            updateMetrics();
        } catch (e) {
            console.error('[raf] tick failed', e);
        }
        requestAnimationFrame(rafTick);
    }

    // ── Status poller (slow path) ────────────────────────────────────
    async function pollStatus() {
        try {
            const r = await fetch('/api/status', { cache: 'no-store' });
            if (r.ok) state.stats = await r.json();
        } catch (_) { /* ignore */ }
        setTimeout(pollStatus, 2000);
    }

    // ── Boot ──────────────────────────────────────────────────────────
    connectWS();
    pollStatus();
    requestAnimationFrame(rafTick);
})();
