// The schematic traffic view panel. Loads after app.js (window.Monitor).
(() => {
  const { $, api, state, ago } = window.Monitor;
  // ---- traffic view ------------------------------------------------------
  const POLL_MS = 400;
  // A frame older than this (or three publish intervals, when the run
  // publishes slower) is a frozen picture, not the simulation: it stays on
  // screen, dimmed, and the HUD says so.
  const STALE_S = 3;
  const DIM_ALPHA = 0.35;
  const fresh = (runId) => ({ runId, geometry: null, geoRetryAt: 0, retryAt: 0, backoff: 0, lastWall: -Infinity });
  // How current the frame is: "ended" when the run's metrics store was
  // closed after this frame was written (meta.closed_at, from the run
  // list), "stale" when no newer frame came for too long, else "live".
  function frameStatus(id, st, age) {
    const run = state.runs.find(r => r.run_id === id);
    const closedAt = run && run.meta ? run.meta.closed_at : null;
    if (Number.isFinite(closedAt) && Number.isFinite(st.wall) && closedAt >= st.wall) return "ended";
    const interval = Number.isFinite(st.interval) && st.interval > 0 ? st.interval : 0;
    return age > Math.max(STALE_S, 3 * interval) ? "stale" : "live";
  }
  let tv = fresh(null);
  function tvRun() { return state.selected[state.selected.length - 1] || null; }
  function clearTV() { const cv = $("#tv"), g = cv.getContext("2d"); g.clearRect(0, 0, cv.width, cv.height); }
  async function tvTick() {
    const id = tvRun();
    $("#tvfor").textContent = id ? `(${id})` : "";
    if (!id) { if (tv.runId !== null) { tv = fresh(null); clearTV(); } $("#tvhud").textContent = "no run selected"; return; }
    if (!$("#tvon").checked || document.visibilityState !== "visible") return;
    if (tv.runId !== id) { tv = fresh(id); clearTV(); }
    if (state.gone.has(id)) { $("#tvhud").textContent = "this run is no longer under the monitor root"; return; }
    const view = tv;
    if (Date.now() < view.retryAt) return;
    let st;
    try {
      st = await api(`/api/runs/${encodeURIComponent(id)}/traffic_state`);
    } catch (e) {
      if (tv !== view) return;          // selection changed while waiting
      // Never latch: the first probes of a fresh run 404 by design (the run
      // writes its first frame only after its metrics store exists), and a
      // monitor restart or a dropped tunnel recovers. Retry, backing off
      // from 1 s to 5 s, and say which of the two it is.
      view.backoff = Math.min(5000, view.backoff ? view.backoff * 2 : 1000);
      view.retryAt = Date.now() + view.backoff;
      $("#tvhud").textContent = e.status === 404 ? "waiting for the first frame of this run"
        : e.status ? `traffic view error (${e.status}): ${e.message} · retrying`
        : "monitor unreachable · retrying";
      if (view.lastWall === -Infinity) clearTV();
      return;
    }
    if (tv !== view) return;
    view.backoff = 0; view.retryAt = 0;
    // Drawn in the writer's order: a state older than the one on screen is
    // dropped. Ordered by wall time, not (episode, step), which restart from
    // zero when a run is relaunched into the same directory.
    if (!(st.wall >= view.lastWall)) return;
    view.lastWall = st.wall;
    // The geometry is fetched on its own: until the run writes it (or when it
    // fails) the vehicles are drawn without lanes and it is asked for again.
    if (!view.geometry && Date.now() >= view.geoRetryAt) {
      try { view.geometry = await api(`/api/runs/${encodeURIComponent(id)}/traffic_geometry`); }
      catch (e) { view.geoRetryAt = Date.now() + 2000; }
      if (tv !== view) return;
    }
    const n = (st.vehicles || []).length;
    const age = Math.max(0, Date.now() / 1000 - st.wall);
    const status = frameStatus(id, st, age);
    drawTV(view.geometry, st, status !== "live");
    const when = status === "ended" ? `run ended · last frame ${ago(st.wall)}`
      : status === "stale" ? `stale · last frame ${ago(st.wall)}` : "live";
    $("#tvhud").textContent = `episode ${st.episode} · step ${st.step} · ${n} vehicle(s) · ${when}`
      + (view.geometry ? "" : " · lanes not published yet");
  }
  function drawTV(geo, st, dim) {
    const cv = $("#tv"), dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#15171b"; g.fillRect(0, 0, W, H);
    const lanes = (geo && geo.lanes) || [];
    const vehicles = (st.vehicles || []).filter(v => Number.isFinite(v.x) && Number.isFinite(v.y));
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    const feed = (x, y) => { xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); };
    for (const lane of lanes) for (const [x, y] of lane) feed(x, y);
    for (const v of vehicles) feed(v.x, v.y);
    if (!Number.isFinite(xmin)) { g.fillStyle = "#9aa0a6"; g.fillText("no geometry yet", 10, 20); return; }
    const pad = 14, spanX = Math.max(1, xmax - xmin), spanY = Math.max(1, ymax - ymin);
    const k = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanY);
    const flip = geo ? geo.y_up !== false : true;   // world y-up -> canvas y-down
    const sx = (x) => pad + (x - xmin) * k + (W - 2 * pad - spanX * k) / 2;
    const sy = (y) => flip ? H - pad - (y - ymin) * k - (H - 2 * pad - spanY * k) / 2
                           : pad + (y - ymin) * k + (H - 2 * pad - spanY * k) / 2;
    g.strokeStyle = "#3a3f4b"; g.lineWidth = Math.max(1, 3.4 * k); g.lineCap = "round"; g.lineJoin = "round";
    for (const lane of lanes) {
      g.beginPath();
      lane.forEach(([x, y], i) => i ? g.lineTo(sx(x), sy(y)) : g.moveTo(sx(x), sy(y)));
      g.stroke();
    }
    g.globalAlpha = dim ? DIM_ALPHA : 1;
    for (const v of vehicles) {
      const px = sx(v.x), py = sy(v.y);
      const yaw0 = Number.isFinite(v.yaw) ? v.yaw : 0;
      const yaw = flip ? -yaw0 : yaw0;             // canvas y-down mirrors angles
      const L = Math.max(6, 4.6 * k), Wv = Math.max(3, 2.0 * k);
      g.save(); g.translate(px, py); g.rotate(yaw);
      g.fillStyle = v.role === "agent" ? "#4ea1ff" : "#9aa0a6";
      g.beginPath(); g.moveTo(L / 2, 0); g.lineTo(-L / 2, -Wv / 2); g.lineTo(-L / 2, Wv / 2); g.closePath(); g.fill();
      g.restore();
    }
    g.globalAlpha = 1;
  }
  // Serialised polling: the next tick is armed only after this one settled,
  // so a slow response (an ssh tunnel) cannot overlap a newer one.
  async function loop() {
    try { await tvTick(); } catch (e) { /* a drawing error must not stop the loop */ }
    setTimeout(loop, POLL_MS);
  }
  loop();

})();
