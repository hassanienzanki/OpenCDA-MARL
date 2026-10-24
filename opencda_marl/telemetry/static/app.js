// OpenCDA-MARL monitor — runs, tags, charts, episodes. No build step.
(() => {
  const $ = (s) => document.querySelector(s);
  const COLORS = ["#4ea1ff","#ff7f50","#7ee787","#f778ba","#e3b341","#a371f7","#79c0ff","#ffa657"];
  const state = {
    runs: [], selected: [], tags: new Set(), series: {}, timer: null,
    gone: new Set(),     // selected runs that /api/runs no longer lists
    sessions: {},        // runId -> writer session last seen (meta.session)
    pickRule: null,      // tag predicate that also applies to tags appearing later
    tagNames: "",        // the tag list the #tags panel currently shows
    tagsAt: 0,           // Date.now() of the last tag listing
    tagsKey: null,       // selected runs' scalar counts at that listing
    metaFor: null,       // run whose metadata #meta shows
  };
  // series[runId][tag] = {points:[[step,value]], last:number}
  // Text from the server (run ids, metadata, tag names, episode fields) is
  // only ever placed with textContent / text nodes, never parsed as HTML.

  // Rejects with an Error whose .status is the HTTP status, or 0 when the
  // monitor could not be reached at all.
  async function api(p) {
    let r;
    try { r = await fetch(p, {cache:"no-store"}); }
    catch (e) { const err = new Error("monitor unreachable"); err.status = 0; throw err; }
    if (r.ok) return r.json();
    let msg = r.statusText || `HTTP ${r.status}`;
    try { const body = await r.json(); if (body && body.error) msg = body.error; } catch (_) { /* not JSON */ }
    const err = new Error(msg); err.status = r.status; throw err;
  }
  const el = (tag, props, ...kids) => { const n = Object.assign(document.createElement(tag), props || {}); n.append(...kids); return n; };
  const color = (runId) => COLORS[state.selected.indexOf(runId) % COLORS.length];
  const fmt = (v) => (typeof v !== "number" || !Number.isFinite(v)) ? "—" : (Math.abs(v) >= 1000 ? v.toFixed(0) : (Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3)));
  const ago = (w) => { if (!w) return "—"; const s = Math.max(0, Date.now()/1000 - w); return s < 90 ? `${s.toFixed(0)} s ago` : s < 5400 ? `${(s/60).toFixed(0)} min ago` : `${(s/3600).toFixed(1)} h ago`; };
  const runUrl = (id, what) => `/api/runs/${encodeURIComponent(id)}/${what}`;

  // ---- runs -------------------------------------------------------------
  async function loadRuns() {
    let runs;
    try { runs = await api("/api/runs"); }
    catch (e) { $("#status").textContent = "runs: " + e.message; return; }
    // Kept only when it is a run list (another service may answer the port):
    // every refresh reads state.runs as one.
    if (!Array.isArray(runs)) { $("#status").textContent = "runs: unexpected response"; return; }
    state.runs = runs;
    const listed = new Set(state.runs.map(r => r.run_id));
    state.gone = new Set(state.selected.filter(id => !listed.has(id)));
    for (const r of state.runs) {
      const sess = r.meta && r.meta.session;
      if (!sess) continue;
      // A new writer process on a selected run (a relaunch into the same
      // store): the step cursor no longer means anything, so start over.
      if (state.selected.includes(r.run_id) && state.sessions[r.run_id] && state.sessions[r.run_id] !== sess) {
        state.series[r.run_id] = {};
        if (state.metaFor === r.run_id) state.metaFor = null;
      }
      state.sessions[r.run_id] = sess;
    }
    renderRuns();
    $("#status").textContent = `${state.runs.length} run(s) · ${new Date().toLocaleTimeString()}`;
  }
  const runRows = new Map();   // runId -> row element, updated in place
  function runRow(id) {
    let row = runRows.get(id);
    if (row) return row;
    const sw = el("div", {className: "sw"}); sw.style.border = "1px solid var(--line)";
    row = el("div", {className: "run", tabIndex: 0}, sw,
             el("div", {}, el("div", {className: "id", textContent: id}), el("div", {className: "sub"})));
    row.setAttribute("role", "button");
    row.onclick = () => toggleRun(id);
    row.onkeydown = (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggleRun(id); } };
    runRows.set(id, row);
    return row;
  }
  function renderRuns() {
    const f = $("#filter").value.toLowerCase();
    const rows = [];
    for (const r of state.runs) {
      if (f && !r.run_id.toLowerCase().includes(f)) continue;
      const m = r.meta || {}, c = r.counts || {};
      rows.push([r.run_id, `${m.algorithm || "?"} · ${m.mode || "?"} · ${c.episodes} ep · ${c.scalars} pts · ${ago(c.last_wall)}`]);
    }
    const listed = rows.length;
    // Still selected but gone from disk: shown greyed so it can be clicked away.
    for (const id of state.gone) rows.push([id, "no longer under the monitor root · click to remove"]);
    const keep = new Set(rows.map(r => r[0]));
    for (const [id, row] of runRows) if (!keep.has(id)) { row.remove(); runRows.delete(id); }
    const box = $("#runs"), focused = document.activeElement;
    rows.forEach(([id, sub], i) => {
      const row = runRow(id), sel = state.selected.includes(id);
      row.className = "run" + (sel ? " sel" : "");
      row.style.opacity = state.gone.has(id) ? "0.55" : "";
      row.setAttribute("aria-pressed", sel ? "true" : "false");
      row.firstChild.style.background = sel ? color(id) : "transparent";
      const s = row.querySelector(".sub"); if (s.textContent !== sub) s.textContent = sub;
      // Moved only when out of place, so focus and hover survive a refresh.
      if (box.children[i] !== row) box.insertBefore(row, box.children[i] || null);
    });
    // A focused row that had to move (its run became the most recent) keeps focus.
    if (focused && box.contains(focused) && document.activeElement !== focused) focused.focus({preventScroll: true});
    $("#runcount").textContent = `(${listed})`;
  }
  async function toggleRun(id) {
    const i = state.selected.indexOf(id);
    if (i >= 0) { state.selected.splice(i, 1); delete state.series[id]; state.gone.delete(id); }
    else { state.selected.push(id); state.series[id] = {}; }
    // Nothing charted and no rule chosen: the default pick applies to the
    // tags of the new selection, including tags that only appear later.
    if (state.tags.size === 0 && !state.pickRule) state.pickRule = DEFAULT_PICK;
    renderRuns();
    await refreshAll(true);
  }

  // ---- tags -------------------------------------------------------------
  const DEFAULT_PICK = (t) => t.startsWith("Episode/reward") || t.startsWith("Episode/success") || t.startsWith("Loss/");
  state.pickRule = DEFAULT_PICK;
  const tagBoxes = new Map();   // tag -> {check, count}
  async function loadTags() {
    const all = new Map();
    for (const id of state.selected) {
      if (state.gone.has(id)) continue;
      try { for (const t of await api(runUrl(id, "tags"))) all.set(t.tag, (all.get(t.tag) || 0) + t.count); }
      catch (e) { $("#status").textContent = `${id}: ${e.message}`; }
    }
    const names = [...all.keys()].sort();
    const key = names.join("\n");
    if (key !== state.tagNames) {
      // Rebuilt only when the set of names changes (counts change every
      // tick); the active pick rule extends to the names that are new.
      if (state.pickRule) for (const t of names) if (!tagBoxes.has(t) && state.pickRule(t)) state.tags.add(t);
      renderTagList(names);
      state.tagNames = key;
    }
    for (const [t, n] of all) { const b = tagBoxes.get(t); if (b) b.count.textContent = ` ${n}`; }
  }
  function renderTagList(names) {
    const box = $("#tags"); box.textContent = "";
    tagBoxes.clear();
    for (const t of names) {
      const check = el("input", {type: "checkbox", checked: state.tags.has(t)});
      check.onchange = () => {
        state.pickRule = null;          // a hand-made selection: stop auto-adding
        if (check.checked) state.tags.add(t);
        else { state.tags.delete(t); for (const id in state.series) delete state.series[id][t]; }
        renderCharts(); refreshAll(false);
      };
      const count = el("span", {className: "muted"});
      box.appendChild(el("label", {}, check, t, count));
      tagBoxes.set(t, {check, count});
    }
  }
  function renderTagChecks() { for (const [t, b] of tagBoxes) b.check.checked = state.tags.has(t); }
  function pick(pred) {
    state.pickRule = pred;              // also picks matching tags that appear later
    state.tags = new Set([...tagBoxes.keys()].filter(pred));
    renderTagChecks();
    state.series = Object.fromEntries(state.selected.map(id => [id, {}]));
    renderCharts(); refreshAll(true);
  }
  $("#pick-episode").onclick = () => pick(t => t.startsWith("Episode/"));
  $("#pick-loss").onclick = () => pick(t => t.startsWith("Loss/") || t.startsWith("Policy/"));
  $("#pick-none").onclick = () => pick(() => false);

  // ---- data -------------------------------------------------------------
  // One refresh at a time: a call made while one is running is folded into
  // a single follow-up run (full if any caller asked for full).
  let busy = false, queued = null;
  async function refreshAll(full) {
    if (busy) { queued = queued === true || !!full; return; }
    busy = true;
    try { await refreshOnce(!!full); }
    finally {
      busy = false;
      if (queued !== null) { const again = queued; queued = null; await refreshAll(again); }
    }
  }
  // Listing tags groups a run's whole scalars table on the server, so a tick
  // lists them again (for tags that appeared meanwhile) only when a selected
  // run wrote scalars since the last listing, and at most every
  // TAGS_EVERY_MS; a full refresh (selection, preset, reload) always does.
  const TAGS_EVERY_MS = 10000;
  const tagsKey = () => state.selected.map(id => {
    const r = state.runs.find(x => x.run_id === id);
    return r && r.counts ? r.counts.scalars : "";
  }).join();
  async function refreshOnce(full) {
    const key = tagsKey();
    if (full || (Date.now() - state.tagsAt >= TAGS_EVERY_MS && key !== state.tagsKey)) {
      await loadTags();
      state.tagsAt = Date.now(); state.tagsKey = key;
    }
    const maxPoints = +$("#maxpts").value || 1500;
    for (const id of state.selected) {
      if (state.gone.has(id)) continue;
      const runSeries = state.series[id] = state.series[id] || {};
      for (const tag of state.tags) {
        const cur = runSeries[tag];
        // Incremental unless asked for a full fetch or the live buffer outgrew
        // twice the budget; a full fetch is min/max-downsampled server-side,
        // so spikes survive where a client-side stride would drop them.
        const incremental = !full && cur && cur.last !== null && cur.points.length <= maxPoints * 2;
        const since = incremental ? `&since=${cur.last}` : "";
        try {
          const d = await api(`${runUrl(id, "scalars")}?tag=${encodeURIComponent(tag)}&max_points=${maxPoints}${since}`);
          if (state.series[id] !== runSeries) continue;     // deselected or reset meanwhile
          if (incremental) {
            const fresh = d.points.filter(p => p[0] > cur.last);   // never append a point twice
            if (fresh.length) { cur.points.push(...fresh); cur.last = fresh[fresh.length - 1][0]; }
          } else runSeries[tag] = { points: d.points, last: d.last_step };
        } catch (e) { $("#status").textContent = `${tag}: ${e.message}`; }
      }
    }
    renderCharts();
    await loadEpisodes();
    await loadRuns();
  }

  // ---- charts -----------------------------------------------------------
  const canvases = {};
  function renderCharts() {
    const box = $("#charts");
    const wanted = [...state.tags].sort();
    for (const k of Object.keys(canvases)) if (!wanted.includes(k)) { canvases[k].wrap.remove(); delete canvases[k]; }
    for (const tag of wanted) {
      if (!canvases[tag]) {
        const v = el("span", {className: "v"});
        const cv = el("canvas");
        cv.setAttribute("role", "img"); cv.setAttribute("aria-label", `chart: ${tag}`);
        const wrap = el("div", {className: "chart"}, el("div", {className: "t"}, el("span", {textContent: tag}), v), cv);
        box.appendChild(wrap);
        canvases[tag] = { wrap, cv, v };
        cv.onmousemove = (ev) => hover(tag, ev); cv.onmouseleave = () => draw(tag);
      }
      draw(tag);
    }
  }
  function seriesFor(tag) {
    const alpha = +$("#smooth").value || 0;
    return state.selected.map(id => {
      const s = state.series[id] && state.series[id][tag];
      let pts = s ? s.points : [];
      if (alpha > 0 && pts.length) {
        let m = null;                   // gaps (null values) are kept, not averaged in
        pts = pts.map(([x, y]) => { if (!Number.isFinite(y)) return [x, y]; m = m === null ? y : alpha * m + (1 - alpha) * y; return [x, m]; });
      }
      return { id, color: color(id), pts };
    });
  }
  function draw(tag, cursor) {
    const { cv, v } = canvases[tag]; const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#1e2127"; g.fillRect(0, 0, W, H);
    const series = seriesFor(tag).filter(s => s.pts.length);
    if (!series.length) { g.fillStyle = "#9aa0a6"; g.fillText("no data", 10, 20); v.textContent = ""; return; }
    const logy = $("#logy").checked;
    // Number.isFinite, not the global variant: the server sends non-finite
    // values as null, which the global one coerces to 0 and accepts.
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of series) for (const [x, y] of s.pts) { if (!Number.isFinite(y)) continue; if (logy && y <= 0) continue; xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
    if (!Number.isFinite(xmin)) { g.fillStyle = "#9aa0a6"; g.fillText("no finite data", 10, 20); return; }
    if (xmax === xmin) xmax = xmin + 1;
    const ty = (y) => logy ? Math.log10(y) : y;
    let lo = ty(ymin), hi = ty(ymax); if (hi === lo) { hi = lo + 1; lo = lo - 1; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const L = 52, R = 10, T = 8, B = 22;
    const sx = (x) => L + (x - xmin) / (xmax - xmin) * (W - L - R);
    const sy = (y) => T + (hi - ty(y)) / (hi - lo) * (H - T - B);
    // grid + axes
    g.strokeStyle = "#2c3039"; g.fillStyle = "#9aa0a6"; g.font = "10px system-ui"; g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const yy = T + i * (H - T - B) / 4; g.beginPath(); g.moveTo(L, yy); g.lineTo(W - R, yy); g.stroke();
      const val = hi - i * (hi - lo) / 4; g.textAlign = "right"; g.fillText(fmt(logy ? Math.pow(10, val) : val), L - 4, yy + 3);
    }
    for (let i = 0; i <= 4; i++) { const xx = L + i * (W - L - R) / 4; g.textAlign = "center"; g.fillText(fmt(xmin + i * (xmax - xmin) / 4), xx, H - 6); }
    // lines
    for (const s of series) {
      g.strokeStyle = s.color; g.lineWidth = 1.4; g.beginPath(); let first = true;
      for (const [x, y] of s.pts) { if (!Number.isFinite(y) || (logy && y <= 0)) continue; const px = sx(x), py = sy(y); first ? g.moveTo(px, py) : g.lineTo(px, py); first = false; }
      g.stroke();
    }
    // readout
    if (cursor !== undefined) {
      const xv = xmin + (cursor - L) / (W - L - R) * (xmax - xmin);
      g.strokeStyle = "#555"; g.beginPath(); g.moveTo(cursor, T); g.lineTo(cursor, H - B); g.stroke();
      v.textContent = series.map(s => { const p = nearest(s.pts, xv); return p ? `${s.id.split("/").pop().slice(0, 28)}: ${fmt(p[1])} @${p[0]}` : ""; }).filter(Boolean).join("   ");
    } else {
      v.textContent = series.map(s => `${fmt(s.pts[s.pts.length - 1][1])}`).join(" · ");
    }
  }
  function nearest(pts, x) { let best = null, bd = Infinity; for (const p of pts) { const d = Math.abs(p[0] - x); if (d < bd) { bd = d; best = p; } } return best; }
  function hover(tag, ev) { const r = canvases[tag].cv.getBoundingClientRect(); draw(tag, ev.clientX - r.left); }

  // ---- episodes + meta ---------------------------------------------------
  // Numbers are formatted per column, since JSON does not tell 3 from 3.0: a
  // column whose every finite value in the rows shown is whole (counts,
  // episode and step numbers) prints integers, and any other column keeps
  // fmt's precision on every row, so a reward of exactly 3 reads 3.000 beside
  // 2.537. The rule is decided again on every refresh, over the rows shown.
  const wholeColumn = (rows, c) => rows.every(e => !Number.isFinite(e[c]) || Number.isInteger(e[c]));
  const cellText = (v, whole) => typeof v === "number" ? (whole && Number.isInteger(v) ? String(v) : fmt(v)) : (v === null || v === undefined) ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
  const EPISODE_COLS = ["episode", "reward", "length", "success_rate", "collision_rate", "training_step"];
  // Outcome, efficiency and safety columns come first, in this order, then
  // the reward components; whatever else a run logs follows alphabetically
  // up to EXTRA_COLS in all. Sorted alphabetically alone, a dozen intent_*
  // diagnostics used to push throughput, TTC and near-miss off the table.
  const PRIORITY_COLS = ["timeout_rate", "successes", "collisions", "timeouts",
    "success", "collision", "active_agents", "traffic/throughput", "throughput",
    "ttc_violation_rate", "near_miss_count"];
  const PRIORITY_PREFIXES = ["reward_component_"];
  const EXTRA_COLS = 24;
  function episodeColumns(eps) {
    const extra = new Set();
    for (const e of eps) for (const k of Object.keys(e)) if (!EPISODE_COLS.includes(k) && k !== "wall" && typeof e[k] === "number") extra.add(k);
    const first = PRIORITY_COLS.filter(k => extra.has(k));
    const rest = [...extra].filter(k => !first.includes(k)).sort();
    for (const p of PRIORITY_PREFIXES) for (const k of rest.filter(k => k.startsWith(p))) first.push(k);
    const others = rest.filter(k => !first.includes(k));
    return EPISODE_COLS.concat(first, others).slice(0, EPISODE_COLS.length + Math.max(EXTRA_COLS, first.length));
  }
  async function loadEpisodes() {
    const id = state.selected[state.selected.length - 1];
    const tbl = $("#episodes"); $("#epfor").textContent = id ? `(${id})` : "";
    if (!id) { tbl.textContent = ""; $("#meta").textContent = "—"; state.metaFor = null; return; }
    if (state.gone.has(id)) return;
    try {
      const eps = await api(`${runUrl(id, "episodes")}?limit=200`);
      const all = episodeColumns(eps);
      const whole = all.map(c => wholeColumn(eps, c));
      const head = el("tr", {}, ...all.map(c => el("th", {textContent: c})));
      tbl.replaceChildren(head, ...eps.slice().reverse().map(e => el("tr", {}, ...all.map((c, i) => el("td", {textContent: cellText(e[c], whole[i])})))));
      if (state.metaFor !== id) {         // run metadata is static: once per selection
        const m = await api(runUrl(id, "meta"));
        $("#meta").textContent = JSON.stringify(m, null, 2);
        state.metaFor = id;
      }
    } catch (e) { $("#status").textContent = `${id}: ${e.message}`; }
  }

  // ---- wiring ------------------------------------------------------------
  // The next tick is armed only after the previous refresh settled, and a
  // hidden tab does not poll at all. A refresh that throws outside its
  // per-request handlers (a drawing error, say) must not end the loop.
  function schedule() {
    clearTimeout(state.timer); state.timer = null;
    const ms = +$("#refresh").value;
    if (ms > 0) state.timer = setTimeout(tick, ms);
  }
  async function tick() {
    try { if (!document.hidden) await refreshAll(false); }
    catch (e) { $("#status").textContent = "refresh failed: " + ((e && e.message) || e); }
    finally { schedule(); }
  }
  document.addEventListener("visibilitychange", () => { if (!document.hidden && +$("#refresh").value > 0) refreshAll(false); });
  $("#refresh").onchange = schedule;
  $("#smooth").onchange = () => renderCharts();
  $("#logy").onchange = () => renderCharts();
  $("#maxpts").onchange = () => refreshAll(true);
  $("#filter").oninput = renderRuns;
  $("#reload").onclick = () => refreshAll(true);
  window.addEventListener("resize", () => renderCharts());
  loadRuns().then(() => { if (state.runs.length) toggleRun(state.runs[0].run_id); });
  schedule();
  // shared namespace for the other static modules (traffic.js)
  window.Monitor = { $, api, state, ago, color, episodeColumns };
})();
