"""The monitor must not show a frozen picture as the live simulation.

* The traffic view keeps the last frame on screen but dims the vehicles and
  says so in the HUD once no newer frame came for ``max(3 s, 3 *
  interval)`` ("stale"), or as soon as the run's metrics store was closed
  after the frame was written ("run ended"). The publisher writes its
  ``interval`` into every frame; a store records ``closed_at`` when its
  writer closes it and drops it when a writer reopens it.
* The episode table lists outcome, throughput, TTC, near-miss and reward
  component columns before the alphabetical rest; sorted alphabetically
  alone, a dozen ``intent_*`` diagnostics pushed them off the table.

The page scripts run in Node.js against a minimal DOM; those tests are
skipped when ``node`` is not on PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import opencda_marl.telemetry.monitor as monitor_mod
from opencda_marl.telemetry import MetricsStore, discover_runs
from opencda_marl.telemetry.traffic_view import STATE_NAME, TrafficViewPublisher

NODE = shutil.which("node")
STATIC = Path(monitor_mod.__file__).resolve().parent / "static"


# --------------------------------------------------------------------- #
# Writer side
# --------------------------------------------------------------------- #
def test_every_frame_carries_the_publish_interval(tmp_path):
    TrafficViewPublisher(tmp_path, interval=0.5).publish_state([], episode=0, step=1)
    state = json.loads((tmp_path / STATE_NAME).read_text(encoding="utf-8"))
    assert state["interval"] == pytest.approx(0.5)


def test_a_closed_store_says_so_until_a_writer_reopens_it(tmp_path):
    run = tmp_path / "masac" / "r"
    before = time.time()
    store = MetricsStore(run, meta={"algorithm": "masac"})
    assert "closed_at" not in store.meta()
    store.close()
    (info,) = discover_runs(tmp_path)
    assert info.meta["closed_at"] >= before

    reader = MetricsStore(run, read_only=True)             # a reader never marks it
    reader.close()

    reopened = MetricsStore(run)                           # a resumed run is live again
    try:
        assert "closed_at" not in reopened.meta()
    finally:
        reopened.close()


# --------------------------------------------------------------------- #
# Page side (Node.js)
# --------------------------------------------------------------------- #
_DOM = r"""
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const env = { now: 1000000000, fills: [] };
class El {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.parentNode = null;
    this.style = {}; this.attrs = {}; this.own = ""; this.value = ""; this.checked = false;
    this.className = ""; this.clientWidth = 320; this.clientHeight = 160;
  }
  get textContent() { return this.own + this.children.map(c => c.textContent).join(""); }
  set textContent(v) { for (const c of this.children) c.parentNode = null; this.children = []; this.own = String(v); }
  get firstChild() { return this.children[0] || null; }
  append(...kids) { for (const k of kids) this.appendChild(typeof k === "string" ? textNode(k) : k); }
  appendChild(k) { k.remove(); k.parentNode = this; this.children.push(k); return k; }
  insertBefore(k, ref) {
    k.remove(); k.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(k); else this.children.splice(i, 0, k);
    return k;
  }
  remove() {
    if (!this.parentNode) return;
    const c = this.parentNode.children; c.splice(c.indexOf(this), 1); this.parentNode = null;
  }
  replaceChildren(...kids) { this.textContent = ""; this.append(...kids); }
  contains(n) { for (let x = n; x; x = x.parentNode) if (x === this) return true; return false; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  querySelector(sel) {
    const cls = sel.replace(/^\./, "");
    for (const c of this.children) { if (c.className === cls) return c; const hit = c.querySelector(sel); if (hit) return hit; }
    return null;
  }
  getContext() {
    // Records the alpha every filled shape is drawn with.
    const g = { globalAlpha: 1, fill() { env.fills.push(this.globalAlpha); } };
    return new Proxy(g, { get: (t, p) => (p in t ? t[p] : () => {}), set: (t, p, v) => { t[p] = v; return true; } });
  }
  getBoundingClientRect() { return { left: 0, top: 0 }; }
  focus() {}
}
function textNode(s) { const n = new El("#text"); n.own = s; return n; }
const ids = {};
const byId = (sel) => ids[sel] || (ids[sel] = new El(sel));
const document = {
  hidden: false, visibilityState: "visible", activeElement: null,
  querySelector: byId, createElement: (tag) => new El(tag), addEventListener() {},
};
const timers = [];
const settle = async () => { for (let i = 0; i < 8; i++) await new Promise(r => setImmediate(r)); };
const baseCtx = (fetchImpl) => ({
  document, console,
  window: { addEventListener() {}, devicePixelRatio: 1 },
  fetch: fetchImpl,
  setTimeout: (fn) => { timers.push(fn); return timers.length; },
  clearTimeout: () => {},
  Date: class extends Date { static now() { return env.now; } },
});
"""

_TRAFFIC = _DOM + r"""
// Each case is its own run, so a fresh view state draws it (a frame older
// than the one on screen is never drawn).
const now = () => env.now / 1000;
const cases = {
  fresh:     { st: { wall: now() - 0.5, interval: 0.2 }, meta: {} },
  quiet:     { st: { wall: now() - 10, interval: 0.2 }, meta: {} },
  slow:      { st: { wall: now() - 10, interval: 5 }, meta: {} },
  old_file:  { st: { wall: now() - 10 }, meta: {} },
  ended:     { st: { wall: now() - 1, interval: 0.2 }, meta: { closed_at: now() - 0.5 } },
  relaunch:  { st: { wall: now() - 0.5, interval: 0.2 }, meta: { closed_at: now() - 60 } },
};
const state = { runs: Object.entries(cases).map(([id, c]) => ({ run_id: id, meta: c.meta, counts: {} })),
                selected: [], gone: new Set() };
async function api(p) {
  const m = p.match(/^\/api\/runs\/([^/]+)\/(traffic_state|traffic_geometry)$/);
  if (!m) throw new Error("unexpected request " + p);
  const c = cases[decodeURIComponent(m[1])];
  if (m[2] === "traffic_geometry") return { lanes: [[[0, 0], [50, 0]]], y_up: false };
  return Object.assign({ episode: 2, step: 40,
    vehicles: [{ id: "a", x: 10, y: 1, yaw: 0, role: "agent" }, { id: "b", x: 30, y: -1, yaw: 3.1, role: "traffic" }] }, c.st);
}
const ago = (w) => `${Math.round(now() - w)} s ago`;
byId("#tvon").checked = true;

(async () => {
  const ctx = baseCtx(async () => { throw new Error("no fetch here"); });
  ctx.window.Monitor = { $: byId, api, state, ago };
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "traffic.js" });
  await settle();
  const seen = {};
  for (const id of Object.keys(cases)) {
    state.selected = [id];
    env.fills = [];
    timers.shift()();                        // the next poll
    await settle();
    seen[id] = { hud: byId("#tvhud").textContent, alphas: [...new Set(env.fills)] };
  }
  const live = (id) => { assert.match(seen[id].hud, /· live$/, id); assert.deepEqual(seen[id].alphas, [1], id); };
  const dim = (id, word) => {
    assert.match(seen[id].hud, new RegExp(`· ${word} · last frame \\d+ s ago$`), id);
    assert.deepEqual(seen[id].alphas, [0.35], id);
  };
  live("fresh");
  dim("quiet", "stale");
  live("slow");                              // 10 s is not stale at a 5 s publish interval
  dim("old_file", "stale");                  // a frame without interval: the 3 s floor
  dim("ended", "run ended");
  live("relaunch");                          // a reopened run's new frames are newer than its old close
  assert.match(seen.fresh.hud, /^episode 2 · step 40 · 2 vehicle\(s\)/);
  console.log("ok");
})().catch((e) => { console.error((e && e.stack) || e); process.exit(1); });
"""

_EPISODES = _DOM + r"""
const row = { episode: 0, wall: 1, reward: 1, length: 10, success_rate: 50, collision_rate: 10,
              training_step: 7, buffer_size: 100, near_miss_count: 3, ttc_violation_rate: 0.2,
              timeout_rate: 40, successes: 5, collisions: 1, timeouts: 4,
              "traffic/throughput": 720, "traffic/avg_speed": 30,
              reward_component_progress: 3, reward_component_collision: -1, note: "text" };
for (let i = 0; i < 14; i++) row[`intent_diag_${String(i).padStart(2, "0")}`] = 0;
function respond(url) {
  const p = new URL(url, "http://monitor").pathname;
  if (p === "/api/runs") return [{ run_id: "masac/r", meta: { algorithm: "masac", mode: "train" }, counts: { scalars: 0, episodes: 1 } }];
  if (p.endsWith("/tags")) return [];
  if (p.endsWith("/episodes")) return [row];
  if (p.endsWith("/meta")) return { meta: {}, counts: {} };
  throw new Error("unexpected request " + url);
}
byId("#refresh").value = "0";
byId("#maxpts").value = "1500";
byId("#smooth").value = "0";
(async () => {
  const ctx = baseCtx(async (url) => { const body = respond(url); return { ok: true, status: 200, json: async () => body }; });
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "app.js" });
  await settle();
  // Array.from: a main-realm copy of the page realm's array, for deepEqual.
  const cols = Array.from(ctx.window.Monitor.episodeColumns([row]));
  assert.deepEqual(cols.slice(0, 16), [
    "episode", "reward", "length", "success_rate", "collision_rate", "training_step",
    "timeout_rate", "successes", "collisions", "timeouts", "traffic/throughput",
    "ttc_violation_rate", "near_miss_count",
    "reward_component_collision", "reward_component_progress",
    "buffer_size"]);
  assert.ok(!cols.includes("note") && !cols.includes("wall"), "numbers only, no wall");
  assert.equal(cols.length, 6 + 24, "the rest is capped");
  assert.ok(cols.includes("intent_diag_13") && !cols.includes("traffic/avg_speed"), "the alphabetical tail is cut");
  // Only what a run logs is shown: an older run without the new columns.
  assert.deepEqual(Array.from(ctx.window.Monitor.episodeColumns([{ episode: 0, reward: 1, intent_x: 0, throughput: 5 }])),
    ["episode", "reward", "length", "success_rate", "collision_rate", "training_step", "throughput", "intent_x"]);
  // ...and the table the page rendered uses that order.
  const head = byId("#episodes").children[0].children.map(c => c.textContent);
  assert.deepEqual(head, cols);
  console.log("ok");
})().catch((e) => { console.error((e && e.stack) || e); process.exit(1); });
"""


def _node(tmp_path, harness: str, script: str):
    path = tmp_path / "harness.js"
    path.write_text(harness, encoding="utf-8")
    result = subprocess.run([NODE, str(path), str(STATIC / script)], capture_output=True, text=True,
                            encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(NODE is None, reason="needs Node.js on PATH to run the page script")
def test_a_stale_or_ended_frame_is_dimmed_and_labelled(tmp_path):
    _node(tmp_path, _TRAFFIC, "traffic.js")


@pytest.mark.skipif(NODE is None, reason="needs Node.js on PATH to run the page script")
def test_the_episode_table_shows_outcome_and_safety_columns_first(tmp_path):
    _node(tmp_path, _EPISODES, "app.js")
