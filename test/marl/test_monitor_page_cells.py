"""How the monitor page prints numbers in the episode table.

* A column whose every finite value in the rows shown is a whole number
  (episode numbers, step counts, success / collision / timeout counts)
  prints integers: ``0`` and ``2``, not ``0.00`` and ``2.000``.
* Any other column keeps the page's precision on every row, including a
  row whose value happens to be whole: JSON does not tell ``3`` from
  ``3.0``, so a reward of exactly 3 reads ``3.000`` beside ``2.537``.
* A missing or non-finite value prints a dash in either kind of column.
* The charts keep their own formatting: a readout of a whole value still
  shows the chart precision.

The page script runs in Node.js against a minimal DOM; skipped when
``node`` is not on PATH.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

import opencda_marl.telemetry.monitor as monitor_mod

NODE = shutil.which("node")
APP_JS = Path(monitor_mod.__file__).resolve().parent / "static" / "app.js"

_HARNESS = r"""
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

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
  getContext() { return new Proxy({}, { get: (t, p) => (p in t ? t[p] : () => {}), set: (t, p, v) => { t[p] = v; return true; } }); }
  getBoundingClientRect() { return { left: 0, top: 0 }; }
  focus() {}
}
function textNode(s) { const n = new El("#text"); n.own = s; return n; }

const ids = {};
const byId = (sel) => ids[sel] || (ids[sel] = new El(sel));
byId("#refresh").value = "0";          // no polling: one load, then inspect
byId("#maxpts").value = "1500";
byId("#smooth").value = "0";
const document = {
  hidden: false, activeElement: null,
  querySelector: byId,
  createElement: (tag) => new El(tag),
  addEventListener() {},
};

// Oldest first, as the server answers; the table shows the newest on top.
// The stub hands the objects over as they are, so non-finite numbers reach
// the page here too (over HTTP the server sends them as null).
const EPISODES = [
  { episode: 0, wall: 1, reward: 2.5371, length: 1200, success_rate: 0.5, collision_rate: 0.125,
    training_step: 0, successes: 2, collisions: null, timeouts: 0, near_miss_count: 12 },
  { episode: 1, wall: 2, reward: 3, length: 900, success_rate: 1, collision_rate: 0.25,
    training_step: 12000, successes: 3, collisions: 1, timeouts: NaN, near_miss_count: Infinity },
];
function respond(url) {
  const p = new URL(url, "http://monitor").pathname;
  if (p === "/api/runs") return [{ run_id: "a/r", meta: { algorithm: "x", mode: "train" },
                                   counts: { scalars: 2, episodes: 2, last_wall: 0 } }];
  if (p.endsWith("/tags")) return [{ tag: "Episode/reward", count: 2, last_step: 2, last_value: 3 }];
  if (p.endsWith("/scalars")) return { tag: "Episode/reward", points: [[1, 2.5371], [2, 3]], last_step: 2 };
  if (p.endsWith("/episodes")) return EPISODES;
  if (p.endsWith("/meta")) return { meta: {}, counts: {} };
  throw new Error("unexpected request " + url);
}

const ctx = {
  document, console,
  window: { addEventListener() {}, devicePixelRatio: 1 },
  fetch: async (url) => { const body = respond(url); return { ok: true, status: 200, statusText: "OK", json: async () => body }; },
  setTimeout: () => 0, clearTimeout: () => {},
  Date,
};

(async () => {
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "app.js" });
  for (let i = 0; i < 10; i++) await new Promise(r => setImmediate(r));

  const [head, ...rows] = byId("#episodes").children;
  const cols = head.children.map(c => c.textContent);
  const cells = rows.map(r => Object.fromEntries(r.children.map((c, i) => [cols[i], c.textContent])));
  assert.equal(cells.length, 2);
  const [newest, oldest] = cells;

  // whole-number columns print integers
  assert.equal(newest.episode, "1");
  assert.equal(oldest.episode, "0");
  assert.equal(newest.successes, "3");
  assert.equal(oldest.successes, "2");
  assert.equal(newest.length, "900");
  assert.equal(oldest.length, "1200");
  assert.equal(newest.training_step, "12000");
  assert.equal(oldest.training_step, "0");
  assert.equal(oldest.timeouts, "0");
  assert.equal(oldest.near_miss_count, "12");

  // other columns keep their precision, on the whole-valued row too
  assert.equal(oldest.reward, "2.537");
  assert.equal(newest.reward, "3.000");
  assert.equal(oldest.success_rate, "0.500");
  assert.equal(newest.success_rate, "1.000");
  assert.equal(newest.collision_rate, "0.250");
  assert.equal(oldest.collision_rate, "0.125");

  // missing and non-finite values print the dash
  assert.equal(oldest.collisions, "—", "null");
  assert.equal(newest.collisions, "1");
  assert.equal(newest.timeouts, "—", "NaN");
  assert.equal(newest.near_miss_count, "—", "Infinity");

  // the chart readout keeps the chart precision
  assert.equal(byId("#charts").textContent, "Episode/reward3.000");
  console.log("ok");
})().catch((e) => { console.error((e && e.stack) || e); process.exit(1); });
"""


@pytest.mark.skipif(NODE is None, reason="needs Node.js on PATH to run the page script")
def test_episode_table_prints_whole_columns_as_integers_and_keeps_float_precision(tmp_path):
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    result = subprocess.run([NODE, str(harness), str(APP_JS)], capture_output=True, text=True,
                            encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
