"""The monitor page's refresh loop, run in Node.js against a minimal DOM.

* Tags are listed again on a tick only when a selected run wrote scalars
  since the last listing, and at most every ``TAGS_EVERY_MS``: a listing
  groups the whole scalars table on the server, so listing on every tick
  kept the training node busy for every run a page watches. A tag that
  appears later in a live run still shows up, and the default pick extends
  to it; a full refresh (selection, preset, reload) always lists.
* A refresh that throws outside its per-request handlers (a drawing error)
  does not end the polling loop, and a ``/api/runs`` answer that is not a
  run list keeps the last list instead of breaking every later refresh.

Skipped when ``node`` is not on PATH.
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

const env = { now: 1000000, scalars: 0, tags: [], tagCalls: 0, runsBody: null, drawFails: false };

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
    if (env.drawFails) throw new Error("canvas lost");
    return new Proxy({}, { get: (t, p) => (p in t ? t[p] : () => {}), set: (t, p, v) => { t[p] = v; return true; } });
  }
  getBoundingClientRect() { return { left: 0, top: 0 }; }
  focus() {}
}
function textNode(s) { const n = new El("#text"); n.own = s; return n; }

const ids = {};
const byId = (sel) => ids[sel] || (ids[sel] = new El(sel));
byId("#refresh").value = "2000";
byId("#maxpts").value = "1500";
byId("#smooth").value = "0";
const document = {
  hidden: false, activeElement: null,
  querySelector: byId,
  createElement: (tag) => new El(tag),
  addEventListener() {},
};

function respond(url) {
  const u = new URL(url, "http://monitor");
  const p = u.pathname;
  if (p === "/api/runs") {
    if (env.runsBody !== null) return env.runsBody;
    return [{ run_id: "a/r", meta: { algorithm: "x", mode: "train", session: "s1" },
              counts: { scalars: env.scalars, episodes: 0, last_wall: 0 } }];
  }
  if (p.endsWith("/tags")) { env.tagCalls++; return env.tags; }
  if (p.endsWith("/scalars")) return { tag: u.searchParams.get("tag"), points: [[1, 0.5]], last_step: 1 };
  if (p.endsWith("/episodes")) return [];
  if (p.endsWith("/meta")) return { meta: {}, counts: {} };
  throw new Error("unexpected request " + url);
}

const timers = new Map();
let seq = 0;
const ctx = {
  document, console,
  window: { addEventListener() {}, devicePixelRatio: 1 },
  fetch: async (url) => { const body = respond(url); return { ok: true, status: 200, statusText: "OK", json: async () => body }; },
  setTimeout: (fn) => { timers.set(++seq, fn); return seq; },
  clearTimeout: (id) => { timers.delete(id); },
  Date: class extends Date { static now() { return env.now; } },
};

const settle = async () => { for (let i = 0; i < 5; i++) await new Promise(r => setImmediate(r)); };
async function tick(ms) {
  env.now += ms;
  assert.equal(timers.size, 1, "exactly one tick is armed");
  const [[id, fn]] = timers;
  timers.delete(id);
  fn();
  await settle();
}

(async () => {
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "app.js" });
  await settle();
  const state = ctx.window.Monitor.state;
  assert.deepEqual([...state.selected], ["a/r"], "the newest run is selected on load");
  assert.equal(env.tagCalls, 1, "a new selection lists the tags");

  await tick(2000);
  assert.equal(env.tagCalls, 1, "a run that wrote nothing is not listed again");

  env.scalars = 5;
  env.tags = [{ tag: "Loss/actor", count: 5, last_step: 4, last_value: 0.1 }];
  await tick(2000);
  await tick(2000);
  assert.equal(env.tagCalls, 1, "a live run is not listed on every tick");
  await tick(6000);
  assert.equal(env.tagCalls, 2, "a run that wrote scalars is listed once the period has passed");
  assert.ok(state.tags.has("Loss/actor"), "the default pick extends to a tag that appeared later");
  assert.match(byId("#tags").textContent, /Loss\/actor/);

  await tick(30000);
  assert.equal(env.tagCalls, 2, "a run whose counts did not change is not listed again");

  byId("#reload").onclick();
  await settle();
  assert.equal(env.tagCalls, 3, "a full refresh always lists");

  env.drawFails = true;
  await tick(2000);
  assert.match(byId("#status").textContent, /^refresh failed: canvas lost/);
  assert.equal(timers.size, 1, "a failed refresh re-arms the loop");
  env.drawFails = false;
  await tick(2000);
  assert.match(byId("#status").textContent, /^1 run\(s\)/, "the loop recovers");

  env.runsBody = { error: "not a run list" };
  await tick(2000);
  assert.equal(byId("#status").textContent, "runs: unexpected response");
  assert.equal(state.runs.length, 1, "the last run list is kept");
  env.runsBody = null;
  await tick(2000);
  assert.match(byId("#status").textContent, /^1 run\(s\)/);
  assert.equal(timers.size, 1);
  console.log("ok");
})().catch((e) => { console.error((e && e.stack) || e); process.exit(1); });
"""


@pytest.mark.skipif(NODE is None, reason="needs Node.js on PATH to run the page script")
def test_page_refresh_lists_tags_sparingly_and_survives_a_failed_refresh(tmp_path):
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    result = subprocess.run([NODE, str(harness), str(APP_JS)], capture_output=True, text=True,
                            encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "ok"
