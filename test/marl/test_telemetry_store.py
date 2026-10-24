"""MetricsStore + monitor: the TensorBoard replacement.

Round trip, buffered writes, concurrent reader, downsampling that keeps
peaks, run discovery, the learner integration (every learner writes to the
store through TrainingLogger), and the monitor's JSON API.
"""
import http.client
import json
import os
import sys
import threading
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from opencda_marl.telemetry import MetricsStore, discover_runs
from opencda_marl.telemetry.store import _readonly_uri, downsample


def test_round_trip_and_buffering(tmp_path):
    db = tmp_path / "run" / "metrics.sqlite"
    with MetricsStore(db, meta={"algorithm": "masac", "mode": "train"}, flush_every=1000,
                      flush_interval_s=1000) as store:
        for step in range(50):
            store.log_scalar("Loss/critic", 10.0 / (step + 1), step)
        store.log_episode(0, {"reward": -12.5, "length": 400, "nested": {"a": np.float32(1.5)}})
        store.log_event("checkpoint", {"path": "x.pth", "episode": np.int64(3)})
        # Buffered: a fresh reader sees nothing yet...
        reader = MetricsStore(db, read_only=True)
        assert reader.counts()["scalars"] == 0
        store.flush()
        assert reader.counts()["scalars"] == 50
        reader.close()
    reader = MetricsStore(db, read_only=True)
    try:
        assert reader.meta()["algorithm"] == "masac" and reader.meta()["mode"] == "train"
        assert "created" in reader.meta()
        tags = reader.tags()
        assert tags[0]["tag"] == "Loss/critic" and tags[0]["count"] == 50 and tags[0]["last_step"] == 49
        rows = reader.scalars("Loss/critic")
        assert [r[0] for r in rows] == list(range(50))
        assert reader.scalars("Loss/critic", since_step=47) == rows[48:]
        eps = reader.episodes()
        assert eps[0]["episode"] == 0 and eps[0]["reward"] == -12.5 and eps[0]["nested"]["a"] == 1.5
        ev = reader.events()
        assert ev[0]["kind"] == "checkpoint" and ev[0]["payload"]["episode"] == 3
        with pytest.raises(RuntimeError):
            reader.log_scalar("x", 1.0, 0)
    finally:
        reader.close()


def test_downsample_keeps_peaks():
    rows = [(i, 0.0, 0.0) for i in range(10_000)]
    rows[4321] = (4321, 0.0, 99.0)           # one spike
    rows[7777] = (7777, 0.0, -50.0)
    out = downsample(rows, 200)
    assert len(out) <= 202
    values = [r[2] for r in out]
    assert 99.0 in values and -50.0 in values, "spikes must survive downsampling"
    assert out[-1] == rows[-1]
    assert [r[0] for r in out] == sorted(r[0] for r in out)
    assert downsample(rows[:10], 200) == rows[:10]


def test_concurrent_writer_and_reader(tmp_path):
    db = tmp_path / "metrics.sqlite"
    store = MetricsStore(db, flush_every=10, flush_interval_s=0.05)
    stop = threading.Event()
    errors = []

    def writer():
        step = 0
        try:
            while not stop.is_set():
                store.log_scalar("a", float(step), step)
                step += 1
                time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    try:
        time.sleep(0.3)
        reader = MetricsStore(db, read_only=True)
        seen = []
        for _ in range(5):
            seen.append(reader.counts()["scalars"])
            time.sleep(0.05)
        reader.close()
    finally:
        stop.set()
        t.join(5)
        store.close()
    assert not errors
    assert seen[-1] > seen[0] > 0, seen


def test_discover_runs_orders_newest_first(tmp_path):
    for name, created in (("masac/old", 100.0), ("pcmappo/new", 200.0)):
        with MetricsStore(tmp_path / name, meta={"algorithm": name.split("/")[0], "created": created}) as s:
            s.log_scalar("x", 1.0, 0)
            s.flush()
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "metrics.sqlite").write_bytes(b"not a database")
    runs = discover_runs(tmp_path)
    assert [r.run_id for r in runs] == ["pcmappo/new", "masac/old"]
    assert runs[0].meta["algorithm"] == "pcmappo" and runs[0].counts["scalars"] == 1


def test_learner_writes_to_the_store_through_training_logger(tmp_path):
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    cfg = {"warmup_steps": 0, "memory_size": 256, "batch_size": 16,
           "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16], "lstm_hidden_size": 8,
           "tensorboard": {"enabled": False},
           "metrics_store": {"enabled": True, "log_dir": str(tmp_path), "run_name": "cell_a",
                             "scenario": "momarl/intersection/x", "log_frequency": 1}}
    algo = SACAlgorithm(cfg, 5, 1)
    assert algo.metrics_store is not None
    rng = np.random.default_rng(0)
    for _ in range(40):
        obs = {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(3)}
        algo.store_transition(obs, "v0", 30.0, 1.0, obs, False)
    for _ in range(3):
        algo.update()
    algo.log_episode_metrics(episode_reward=-5.0, episode_length=40, success_rate=0.5,
                             collision_rate=0.1, traffic_metrics={"avg_speed": 12.0})
    algo.close()
    assert algo.metrics_store is None
    runs = discover_runs(tmp_path)
    assert runs and runs[0].run_id == "sac/cell_a"
    reader = MetricsStore(runs[0].path, read_only=True)
    try:
        tags = {t["tag"] for t in reader.tags()}
        assert any(t.startswith("Loss/") or t.startswith("SAC/") for t in tags), tags
        assert any(t.startswith("Episode/") for t in tags), tags
        eps = reader.episodes()
        assert eps and eps[0]["reward"] == -5.0 and eps[0]["traffic/avg_speed"] == 12.0
        assert reader.meta()["scenario"] == "momarl/intersection/x" and reader.meta()["mode"] == "train"
    finally:
        reader.close()


def test_store_disabled_by_config(tmp_path):
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    cfg = {"warmup_steps": 0, "memory_size": 64, "batch_size": 8,
           "actor_hidden_dims": [8, 8], "critic_hidden_dims": [8, 8], "lstm_hidden_size": 4,
           "tensorboard": {"enabled": False}, "metrics_store": {"enabled": False}}
    algo = SACAlgorithm(cfg, 5, 1)
    assert algo.metrics_store is None
    algo.log_scalar("Loss/x", 1.0, 0)          # no sink: must be a no-op, not a crash
    algo.close()


def _strict_json(body):
    """``json.loads`` that rejects the NaN/Infinity tokens a browser rejects."""
    def _reject(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(body, parse_constant=_reject)


def test_non_finite_scalars_are_skipped_and_counted_without_losing_the_batch(tmp_path):
    """One NaN scalar used to fail the single flush transaction (``value`` is
    REAL NOT NULL; SQLite binds NaN as NULL), and the buffers were cleared
    anyway -- the finite scalars, the episode summary and the events of that
    batch were lost. Non-finite scalars are now skipped and counted, and
    non-finite floats inside JSON payloads are stored as null."""
    db = tmp_path / "run"
    with MetricsStore(db, flush_every=1000, flush_interval_s=1000) as store:
        store.log_scalar("Loss/a", 1.0, 0)
        store.log_scalar("Loss/a", float("nan"), 1)
        store.log_scalar("Loss/a", float("inf"), 2)
        store.log_scalar("Loss/a", np.float32("-inf"), 3)
        store.log_scalar("Loss/a", 4.0, 4)
        store.log_episode(0, {"reward": float("nan"), "length": 10, "q": np.float64("inf"),
                              "series": np.array([1.0, np.nan])})
        store.log_event("diverged", {"loss": float("-inf")})
        store.set_meta(note=float("nan"))
    reader = MetricsStore(db, read_only=True)
    try:
        assert [(s, v) for s, _w, v in reader.scalars("Loss/a")] == [(0, 1.0), (4, 4.0)]
        assert reader.meta()["skipped_nonfinite"] == 3
        assert reader.meta()["note"] is None
        (ep,) = reader.episodes()
        assert ep["length"] == 10 and ep["reward"] is None and ep["q"] is None
        assert ep["series"] == [1.0, None]
        (ev,) = reader.events()
        assert ev["kind"] == "diverged" and ev["payload"] == {"loss": None}
        raw = reader._conn.execute("SELECT summary FROM episodes").fetchone()[0]
        _strict_json(raw)                          # stored as strict JSON text
    finally:
        reader.close()
    with MetricsStore(db) as store:                # a reopened store keeps counting
        store.log_scalar("Loss/a", float("nan"), 5)
    reader = MetricsStore(db, read_only=True)
    try:
        assert reader.meta()["skipped_nonfinite"] == 4
    finally:
        reader.close()


def test_a_row_the_schema_rejects_costs_only_itself(tmp_path):
    """Backstop for rows that reach flush() invalid by any other route: the
    batch is retried per table and then row by row, so only the rejected row
    is dropped."""
    store = MetricsStore(tmp_path / "run", flush_every=1000, flush_interval_s=1000)
    try:
        store.log_scalar("x", 1.0, 0)
        store._scalar_buf.append(("x", 1, time.time(), float("nan")))   # bypasses log_scalar
        store.log_scalar("x", 2.0, 2)
        store.log_episode(7, {"reward": 1.5})
        store.log_event("checkpoint", {"path": "a.pth"})
        store.flush()
        assert [r[0] for r in store.scalars("x")] == [0, 2]
        assert [e["episode"] for e in store.episodes()] == [7]
        assert [e["kind"] for e in store.events()] == ["checkpoint"]
    finally:
        store.close()


def test_counts_reads_the_newest_row_without_scanning(tmp_path):
    with MetricsStore(tmp_path / "run") as store:
        for step in range(5):
            store.log_scalar("a", float(step), step, wall=1000.0 + step)
        store.log_scalar("b", 9.0, 0, wall=2000.0)
        store.log_episode(0, {"reward": 1.0})
        store.log_episode(1, {"reward": 2.0})
        store.flush()
        assert store.counts() == {"scalars": 6, "episodes": 2, "last_wall": 2000.0}
    with MetricsStore(tmp_path / "empty") as store:
        assert store.counts() == {"scalars": 0, "episodes": 0, "last_wall": 0.0}


def test_each_writer_process_records_a_session_id(tmp_path):
    from opencda_marl.telemetry.store import SESSION_ID

    with MetricsStore(tmp_path / "run") as store:
        assert store.meta()["session"] == SESSION_ID
    reader = MetricsStore(tmp_path / "run", read_only=True)
    try:
        assert reader.meta()["session"] == SESSION_ID
    finally:
        reader.close()


@pytest.mark.parametrize("name", ["cell#2", "pct%41run", "space here"])
def test_read_only_open_survives_uri_characters_in_the_path(tmp_path, name):
    """The read-only URI used to be ``file:<path>?mode=ro`` unescaped: '#'
    began a fragment that also swallowed ``?mode=ro``, '%41' was decoded,
    and the truncated path was opened read-write-create -- the run vanished
    from discovery and a stray empty file appeared beside it."""
    with MetricsStore(tmp_path / "sac" / name) as store:
        store.log_scalar("x", 1.0, 0)
    before = sorted(p.name for p in (tmp_path / "sac").iterdir())
    runs = discover_runs(tmp_path)
    assert [r.run_id for r in runs] == [f"sac/{name}"]
    assert runs[0].counts["scalars"] == 1
    assert sorted(p.name for p in (tmp_path / "sac").iterdir()) == before, "no stray file"


@pytest.mark.parametrize("path, uri", [
    (PureWindowsPath(r"\\host\share\a#b\metrics.sqlite"), "file:////host/share/a%23b/metrics.sqlite?mode=ro"),
    (PureWindowsPath(r"Y:\runs\metrics.sqlite"), "file:///Y:/runs/metrics.sqlite?mode=ro"),
    (PurePosixPath("/srv/a b/pct%41/metrics.sqlite"), "file:///srv/a%20b/pct%2541/metrics.sqlite?mode=ro"),
])
def test_read_only_uri_keeps_drives_and_gives_unc_paths_an_empty_authority(path, uri):
    """A drive path keeps its letter (it is made absolute, not resolved, so a
    mapped network drive is not turned into its UNC path), and a UNC path is
    written with an empty URI authority: ``file://host/share`` names 'host'
    as the authority, which SQLite refuses to open."""
    assert _readonly_uri(path) == uri


@pytest.mark.skipif(sys.platform != "win32", reason="UNC paths are a Windows path form")
def test_a_store_under_a_unc_root_opens_read_only(tmp_path):
    """A mapped network drive resolves to a UNC path, and the monitor resolves
    every run directory it opens; a read-only open must therefore work on a
    ``\\\\host\\share`` path. Reached here through the local admin share."""
    from opencda_marl.telemetry.monitor import MonitorApp

    with MetricsStore(tmp_path / "sac" / "cell#2") as store:
        store.log_scalar("x", 1.0, 0)
    drive, rest = os.path.splitdrive(str(tmp_path.resolve()))
    if len(drive) != 2 or drive[1] != ":":
        pytest.skip("tmp_path is not on a drive letter")
    unc = Path(f"\\\\localhost\\{drive[0]}$" + rest)
    try:
        reachable = unc.is_dir()
    except OSError:
        reachable = False
    if not reachable:
        pytest.skip("the local administrative share is not reachable")
    runs = discover_runs(unc)
    assert [r.run_id for r in runs] == ["sac/cell#2"]
    assert runs[0].counts["scalars"] == 1
    assert MonitorApp(unc).handle("/api/runs/sac/cell%232/meta", {})["counts"]["scalars"] == 1


def test_discover_runs_cache_reopens_only_changed_stores(tmp_path, monkeypatch):
    import opencda_marl.telemetry.store as store_mod

    for name in ("a/one", "a/two"):
        with MetricsStore(tmp_path / name) as s:
            s.log_scalar("x", 1.0, 0)
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "metrics.sqlite").write_bytes(b"not a database")

    opened = []
    real = store_mod.MetricsStore

    def counting(path, **kw):
        opened.append(Path(path).parent.name)
        return real(path, **kw)

    monkeypatch.setattr(store_mod, "MetricsStore", counting)
    warnings = []
    sink = store_mod.logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        cache = {}
        first = discover_runs(tmp_path, cache=cache)
        assert sorted(r.run_id for r in first) == ["a/one", "a/two"]
        assert sorted(opened) == ["junk", "one", "two"]
        opened.clear()
        assert [r.run_id for r in discover_runs(tmp_path, cache=cache)] == [r.run_id for r in first]
        assert opened == [], "unchanged stores are served from the cache"
        assert sum("junk" in w for w in warnings) == 1, "an unreadable store is reported once"

        with real(tmp_path / "a" / "two") as s:   # a write changes the WAL
            s.log_scalar("x", 2.0, 1)
        runs = discover_runs(tmp_path, cache=cache)
        assert opened == ["two"]
        assert {r.run_id: r.counts["scalars"] for r in runs} == {"a/one": 1, "a/two": 2}
    finally:
        store_mod.logger.remove(sink)


@pytest.fixture
def monitor(tmp_path):
    """The monitor served over one keep-alive connection, as a browser uses it.

    Measured on Windows loopback against this server (HTTP/1.1, 100 requests
    each): a fresh connection per request that announces ``Connection:
    close`` -- what ``urllib.request`` sends -- is reset roughly a third of
    the time (36/100; urllib 27/100), while fresh connections without that
    header and requests over one kept-alive connection both fail 0/100.
    This agrees with the note on ``monitor._probe``. Browsers keep the
    connection alive, so the monitor is exercised the way it is actually
    used, and the suite does not depend on socket teardown timing.
    """
    from opencda_marl.telemetry.monitor import serve

    with MetricsStore(tmp_path / "masac" / "cell_a", meta={"algorithm": "masac"}) as s:
        for step in range(300):
            s.log_scalar("Loss/critic", float(step % 7), step)
        s.log_episode(0, {"reward": 1.0, "length": 10})
        s.flush()
    server = serve(tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    try:
        yield conn
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(5)


def _get(conn, path):
    conn.request("GET", path)
    response = conn.getresponse()
    return response.status, response.headers.get("Content-Type", ""), response.read()


def test_monitor_api_and_page(monitor):
    status, ctype, body = _get(monitor, "/")
    assert status == 200 and "text/html" in ctype and b"OpenCDA-MARL Monitor" in body
    status, _, body = _get(monitor, "/api/runs")
    runs = json.loads(body)
    assert status == 200 and runs[0]["run_id"] == "masac/cell_a" and runs[0]["counts"]["scalars"] == 300
    _, _, body = _get(monitor, "/api/runs/masac/cell_a/tags")
    assert json.loads(body)[0]["tag"] == "Loss/critic"
    _, _, body = _get(monitor, "/api/runs/masac/cell_a/scalars?tag=Loss/critic&max_points=50")
    d = json.loads(body)
    assert d["tag"] == "Loss/critic" and 2 <= len(d["points"]) <= 52 and d["last_step"] == 299
    _, _, body = _get(monitor, "/api/runs/masac/cell_a/scalars?tag=Loss/critic&since=295")
    assert [p[0] for p in json.loads(body)["points"]] == [296, 297, 298, 299]
    _, _, body = _get(monitor, "/api/runs/masac/cell_a/episodes")
    assert json.loads(body)[0]["reward"] == 1.0
    _, _, body = _get(monitor, "/api/runs/masac/cell_a/meta")
    assert json.loads(body)["meta"]["algorithm"] == "masac"

    status, _, _ = _get(monitor, "/api/runs/masac/cell_a/scalars")       # tag required
    assert status == 400
    status, _, _ = _get(monitor, "/api/runs/../../etc/passwd/tags")      # no escaping the root
    assert status == 404
    status, _, _ = _get(monitor, "/api/runs/nope/tags")
    assert status == 404


def test_episode_category_is_not_thinned_by_the_step_log_frequency(tmp_path):
    """`log_frequency` is a per-*step* stride. Applying it to the episode
    category (stepped by episode number) logged Episode/* once per 100
    episodes with the default config -- in every TensorBoard mode, and it
    silently starved the store too."""
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    cfg = {"warmup_steps": 0, "memory_size": 64, "batch_size": 8,
           "actor_hidden_dims": [8, 8], "critic_hidden_dims": [8, 8], "lstm_hidden_size": 4,
           "tensorboard": {"enabled": False, "mode": "episode_first", "log_frequency": 100},
           "metrics_store": {"enabled": True, "log_dir": str(tmp_path), "run_name": "r", "log_frequency": 1}}
    algo = SACAlgorithm(cfg, 5, 1)
    assert algo.tb_category_frequencies["episode"] == 1
    assert algo.tb_category_frequencies["losses"] == 500      # per-step categories keep the stride
    for ep in range(3):
        algo.log_episode_metrics(episode_reward=-float(ep), episode_length=10)
        algo.reset_episode()
    algo.close()
    reader = MetricsStore(tmp_path / "sac" / "r", read_only=True)
    try:
        assert [e["episode"] for e in reader.episodes()] == [0, 1, 2]
        reward = reader.scalars("Episode/reward")
        assert [r[0] for r in reward] == [0, 1, 2]
    finally:
        reader.close()


def test_store_stride_for_per_step_categories(tmp_path):
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    cfg = {"warmup_steps": 0, "memory_size": 64, "batch_size": 8,
           "actor_hidden_dims": [8, 8], "critic_hidden_dims": [8, 8], "lstm_hidden_size": 4,
           "tensorboard": {"enabled": False},
           "metrics_store": {"enabled": True, "log_dir": str(tmp_path), "run_name": "r", "log_frequency": 5}}
    algo = SACAlgorithm(cfg, 5, 1)
    for step in range(20):
        algo.training_step = step
        algo.log_scalar("Loss/x", float(step), category="losses")
        algo.log_scalar("Episode/y", float(step), step=step, category="episode")
    algo.close()
    reader = MetricsStore(tmp_path / "sac" / "r", read_only=True)
    try:
        assert [r[0] for r in reader.scalars("Loss/x")] == [0, 5, 10, 15]
        assert len(reader.scalars("Episode/y")) == 20
    finally:
        reader.close()
