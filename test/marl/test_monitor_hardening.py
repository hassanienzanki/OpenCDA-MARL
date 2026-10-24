"""The monitor's API under hostile or degenerate input.

* Responses are strict JSON: a stored +inf scalar or a NaN inside a stored
  summary is served as ``null``, never as a bare ``Infinity``/``NaN`` token
  that the browser's JSON parser rejects (it used to empty the whole tag
  list or episode table of the run).
* A run id is validated lexically before anything touches the filesystem:
  on Windows ``%5C%5Chost%5Cshare`` decodes to a UNC path, and resolving it
  opens an SMB connection to that host before any containment check.
* Bound to loopback, the service answers only requests addressed to a
  loopback name, so a page cannot read it by rebinding its own DNS name.
* ``/api/runs`` is rescanned at most once per cache period, by one request.
* A run reuses its monitor even when one ``/api/root`` probe is lost, and a
  freshly started monitor counts as started only when it serves the run's
  root.
* The traffic view writer survives a reader holding the file on Windows.
* The page builds its DOM from server text with ``textContent`` only.
"""
import http.client
import json
import pathlib
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import opencda_marl.telemetry.monitor as monitor_mod
from opencda_marl.telemetry import MetricsStore
from opencda_marl.telemetry.monitor import MonitorApp, ensure_service, serve
from opencda_marl.telemetry.traffic_view import STATE_NAME, TrafficViewPublisher

HOST = "127.0.0.1"
STATIC = Path(monitor_mod.__file__).resolve().parent / "static"


def _strict_json(body):
    def _reject(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(body, parse_constant=_reject)


@pytest.fixture
def server_port(tmp_path):
    """A monitor over ``tmp_path`` with one run that holds non-finite data
    the way an older store (or a direct writer) left it."""
    run = tmp_path / "masac" / "cell_a"
    with MetricsStore(run, meta={"algorithm": "masac"}) as s:
        for step in range(5):
            s.log_scalar("Loss/critic", float(step), step)
        s.log_episode(0, {"reward": 1.0, "length": 10})
    conn = sqlite3.connect(str(run / "metrics.sqlite"))
    with conn:
        conn.execute("INSERT INTO scalars (tag, step, wall, value) VALUES (?, ?, ?, ?)",
                     ("Grad/norm", 4, time.time(), float("inf")))
        conn.execute("INSERT OR REPLACE INTO episodes (episode, wall, summary) VALUES (?, ?, ?)",
                     (1, time.time(), '{"reward": NaN, "length": 12, "extra": [Infinity, 1.0]}'))
        conn.execute("INSERT INTO events (wall, kind, payload) VALUES (?, ?, ?)",
                     (time.time(), "diverged", '{"loss": -Infinity}'))
    conn.close()
    server = serve(tmp_path, HOST, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def _get(port, path, headers=None):
    conn = http.client.HTTPConnection(HOST, port, timeout=10)
    try:
        conn.request("GET", path, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def test_non_finite_values_are_served_as_null(server_port):
    status, body = _get(server_port, "/api/runs/masac/cell_a/tags")
    assert status == 200
    tags = {t["tag"]: t for t in _strict_json(body)}
    assert tags["Grad/norm"]["last_value"] is None and tags["Loss/critic"]["last_value"] == 4.0

    status, body = _get(server_port, "/api/runs/masac/cell_a/scalars?tag=Grad/norm")
    assert status == 200 and _strict_json(body)["points"] == [[4, None]]

    status, body = _get(server_port, "/api/runs/masac/cell_a/episodes")
    assert status == 200
    eps = _strict_json(body)
    assert eps[1]["reward"] is None and eps[1]["length"] == 12 and eps[1]["extra"] == [None, 1.0]

    status, body = _get(server_port, "/api/runs/masac/cell_a/events")
    assert status == 200 and _strict_json(body)[0]["payload"] == {"loss": None}

    for path in ("/api/runs", "/api/runs/masac/cell_a/meta"):
        status, body = _get(server_port, path)
        assert status == 200
        _strict_json(body)


def test_encode_refuses_what_it_cannot_make_strict():
    assert monitor_mod._encode({"a": [float("nan"), {"b": float("-inf")}], "c": 1}) == \
        b'{"a": [null, {"b": null}], "c": 1}'

    class _Weird(float):                       # a float subclass still maps to null
        pass

    assert monitor_mod._encode([_Weird("inf")]) == b"[null]"


@pytest.mark.parametrize("run_id", [
    "%5C%5Cattacker%5Cshare",                  # \\attacker\share: a UNC path on Windows
    "%5C%5C%3F%5CUNC%5Chost%5Cx",              # \\?\UNC\host\x
    "C%3A%5CWindows",                          # C:\Windows
    "C%3A",                                    # a drive
    "%2Fetc%2Fpasswd",                         # absolute
    "masac%2F..%2F..%2Fx",                     # .. segments
    "masac%2F.%2Fcell_a",                      # . segment
    "masac%2F%2Fcell_a",                       # empty segment
    "",
])
def test_run_ids_are_rejected_before_the_filesystem_is_touched(tmp_path, monkeypatch, run_id):
    touched = []
    real_resolve = pathlib.Path.resolve

    def spy(self, *a, **kw):
        touched.append(str(self))
        return real_resolve(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "resolve", spy)
    app = MonitorApp(tmp_path)
    with pytest.raises(FileNotFoundError):
        app._run_dir(run_id)
    assert touched == []


def test_valid_run_ids_still_resolve_inside_the_root(tmp_path):
    (tmp_path / "masac" / "cell_a").mkdir(parents=True)
    app = MonitorApp(tmp_path)
    assert app._run_dir("masac/cell_a") == (tmp_path / "masac" / "cell_a").resolve()
    assert app._run_dir("masac%2Fcell_a") == (tmp_path / "masac" / "cell_a").resolve()


def test_a_symlink_out_of_the_root_is_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks needs a privilege this account lacks")
    with pytest.raises(FileNotFoundError):
        MonitorApp(root)._run_dir("link")


def test_hostile_run_ids_404_over_http(server_port):
    for path in ("/api/runs/%5C%5Cattacker%5Cx/tags", "/api/runs/C%3A%5CWindows/tags",
                 "/api/runs/%5C%5Cattacker%5Cx/traffic_state", "/api/runs/../../etc/passwd/tags"):
        status, _ = _get(server_port, path)
        assert status == 404, path
    status, _ = _get(server_port, "/C:/Windows/win.ini")
    assert status == 404
    status, body = _get(server_port, "/app.js")
    assert status == 200 and b"window.Monitor" in body


def test_loopback_monitor_answers_only_loopback_host_names(server_port):
    for host in (f"127.0.0.1:{server_port}", f"localhost:{server_port}", "localhost",
                 f"[::1]:{server_port}", f"monitor.localhost:{server_port}"):
        status, _ = _get(server_port, "/api/root", {"Host": host})
        assert status == 200, host
    for host in (f"attacker.example:{server_port}", "attacker.example", f"10.0.0.5:{server_port}"):
        status, _ = _get(server_port, "/api/root", {"Host": host})
        assert status == 403, host
    # The refusal names the switch that admits the name (a proxy's Host).
    status, body = _get(server_port, "/api/runs", {"Host": f"proxy.internal:{server_port}"})
    assert status == 403
    assert _strict_json(body)["error"] == ("host not allowed: proxy.internal; "
                                           "start the monitor with --allow-host proxy.internal")


def test_extra_host_names_can_be_allowed(tmp_path):
    server = serve(tmp_path, HOST, 0, allow_hosts=["proxy.internal"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        assert _get(port, "/api/root", {"Host": "proxy.internal:443"})[0] == 200
        assert _get(port, "/api/root", {"Host": "other.example"})[0] == 403
        root = _strict_json(_get(port, "/api/root")[1])
        assert Path(root["root"]) == tmp_path.resolve() and isinstance(root["pid"], int)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_run_discovery_is_cached_and_single_flight(tmp_path, monkeypatch):
    calls = []

    def slow_discover(root, **kw):
        calls.append(kw.get("cache") is not None)
        time.sleep(0.2)
        return []

    monkeypatch.setattr(monitor_mod, "discover_runs", slow_discover)
    app = MonitorApp(tmp_path, cache_s=60.0)
    threads = [threading.Thread(target=app.runs) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    app.runs()
    assert calls == [True], "one scan serves concurrent and later requests, with a per-store cache"

    expired = MonitorApp(tmp_path, cache_s=0.0)
    expired.runs()
    expired.runs()
    assert len(calls) == 3


def test_a_lost_root_probe_does_not_make_the_own_monitor_a_stranger(tmp_path, monkeypatch):
    answers = iter([None, str(tmp_path.resolve())])
    monkeypatch.setattr(monitor_mod, "_service_alive", lambda url, timeout=0.5: True)
    monkeypatch.setattr(monitor_mod, "_service_root", lambda url, timeout=0.5: next(answers))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: pytest.fail("no monitor should start"))
    assert ensure_service(tmp_path, HOST, 8765) == (f"http://{HOST}:8765", False)


def test_a_started_child_counts_only_when_the_port_serves_its_root(tmp_path, monkeypatch):
    """Two runs under different roots race for one port: the other run's
    monitor answers there and this run's child exits (it cannot bind). The
    run must move on rather than report the other root's monitor as its own."""
    class _DiesSoon:
        polls = 0

        def poll(self):
            _DiesSoon.polls += 1
            return 1 if _DiesSoon.polls > 2 else None

    class _Alive:
        def poll(self):
            return None

    launched = []
    monkeypatch.setattr("subprocess.Popen",
                        lambda args, **kw: launched.append(args) or (_DiesSoon() if len(launched) == 1 else _Alive()))
    # Once our child is launched on 8765, 8765 answers -- but it is the other
    # root's monitor that won the bind.
    monkeypatch.setattr(monitor_mod, "_service_alive",
                        lambda url, timeout=0.5: url.endswith(":8765") and bool(launched))
    monkeypatch.setattr(monitor_mod, "_service_root",
                        lambda url, timeout=0.5: str(tmp_path / "theirs") if url.endswith(":8765") else None)
    monkeypatch.setattr(monitor_mod, "_START_WAIT_S", 0.5)
    url, started = ensure_service(tmp_path / "mine", HOST, 8765)
    assert (url, started) == (f"http://{HOST}:8766", True)
    assert [a[a.index("--port") + 1] for a in launched] == ["8765", "8766"]


def test_a_started_monitor_is_given_the_allowed_host_names(tmp_path, monkeypatch):
    """``ensure_service(allow_hosts=...)`` reaches the started service as one
    ``--allow-host`` per name (blank names dropped; a bare string is one
    name), so a monitor started by a run can be served under a proxy name."""
    class _Alive:
        def poll(self):
            return None

    launched = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: launched.append(args) or _Alive())
    monkeypatch.setattr(monitor_mod, "_service_alive", lambda url, timeout=0.5: bool(launched))
    monkeypatch.setattr(monitor_mod, "_service_root", lambda url, timeout=0.5: str(tmp_path.resolve()))

    def allowed(args):
        return [args[i + 1] for i, a in enumerate(args) if a == "--allow-host"]

    assert ensure_service(tmp_path, HOST, 8765, allow_hosts=["proxy.internal", " ", "lab.example"]) \
        == (f"http://{HOST}:8765", True)
    assert allowed(launched[0]) == ["proxy.internal", "lab.example"]
    launched.clear()
    ensure_service(tmp_path, HOST, 8765, allow_hosts="proxy.internal")
    assert allowed(launched[0]) == ["proxy.internal"]
    launched.clear()
    ensure_service(tmp_path, HOST, 8765)
    assert allowed(launched[0]) == []


def test_traffic_state_write_retries_while_a_reader_holds_the_file(tmp_path, monkeypatch):
    import opencda_marl.telemetry.traffic_view as tv_mod

    real_replace = tv_mod.os.replace
    failures = {"left": 2}

    def flaky_replace(src, dst):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(tv_mod.os, "replace", flaky_replace)
    pub = TrafficViewPublisher(tmp_path, interval=0.05)
    assert pub.publish_state([], episode=1, step=7)
    assert json.loads((tmp_path / STATE_NAME).read_text())["step"] == 7


def test_a_frame_held_by_a_reader_is_skipped_quietly(tmp_path, monkeypatch):
    import opencda_marl.telemetry.traffic_view as tv_mod

    def always_held(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(tv_mod.os, "replace", always_held)
    monkeypatch.setattr(tv_mod, "_REPLACE_PAUSE_S", 0.0)
    warnings = []
    sink = tv_mod.logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    try:
        pub = TrafficViewPublisher(tmp_path, interval=0.05)
        assert pub.publish_state([], episode=1, step=7) is False
        pub.publish_geometry({"lanes": [[[0, 0], [1, 0]]]})
        assert not pub.geometry_written, "geometry is retried on a later tick"
    finally:
        tv_mod.logger.remove(sink)
    assert warnings == []


def test_traffic_view_read_retries_while_the_writer_replaces_the_file(tmp_path, monkeypatch):
    run = tmp_path / "sac" / "r"
    MetricsStore(run).close()
    TrafficViewPublisher(run, interval=0.05).publish_state([], episode=2, step=3)
    real_read = pathlib.Path.read_text
    left = {"n": 2}

    def mid_replace(self, *a, **kw):
        if self.name == STATE_NAME and left["n"]:
            left["n"] -= 1
            raise PermissionError(13, "The process cannot access the file")
        return real_read(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "read_text", mid_replace)
    app = MonitorApp(tmp_path)
    assert app.handle("/api/runs/sac/r/traffic_state", {})["step"] == 3
    with pytest.raises(FileNotFoundError) as missing:     # never written: still a 404
        app.handle("/api/runs/sac/r/traffic_geometry", {})
    assert str(tmp_path) not in str(missing.value)


def test_page_never_parses_server_text_as_html():
    """Run ids, metadata, tag names and episode fields are user data (a copied
    artifacts tree can carry anything); the page places them as text only, and
    tests values with Number.isFinite (isFinite(null) is true)."""
    for name in ("app.js", "traffic.js"):
        source = (STATIC / name).read_text(encoding="utf-8")
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            assert sink not in source, f"{name} uses {sink}"
        assert "isFinite(" not in source.replace("Number.isFinite(", ""), f"{name} uses global isFinite"
