"""One monitor per port, and a run reuses only the monitor that serves its root.

Every run auto-starts a monitor on its configured port unless one already
answers there. The stdlib server binds with ``SO_REUSEADDR``, which on
Windows lets a second process bind a port that is already being served:
several monitors could bind the same port, each started by a different run
and answering for whichever root it had been given, so the same URL listed
different runs from one request to the next.

Now the port is bound exclusively, the service reports the directory it
serves, and ``ensure_service`` reuses a monitor only when it serves the
run's own root -- otherwise it moves to the next port, so parallel runs on
different roots each get their own service.
"""
import http.client
import json
import threading
from pathlib import Path

import pytest

import opencda_marl.telemetry.monitor as monitor_mod
from opencda_marl.telemetry.monitor import ensure_service, serve

HOST = "127.0.0.1"


@pytest.fixture
def running():
    """Start monitors on OS-chosen ports; returns ``start(root) -> port``."""
    servers = []

    def start(root):
        server = serve(root, HOST, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server.server_address[1]

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(5)


def _get_json(port, path):
    conn = http.client.HTTPConnection(HOST, port, timeout=10)
    try:
        conn.request("GET", path)
        return json.loads(conn.getresponse().read().decode("utf-8"))
    finally:
        conn.close()


def _never_start(*args, **kwargs):
    pytest.fail("no monitor should have been started")


def test_the_service_says_which_directory_it_serves(tmp_path, running):
    port = running(tmp_path)
    assert Path(_get_json(port, "/api/root")["root"]) == tmp_path.resolve()


def test_a_second_monitor_cannot_bind_a_served_port(tmp_path, running):
    port = running(tmp_path)
    with pytest.raises(OSError):
        serve(tmp_path, HOST, port)


def test_a_run_reuses_the_monitor_that_serves_its_root(tmp_path, running, monkeypatch):
    port = running(tmp_path)
    monkeypatch.setattr("subprocess.Popen", _never_start)
    assert ensure_service(tmp_path, HOST, port) == (f"http://{HOST}:{port}", False)


def test_a_monitor_of_another_root_is_left_alone_and_the_next_port_is_used(tmp_path, running, monkeypatch):
    theirs = tmp_path / "theirs"
    theirs.mkdir()
    mine = tmp_path / "mine"
    port = running(theirs)

    class _SlowChild:                      # started, not yet answering, still alive
        def poll(self):
            return None

    launched = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: launched.append(args) or _SlowChild())
    monkeypatch.setattr(monitor_mod, "_START_WAIT_S", 0.2)

    url, started = ensure_service(mine, HOST, port)

    assert (url, started) == (f"http://{HOST}:{port + 1}", True)
    (args,) = launched
    assert args[args.index("--port") + 1] == str(port + 1)
    assert Path(args[args.index("--root") + 1]) == mine.resolve()
    assert (mine / "monitor.log").exists(), "the child's log lives under the root it serves"


def test_a_child_that_dies_before_answering_moves_the_search_to_the_next_port(tmp_path, monkeypatch):
    class _Dead:                            # exited at once: it could not bind
        def poll(self):
            return 2

    class _Alive:
        def poll(self):
            return None

    launched = []
    monkeypatch.setattr("subprocess.Popen",
                        lambda args, **kw: launched.append(args) or (_Dead() if len(launched) == 1 else _Alive()))
    monkeypatch.setattr(monitor_mod, "_service_alive", lambda url, timeout=0.5: False)
    monkeypatch.setattr(monitor_mod, "_START_WAIT_S", 0.2)

    url, started = ensure_service(tmp_path, HOST, 8765)

    assert (url, started) == (f"http://{HOST}:8766", True)
    assert [a[a.index("--port") + 1] for a in launched] == ["8765", "8766"]


def test_when_every_port_serves_another_root_nothing_is_started(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor_mod, "_service_alive", lambda url, timeout=0.5: True)
    monkeypatch.setattr(monitor_mod, "_service_root", lambda url, timeout=0.5: str(tmp_path / "elsewhere"))
    monkeypatch.setattr("subprocess.Popen", _never_start)
    assert ensure_service(tmp_path, HOST, 8765) == (None, False)


def test_a_monitor_too_old_to_name_its_root_is_not_trusted(tmp_path, monkeypatch):
    """A service from before ``/api/root`` answers ``/api/runs`` but not
    ``/api/root``; it may serve anything, so it is skipped like a stranger."""
    monkeypatch.setattr(monitor_mod, "_service_alive", lambda url, timeout=0.5: url.endswith(":8765"))
    monkeypatch.setattr(monitor_mod, "_service_root", lambda url, timeout=0.5: None)
    launched = []

    class _Alive:
        def poll(self):
            return None

    monkeypatch.setattr("subprocess.Popen", lambda args, **kw: launched.append(args) or _Alive())
    monkeypatch.setattr(monitor_mod, "_START_WAIT_S", 0.2)
    url, started = ensure_service(tmp_path, HOST, 8765)
    assert (url, started) == (f"http://{HOST}:8766", True)
