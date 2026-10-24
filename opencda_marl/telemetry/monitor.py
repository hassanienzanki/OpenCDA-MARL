"""The training monitor: serves the metrics stores under a root to a browser.

    pixi run monitor                      # artifacts/, http://127.0.0.1:8765
    python -m opencda_marl.telemetry.monitor --root artifacts --port 8765
    python -m opencda_marl.telemetry.monitor list --root artifacts
    python -m opencda_marl.telemetry.monitor tail <run_id> --root artifacts

Standard library only (``http.server``), so it runs on a headless training
node; reach it through ``ssh -L 8765:127.0.0.1:8765 <node>``. Runs are
rediscovered at most every ``cache_s`` seconds (one scan serves every tab),
so a cell that starts while the page is open appears on the next refresh;
charts poll the API and ask only for points after the last step they
already have.

Responses are strict JSON: non-finite floats are served as ``null``. Run
ids are checked lexically before any filesystem access. A monitor bound to
a loopback address answers only requests whose ``Host`` names a loopback
host, so a web page cannot read it through DNS rebinding; a proxy that
forwards under its own host name is admitted with ``--allow-host NAME``.

API (JSON):
    GET /api/root                              the served root and the service pid
    GET /api/runs                              discovered runs, newest first
    GET /api/runs/<id>/tags                    scalar tags with counts
    GET /api/runs/<id>/scalars?tag=T&since=S&max_points=N
    GET /api/runs/<id>/episodes?limit=N
    GET /api/runs/<id>/events?limit=N
    GET /api/runs/<id>/meta
    GET /api/runs/<id>/traffic_geometry      schematic lane polylines
    GET /api/runs/<id>/traffic_state         live vehicle poses (throttled writer)
"""
from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import math
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .store import DB_NAME, MetricsStore, discover_runs

_HERE = Path(__file__).resolve().parent
_STATIC = _HERE / "static"
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


def _relative_parts(rel: str) -> list:
    """The segments of a relative POSIX path, checked before it touches the
    filesystem; ``FileNotFoundError`` for anything else.

    Rejected: a backslash or a colon (on Windows ``\\\\host\\share`` joins
    into a UNC path, and resolving one opens a network connection to that
    host before any containment check can run; ``C:`` names a drive), an
    absolute path, a NUL, and empty, ``.`` or ``..`` segments.
    """
    if not rel or "\\" in rel or ":" in rel or "\x00" in rel or PurePosixPath(rel).is_absolute():
        raise FileNotFoundError(rel)
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise FileNotFoundError(rel)
    return parts


def _contained(root: Path, parts: list) -> Path:
    """``root / parts``, resolved, or ``FileNotFoundError`` when it is not
    inside ``root``: lexically first, then after resolving (a symlink)."""
    base = os.path.normpath(os.path.abspath(root))
    joined = os.path.normpath(os.path.join(base, *parts))
    if os.path.commonpath([base, joined]) != base:
        raise FileNotFoundError("/".join(parts))
    resolved, base_resolved = Path(joined).resolve(), Path(base).resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise FileNotFoundError("/".join(parts))
    return resolved


def _static_file(path: str):
    """Resolve a request path to a file inside static/ (traversal-safe)."""
    try:
        candidate = _contained(_STATIC, _relative_parts(path.lstrip("/") or "index.html"))
    except FileNotFoundError:
        return None, None
    if not candidate.is_file():
        return None, None
    return candidate.read_bytes(), _STATIC_TYPES.get(candidate.suffix, "application/octet-stream")


def _finite(value: Any) -> Any:
    """``value`` with every non-finite float replaced by ``None``.

    A stored +inf scalar, or a NaN inside a summary that an older store kept
    as a bare ``NaN`` token, would otherwise be served as ``Infinity`` /
    ``NaN``, which the browser's JSON parser rejects -- and with it the whole
    tag list or episode table of the run."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _encode(payload: Any) -> bytes:
    """Strict JSON. ``allow_nan=False`` only backs up :func:`_finite`: a
    non-finite value that slipped through fails loudly here (a 500, not the
    400 a ``ValueError`` would map to) rather than shipping a body the page
    cannot parse."""
    try:
        return json.dumps(_finite(payload), allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise RuntimeError(f"response is not strict JSON: {exc}") from exc


def _is_loopback(host: str) -> bool:
    name = host.strip().lower().strip("[]")
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _host_name(header: str) -> str:
    """The host part of a ``Host`` header: ``127.0.0.1:8765`` -> ``127.0.0.1``,
    ``[::1]:8765`` -> ``::1``."""
    header = header.strip()
    if header.startswith("["):
        return header[1:header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


_VIEW_READ_ATTEMPTS = 4


def _read_view_file(path: Path) -> Any:
    """A traffic view file, read while its writer may be replacing it.

    On Windows a read that meets the writer's ``os.replace`` fails with
    ``PermissionError``, or briefly finds no file at all, so reads that
    overlap a writer replacing the file back to back fail intermittently.
    Retried a few times before the error is reported (a 404 then means the
    run has not written the file)."""
    for attempt in range(_VIEW_READ_ATTEMPTS):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError) as exc:
            if attempt + 1 < _VIEW_READ_ATTEMPTS:
                time.sleep(0.01)
            elif isinstance(exc, FileNotFoundError):
                raise FileNotFoundError(path.name) from None
            else:
                raise


class MonitorApp:
    """Request handlers over a root directory of runs."""

    def __init__(self, root: Path, *, cache_s: float = 2.0):
        self.root = Path(root)
        # At least the page's default refresh period, so every open tab
        # shares one discovery scan per period.
        self._cache_s = cache_s
        self._runs_cache: tuple[float, list] = (float("-inf"), [])
        self._store_cache: dict = {}            # per-store (meta, counts), see discover_runs
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- #
    def runs(self) -> list:
        # Single flight: a request that arrives while a scan runs waits for
        # it and shares its result instead of starting another scan.
        with self._lock:
            stamp, cached = self._runs_cache
            if time.monotonic() - stamp < self._cache_s:
                return cached
            runs = [r.as_dict() for r in discover_runs(self.root, cache=self._store_cache)]
            self._runs_cache = (time.monotonic(), runs)
            return runs

    def _run_dir(self, run_id: str) -> Path:
        """The directory of ``run_id`` (as listed by ``/api/runs``) inside
        the root. The id is validated lexically before anything resolves it
        (see :func:`_relative_parts`)."""
        return _contained(self.root, _relative_parts(unquote(run_id)))

    def _open(self, run_id: str) -> MetricsStore:
        candidate = self._run_dir(run_id)
        db = candidate / DB_NAME if candidate.is_dir() else candidate
        if not db.exists():
            raise FileNotFoundError(run_id)
        return MetricsStore(db, read_only=True)

    def handle(self, path: str, query: Dict[str, list]) -> Any:
        parts = [p for p in path.split("/") if p]
        if parts == ["api", "root"]:
            # Which directory this service watches, so a run can tell a
            # monitor of its own root from a stranger's on the same port;
            # the pid names the process to stop.
            return {"root": str(self.root.resolve()), "pid": os.getpid()}
        if parts[:2] == ["api", "runs"] and len(parts) == 2:
            return self.runs()
        if parts[:2] == ["api", "runs"] and len(parts) >= 4:
            run_id = "/".join(parts[2:-1])
            what = parts[-1]
            if what in ("traffic_geometry", "traffic_state"):
                name = ("traffic_geometry.json" if what == "traffic_geometry"
                        else "traffic_state.json")
                return _read_view_file(self._run_dir(run_id) / name)
            store = self._open(run_id)
            try:
                if what == "tags":
                    return store.tags()
                if what == "meta":
                    return {"meta": store.meta(), "counts": store.counts()}
                if what == "scalars":
                    tag = query.get("tag", [None])[0]
                    if not tag:
                        raise ValueError("tag is required")
                    since = query.get("since", [None])[0]
                    max_points = int(query.get("max_points", ["1500"])[0])
                    rows = store.scalars(tag, since_step=int(since) if since not in (None, "") else None,
                                         max_points=max_points)
                    return {"tag": tag, "points": [[int(s), float(v)] for s, _w, v in rows],
                            "last_step": rows[-1][0] if rows else (int(since) if since else None)}
                if what == "episodes":
                    limit = int(query.get("limit", ["200"])[0])
                    return store.episodes(limit=limit)
                if what == "events":
                    limit = int(query.get("limit", ["200"])[0])
                    return store.events(limit=limit)
            finally:
                store.close()
        raise FileNotFoundError(path)


def _make_handler(app: MonitorApp, *, check_host: bool = False, extra_hosts=()):
    """``check_host``: answer only requests whose ``Host`` header names a
    loopback host (or one of ``extra_hosts``). A page on another origin can
    point its own DNS name at 127.0.0.1 and then read a loopback service as
    same-origin; its requests still carry that name in ``Host``."""
    allowed = {h.strip().lower() for h in extra_hosts if h and h.strip()}

    def _host_ok(header: Optional[str]) -> bool:
        if not check_host or header is None:     # no Host: not a browser
            return True
        name = _host_name(header).lower()
        return _is_loopback(name) or name in allowed

    class Handler(BaseHTTPRequestHandler):
        server_version = "OpenCDA-MARL-monitor/1.0"
        # Every response carries an exact Content-Length, so HTTP/1.1 is
        # safe — and it matters: under HTTP/1.0 the server closes the
        # socket after each response, and on Windows a close with request
        # bytes still unread makes the peer see a connection reset instead
        # of the response it was sent.
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):      # quiet by default
            if "--verbose" in sys.argv:
                super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            header = self.headers.get("Host")
            if not _host_ok(header):
                # Says how to serve the name, for a proxy that forwards
                # under its own Host (a notebook or IDE proxy, a tunnel).
                name = _host_name(header)
                self._send(HTTPStatus.FORBIDDEN,
                           json.dumps({"error": f"host not allowed: {name}; "
                                                f"start the monitor with --allow-host {name}"}).encode(),
                           "application/json")
                return
            parsed = urlparse(self.path)
            try:
                if parsed.path.startswith("/api/"):
                    payload = app.handle(parsed.path, parse_qs(parsed.query))
                    self._send(HTTPStatus.OK, _encode(payload), "application/json; charset=utf-8")
                    return
                body, content_type = _static_file(parsed.path)
                if body is not None:
                    self._send(HTTPStatus.OK, body, content_type)
                    return
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            except FileNotFoundError as exc:
                self._send(HTTPStatus.NOT_FOUND, json.dumps({"error": f"not found: {exc}"}).encode(),
                           "application/json")
            except ValueError as exc:
                self._send(HTTPStatus.BAD_REQUEST, json.dumps({"error": str(exc)}).encode(),
                           "application/json")
            except Exception as exc:  # noqa: BLE001
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR,
                           json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(),
                           "application/json")

    return Handler


class _MonitorServer(ThreadingHTTPServer):
    """One monitor per port.

    The stdlib server sets ``SO_REUSEADDR``, which on Windows lets a second
    process bind a port that is already being served, so several monitors
    can share one port, each answering for whichever root had started it.
    Binding exclusively there makes the second bind fail instead, so
    :func:`ensure_service` moves on to the next port. On POSIX the flag only
    relaxes ``TIME_WAIT`` and a second live bind fails regardless.
    """
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A browser that left mid-response, or this loopback's habit of
        # resetting fresh connections, is not worth a traceback in the log;
        # anything else still is.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve(root: Path, host: str = "127.0.0.1", port: int = 8765,
          allow_hosts=()) -> ThreadingHTTPServer:
    """Create (but do not run) the server; call ``serve_forever`` on it.

    Bound to a loopback address, the server answers only requests whose
    ``Host`` is a loopback name or one of ``allow_hosts`` (for a proxy that
    forwards under its own name). Raises ``OSError`` when the port is taken."""
    handler = _make_handler(MonitorApp(root), check_host=_is_loopback(host), extra_hosts=allow_hosts)
    return _MonitorServer((host, port), handler)


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def _cmd_list(root: Path) -> int:
    runs = discover_runs(root)
    if not runs:
        print(f"no runs under {root}")
        return 1
    print(f"{'run':60s} {'algo':10s} {'mode':6s} {'scalars':>9s} {'episodes':>9s}  last activity")
    for r in runs:
        meta, counts = r.meta, r.counts
        last = counts.get("last_wall") or meta.get("created") or 0
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last)) if last else "-"
        print(f"{r.run_id[:60]:60s} {str(meta.get('algorithm', '?'))[:10]:10s} "
              f"{str(meta.get('mode', '?'))[:6]:6s} {counts.get('scalars', 0):>9d} "
              f"{counts.get('episodes', 0):>9d}  {stamp}")
    return 0


def _cmd_tail(root: Path, run_id: str, n: int) -> int:
    # A typed id may use the platform separator or a trailing slash; the API
    # accepts only the canonical form that `list` prints.
    store = MonitorApp(root)._open(run_id.replace("\\", "/").strip("/"))
    try:
        print(json.dumps({"meta": store.meta(), "counts": store.counts()}, indent=2)[:2000])
        print("--- tags")
        for t in store.tags():
            print(f"  {t['tag']:40s} n={t['count']:<7d} last_step={t['last_step']:<8d} last={t['last_value']:.6g}")
        print(f"--- last {n} episodes")
        for ep in store.episodes(limit=n):
            keys = ("episode", "reward", "length", "success_rate", "collision_rate", "training_step")
            print("  " + "  ".join(f"{k}={ep.get(k)}" for k in keys if k in ep))
    finally:
        store.close()
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="OpenCDA-MARL training monitor")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve", "list", "tail"])
    parser.add_argument("run_id", nargs="?", default=None)
    parser.add_argument("--root", default="artifacts", help="directory searched for metrics.sqlite files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                        help="also answer requests addressed to NAME (a proxy's host name); "
                             "loopback names are always accepted")
    parser.add_argument("-n", type=int, default=10, help="episodes to show for `tail`")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "list":
        return _cmd_list(root)
    if args.command == "tail":
        if not args.run_id:
            parser.error("tail needs a run id (see `list`)")
        return _cmd_tail(root, args.run_id, args.n)
    try:
        server = serve(root, args.host, args.port, allow_hosts=args.allow_host)
    except OSError as exc:
        print(f"monitor: cannot bind {args.host}:{args.port} ({exc}); "
              f"another service holds it -- pass --port", file=sys.stderr, flush=True)
        return 2
    print(f"OpenCDA-MARL monitor: http://{args.host}:{args.port}/  (root: {root.resolve()})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())


def _probe(url: str, path: str, timeout: float):
    """GET ``url + path`` over a plain connection; the body, or ``None`` when
    nothing answers there or the answer is not 200.

    On Windows loopback a fresh connection that announces
    ``Connection: close`` -- what ``urllib.request`` sends -- is often reset
    by the peer, while the same request without that header is not. The
    probe therefore sends the plain request: a spurious "no monitor here"
    would start another monitor.
    """
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        return body if response.status == HTTPStatus.OK else None
    except Exception:  # noqa: BLE001 - not alive, whatever the reason
        return None
    finally:
        conn.close()


def _service_alive(url: str, timeout: float = 0.5) -> bool:
    return _probe(url, "/api/runs", timeout) is not None


def _service_root(url: str, timeout: float = 0.5):
    """The directory a live monitor serves, or ``None`` when it is not
    alive or too old to say."""
    body = _probe(url, "/api/root", timeout)
    if body is None:
        return None
    try:
        return json.loads(body.decode("utf-8")).get("root")
    except ValueError:
        return None


_PORT_SPAN = 8         # ports tried from the configured one upward
_START_WAIT_S = 3.0    # how long a freshly started service may take to answer
_ROOT_PROBES = 3       # /api/root attempts before a live monitor counts as a stranger


def _served_root(url: str):
    """:func:`_service_root`, asked up to ``_ROOT_PROBES`` times: one lost
    probe must not make a run's own monitor look like a stranger's (and
    start a duplicate on the next port)."""
    for attempt in range(_ROOT_PROBES):
        served = _service_root(url)
        if served is not None:
            return served
        if attempt + 1 < _ROOT_PROBES:
            time.sleep(0.1)
    return None


def ensure_service(root: Path, host: str = "127.0.0.1", port: int = 8765, allow_hosts=()):
    """Start the monitor for ``root`` as a detached process, or reuse one.

    The shell calls this when a simulation starts (``outputs.monitor``), so
    watching a run never requires launching anything by hand. The service
    OUTLIVES the run -- a finished run stays inspectable -- and the next run
    under the same root reuses it. Its log goes to ``<root>/monitor.log``.
    To stop it, end the process: ``/api/root`` reports its ``pid``. On
    Windows the running service keeps ``<root>`` (its working directory and
    log) in use, so stop it before deleting or renaming the root.

    One service serves one root. A live monitor on a port is reused only
    when ``/api/root`` says it serves *this* root; otherwise the next port
    is tried (``port`` up to ``port + 7``), so runs under different roots
    each get their own monitor instead of one service answering for a
    directory it does not watch. Returns ``(url, started)``, or
    ``(None, False)`` when every port in the span serves another root.

    ``allow_hosts`` are passed to a started service as ``--allow-host``
    (see :func:`serve`), for a page reached through a proxy under its own
    host name. A monitor that already serves this root is reused as it was
    started, with its own allowed names.
    """
    root = Path(root).resolve()
    if isinstance(allow_hosts, str):     # one name, as a config may give it
        allow_hosts = [allow_hosts]
    for candidate in range(port, port + _PORT_SPAN):
        url = f"http://{host}:{candidate}"
        if _service_alive(url):
            served = _served_root(url)
            if served is not None and Path(served) == root:
                return url, False
            continue          # another root, or a monitor too old to say which
        if _start_service(root, host, candidate, url, allow_hosts):
            return url, True
        # the child died, or another root's monitor won the port first:
        # try the next one
    return None, False


def _start_service(root: Path, host: str, port: int, url: str, allow_hosts=()) -> bool:
    """Detach a monitor on ``port``; True once a monitor of ``root`` answers
    there (or the child is merely slow), False when the child exits first --
    it could not bind, for instance because a run under another root
    started a monitor on the same port a moment earlier."""
    import subprocess
    import sys as _sys
    import time as _time

    # Resolved before use: the child is started *in* this directory and is
    # also told to serve it, so a relative path would resolve twice and the
    # service would watch <root>/<root> -- a directory that does not exist.
    root.mkdir(parents=True, exist_ok=True)
    log = open(root / "monitor.log", "ab")
    kwargs: dict = {"stdout": log, "stderr": log, "stdin": subprocess.DEVNULL,
                    "cwd": str(root)}
    if _sys.platform == "win32":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True
    args = [_sys.executable, "-m", "opencda_marl.telemetry.monitor",
            "--root", str(root), "--host", host, "--port", str(port)]
    for name in allow_hosts or ():
        if name and str(name).strip():
            args += ["--allow-host", str(name).strip()]
    child = subprocess.Popen(args, **kwargs)
    log.close()
    deadline = _time.monotonic() + _START_WAIT_S
    while _time.monotonic() < deadline:
        if _service_alive(url, timeout=0.3):
            # Something answers; it is ours only if it serves this root.
            served = _service_root(url, timeout=0.3)
            if served is not None and Path(served) == root:
                return True
        if child.poll() is not None:
            return False
        _time.sleep(0.15)
    served = _service_root(url, timeout=0.3)
    if served is not None and Path(served) != root:
        return False      # another root's monitor holds the port; ours cannot bind
    # slow start: report the URL anyway; the page retries on refresh
    return True
