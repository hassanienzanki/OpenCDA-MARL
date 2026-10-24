"""One SQLite file per run: scalars, episode summaries, events, run metadata.

Write side
    ``log_scalar`` / ``log_episode`` / ``log_event`` append to an in-memory
    buffer; the buffer is flushed in one transaction every ``flush_every``
    rows or ``flush_interval_s`` seconds, and on ``close``. Thread-safe.

    Non-finite scalars (NaN, +/-inf) are not stored: they are counted in the
    ``skipped_nonfinite`` meta key instead. Non-finite floats inside episode
    summaries, events and meta become ``null``, so every stored payload is
    strict JSON. Each writer also records a ``session`` id in meta (one per
    process), so a reader can tell that a new process started appending,
    and ``closed_at`` (wall time) when it closes the store, removed again
    when a writer reopens it: a reader can tell a finished run from one
    that is merely quiet.

Read side
    ``tags`` / ``scalars`` / ``episodes`` / ``events`` / ``meta``. ``scalars``
    downsamples server-side (min/max per bucket, so spikes survive) when a
    tag holds more than ``max_points`` rows. Readers open the file read-only
    and never block the writer (WAL).

Discovery
    ``discover_runs(root)`` lists every ``metrics.sqlite`` under a directory
    with its metadata, newest first. A caller that polls (the monitor) passes
    a ``cache`` dict, so a store whose files did not change is not reopened.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

SCHEMA_VERSION = 1
DB_NAME = "metrics.sqlite"

# One id per writing process: a relaunch that reopens an existing store gets
# a new one, so a reader holding a step cursor knows to start over.
SESSION_ID = uuid.uuid4().hex

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scalars (
    id    INTEGER PRIMARY KEY,
    tag   TEXT    NOT NULL,
    step  INTEGER NOT NULL,
    wall  REAL    NOT NULL,
    value REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS scalars_tag_step ON scalars (tag, step);
CREATE TABLE IF NOT EXISTS episodes (
    episode INTEGER PRIMARY KEY,
    wall    REAL    NOT NULL,
    summary TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY,
    wall    REAL NOT NULL,
    kind    TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def _json_safe(value: Any) -> Any:
    """Coerce numpy scalars/arrays and other odd values into strict-JSON ones.

    Non-finite floats (Python or numpy) become ``None``: ``json.dumps`` would
    otherwise write bare ``NaN``/``Infinity`` tokens, which a browser's JSON
    parser rejects.
    """
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _readonly_uri(path: os.PathLike | str) -> str:
    """The SQLite URI that opens ``path`` read-only.

    ``as_uri()`` percent-encodes the path: a raw '#' would start a URI
    fragment that also swallows ``?mode=ro``, and the truncated path would
    then be opened read-write-create. ``absolute()``, not ``resolve()``: on
    Windows, resolving turns a mapped network drive into its UNC path. A UNC
    path (``\\\\host\\share\\...``) becomes ``file://host/share/...``, whose
    host SQLite reads as a URI authority and refuses to open; with an empty
    authority (``file:////host/share/...``) SQLite opens the UNC path itself.
    An already-absolute pure path is mapped as given, on any platform.
    """
    if not (isinstance(path, PurePath) and path.is_absolute()):
        path = Path(path).absolute()
    uri = path.as_uri()
    if not uri.startswith("file:///"):
        uri = "file:////" + uri[len("file://"):]
    return uri + "?mode=ro"


class MetricsStore:
    """Writer and reader for one run's metrics file."""

    def __init__(self, path: os.PathLike | str, *, read_only: bool = False,
                 flush_every: int = 200, flush_interval_s: float = 2.0,
                 meta: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        # A run *directory* (existing or not) gets the standard file name; only
        # an explicit .sqlite/.db path is taken as the file itself.
        if self.path.suffix.lower() not in (".sqlite", ".sqlite3", ".db"):
            self.path = self.path / DB_NAME
        self.read_only = bool(read_only)
        self._flush_every = max(1, int(flush_every))
        self._flush_interval_s = float(flush_interval_s)
        self._lock = threading.RLock()
        self._scalar_buf: List[Tuple[str, int, float, float]] = []
        self._episode_buf: List[Tuple[int, float, str]] = []
        self._event_buf: List[Tuple[float, str, str]] = []
        self._last_flush = time.monotonic()
        self._closed = False
        self._skipped_nonfinite = 0          # non-finite scalars dropped so far
        self._skipped_recorded = 0           # ...of which meta already says so
        self._warned_nonfinite = False
        if self.read_only:
            if not self.path.exists():
                raise FileNotFoundError(self.path)
            self._conn = sqlite3.connect(_readonly_uri(self.path), uri=True,
                                         check_same_thread=False, timeout=5.0)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=5.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            # A writer reopening a closed store (a resumed run) makes it
            # live again.
            self._conn.execute("DELETE FROM meta WHERE key = 'closed_at'")
            self._conn.commit()
            self.set_meta(schema_version=SCHEMA_VERSION, session=SESSION_ID)
            if meta:
                self.set_meta(**meta)
            existing = self.meta()
            if "created" not in existing:
                self.set_meta(created=time.time())
            try:                             # a reopened store keeps counting
                self._skipped_nonfinite = self._skipped_recorded = int(
                    existing.get("skipped_nonfinite") or 0)
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------ #
    # Write side
    # ------------------------------------------------------------------ #
    def set_meta(self, **values: Any) -> None:
        self._check_writable()
        rows = [(str(k), json.dumps(_json_safe(v))) for k, v in values.items()]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", rows)
            self._conn.commit()

    def log_scalar(self, tag: str, value: float, step: int, wall: Optional[float] = None) -> None:
        self._check_writable()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            # ``value`` is REAL NOT NULL and SQLite binds NaN as NULL, so one
            # NaN would fail the whole flush transaction; +/-inf would be
            # stored but cannot be served as JSON. Count it instead.
            with self._lock:
                self._skipped_nonfinite += 1
                if not self._warned_nonfinite:
                    self._warned_nonfinite = True
                    logger.warning(f"metrics store skips non-finite scalars (first: {tag}={value} "
                                   f"at step {step}); the count is kept in meta 'skipped_nonfinite'")
            return
        wall = float(wall) if wall else time.time()
        if not math.isfinite(wall):
            wall = time.time()
        with self._lock:
            self._scalar_buf.append((str(tag), int(step), wall, value))
            self._maybe_flush()

    def log_scalars(self, values: Dict[str, float], step: int, prefix: str = "") -> None:
        for tag, value in values.items():
            self.log_scalar(f"{prefix}{tag}", value, step)

    def log_episode(self, episode: int, summary: Dict[str, Any]) -> None:
        self._check_writable()
        payload = json.dumps(_json_safe(summary))
        with self._lock:
            self._episode_buf.append((int(episode), time.time(), payload))
            self._maybe_flush()

    def log_event(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._check_writable()
        with self._lock:
            self._event_buf.append((time.time(), str(kind), json.dumps(_json_safe(payload or {}))))
            self._maybe_flush()

    def flush(self) -> None:
        if self.read_only or self._closed:
            return
        with self._lock:
            skipped = self._skipped_nonfinite
            if not (self._scalar_buf or self._episode_buf or self._event_buf
                    or skipped != self._skipped_recorded):
                self._last_flush = time.monotonic()
                return
            batches = [
                ("scalar", "INSERT INTO scalars (tag, step, wall, value) VALUES (?, ?, ?, ?)",
                 self._scalar_buf),
                ("episode", "INSERT OR REPLACE INTO episodes (episode, wall, summary) VALUES (?, ?, ?)",
                 self._episode_buf),
                ("event", "INSERT INTO events (wall, kind, payload) VALUES (?, ?, ?)",
                 self._event_buf),
            ]
            if skipped != self._skipped_recorded:
                batches.append(("meta", "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                                [("skipped_nonfinite", json.dumps(skipped))]))
            try:
                try:
                    with self._conn:             # one transaction
                        for _kind, sql, rows in batches:
                            if rows:
                                self._conn.executemany(sql, rows)
                except sqlite3.IntegrityError:
                    # A row the schema rejects must not cost its neighbours:
                    # retry each table on its own, then row by row.
                    for kind, sql, rows in batches:
                        self._insert_salvaging(kind, sql, rows)
                self._skipped_recorded = skipped
            except sqlite3.Error as exc:         # never take training down for telemetry
                logger.warning(f"metrics store flush failed ({self.path}): {exc}")
            finally:
                self._scalar_buf.clear()
                self._episode_buf.clear()
                self._event_buf.clear()
                self._last_flush = time.monotonic()

    def _insert_salvaging(self, kind: str, sql: str, rows: Sequence[tuple]) -> None:
        """Insert ``rows`` in one transaction, or one by one when the batch
        violates a constraint; rows that still fail are dropped and counted
        in one warning. Other SQLite errors propagate to ``flush``."""
        if not rows:
            return
        try:
            with self._conn:
                self._conn.executemany(sql, rows)
            return
        except sqlite3.IntegrityError:
            pass
        dropped = 0
        for row in rows:
            try:
                with self._conn:
                    self._conn.execute(sql, row)
            except sqlite3.IntegrityError:
                dropped += 1
        if dropped:
            logger.warning(f"metrics store dropped {dropped} {kind} row(s) the schema rejects ({self.path})")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.flush()
            if not self.read_only:
                try:
                    self.set_meta(closed_at=time.time())
                except sqlite3.Error as exc:     # telemetry never fails a run
                    logger.warning(f"metrics store close marker not written ({self.path}): {exc}")
        finally:
            self._closed = True
            with self._lock:
                self._conn.close()

    def __enter__(self) -> "MetricsStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Read side
    # ------------------------------------------------------------------ #
    def meta(self) -> Dict[str, Any]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM meta").fetchall()
        out = {}
        for key, value in rows:
            try:
                out[key] = json.loads(value)
            except json.JSONDecodeError:
                out[key] = value
        return out

    def tags(self) -> List[Dict[str, Any]]:
        """Every scalar tag with its row count and last step/value."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tag, COUNT(*), MAX(step), MIN(step) FROM scalars GROUP BY tag ORDER BY tag"
            ).fetchall()
            out = []
            for tag, count, last_step, first_step in rows:
                last = self._conn.execute(
                    "SELECT value, wall FROM scalars WHERE tag = ? ORDER BY step DESC, id DESC LIMIT 1",
                    (tag,)).fetchone()
                out.append({"tag": tag, "count": int(count), "first_step": int(first_step),
                            "last_step": int(last_step),
                            "last_value": float(last[0]) if last else None,
                            "last_wall": float(last[1]) if last else None})
        return out

    def scalars(self, tag: str, *, since_step: Optional[int] = None,
                max_points: Optional[int] = None) -> List[Tuple[int, float, float]]:
        """``[(step, wall, value), ...]`` for ``tag``, oldest first.

        With ``max_points`` the series is reduced to about that many points
        by keeping the min and the max of each step bucket (plus the last
        point), so peaks are not averaged away.
        """
        query = "SELECT step, wall, value FROM scalars WHERE tag = ?"
        params: list = [tag]
        if since_step is not None:
            query += " AND step > ?"
            params.append(int(since_step))
        query += " ORDER BY step, id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return downsample(rows, max_points) if max_points else rows

    def episodes(self, *, since: Optional[int] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT episode, wall, summary FROM episodes"
        params: list = []
        if since is not None:
            query += " WHERE episode > ?"
            params.append(int(since))
        query += " ORDER BY episode DESC"
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        out = []
        for episode, wall, summary in reversed(rows):
            try:
                data = json.loads(summary)
            except json.JSONDecodeError:
                data = {"raw": summary}
            data = dict(data)
            data["episode"] = int(episode)
            data["wall"] = float(wall)
            out.append(data)
        return out

    def events(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT wall, kind, payload FROM events ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for wall, kind, payload in reversed(rows):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"raw": payload}
            out.append({"wall": float(wall), "kind": kind, "payload": data})
        return out

    def counts(self) -> Dict[str, int]:
        """Row counts and the wall time of the newest scalar.

        Polled for every run on every monitor refresh, so it reads only the
        end of the scalars table: ``id`` is an append-only INTEGER PRIMARY
        KEY (rows are never deleted), so ``MAX(id)`` is the row count and the
        highest id is the newest row. ``COUNT(*)`` and ``MAX(wall)`` would
        scan the whole table instead.
        """
        with self._lock:
            scalars = self._conn.execute("SELECT MAX(id) FROM scalars").fetchone()[0]
            episodes = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            last = self._conn.execute("SELECT wall FROM scalars ORDER BY id DESC LIMIT 1").fetchone()
        return {"scalars": int(scalars or 0), "episodes": int(episodes),
                "last_wall": float(last[0]) if last and last[0] else 0.0}

    # ------------------------------------------------------------------ #
    def _maybe_flush(self) -> None:
        pending = len(self._scalar_buf) + len(self._episode_buf) + len(self._event_buf)
        if pending >= self._flush_every or (time.monotonic() - self._last_flush) >= self._flush_interval_s:
            self.flush()

    def _check_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("metrics store opened read-only")
        if self._closed:
            raise RuntimeError("metrics store is closed")


def downsample(rows: Sequence[Tuple[int, float, float]], max_points: int) -> List[Tuple[int, float, float]]:
    """Reduce ``rows`` (sorted by step) to ~``max_points`` keeping bucket min/max."""
    n = len(rows)
    if max_points <= 0 or n <= max_points:
        return list(rows)
    buckets = max(1, max_points // 2)
    size = n / buckets
    out: List[Tuple[int, float, float]] = []
    for b in range(buckets):
        lo, hi = int(b * size), int((b + 1) * size) if b < buckets - 1 else n
        chunk = rows[lo:hi]
        if not chunk:
            continue
        lo_row = min(chunk, key=lambda r: r[2])
        hi_row = max(chunk, key=lambda r: r[2])
        pair = sorted({lo_row, hi_row}, key=lambda r: r[0])
        out.extend(pair)
    if out[-1] != rows[-1]:
        out.append(rows[-1])
    return out


@dataclass
class RunInfo:
    run_id: str
    path: Path
    meta: Dict[str, Any]
    counts: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "path": str(self.path), "meta": self.meta, "counts": self.counts}


_CACHE_MAX_AGE_S = 30.0   # a cached store is re-read at least this often


def _file_stamp(db: Path) -> tuple:
    """What changes when a store is written: size and mtime of the database
    and of its WAL (a WAL-mode commit touches only the ``-wal`` file). An
    empty WAL counts as none: a read-only open of a closed store creates
    one, and that must not look like a write."""
    out = []
    for path in (db, db.with_name(db.name + "-wal")):
        try:
            st = path.stat()
        except OSError:
            out.append(None)
            continue
        out.append((st.st_mtime_ns, st.st_size) if st.st_size or path == db else None)
    return tuple(out)


def _read_run(db: Path, cache: Optional[dict]):
    """``(meta, counts)`` of one store, or ``None`` when it cannot be read.

    With ``cache``, a store whose files did not change since the last call
    (and was read less than ``_CACHE_MAX_AGE_S`` ago) is not reopened, and an
    unreadable file is reported once rather than on every poll."""
    key = str(db)
    stamp = _file_stamp(db) if cache is not None else None
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and hit[0] == stamp and time.monotonic() - hit[1] < _CACHE_MAX_AGE_S:
            return hit[2]
    try:
        store = MetricsStore(db, read_only=True)
        try:
            result = (store.meta(), store.counts())
        finally:
            store.close()
    except (sqlite3.Error, OSError) as exc:
        if cache is None or key not in cache or cache[key][2] is not None:
            logger.warning(f"skipping unreadable metrics store {db}: {exc}")
        result = None
    if cache is not None:
        cache[key] = (stamp, time.monotonic(), result)
    return result


def discover_runs(root: os.PathLike | str, *, limit: Optional[int] = None,
                  cache: Optional[dict] = None) -> List[RunInfo]:
    """Every ``metrics.sqlite`` under ``root`` (newest activity first).

    ``cache`` is an optional dict the caller keeps between calls (see
    :func:`_read_run`); entries of stores that disappeared are dropped."""
    root = Path(root)
    if not root.exists():
        return []
    runs: List[RunInfo] = []
    seen = set()
    for db in root.rglob(DB_NAME):
        seen.add(str(db))
        read = _read_run(db, cache)
        if read is None:
            continue
        meta, counts = read
        rel = db.parent.relative_to(root).as_posix()
        runs.append(RunInfo(run_id=rel, path=db, meta=meta, counts=counts))
    if cache is not None:
        for key in [k for k in cache if k not in seen]:
            del cache[key]
    runs.sort(key=lambda r: (r.counts.get("last_wall") or r.meta.get("created") or 0.0), reverse=True)
    return runs[:limit] if limit else runs
