"""Schematic live traffic view for headless runs.

SUMO-GUI-style lines instead of a rendered camera: the run writes two
small JSON files into its metrics-store run directory —
``traffic_geometry.json`` (lane polylines, once per world) and
``traffic_state.json`` (vehicle poses, throttled) — and the web monitor
draws them on a canvas next to the metrics. No CARLA rendering, no extra
server round trips (the CARLA state comes from the per-step world
snapshot the agents already captured), no dependencies.

File shapes:

``traffic_geometry.json``::

    {"lanes": [[[x, y], ...], ...], "junction": [cx, cy] | null,
     "y_up": bool (default true)}

``traffic_state.json``::

    {"episode": int, "step": int, "wall": float, "interval": float,
     "vehicles": [{"id": str, "x": float, "y": float,
                   "yaw": float (radians, 0 = +x, positive from +x toward +y),
                   "speed": float (m/s), "role": "agent" | "traffic"}]}

``y_up`` states the handedness of the world frame as seen from above:
``true`` when +y points up with +x to the right (a right-handed map frame,
as in SUMO), ``false`` when +y points down (a left-handed frame, as in
CARLA). The page flips the drawing to match, so a backend that leaves it
out is drawn as ``y_up: true`` and a left-handed one would come out
mirrored. ``yaw`` is measured in the same frame, so it needs no conversion
beyond radians.

``wall`` is the writer's ``time.time()``; the page orders states by it, so
a relaunch whose episode and step counters restart from zero still counts
as newer. ``interval`` is the publish throttle in seconds (absent in files
of older runs). The page keeps the last frame on screen but dims it once
no newer one came for ``max(3 s, 3 * interval)`` ("stale"), or as soon as
the run's metrics store was closed after it (``closed_at`` in the store
meta: "run ended"), so a frozen picture never looks live.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

GEOMETRY_NAME = "traffic_geometry.json"
STATE_NAME = "traffic_state.json"


_REPLACE_ATTEMPTS = 5
_REPLACE_PAUSE_S = 0.015


def _write_atomic(path: Path, payload: Dict[str, Any]) -> bool:
    """Write ``payload`` to a temporary file and move it over ``path``.

    On Windows the move fails with ``PermissionError`` while a reader (the
    monitor serving the file) has ``path`` open, so it is retried a few
    times; when the reader still holds it, this frame is skipped -- the next
    due tick writes a newer one -- and False is returned. Errors writing the
    temporary file propagate."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError as exc:
            if attempt + 1 == _REPLACE_ATTEMPTS:
                logger.debug(f"traffic view: {path.name} held by a reader, frame skipped: {exc}")
                return False
            time.sleep(_REPLACE_PAUSE_S)
    return False


class TrafficViewPublisher:
    """Throttled writer of the two view files into one run directory."""

    def __init__(self, run_dir: os.PathLike | str, *, interval: float = 0.2):
        self.run_dir = Path(run_dir)
        self.interval = max(0.05, float(interval))
        self._last_publish = 0.0
        self._geometry_written = False
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def publish_geometry(self, geometry: Optional[Dict[str, Any]]) -> None:
        """Write the static lane geometry (idempotent; first call wins)."""
        if self._geometry_written or not geometry or not geometry.get("lanes"):
            return
        try:
            if not _write_atomic(self.run_dir / GEOMETRY_NAME, geometry):
                return                       # held by a reader: retried next tick
            self._geometry_written = True
            logger.info(f"Traffic view geometry written: "
                        f"{len(geometry['lanes'])} lanes -> {self.run_dir / GEOMETRY_NAME}")
        except OSError as exc:
            logger.warning(f"traffic view geometry write failed: {exc}")

    def invalidate_geometry(self) -> None:
        """Force a re-write on the next publish (after a world reload)."""
        self._geometry_written = False

    @property
    def geometry_written(self) -> bool:
        return self._geometry_written

    def due(self) -> bool:
        """True when the throttle interval elapsed — check BEFORE collecting
        vehicle states, so non-publishing steps cost nothing."""
        return time.monotonic() - self._last_publish >= self.interval

    def publish_state(self, vehicles: list, *, episode: int, step: int) -> bool:
        """Write the vehicle poses if the throttle interval elapsed.

        Returns True when a write happened — callers may lazily collect
        geometry only on publishing ticks.
        """
        now = time.monotonic()
        if now - self._last_publish < self.interval:
            return False
        self._last_publish = now
        try:
            return _write_atomic(self.run_dir / STATE_NAME, {
                "episode": int(episode),
                "step": int(step),
                "wall": time.time(),
                "interval": self.interval,
                "vehicles": vehicles,
            })
        except OSError as exc:
            logger.warning(f"traffic view state write failed: {exc}")
            return False
