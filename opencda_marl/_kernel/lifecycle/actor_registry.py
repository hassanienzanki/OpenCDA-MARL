"""Actor lifecycle registry: the single owner of CARLA actor destruction.

Every RL vehicle actor is registered at spawn time — before any adapter or
sensor construction — so a failure anywhere later can never orphan it. Sensor
actor ids are adopted per vehicle once the vehicle's managers exist. Teardown
is verified by reconciliation against the server (``world.get_actors``)
instead of trusting Python-side bookkeeping: still-alive actors are destroyed
and reported.

Reconciliation modes:

- warn (default): destroy stragglers, log a warning, keep going.
- strict: additionally raise :class:`ActorLeakError` after the sweep — for
  runs whose data must not come from a leaking process.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import carla
from loguru import logger


class SpawnCollisionError(RuntimeError):
    """Spawn failed because the spawn point was occupied (retryable)."""


class ActorLeakError(RuntimeError):
    """Reconciliation found live actors after teardown (strict mode only)."""


class ActorRegistry:
    """Tracks every spawned RL vehicle (and its sensors) and guarantees
    destruction by verifying against the simulator server."""

    def __init__(self, world, client=None, *, strict: bool = False):
        self.world = world
        self.client = client
        self.strict = bool(strict)
        # vehicle_id -> {"vehicle": actor, "sensor_ids": [int, ...]}
        self._entries: Dict[int, Dict[str, Any]] = {}
        # Entries whose destroy path ran but whose disappearance the server
        # has not yet been asked to confirm. In synchronous mode a destroy
        # only materializes after the next tick, so verifying immediately
        # reports every actor as a false-positive leak (and the redundant
        # re-destroy fails with "not found"). Verification is therefore
        # deferred to sweep_pending(), called after an intervening tick.
        self._pending_verify: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "spawned": 0,
            "destroyed": 0,
            "aborted": 0,
            "leaks_reclaimed": 0,
        }

    # ------------------------------------------------------------------ #
    # Spawn side
    # ------------------------------------------------------------------ #
    def spawn_vehicle(self, blueprint, transform):
        """Spawn and register a vehicle actor in one step.

        Registration happens before this method returns, so the registry can
        always find the actor no matter what fails afterwards. The CARLA
        occupied-spawn-point error is translated to
        :class:`SpawnCollisionError` (the one retryable failure); everything
        else propagates unchanged.
        """
        try:
            vehicle = self.world.spawn_actor(blueprint, transform)
        except Exception as exc:
            if "collision at spawn" in str(exc).lower():
                raise SpawnCollisionError(str(exc)) from exc
            raise
        self._entries[int(vehicle.id)] = {"vehicle": vehicle, "sensor_ids": []}
        self.stats["spawned"] += 1
        return vehicle

    def adopt_sensors(self, vehicle_id: int, sensor_ids: Iterable[int]) -> None:
        """Record the sensor actors owned by an already-registered vehicle."""
        entry = self._entries.get(int(vehicle_id))
        if entry is None:
            logger.warning(f"adopt_sensors: vehicle {vehicle_id} is not registered")
            return
        entry["sensor_ids"] = [int(s) for s in sensor_ids]

    def abort_spawn(self, vehicle) -> None:
        """Destroy a just-spawned vehicle whose setup failed.

        Sensors may already have been attached by partially-constructed
        managers, so they are discovered from the server (by parent id)
        rather than from Python-side objects that may not exist.
        """
        vehicle_id = int(getattr(vehicle, "id", -1))
        destroyed_sensors = 0
        try:
            for actor in self.world.get_actors().filter("sensor.*"):
                parent = getattr(actor, "parent", None)
                if parent is not None and int(getattr(parent, "id", -1)) == vehicle_id:
                    self._stop_and_destroy(actor)
                    destroyed_sensors += 1
        except Exception as exc:
            logger.warning(
                f"abort_spawn: sensor sweep for vehicle {vehicle_id} failed: {exc}")
        self._stop_and_destroy(vehicle)
        self._entries.pop(vehicle_id, None)
        self.stats["aborted"] += 1
        logger.warning(
            f"Aborted spawn of vehicle {vehicle_id}: destroyed the actor and "
            f"{destroyed_sensors} attached sensor(s) after a setup failure")

    # ------------------------------------------------------------------ #
    # Destroy side
    # ------------------------------------------------------------------ #
    def tracked_vehicle_ids(self) -> List[int]:
        return list(self._entries.keys())

    def finalize_vehicle(self, vehicle_id: int) -> None:
        """Mark one vehicle's teardown complete, pending server verification.

        The owner's destroy chain just ran; the entry is queued for
        verification by the next :meth:`sweep_pending` (after a world tick),
        which destroys and reports anything that genuinely survived.
        """
        entry = self._entries.pop(int(vehicle_id), None)
        if entry is None:
            return
        self._pending_verify.append(entry)
        self.stats["destroyed"] += 1

    def destroy_all(self) -> None:
        """Batch-destroy every tracked vehicle and queue verification.

        Sensor teardown belongs to the per-vehicle managers and must run
        before this call. The caller should tick the world afterwards and
        then call :meth:`sweep_pending` — in synchronous mode destruction
        only materializes on the next tick.
        """
        entries = list(self._entries.values())
        vehicle_ids = list(self._entries.keys())
        if self.client is not None and vehicle_ids:
            try:
                responses = self.client.apply_batch_sync(
                    [carla.command.DestroyActor(vid) for vid in vehicle_ids],
                    False,
                )
                for vid, response in zip(vehicle_ids, responses):
                    if getattr(response, "error", None):
                        logger.warning(
                            f"Batch destroy failed for actor {vid}: {response.error}")
            except Exception as exc:
                logger.warning(
                    "Batch vehicle destroy failed; the verification sweep "
                    f"will destroy individually: {exc}")
        self._pending_verify.extend(entries)
        self.stats["destroyed"] += len(vehicle_ids)
        self._entries.clear()

    #: Sweeps an entry must wait before verification. The world snapshot
    #: behind ``get_actors`` can lag destroy processing by a full tick, so
    #: verifying too early reports actors that are already dying.
    VERIFY_AFTER_SWEEPS = 2

    def sweep_pending(self, *, force: bool = False) -> List[int]:
        """Verify aged teardowns against the server; reclaim true leaks.

        Entries are verified once :attr:`VERIFY_AFTER_SWEEPS` sweeps (one per
        step) have passed since their destroy chain ran, so the server
        snapshot has definitely refreshed. ``force=True`` (cleanup) verifies
        everything immediately. A genuine leak is an actor whose ``destroy()``
        here returns True — we actually killed something the owner chain
        missed; already-dead actors that the snapshot still lists are not
        leaks. Returns the genuinely leaked ids (raising in strict mode).
        """
        if not self._pending_verify:
            return []
        due: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for entry in self._pending_verify:
            entry["age"] = entry.get("age", 0) + 1
            if force or entry["age"] >= self.VERIFY_AFTER_SWEEPS:
                due.append(entry)
            else:
                remaining.append(entry)
        self._pending_verify = remaining
        if not due:
            return []
        # One full-list snapshot per sweep: get_actors(ids) is a registry
        # lookup that still returns dead actors, but the enumerated snapshot
        # only contains the genuinely alive — so no destroy RPC is wasted on
        # actors that are already gone.
        try:
            snapshot = {int(a.id): a for a in self.world.get_actors()}
        except Exception as exc:
            logger.warning(f"Verification snapshot query failed: {exc}")
            snapshot = {}
        leaked: List[int] = []
        for entry in due:
            leaked.extend(self._reclaim_entry(entry, snapshot))
        if leaked:
            self._report_leak(leaked, context="post-tick verification")
        return leaked

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _reclaim_entry(self, entry: Dict[str, Any],
                       snapshot: Dict[int, Any]) -> List[int]:
        """Destroy any of the entry's actors present in the live snapshot.

        Sensors go first (children before parent). ``destroy()`` returning
        True is the leak criterion — a race can still hand us an actor that
        just died, in which case destroy returns False and it is not a leak.
        """
        vehicle = entry.get("vehicle")
        vehicle_id = int(getattr(vehicle, "id", -1))
        candidate_ids = [int(s) for s in entry.get("sensor_ids", [])] + [vehicle_id]
        listed = [snapshot[cid] for cid in candidate_ids if cid in snapshot]
        # children before parent
        ordered = [a for a in listed if int(a.id) != vehicle_id] + \
                  [a for a in listed if int(a.id) == vehicle_id]
        leaked: List[int] = []
        for actor in ordered:
            if self._stop_and_destroy(actor):
                leaked.append(int(actor.id))
                self.stats["leaks_reclaimed"] += 1
            else:
                logger.debug(
                    f"Actor {getattr(actor, 'id', '?')} was already gone "
                    "(snapshot race) — not a leak")
        return leaked

    @staticmethod
    def _stop_and_destroy(actor) -> bool:
        """Best-effort stop+destroy. True only when destroy() reports it
        actually destroyed a live actor."""
        if actor is None:
            return False
        try:
            if not getattr(actor, "is_alive", True):
                return False
            if str(getattr(actor, "type_id", "")).startswith("sensor") and \
                    hasattr(actor, "stop"):
                try:
                    actor.stop()
                except Exception as exc:
                    logger.warning(f"sensor.stop failed for actor "
                                   f"{getattr(actor, 'id', '?')}: {exc}")
            return bool(actor.destroy())
        except Exception as exc:
            logger.warning(
                f"destroy failed for actor {getattr(actor, 'id', '?')}: {exc}")
            return False

    def _report_leak(self, leaked_ids: List[int], *, context: str) -> None:
        message = (
            f"ActorRegistry reclaimed {len(leaked_ids)} leaked actor(s) during "
            f"{context}: {leaked_ids}. The normal teardown path failed to "
            "destroy them — investigate the owning destroy chain.")
        if self.strict:
            raise ActorLeakError(message)
        logger.warning(message)
