"""Per-step snapshot of every vehicle's kinematics — one server round trip.

Observation assembly used to be O(N^2) in RPCs: each of N vehicles fetched
the full actor list *twice* (nearby search + front-vehicle search) and then
queried location/velocity/transform of every neighbour, with TTC fetching
them all again — roughly 1,600 round trips per step at 40 vehicles. The
snapshot fetches the actor list once, reads each vehicle's kinematics once,
and serves every consumer from NumPy arrays; a step's worth of observation
math becomes a few vectorised operations.

Semantics are unchanged: the snapshot is taken at the start of observation
assembly, so all vehicles observe the same instant (previously neighbours
were sampled at slightly different moments across the loop, which was never
intended).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

import numpy as np


@dataclass
class VehicleKinematics:
    """Kinematic state of one vehicle at the snapshot instant.

    Carries the CARLA-native objects read once at capture (``transform``,
    ``velocity``, ``bounding_box``) so downstream consumers — including the
    vendored perception layer via ``PerceptionManager.set_world_snapshot`` —
    never re-query the server for them.
    """

    actor_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    yaw_rad: float
    type_id: str = ""
    transform: Any = field(default=None, repr=False)
    velocity: Any = field(default=None, repr=False)
    bounding_box: Any = field(default=None, repr=False)
    #: Original actor handle for consumers that genuinely need it.
    actor: Any = field(default=None, repr=False)

    @property
    def location(self):
        return self.transform.location if self.transform is not None else None


class WorldSnapshot:
    """All live vehicles' kinematics captured in one pass.

    Build once per step with :meth:`capture`; query with :meth:`nearby`,
    :meth:`front_vehicle_distance` and :attr:`by_id`. Missing/dead actors are
    skipped at capture time so consumers never touch a stale handle.
    """

    #: Monotonic id per capture; consumers memoize per-tick work on it.
    _seq = 0

    def __init__(self, vehicles: Sequence[VehicleKinematics]):
        WorldSnapshot._seq += 1
        self.seq: int = WorldSnapshot._seq
        self.vehicles: List[VehicleKinematics] = list(vehicles)
        self.by_id: Dict[int, VehicleKinematics] = {v.actor_id: v for v in self.vehicles}
        n = len(self.vehicles)
        self.ids = np.fromiter((v.actor_id for v in self.vehicles), dtype=np.int64, count=n)
        self.pos = np.array([(v.x, v.y) for v in self.vehicles], dtype=np.float64).reshape(n, 2)
        self.vel = np.array([(v.vx, v.vy) for v in self.vehicles], dtype=np.float64).reshape(n, 2)
        self.yaw = np.fromiter((v.yaw_rad for v in self.vehicles), dtype=np.float64, count=n)
        self._row_of: Dict[int, int] = {int(v.actor_id): row for row, v in enumerate(self.vehicles)}
        self._blocks_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def capture(cls, world) -> "WorldSnapshot":
        """One ``get_actors`` + one kinematics read per vehicle."""
        records: List[VehicleKinematics] = []
        try:
            actors = world.get_actors().filter("vehicle.*")
        except Exception:
            return cls([])
        deg2rad = float(np.pi) / 180.0
        skipped = 0
        for actor in actors:
            try:
                if not actor.is_alive:
                    continue
                transform = actor.get_transform()
                velocity = actor.get_velocity()
                loc = transform.location
                records.append(VehicleKinematics(
                    actor_id=int(actor.id),
                    x=float(loc.x), y=float(loc.y), z=float(loc.z),
                    vx=float(velocity.x), vy=float(velocity.y),
                    yaw_rad=float(transform.rotation.yaw) * deg2rad,
                    type_id=actor.type_id,
                    transform=transform,
                    velocity=velocity,
                    # Only the perception layer needs the box; tolerate
                    # actors (or test doubles) without one.
                    bounding_box=getattr(actor, "bounding_box", None),
                    actor=actor,
                ))
            except Exception as exc:
                # An actor that died between listing and read is expected;
                # anything else silently vanishing from every other vehicle's
                # view is not, so say so.
                skipped += 1
                logger.debug(f"WorldSnapshot: skipped actor "
                             f"{getattr(actor, 'id', '?')}: {exc}")
        if skipped:
            logger.warning(f"WorldSnapshot.capture skipped {skipped} actor(s) "
                           "that failed to read (see debug log)")
        return cls(records)

    def publish_to_perception(self) -> None:
        """Expose this snapshot to the vendored perception layer for the
        current step (see ``PerceptionManager.set_world_snapshot``)."""
        from opencda.core.sensing.perception.perception_manager import PerceptionManager

        PerceptionManager.set_world_snapshot(self.by_id)

    @staticmethod
    def unpublish_from_perception() -> None:
        from opencda.core.sensing.perception.perception_manager import PerceptionManager

        PerceptionManager.clear_world_snapshot()

    def __len__(self) -> int:
        return len(self.vehicles)

    # ------------------------------------------------------------------ #
    def index_of(self, actor_id: int) -> int:
        """Row index of *actor_id* in the arrays, or -1."""
        return self._row_of.get(int(actor_id), -1)

    def nearby(self, actor_id: int, radius: float, k: int) -> List[Tuple[int, float]]:
        """The k nearest other vehicles within *radius*: ``[(row, distance)]``.

        Uses a bounding-box prefilter then exact distance, matching the
        previous scalar implementation's semantics.
        """
        i = self.index_of(actor_id)
        if i < 0 or len(self) < 2 or k <= 0:
            return []
        delta = self.pos - self.pos[i]
        box = (np.abs(delta[:, 0]) <= radius) & (np.abs(delta[:, 1]) <= radius)
        box[i] = False
        if not box.any():
            return []
        candidates = np.nonzero(box)[0]
        dist = np.hypot(delta[candidates, 0], delta[candidates, 1])
        keep = dist <= radius
        candidates, dist = candidates[keep], dist[keep]
        if candidates.size == 0:
            return []
        order = np.argsort(dist, kind="stable")[:k]
        return [(int(candidates[j]), float(dist[j])) for j in order]

    def ttc(self, i: int, j: int, collision_radius: float = 4.0) -> float:
        """Constant-velocity time-to-collision between rows i and j (inf if none)."""
        rel = self.pos[j] - self.pos[i]
        rel_v = self.vel[j] - self.vel[i]
        a = float(rel_v @ rel_v)
        if abs(a) < 1e-6:
            return float("inf")
        b = 2.0 * float(rel @ rel_v)
        c = float(rel @ rel) - collision_radius ** 2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return float("inf")
        sq = float(np.sqrt(disc))
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        if t1 > 0:
            return t1
        if t2 > 0:
            return t2
        return float("inf")

    def front_vehicle_distance(
        self,
        actor_id: int,
        carla_map,
        max_distance: float = 100.0,
        default: float = 999.0,
    ) -> float:
        """Distance to the nearest vehicle ahead in the same road+lane.

        Waypoint lookups (the only per-neighbour map query left) are done
        only for vehicles inside *max_distance*, exactly as before, but the
        candidate set now comes from the snapshot instead of an RPC per
        neighbour.
        """
        i = self.index_of(actor_id)
        if i < 0:
            return default
        ego = self.vehicles[i]
        ego_loc = ego.location
        if ego_loc is None:
            return default
        try:
            ego_wp = carla_map.get_waypoint(ego_loc)
        except Exception:
            return default
        if not ego_wp:
            return default

        delta = self.pos - self.pos[i]
        dist = np.hypot(delta[:, 0], delta[:, 1])
        dist[i] = np.inf
        candidates = np.nonzero(dist <= max_distance)[0]
        if candidates.size == 0:
            return default

        forward = np.array([np.cos(ego.yaw_rad), np.sin(ego.yaw_rad)])
        best = max_distance
        found = False
        for j in candidates[np.argsort(dist[candidates])]:
            if float(delta[j] @ forward) <= 0:
                continue  # behind
            other = self.vehicles[j]
            other_loc = other.location
            if other_loc is None:
                continue
            try:
                wp = carla_map.get_waypoint(other_loc)
            except Exception:
                continue
            if wp and wp.lane_id == ego_wp.lane_id and wp.road_id == ego_wp.road_id:
                if dist[j] < best:
                    best = float(dist[j])
                    found = True
                break  # sorted by distance: first same-lane hit ahead is nearest
        return best if found else default

    def neighbour_features(
        self,
        actor_id: int,
        *,
        radius: float,
        k: int,
        max_relative_velocity: float,
        max_ttc: float,
        features_per_vehicle: int = 7,
    ) -> Tuple[List[float], float]:
        """K-slot normalised neighbour feature vector + min TTC.

        Layout per slot: rel_x, rel_y, rel_vx, rel_vy, heading_diff,
        distance, ttc — identical to the pre-snapshot implementation, so
        trained checkpoints read the same vector.

        Served from :meth:`neighbour_blocks`: the first ego of a step pays
        for one vectorised pass over every pair, the others index into it.
        The per-ego loop this replaced made four TTC solves per call;
        :meth:`_neighbour_features_scalar` keeps it as the reference the
        batched pass is tested against.
        """
        i = self.index_of(actor_id)
        if i < 0:
            return _EMPTY_SLOT * k, float("inf")
        blocks, min_ttc = self.neighbour_blocks(
            radius=radius, k=k, max_relative_velocity=max_relative_velocity,
            max_ttc=max_ttc, features_per_vehicle=features_per_vehicle)
        return blocks[i].tolist(), float(min_ttc[i])

    def neighbour_blocks(
        self,
        *,
        radius: float,
        k: int,
        max_relative_velocity: float,
        max_ttc: float,
        features_per_vehicle: int = 7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Every vehicle's neighbour block at once: ``(blocks, min_ttc)`` with
        ``blocks[row]`` the ``k * 7`` features of that row's vehicle and
        ``min_ttc[row]`` its minimum TTC over the selected neighbours.
        Cached per snapshot and parameter set."""
        key = (float(radius), int(k), float(max_relative_velocity), float(max_ttc), int(features_per_vehicle))
        hit = self._blocks_cache.get(key)
        if hit is None:
            hit = self._compute_neighbour_blocks(*key[:4])
            self._blocks_cache[key] = hit
        return hit

    def _compute_neighbour_blocks(
        self, radius: float, k: int, max_relative_velocity: float, max_ttc: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The vectorised equivalent of :meth:`_neighbour_features_scalar`
        for every row: the k nearest others within *radius*, ascending
        distance, ties by ascending row, then the same normalised features
        and the same constant-velocity TTC."""
        n = len(self)
        blocks = np.tile(np.asarray(_EMPTY_SLOT, dtype=np.float64), (n, k))
        min_ttc = np.full(n, np.inf)
        if n < 2 or k <= 0:
            return blocks, min_ttc
        pos, vel, yaw = self.pos, self.vel, self.yaw
        delta = pos[None, :, :] - pos[:, None, :]              # [i, j] = pos_j - pos_i
        dist = np.hypot(delta[..., 0], delta[..., 1])
        eligible = dist <= radius
        np.fill_diagonal(eligible, False)
        count = np.minimum(eligible.sum(axis=1), k)
        if not count.any():
            return blocks, min_ttc
        # Ascending distance with ties by ascending row: the stable argsort
        # over a row whose excluded entries are +inf.
        order = np.argsort(np.where(eligible, dist, np.inf), axis=1, kind="stable")[:, :k]
        width = order.shape[1]
        slot = np.arange(width)[None, :]
        filled = slot < count[:, None]                          # [i, slot] holds a neighbour
        rows = np.arange(n)[:, None]
        rel = delta[rows, order]                                # [i, slot, 2]
        rel_v = vel[order] - vel[:, None, :]
        heading = yaw[order] - yaw[:, None]
        heading = (heading + np.pi) % (2.0 * np.pi) - np.pi
        distance = dist[rows, order]
        ttc = _ttc_pairs(rel, rel_v)
        features = np.stack([
            np.clip(rel[..., 0] / radius, -1.0, 1.0),
            np.clip(rel[..., 1] / radius, -1.0, 1.0),
            np.clip(rel_v[..., 0] / max_relative_velocity, -1.0, 1.0),
            np.clip(rel_v[..., 1] / max_relative_velocity, -1.0, 1.0),
            np.clip(heading / np.pi, -1.0, 1.0),
            np.clip(distance / radius, 0.0, 1.0),
            np.clip(np.minimum(ttc, max_ttc) / max_ttc, 0.0, 1.0),
        ], axis=-1)                                             # [i, slot, 7]
        empty = np.asarray(_EMPTY_SLOT, dtype=np.float64)
        features = np.where(filled[..., None], features, empty)
        blocks[:, :width * 7] = features.reshape(n, width * 7)
        min_ttc = np.where(filled, ttc, np.inf).min(axis=1)
        return blocks, min_ttc

    def _neighbour_features_scalar(
        self,
        actor_id: int,
        *,
        radius: float,
        k: int,
        max_relative_velocity: float,
        max_ttc: float,
        features_per_vehicle: int = 7,
    ) -> Tuple[List[float], float]:
        """The per-ego reference implementation of :meth:`neighbour_features`."""
        empty_slot = list(_EMPTY_SLOT)
        i = self.index_of(actor_id)
        if i < 0:
            return empty_slot * k, float("inf")

        neighbours = self.nearby(actor_id, radius, k)
        features: List[float] = []
        min_ttc = float("inf")
        # Plain scalar clamps: np.clip on a Python float goes through NumPy's
        # generic dispatch (~7 us each); at 35 clips per vehicle per step
        # that was the single largest Python-side cost in the profile.
        pi = float(np.pi)
        pos_i = self.pos[i]
        vel_i = self.vel[i]
        yaw_i = float(self.yaw[i])
        for slot in range(k):
            if slot >= len(neighbours):
                features.extend(empty_slot)
                continue
            j, distance = neighbours[slot]
            rel_x = float(self.pos[j, 0] - pos_i[0]) / radius
            rel_y = float(self.pos[j, 1] - pos_i[1]) / radius
            rel_vx = float(self.vel[j, 0] - vel_i[0]) / max_relative_velocity
            rel_vy = float(self.vel[j, 1] - vel_i[1]) / max_relative_velocity
            heading_diff = float(self.yaw[j]) - yaw_i
            heading_diff = (heading_diff + pi) % (2.0 * pi) - pi
            ttc = self.ttc(i, j)
            if ttc < min_ttc:
                min_ttc = ttc
            features.extend([
                _clamp(rel_x, -1.0, 1.0),
                _clamp(rel_y, -1.0, 1.0),
                _clamp(rel_vx, -1.0, 1.0),
                _clamp(rel_vy, -1.0, 1.0),
                _clamp(heading_diff / pi, -1.0, 1.0),
                _clamp(distance / radius, 0.0, 1.0),
                _clamp(min(ttc, max_ttc) / max_ttc, 0.0, 1.0),
            ])
        return features, min_ttc


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


_EMPTY_SLOT = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def _ttc_pairs(rel: np.ndarray, rel_v: np.ndarray, collision_radius: float = 4.0) -> np.ndarray:
    """Constant-velocity time-to-collision for arrays of relative position
    and velocity (last axis x, y), inf where none: the same quadratic as
    :meth:`WorldSnapshot.ttc`, evaluated term by term in the same order."""
    rvx, rvy = rel_v[..., 0], rel_v[..., 1]
    rx, ry = rel[..., 0], rel[..., 1]
    a = rvx * rvx + rvy * rvy
    b = 2.0 * (rx * rvx + ry * rvy)
    c = (rx * rx + ry * ry) - collision_radius ** 2
    disc = b * b - 4.0 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
    ttc = np.where(t1 > 0, t1, np.where(t2 > 0, t2, np.inf))
    ttc = np.where(disc < 0, np.inf, ttc)
    return np.where(np.abs(a) < 1e-6, np.inf, ttc)


__all__ = ["WorldSnapshot", "VehicleKinematics"]
