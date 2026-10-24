'''
Author       : AXIBA leolihao@arizona.edu
FilePath     : /OpenCDA-MARL/opencda_marl/api/route_intent.py
Description  : Semantic route-intent representation, candidate generation, and
               legal feasibility mask construction for post-spawn CAV control.
               Designed for one-shot post-spawn semantic intent selection
               (left / straight / right / keep-current) using the existing
               MARLPlanner route metadata as the route oracle.
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import carla
except ImportError:  # SUMO-only installs: the sumo_adapter shims stand in
    from opencda_marl._kernel.traffic import sumo_adapter as carla


class RouteIntent(str, Enum):
    LEFT = "left"
    STRAIGHT = "straight"
    RIGHT = "right"
    KEEP = "keep-current"


SEMANTIC_TURN_INTENTS: Sequence[RouteIntent] = (
    RouteIntent.LEFT,
    RouteIntent.STRAIGHT,
    RouteIntent.RIGHT,
)
ALL_INTENTS: Sequence[RouteIntent] = (
    RouteIntent.LEFT,
    RouteIntent.STRAIGHT,
    RouteIntent.RIGHT,
    RouteIntent.KEEP,
)


def _intent_from_route_type(route_type: str) -> Optional[RouteIntent]:
    if route_type == "left":
        return RouteIntent.LEFT
    if route_type == "straight":
        return RouteIntent.STRAIGHT
    if route_type == "right":
        return RouteIntent.RIGHT
    return None


@dataclass(frozen=True)
class RouteIntentCandidate:
    intent: RouteIntent
    route_id: int
    junction_id: int
    dest_tf: carla.Transform
    exit_direction: str
    exit_lane_idx: int
    is_original: bool
    route_type: str = ""
    same_exit_direction: bool = True

    @property
    def same_destination(self) -> bool:
        """Backward-compatible alias for the former field name.

        ``same_exit_direction`` is the current name; ``same_destination``
        is the former name for the same exit-direction invariant.
        """
        return self.same_exit_direction


@dataclass(frozen=True)
class CandidateTrafficFeatures:
    """Traffic pressure around one concrete route candidate."""

    active_same_exit_direction_count: int = 0
    active_same_exit_lane_count: int = 0
    queued_same_exit_lane_count: int = 0
    same_route_count: int = 0
    nearest_same_exit_lane_distance_m: float = 999.0
    min_same_exit_lane_ttc_s: float = 10.0

    def as_metadata(self) -> Dict[str, float]:
        return {
            "active_same_exit_direction_count": float(self.active_same_exit_direction_count),
            "active_same_exit_lane_count": float(self.active_same_exit_lane_count),
            "queued_same_exit_lane_count": float(self.queued_same_exit_lane_count),
            "same_route_count": float(self.same_route_count),
            "nearest_same_exit_lane_distance_m": float(self.nearest_same_exit_lane_distance_m),
            "min_same_exit_lane_ttc_s": float(self.min_same_exit_lane_ttc_s),
        }

    def to_feature_vector(
        self,
        *,
        max_active_agents: float = 10.0,
        max_distance_m: float = 50.0,
        max_ttc_s: float = 10.0,
        pressure_score: float = 0.0,
        max_pressure_score: float = 20.0,
    ) -> List[float]:
        denom = max(1.0, float(max_active_agents))
        distance = min(float(self.nearest_same_exit_lane_distance_m), max_distance_m)
        ttc = min(float(self.min_same_exit_lane_ttc_s), max_ttc_s)
        return [
            min(1.0, float(self.active_same_exit_direction_count) / denom),
            min(1.0, float(self.active_same_exit_lane_count) / denom),
            min(1.0, float(self.queued_same_exit_lane_count) / denom),
            min(1.0, float(self.same_route_count) / denom),
            min(1.0, distance / max_distance_m),
            min(1.0, ttc / max_ttc_s),
            min(1.0, float(pressure_score) / max_pressure_score),
        ]


@dataclass
class RouteIntentDecision:
    selected_intent: RouteIntent
    mask: Dict[RouteIntent, bool]
    selected_candidate: RouteIntentCandidate
    original_route_id: int
    used_keep_current: bool
    original_exit_direction: str = ""
    candidate_count_total: int = 0
    candidate_count_same_exit_direction: int = 0
    mask_sparsity: float = 0.0
    direction_preserved: bool = True
    fallback_reason: str = ""
    selected_candidate_score: float = 0.0
    selected_candidate_traffic: Dict[str, float] | None = None
    original_candidate_traffic: Dict[str, float] | None = None
    candidate_resolution: str = "deterministic"
    #: Which path action produced this decision. The built-in turn action
    #: writes ``semantic``; a registered provider tags its own.
    interface: str = "semantic"

    @property
    def admissible(self) -> bool:
        """Whether the lifecycle may execute this decision.

        The framework's own path providers keep the demand exit direction,
        so that is their invariant. A provider with different semantics
        implements this property differently; the lifecycle only asks
        (:mod:`opencda_marl.api.path_action`).
        """
        return bool(self.direction_preserved)

    @property
    def destination(self):
        """The transform the vehicle should be routed to."""
        return getattr(self.selected_candidate, "dest_tf", None)

    @property
    def candidate_count_same_destination(self) -> int:
        """Backward-compatible alias for the former field name."""
        return self.candidate_count_same_exit_direction

    @property
    def same_destination_preserved(self) -> bool:
        """Backward-compatible alias for the former field name."""
        return self.direction_preserved

    def as_metadata(self) -> Dict[str, Any]:
        raw_rt = (self.selected_candidate.route_type or "").lower()
        route_type = raw_rt if raw_rt in {"left", "straight", "right"} else "other"
        same_exit_count = int(self.candidate_count_same_exit_direction)
        direction_preserved = bool(self.direction_preserved)
        return {
            "selected_intent": self.selected_intent.value,
            "mask": {k.value: bool(v) for k, v in self.mask.items()},
            "selected_junction_id": int(self.selected_candidate.junction_id),
            "selected_route_id": int(self.selected_candidate.route_id),
            "selected_exit_direction": self.selected_candidate.exit_direction,
            "selected_exit_lane_idx": int(self.selected_candidate.exit_lane_idx),
            "selected_route_type": route_type,
            "original_route_id": int(self.original_route_id),
            "original_exit_direction": self.original_exit_direction,
            "used_keep_current": bool(self.used_keep_current),
            "candidate_count_total": int(self.candidate_count_total),
            "candidate_count_same_exit_direction": same_exit_count,
            "candidate_count_same_destination": same_exit_count,
            "mask_sparsity": float(self.mask_sparsity),
            "direction_preserved": direction_preserved,
            "same_exit_direction_preserved": direction_preserved,
            "same_destination_preserved": direction_preserved,
            "admissible": bool(self.admissible),
            "fallback_reason": self.fallback_reason,
            "selected_candidate_score": float(self.selected_candidate_score),
            "selected_candidate_traffic": dict(self.selected_candidate_traffic or {}),
            "original_candidate_traffic": dict(self.original_candidate_traffic or {}),
            "route_changed": int(self.selected_candidate.route_id) != int(self.original_route_id),
            "candidate_resolution": self.candidate_resolution,
            "interface": self.interface,
        }


def _same_exit_direction(route: Dict[str, Any], original: Dict[str, Any]) -> bool:
    """Return whether two route records preserve the same exit direction.

    The route action keeps the spawn entry and requested exit direction fixed.
    Different exit lanes or concrete destination transforms within that exit
    direction are valid optimization choices; changing the exit direction is
    outside the benchmark contract.
    """
    original_exit = str(original.get("exit_direction", "") or "")
    route_exit = str(route.get("exit_direction", "") or "")
    return bool(original_exit) and route_exit == original_exit


def _same_destination_region(route: Dict[str, Any], original: Dict[str, Any]) -> bool:
    """Compatibility wrapper for older call sites."""
    return _same_exit_direction(route, original)


def _mask_sparsity(mask: Dict[RouteIntent, bool]) -> float:
    legal = sum(1 for ok in mask.values() if ok)
    return 1.0 - (float(legal) / float(len(ALL_INTENTS)))


def build_candidates_with_diagnostics(
    planner: Any,
    junction_id: int,
    entry_direction: str,
    entry_lane_idx: int,
    original_route_id: int,
) -> Tuple[List[RouteIntentCandidate], Dict[str, Any]]:
    """Build the legal candidate set for a vehicle that just spawned.

    Candidates are derived from the existing MARLPlanner route table:
    - Filter by matching ``entry_direction`` and ``entry_lane_idx``.
    - Keep only planner-feasible alternatives that preserve the original exit
      direction. The concrete exit lane, destination transform, and route id
      may change within that same direction.
    - Map each remaining non-original route's ``route_type`` to a semantic
      intent (left/straight/right). U-turn and unknown route types are skipped.
    - Always add the original ``route_id`` as the ``KEEP`` candidate so that
      ``keep-current`` is legal by definition.
    - If no same-exit-direction alternatives exist, the mask built from these
      candidates collapses to ``KEEP`` only.
    """
    routes = planner.get_routes(junction_id) or {}
    candidates: List[RouteIntentCandidate] = []
    original = routes.get(int(original_route_id))
    original_exit_direction = str(
        (original or {}).get("exit_direction", "") or ""
    )
    considered_total = 0
    changed_exit_direction_count = 0
    same_exit_direction_alternatives = 0

    for route_id, route in routes.items():
        if route.get("entry_direction") != entry_direction:
            continue
        if route.get("entry_lane_idx") != entry_lane_idx:
            continue
        if int(route_id) == int(original_route_id):
            continue
        intent = _intent_from_route_type(route.get("route_type", ""))
        if intent is None:
            continue
        considered_total += 1
        same_exit_direction = (
            original is not None and _same_exit_direction(route, original)
        )
        if not same_exit_direction:
            changed_exit_direction_count += 1
            continue
        same_exit_direction_alternatives += 1
        candidates.append(
            RouteIntentCandidate(
                intent=intent,
                route_id=int(route_id),
                junction_id=int(junction_id),
                dest_tf=route["dest_tf"],
                exit_direction=str(route.get("exit_direction", "")),
                exit_lane_idx=int(route.get("exit_lane_idx", -1)),
                is_original=False,
                route_type=str(route.get("route_type", "")),
                same_exit_direction=True,
            )
        )

    if original is not None:
        candidates.append(
            RouteIntentCandidate(
                intent=RouteIntent.KEEP,
                route_id=int(original_route_id),
                junction_id=int(junction_id),
                dest_tf=original["dest_tf"],
                exit_direction=str(original.get("exit_direction", "")),
                exit_lane_idx=int(original.get("exit_lane_idx", -1)),
                is_original=True,
                route_type=str(original.get("route_type", "")),
                same_exit_direction=True,
            )
        )

    fallback_reason = ""
    if original is None:
        fallback_reason = "missing_original_route"
    elif same_exit_direction_alternatives == 0:
        fallback_reason = "no_same_exit_direction_alternatives"

    candidate_count = len(candidates)

    diagnostics = {
        "original_exit_direction": original_exit_direction,
        "candidate_count_total": considered_total + (1 if original is not None else 0),
        "candidate_count_same_exit_direction": candidate_count,
        "candidate_count_changed_exit_direction": changed_exit_direction_count,
        "same_exit_direction_alternative_count": same_exit_direction_alternatives,
        # Backward-compatible aliases: the former same_destination names.
        "candidate_count_same_destination": candidate_count,
        "candidate_count_changed_destination": changed_exit_direction_count,
        "same_destination_alternative_count": same_exit_direction_alternatives,
        "fallback_reason": fallback_reason,
    }
    return candidates, diagnostics


def build_candidates(
    planner: Any,
    junction_id: int,
    entry_direction: str,
    entry_lane_idx: int,
    original_route_id: int,
) -> List[RouteIntentCandidate]:
    """Build legal route-intent candidates, preserving the exit direction."""
    candidates, _ = build_candidates_with_diagnostics(
        planner=planner,
        junction_id=junction_id,
        entry_direction=entry_direction,
        entry_lane_idx=entry_lane_idx,
        original_route_id=original_route_id,
    )
    return candidates


#: Width of the ``committed_exit_lanes`` observation: how many exit lanes a
#: neighbour's selected route can be counted in. A junction with fewer exit
#: lanes than this simply leaves the upper slots at zero.
DEFAULT_EXIT_LANE_SLOTS = 4


def committed_exit_lane_occupancy(
    spawn_metadata: "Sequence[Dict[str, Any]]",
    n_max: int = DEFAULT_EXIT_LANE_SLOTS,
    *,
    junction_id: Any = None,
    exclude_actor: Any = None,
) -> "list[float]":
    """How many other vehicles have already committed to each exit lane.

    This feeds the ``committed_exit_lanes`` observation, which lets a
    policy take the exit lanes its neighbours have already selected into
    account when it chooses its own. Counting per lane (rather than
    listing neighbours) keeps the signal permutation-invariant and
    independent of how many vehicles are present.

    ``spawn_metadata`` is one mapping per live vehicle, each optionally
    carrying ``route_intent`` (the decision record) and ``junction_id``.
    Vehicles that have not decided yet contribute nothing — an undecided
    neighbour is genuinely unknown, not lane 0.
    """
    occupancy = [0.0] * max(1, int(n_max))
    for metadata in spawn_metadata or ():
        if not metadata:
            continue
        if exclude_actor is not None and metadata.get("actor_id") == exclude_actor:
            continue
        if junction_id is not None:
            other_junction = metadata.get("junction_id")
            if other_junction is not None and str(other_junction) != str(junction_id):
                continue
        intent = metadata.get("route_intent") or {}
        lane = intent.get("selected_exit_lane_idx")
        if lane is None:
            continue
        idx = int(lane)
        if 0 <= idx < len(occupancy):
            occupancy[idx] += 1.0
    return occupancy


def build_mask(candidates: Sequence[RouteIntentCandidate]) -> Dict[RouteIntent, bool]:
    """Mark each semantic intent legal iff at least one candidate exists.

    The mask is computed exclusively from candidate feasibility derived from
    map/lane/planner data. It is intentionally independent from nearby
    vehicles, congestion, or learned risk: those are policy observations,
    not feasibility filters.
    """
    legal = {intent: False for intent in ALL_INTENTS}
    for c in candidates:
        legal[c.intent] = True
    legal[RouteIntent.KEEP] = legal.get(RouteIntent.KEEP, False) or any(
        c.intent == RouteIntent.KEEP for c in candidates
    )
    return legal


def _int_or_default(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def adapter_route_key(adapter: Any) -> Tuple[str, int, int]:
    """Return ``(exit_direction, exit_lane_idx, route_id)`` for an adapter.

    Active vehicles may carry the original spawn route metadata directly or a
    nested ``route_intent`` decision metadata block after the route layer has
    resolved a concrete candidate. Prefer the resolved route when present.
    """
    meta = getattr(adapter, "spawn_metadata", None)
    if not isinstance(meta, dict):
        return "", -1, -1
    route_intent = meta.get("route_intent")
    if isinstance(route_intent, dict):
        exit_direction = str(route_intent.get("selected_exit_direction", "") or "")
        exit_lane_idx = _int_or_default(
            route_intent.get("selected_exit_lane_idx", -1)
        )
        route_id = _int_or_default(route_intent.get("selected_route_id", -1))
        if exit_direction or exit_lane_idx >= 0 or route_id >= 0:
            return exit_direction, exit_lane_idx, route_id
    return (
        str(meta.get("exit_direction", "") or ""),
        _int_or_default(meta.get("exit_lane_idx", -1)),
        _int_or_default(meta.get("route_id", -1)),
    )


def adapter_speed_kmh(adapter: Any) -> float:
    """Best-effort active vehicle speed in km/h for queue diagnostics."""
    speed = getattr(adapter, "current_speed", None)
    if speed is None:
        speed = getattr(adapter, "speed", None)
    if speed is not None:
        return _float_or_default(speed, 0.0)
    vehicle = getattr(adapter, "vehicle", None)
    if vehicle is None:
        return 0.0
    try:
        velocity = vehicle.get_velocity()
        return float(
            math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6
        )
    except Exception:
        return 0.0


def _adapter_location(adapter: Any) -> Any | None:
    vehicle = getattr(adapter, "vehicle", None)
    if vehicle is not None:
        try:
            return vehicle.get_location()
        except Exception:
            pass
    return getattr(adapter, "location", None)


def _adapter_velocity(adapter: Any) -> Any | None:
    vehicle = getattr(adapter, "vehicle", None)
    if vehicle is not None:
        try:
            return vehicle.get_velocity()
        except Exception:
            pass
    return getattr(adapter, "velocity", None)


def _distance_between_adapters(a: Any, b: Any) -> Optional[float]:
    loc_a = _adapter_location(a)
    loc_b = _adapter_location(b)
    if loc_a is None or loc_b is None:
        return None
    try:
        dx = float(loc_b.x) - float(loc_a.x)
        dy = float(loc_b.y) - float(loc_a.y)
        dz = float(getattr(loc_b, "z", 0.0)) - float(getattr(loc_a, "z", 0.0))
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    except Exception:
        return None


def _ttc_between_adapters(
    ego_adapter: Any,
    other_adapter: Any,
    *,
    max_ttc_s: float = 10.0,
    collision_radius_m: float = 4.0,
) -> Optional[float]:
    """Approximate constant-velocity TTC between two adapters.

    Returns ``None`` when geometry or velocity is unavailable, the vehicles are
    not closing, or the closest point within the horizon stays outside the
    collision radius.
    """
    loc_e = _adapter_location(ego_adapter)
    loc_o = _adapter_location(other_adapter)
    vel_e = _adapter_velocity(ego_adapter)
    vel_o = _adapter_velocity(other_adapter)
    if loc_e is None or loc_o is None or vel_e is None or vel_o is None:
        return None
    try:
        rel_pos = (
            float(loc_o.x) - float(loc_e.x),
            float(loc_o.y) - float(loc_e.y),
            float(getattr(loc_o, "z", 0.0)) - float(getattr(loc_e, "z", 0.0)),
        )
        rel_vel = (
            float(vel_o.x) - float(vel_e.x),
            float(vel_o.y) - float(vel_e.y),
            float(getattr(vel_o, "z", 0.0)) - float(getattr(vel_e, "z", 0.0)),
        )
    except Exception:
        return None
    vel_sq = sum(v * v for v in rel_vel)
    if vel_sq <= 1e-6:
        return None
    t_closest = -sum(p * v for p, v in zip(rel_pos, rel_vel)) / vel_sq
    if t_closest < 0.0 or t_closest > float(max_ttc_s):
        return None
    closest_sq = sum(
        (p + t_closest * v) ** 2 for p, v in zip(rel_pos, rel_vel)
    )
    if closest_sq > float(collision_radius_m) ** 2:
        return None
    return float(t_closest)


def candidate_traffic_features(
    candidate: RouteIntentCandidate,
    active_adapters: Sequence[Any],
    *,
    ego_adapter: Any | None = None,
    queue_speed_threshold_kmh: float = 5.0,
    max_ttc_s: float = 10.0,
) -> CandidateTrafficFeatures:
    """Summarize local pressure around a concrete same-direction candidate."""
    same_exit_direction = 0
    same_exit_lane = 0
    queued_same_lane = 0
    same_route = 0
    nearest_distance = 999.0
    min_ttc = float(max_ttc_s)

    for adapter in active_adapters:
        if ego_adapter is not None and adapter is ego_adapter:
            continue
        exit_direction, exit_lane_idx, route_id = adapter_route_key(adapter)
        if exit_direction != candidate.exit_direction:
            continue
        same_exit_direction += 1
        if route_id == candidate.route_id:
            same_route += 1
        if exit_lane_idx != candidate.exit_lane_idx:
            continue
        same_exit_lane += 1
        if adapter_speed_kmh(adapter) < float(queue_speed_threshold_kmh):
            queued_same_lane += 1
        if ego_adapter is not None:
            distance = _distance_between_adapters(ego_adapter, adapter)
            if distance is not None:
                nearest_distance = min(nearest_distance, distance)
            ttc = _ttc_between_adapters(
                ego_adapter, adapter, max_ttc_s=float(max_ttc_s)
            )
            if ttc is not None:
                min_ttc = min(min_ttc, ttc)

    return CandidateTrafficFeatures(
        active_same_exit_direction_count=same_exit_direction,
        active_same_exit_lane_count=same_exit_lane,
        queued_same_exit_lane_count=queued_same_lane,
        same_route_count=same_route,
        nearest_same_exit_lane_distance_m=nearest_distance,
        min_same_exit_lane_ttc_s=min_ttc,
    )


def _normalize_preference(preference: Any) -> Tuple[float, float]:
    try:
        values = list(preference)
    except TypeError:
        values = []
    if len(values) < 2:
        return 0.5, 0.5
    efficiency = max(0.0, _float_or_default(values[0], 0.0))
    risk = max(0.0, _float_or_default(values[1], 0.0))
    total = efficiency + risk
    if total <= 1e-6:
        return 0.5, 0.5
    return efficiency / total, risk / total


def candidate_pressure_score(
    candidate: RouteIntentCandidate,
    features: CandidateTrafficFeatures,
    *,
    preference: Any = None,
    route_change_penalty: float = 0.0,
    max_distance_m: float = 50.0,
    max_ttc_s: float = 10.0,
    same_route_risk_weight: float = 4.0,
) -> float:
    """Lower-is-better score for concrete candidate resolution.

    Route diversity: adds ``same_route_risk_weight * same_route_count``
    to the risk side so candidates that share the exact internal route with
    another active CAV are penalised under a risk-leaning preference. Two
    vehicles on the same concrete route would tail each other through the
    same internal lanes and thus carry follow-distance collision risk that
    is not captured by exit-lane-only counters. Default 4.0 (above the 3.0
    weight on ``active_same_exit_lane_count``) reflects that route-level
    overlap is strictly more committing than lane-level overlap.
    """
    efficiency_w, risk_w = _normalize_preference(preference)
    distance = min(
        float(features.nearest_same_exit_lane_distance_m), float(max_distance_m)
    )
    ttc = min(float(features.min_same_exit_lane_ttc_s), float(max_ttc_s))
    distance_pressure = 1.0 - (distance / float(max_distance_m))
    ttc_pressure = 1.0 - (ttc / float(max_ttc_s))

    risk_cost = (
        3.0 * features.active_same_exit_lane_count
        + 0.75 * features.active_same_exit_direction_count
        + 5.0 * features.queued_same_exit_lane_count
        + 1.5 * distance_pressure
        + 2.0 * ttc_pressure
        + float(same_route_risk_weight) * features.same_route_count
    )
    efficiency_cost = (
        3.0 * features.queued_same_exit_lane_count
        + 1.5 * features.active_same_exit_lane_count
        + 0.5 * features.same_route_count
        + 0.5 * distance_pressure
    )
    change_cost = (
        0.0 if candidate.intent == RouteIntent.KEEP else float(route_change_penalty)
    )
    return float(risk_w * risk_cost + efficiency_w * efficiency_cost + change_cost)


def resolve_route(
    intent: RouteIntent,
    candidates: Sequence[RouteIntentCandidate],
    mask: Dict[RouteIntent, bool],
    *,
    original_route_id: int,
    active_adapters: Sequence[Any] | None = None,
    ego_adapter: Any | None = None,
    preference: Any = None,
    queue_speed_threshold_kmh: float = 5.0,
    route_change_penalty: float = 0.0,
) -> RouteIntentCandidate:
    """Resolve a selected semantic intent to a concrete candidate route.

    Tie-breaking rule:
    - Filter candidates by selected intent.
    - If active traffic context is provided, rank concrete routes by
      candidate pressure, then deterministic lane/route ids.
    - Otherwise sort by ``(exit_lane_idx, route_id)`` ascending.
    - Return the first.

    If the intent is illegal under ``mask`` or has no candidates, the
    function falls back to ``KEEP`` using the original route. Callers must
    not rely on this fallback in place of action masking; it exists only to
    keep the simulation safe if a caller bypasses the mask.
    """
    if not mask.get(intent, False):
        intent = RouteIntent.KEEP

    matching = [c for c in candidates if c.intent == intent]
    if not matching:
        keep_matches = [
            c for c in candidates
            if c.intent == RouteIntent.KEEP and c.route_id == int(original_route_id)
        ]
        if keep_matches:
            return keep_matches[0]
        raise ValueError(
            f"No candidate available for intent={intent} and no keep-current "
            f"fallback for original_route_id={original_route_id}."
        )

    if active_adapters is not None:
        active_list = list(active_adapters)
        scored = []
        for candidate in matching:
            features = candidate_traffic_features(
                candidate,
                active_list,
                ego_adapter=ego_adapter,
                queue_speed_threshold_kmh=float(queue_speed_threshold_kmh),
            )
            score = candidate_pressure_score(
                candidate,
                features,
                preference=preference,
                route_change_penalty=float(route_change_penalty),
            )
            scored.append((score, candidate.exit_lane_idx, candidate.route_id, candidate))
        scored.sort(key=lambda item: item[:3])
        return scored[0][-1]

    matching.sort(key=lambda c: (c.exit_lane_idx, c.route_id))
    return matching[0]


def build_decision(
    *,
    requested_intent: RouteIntent,
    candidate: RouteIntentCandidate,
    mask: Dict[RouteIntent, bool],
    original_route_id: int,
    diagnostics: Optional[Dict[str, Any]] = None,
    selected_candidate_score: float = 0.0,
    selected_candidate_traffic: Optional[Dict[str, Any]] = None,
    original_candidate_traffic: Optional[Dict[str, Any]] = None,
    candidate_resolution: str = "deterministic",
) -> RouteIntentDecision:
    """Create a decision record with direction-preserving diagnostics."""
    diagnostics = diagnostics or {}
    original_exit = str(diagnostics.get("original_exit_direction", "") or "")
    direction_preserved = (
        bool(candidate.is_original)
        or (bool(original_exit) and candidate.exit_direction == original_exit)
    )
    fallback_reason = str(diagnostics.get("fallback_reason", "") or "")
    if candidate.intent == RouteIntent.KEEP and requested_intent != RouteIntent.KEEP:
        fallback_reason = fallback_reason or "resolved_to_keep_current"
    return RouteIntentDecision(
        selected_intent=candidate.intent,
        mask=mask,
        selected_candidate=candidate,
        original_route_id=int(original_route_id),
        used_keep_current=(candidate.intent == RouteIntent.KEEP),
        original_exit_direction=original_exit,
        candidate_count_total=int(diagnostics.get("candidate_count_total", 0) or 0),
        candidate_count_same_exit_direction=int(
            diagnostics.get(
                "candidate_count_same_exit_direction",
                diagnostics.get("candidate_count_same_destination", 0),
            ) or 0
        ),
        mask_sparsity=_mask_sparsity(mask),
        direction_preserved=direction_preserved,
        fallback_reason=fallback_reason,
        interface=str(diagnostics.get("interface", "semantic") or "semantic"),
        selected_candidate_score=float(selected_candidate_score),
        selected_candidate_traffic=dict(selected_candidate_traffic or {}),
        original_candidate_traffic=dict(original_candidate_traffic or {}),
        candidate_resolution=str(candidate_resolution),
    )
