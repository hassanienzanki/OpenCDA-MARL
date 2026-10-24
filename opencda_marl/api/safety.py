from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TtcMetricState:
    near_miss_count: int = 0
    near_miss_agents: set[Any] = field(default_factory=set)
    violation_count: int = 0
    check_count: int = 0

    def reset(self) -> None:
        self.near_miss_count = 0
        self.near_miss_agents.clear()
        self.violation_count = 0
        self.check_count = 0


def track_ttc_metrics(
    reward_params: Mapping[str, Any],
    min_ttc: float,
    agent_id: Any | None,
    ttc_metrics: TtcMetricState,
) -> None:
    safe_threshold = float(reward_params.get("ttc_safe_threshold", 4.0))
    near_miss_threshold = float(reward_params.get("ttc_near_miss_threshold", 2.0))

    ttc_metrics.check_count += 1
    if min_ttc < safe_threshold and not math.isinf(min_ttc):
        ttc_metrics.violation_count += 1
    if min_ttc < near_miss_threshold and agent_id is not None:
        if agent_id not in ttc_metrics.near_miss_agents:
            ttc_metrics.near_miss_count += 1
            ttc_metrics.near_miss_agents.add(agent_id)


def calculate_ttc_reward(
    reward_params: Mapping[str, Any],
    min_ttc: float,
    agent_id: Any | None,
    ttc_metrics: TtcMetricState,
) -> float:
    safe_threshold = float(reward_params.get("ttc_safe_threshold", 4.0))
    max_penalty = float(reward_params.get("ttc_max_penalty", 0.0))
    decay_rate = float(reward_params.get("ttc_decay_rate", 1.5))
    near_miss_threshold = float(reward_params.get("ttc_near_miss_threshold", 2.0))

    track_ttc_metrics(reward_params, min_ttc, agent_id, ttc_metrics)
    if min_ttc > safe_threshold or math.isinf(min_ttc):
        return 0.0

    penalty = -max_penalty * math.exp(-decay_rate * min_ttc)
    if min_ttc < near_miss_threshold and agent_id is not None:
        if agent_id not in ttc_metrics.near_miss_agents:
            ttc_metrics.near_miss_count += 1
            ttc_metrics.near_miss_agents.add(agent_id)
    return penalty


# --------------------------------------------------------------------- #
# Time-to-collision and the step's kinematics, for agents that act
# --------------------------------------------------------------------- #

def constant_velocity_ttc(rel_x: float, rel_y: float, rel_vx: float, rel_vy: float,
                          collision_radius: float = 4.0) -> float:
    """Time until two vehicles closing at constant velocity come within
    *collision_radius*, or ``inf``: the quadratic the world snapshot solves
    for the neighbour features, term by term in the same order, so an agent
    reading it brakes on the number the learner and the metrics see."""
    a = rel_vx * rel_vx + rel_vy * rel_vy
    if abs(a) < 1e-6:
        return math.inf
    b = 2.0 * (rel_x * rel_vx + rel_y * rel_vy)
    c = (rel_x * rel_x + rel_y * rel_y) - collision_radius ** 2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return math.inf
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return math.inf


def min_ttc_to_others(records: Mapping[Any, Any], ego_id: Any, collision_radius: float = 4.0) -> float:
    """The smallest constant-velocity TTC from *ego_id* to every other
    record in *records* (a mapping of ids to objects with ``x``, ``y``,
    ``vx``, ``vy``, as the world snapshot publishes); ``inf`` when the ego
    is absent or alone."""
    ego = records.get(ego_id)
    if ego is None:
        try:
            ego = records.get(int(ego_id))
        except (TypeError, ValueError):
            ego = None
    if ego is None:
        return math.inf
    best = math.inf
    for other_id, other in records.items():
        if other is ego:
            continue
        ttc = constant_velocity_ttc(other.x - ego.x, other.y - ego.y, other.vx - ego.vx, other.vy - ego.vy,
                                    collision_radius)
        if ttc < best:
            best = ttc
    return best


def published_kinematics():
    """The kinematics the agent manager published for the current step --
    id -> record with ``x``, ``y``, ``vx``, ``vy``, ``yaw_rad`` -- or
    ``None`` outside a step. The manager publishes the world snapshot to the
    vendored perception layer around observation collection and around the
    action phase, so an agent's ``run_step`` may read it without querying
    the simulator again."""
    try:
        from opencda.core.sensing.perception.perception_manager import PerceptionManager
    except Exception:  # noqa: BLE001 - no CARLA stack in this environment
        return None
    return getattr(PerceptionManager, "_world_snapshot", None)
