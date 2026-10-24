from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, TypedDict


CANONICAL_REWARD_COMPONENTS = (
    "collision",
    "success",
    "step_penalty",
    "speed",
    "stop",
    "progress",
    "ttc",
    "yielding",
    "clearance",
)

REWARD_MODE_LEARN = "learn"
REWARD_MODE_DISABLED = "disabled"
REWARD_MODE_METRIC_ONLY = "metric_only"
VALID_REWARD_MODES = {
    REWARD_MODE_LEARN,
    REWARD_MODE_DISABLED,
    REWARD_MODE_METRIC_ONLY,
}

DEFAULT_MAX_NEARBY_VEHICLES = 5
DEFAULT_FEATURES_PER_VEHICLE = 7


class NormalizedObservation(TypedDict, total=False):
    vehicle_id: int | str
    speed: float
    physics_speed: float
    physics_speed_planar: float
    physics_speed_3d: float
    vertical_speed: float
    tilt_pitch: float
    tilt_roll: float
    height_above_waypoint: float
    heading_angle: float
    relative_position_to_intersection: Dict[str, float]
    distance_to_intersection: float
    distance_to_front_vehicle: float
    lane_position: float
    waypoint_buffer_size: float
    nearby_vehicles: list[float]
    #: Per exit lane, how many other vehicles at this junction have already
    #: committed to it. Empty unless the run publishes commitments; the
    #: extractor includes it only behind the committed_exit_lanes flag.
    committed_exit_lanes: list[float]
    min_ttc: float
    distance_to_destination: float
    target_speed: float
    location: Dict[str, float]
    velocity: Dict[str, float]
    position_x: float
    position_y: float
    status: Any


def _coerce_component_mode(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_REWARD_MODES:
            return normalized
        if normalized in {"true", "enabled", "on"}:
            return REWARD_MODE_LEARN
        if normalized in {"false", "off"}:
            return REWARD_MODE_DISABLED
    if value:
        return REWARD_MODE_LEARN
    return REWARD_MODE_DISABLED


def _get_default_component_mode(component: str, reward_params: Mapping[str, Any] | None) -> str:
    if not reward_params:
        return REWARD_MODE_LEARN

    if component == "ttc" and float(reward_params.get("ttc_max_penalty", 0.0)) == 0.0:
        return REWARD_MODE_METRIC_ONLY

    return REWARD_MODE_LEARN


@dataclass(frozen=True)
class RewardSpec:
    components: tuple[str, ...] = CANONICAL_REWARD_COMPONENTS
    weights: Dict[str, float] = field(default_factory=dict)
    component_modes: Dict[str, str] = field(default_factory=dict)
    scalarizer: str = "weighted_sum"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        reward_params: Mapping[str, Any] | None = None,
    ) -> "RewardSpec":
        config = dict(config or {})
        configured_components = config.get("components") or CANONICAL_REWARD_COMPONENTS
        components = tuple(dict.fromkeys(configured_components))

        raw_enabled = config.get("enabled", {})
        raw_modes = config.get("component_modes", {})
        weights = {
            component: float(config.get("weights", {}).get(component, 1.0))
            for component in components
        }

        component_modes: Dict[str, str] = {}
        for component in components:
            if component in raw_modes:
                component_modes[component] = _coerce_component_mode(raw_modes[component])
            elif component in raw_enabled:
                component_modes[component] = _coerce_component_mode(raw_enabled[component])
            else:
                component_modes[component] = _get_default_component_mode(component, reward_params)

        return cls(
            components=components,
            weights=weights,
            component_modes=component_modes,
            scalarizer=config.get("scalarizer", "weighted_sum"),
        )

    def mode_for(self, component: str) -> str:
        if component not in self.components:
            return REWARD_MODE_DISABLED
        return self.component_modes.get(component, REWARD_MODE_LEARN)

    def is_enabled(self, component: str) -> bool:
        return self.mode_for(component) == REWARD_MODE_LEARN

    def is_metric_only(self, component: str) -> bool:
        return self.mode_for(component) == REWARD_MODE_METRIC_ONLY

    def weight_for(self, component: str) -> float:
        return float(self.weights.get(component, 1.0))

    def empty_components(self) -> Dict[str, float]:
        return {component: 0.0 for component in self.components}

    def vectorize(self, components: Mapping[str, float]) -> list[float]:
        return [float(components.get(component, 0.0)) for component in self.components]

    def scalarize(self, components: Mapping[str, float]) -> float:
        if self.scalarizer != "weighted_sum":
            raise ValueError(f"Unsupported scalarizer: {self.scalarizer}")

        scalar = 0.0
        for component in self.components:
            if self.is_enabled(component):
                scalar += self.weight_for(component) * float(components.get(component, 0.0))
        return scalar

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": list(self.components),
            "weights": dict(self.weights),
            "enabled": dict(self.component_modes),
            "scalarizer": self.scalarizer,
        }


@dataclass(frozen=True)
class RewardOutput:
    components: Dict[str, float]
    vector: list[float]
    scalar: float
    spec: RewardSpec

    @classmethod
    def from_components(
        cls,
        components: Mapping[str, float],
        spec: RewardSpec,
    ) -> "RewardOutput":
        completed = spec.empty_components()
        for component, value in components.items():
            if component in completed:
                completed[component] = float(value)

        return cls(
            components=completed,
            vector=spec.vectorize(completed),
            scalar=spec.scalarize(completed),
            spec=spec,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": dict(self.components),
            "vector": list(self.vector),
            "scalar": self.scalar,
            "spec": self.spec.to_dict(),
        }


def build_reward_outputs(
    reward_components: Mapping[Any, Mapping[str, float]],
    spec: RewardSpec,
) -> Dict[Any, RewardOutput]:
    return {
        agent_id: RewardOutput.from_components(components, spec)
        for agent_id, components in reward_components.items()
    }


def normalize_observation(
    observation: Mapping[str, Any],
    *,
    max_nearby_vehicles: int = DEFAULT_MAX_NEARBY_VEHICLES,
    features_per_vehicle: int = DEFAULT_FEATURES_PER_VEHICLE,
) -> NormalizedObservation:
    rel_pos = observation.get("relative_position_to_intersection", {})
    location = observation.get("location", {})
    velocity = observation.get("velocity", {})
    safe_nearby = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * max_nearby_vehicles

    normalized: NormalizedObservation = {
        "vehicle_id": observation.get("vehicle_id"),
        "speed": float(observation.get("speed", 0.0)),
        "physics_speed": float(observation.get("physics_speed", observation.get("physics_speed_planar", observation.get("speed", 0.0)))),
        "physics_speed_planar": float(observation.get("physics_speed_planar", observation.get("physics_speed", observation.get("speed", 0.0)))),
        "physics_speed_3d": float(observation.get("physics_speed_3d", observation.get("physics_speed_planar", observation.get("physics_speed", observation.get("speed", 0.0))))),
        "vertical_speed": float(observation.get("vertical_speed", 0.0)),
        "tilt_pitch": float(observation.get("tilt_pitch", 0.0)),
        "tilt_roll": float(observation.get("tilt_roll", 0.0)),
        "height_above_waypoint": float(observation.get("height_above_waypoint", 0.0)),
        "heading_angle": float(observation.get("heading_angle", 0.0)),
        "relative_position_to_intersection": {
            "x": float(rel_pos.get("x", location.get("x", observation.get("position_x", 0.0)))),
            "y": float(rel_pos.get("y", location.get("y", observation.get("position_y", 0.0)))),
        },
        "distance_to_intersection": float(observation.get("distance_to_intersection", 100.0)),
        "distance_to_front_vehicle": float(observation.get("distance_to_front_vehicle", 999.0)),
        "lane_position": float(observation.get("lane_position", 0.0)),
        "waypoint_buffer_size": float(observation.get("waypoint_buffer_size", 0.0)),
        "nearby_vehicles": list(observation.get("nearby_vehicles", safe_nearby)),
        # Per-exit-lane commitment counts (empty unless the run publishes them).
        "committed_exit_lanes": list(observation.get("committed_exit_lanes", []) or []),
        "min_ttc": float(observation.get("min_ttc", float("inf"))),
        "distance_to_destination": float(observation.get("distance_to_destination", 999.0)),
        "target_speed": float(observation.get("target_speed", 0.0)),
        "location": {
            "x": float(location.get("x", rel_pos.get("x", observation.get("position_x", 0.0)))),
            "y": float(location.get("y", rel_pos.get("y", observation.get("position_y", 0.0)))),
            "z": float(location.get("z", 0.0)),
        },
        "velocity": {
            "x": float(velocity.get("x", 0.0)),
            "y": float(velocity.get("y", 0.0)),
            "z": float(velocity.get("z", 0.0)),
        },
        "position_x": float(observation.get("position_x", location.get("x", 0.0))),
        "position_y": float(observation.get("position_y", location.get("y", 0.0))),
        "status": observation.get("status"),
    }

    expected_nearby_size = max_nearby_vehicles * features_per_vehicle
    nearby = list(normalized["nearby_vehicles"])
    if len(nearby) < expected_nearby_size:
        nearby.extend(safe_nearby[len(nearby):expected_nearby_size])
    elif len(nearby) > expected_nearby_size:
        nearby = nearby[:expected_nearby_size]
    normalized["nearby_vehicles"] = nearby

    return normalized


def normalize_observations(
    observations: Mapping[Any, Mapping[str, Any]],
    *,
    max_nearby_vehicles: int = DEFAULT_MAX_NEARBY_VEHICLES,
    features_per_vehicle: int = DEFAULT_FEATURES_PER_VEHICLE,
) -> Dict[Any, NormalizedObservation]:
    return {
        agent_id: normalize_observation(
            observation,
            max_nearby_vehicles=max_nearby_vehicles,
            features_per_vehicle=features_per_vehicle,
        )
        for agent_id, observation in observations.items()
    }


def reward_outputs_to_scalars(reward_outputs: Mapping[Any, RewardOutput]) -> Dict[Any, float]:
    return {agent_id: reward_output.scalar for agent_id, reward_output in reward_outputs.items()}


def reward_outputs_to_components(
    reward_outputs: Mapping[Any, RewardOutput],
) -> Dict[Any, Dict[str, float]]:
    return {
        agent_id: dict(reward_output.components)
        for agent_id, reward_output in reward_outputs.items()
    }


def reward_outputs_to_vectors(reward_outputs: Mapping[Any, RewardOutput]) -> Dict[Any, list[float]]:
    return {agent_id: list(reward_output.vector) for agent_id, reward_output in reward_outputs.items()}
