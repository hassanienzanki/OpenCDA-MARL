from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from opencda_marl.api.contracts import RewardOutput


BUILTIN_OBJECTIVE_PROFILES: Dict[str, Dict[str, Any]] = {
    "cooperative_pref_v1": {
        "objective_spec": {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {
                    "progress": 1.0,
                    "success": 1.0,
                    "step_penalty": 1.0,
                },
                "safety": {
                    "collision": 1.0,
                },
            },
        },
    },
    "competitive_ordering_v1": {
        "objective_spec": {
            "objectives": ["efficiency", "interaction_risk"],
            "groups": {
                "efficiency": {
                    "progress": 1.0,
                    "success": 1.0,
                    "step_penalty": 1.0,
                },
                "interaction_risk": {
                    "collision": 1.0,
                },
            },
        },
    },
}


def competition_metrics_enabled(reward_config: Any, objective_config: Any) -> bool:
    """Whether a run records the per-agent competition metadata and terminal
    records behind the route-intent and preference summaries.

    Any run with an objective spec is preference-conditioned and gets them,
    and so does a reward profile whose name says it is competitive. The
    test is the spec or the name prefix, never a list of the built-in
    profile names, so an objective profile registered through the plugin
    door gets the metrics like any other.
    """
    spec = getattr(objective_config, "objective_spec", None)
    if spec is not None and getattr(spec, "objectives", None):
        return True
    for config in (reward_config, objective_config):
        name = str(getattr(config, "profile_name", "") or "")
        if name.startswith("competitive"):
            return True
    return False


def register_objective_profile(name: str, profile: Mapping[str, Any], *,
                               override: bool = False) -> None:
    """Register an objective profile under ``name`` (``MARL.objective_mode``).

    ``profile`` has the built-in shape: an ``objective_spec`` block naming
    the objectives and grouping reward components under them. Import-time
    registration — the same plugin door as algorithms and components.
    """
    key = str(name)
    if key in BUILTIN_OBJECTIVE_PROFILES and not override:
        raise ValueError(
            f"objective profile {key!r} is already registered; pass "
            "override=True to replace it deliberately.")
    from opencda_marl.plugins import registry as _plugin_registry

    _plugin_registry.guard("objective_profile", key)
    BUILTIN_OBJECTIVE_PROFILES[key] = deepcopy(dict(profile))


def _kind_adapter() -> dict:
    return {
        "describe": lambda: [{"key": k, "source": "builtin"}
                             for k in BUILTIN_OBJECTIVE_PROFILES],
        "remove": lambda key: BUILTIN_OBJECTIVE_PROFILES.pop(key, None),
    }


@dataclass(frozen=True)
class ObjectiveSpec:
    objectives: tuple[str, ...]
    groups: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scalarizer: str = "weighted_sum"

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> "ObjectiveSpec":
        config = dict(config or {})
        configured_objectives = config.get("objectives") or []
        if not configured_objectives:
            raise ValueError("ObjectiveSpec requires at least one objective.")

        objectives = tuple(dict.fromkeys(str(name) for name in configured_objectives))
        raw_groups = dict(config.get("groups", {}) or {})
        groups: Dict[str, Dict[str, float]] = {}
        for objective in objectives:
            component_weights = dict(raw_groups.get(objective, {}) or {})
            if not component_weights:
                raise ValueError(f"Objective '{objective}' must define at least one grouped reward component.")
            groups[objective] = {
                str(component): float(weight)
                for component, weight in component_weights.items()
            }

        return cls(
            objectives=objectives,
            groups=groups,
            scalarizer=str(config.get("scalarizer", "weighted_sum")),
        )

    def empty_components(self) -> Dict[str, float]:
        return {objective: 0.0 for objective in self.objectives}

    def group_reward_components(self, reward_components: Mapping[str, float]) -> Dict[str, float]:
        grouped = self.empty_components()
        for objective in self.objectives:
            total = 0.0
            for component, weight in self.groups.get(objective, {}).items():
                total += float(weight) * float(reward_components.get(component, 0.0))
            grouped[objective] = total
        return grouped

    def vectorize(self, objective_components: Mapping[str, float]) -> list[float]:
        return [float(objective_components.get(objective, 0.0)) for objective in self.objectives]

    def scalarize(
        self,
        objective_components: Mapping[str, float] | Sequence[float],
        preference: Sequence[float],
    ) -> float:
        if self.scalarizer != "weighted_sum":
            raise ValueError(f"Unsupported objective scalarizer: {self.scalarizer}")

        if len(preference) != len(self.objectives):
            raise ValueError(
                f"Preference dimension {len(preference)} does not match objective dimension {len(self.objectives)}."
            )

        if isinstance(objective_components, Mapping):
            vector = self.vectorize(objective_components)
        else:
            vector = [float(value) for value in objective_components]

        return sum(float(weight) * float(value) for weight, value in zip(preference, vector))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": list(self.objectives),
            "groups": deepcopy(self.groups),
            "scalarizer": self.scalarizer,
        }


@dataclass(frozen=True)
class ObjectiveOutput:
    components: Dict[str, float]
    vector: list[float]
    spec: ObjectiveSpec

    @classmethod
    def from_reward_components(
        cls,
        reward_components: Mapping[str, float],
        spec: ObjectiveSpec,
    ) -> "ObjectiveOutput":
        grouped = spec.group_reward_components(reward_components)
        return cls(
            components=grouped,
            vector=spec.vectorize(grouped),
            spec=spec,
        )

    def scalarize(self, preference: Sequence[float]) -> float:
        return self.spec.scalarize(self.components, preference)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": dict(self.components),
            "vector": list(self.vector),
            "spec": self.spec.to_dict(),
        }


@dataclass(frozen=True)
class ResolvedObjectiveConfig:
    profile_name: str
    objective_spec_config: Dict[str, Any]
    objective_spec: ObjectiveSpec | None

    @property
    def enabled(self) -> bool:
        return self.objective_spec is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "objective_spec": deepcopy(self.objective_spec_config),
            "resolved_spec": self.objective_spec.to_dict() if self.objective_spec else None,
        }


def resolve_objective_configuration(config: Mapping[str, Any] | None = None) -> ResolvedObjectiveConfig:
    config = dict(config or {})
    objective_mode = str(config.get("objective_mode") or "").strip()
    objective_spec_config: Dict[str, Any] = {}

    if objective_mode:
        profile = BUILTIN_OBJECTIVE_PROFILES.get(objective_mode)
        if profile is None:
            available = ", ".join(sorted(BUILTIN_OBJECTIVE_PROFILES))
            raise ValueError(
                f"Unknown objective mode '{objective_mode}'. Available objective modes: {available}"
            )
        objective_spec_config = _deep_merge_dicts(objective_spec_config, profile.get("objective_spec", {}))
    else:
        objective_mode = "disabled"

    objective_spec_config = _deep_merge_dicts(
        objective_spec_config,
        dict(config.get("objective_spec", {}) or {}),
    )

    if not objective_spec_config:
        return ResolvedObjectiveConfig(
            profile_name=objective_mode,
            objective_spec_config={},
            objective_spec=None,
        )

    objective_spec = ObjectiveSpec.from_config(objective_spec_config)
    return ResolvedObjectiveConfig(
        profile_name=objective_mode,
        objective_spec_config=objective_spec_config,
        objective_spec=objective_spec,
    )


def build_objective_outputs(
    reward_outputs: Mapping[Any, RewardOutput] | Mapping[Any, Mapping[str, float]],
    spec: ObjectiveSpec,
) -> Dict[Any, ObjectiveOutput]:
    objective_outputs: Dict[Any, ObjectiveOutput] = {}
    for agent_id, reward_output in reward_outputs.items():
        components = reward_output.components if isinstance(reward_output, RewardOutput) else reward_output
        objective_outputs[agent_id] = ObjectiveOutput.from_reward_components(components, spec)
    return objective_outputs


def objective_outputs_to_components(
    objective_outputs: Mapping[Any, ObjectiveOutput],
) -> Dict[Any, Dict[str, float]]:
    return {
        agent_id: dict(objective_output.components)
        for agent_id, objective_output in objective_outputs.items()
    }


def objective_outputs_to_vectors(
    objective_outputs: Mapping[Any, ObjectiveOutput],
) -> Dict[Any, list[float]]:
    return {
        agent_id: list(objective_output.vector)
        for agent_id, objective_output in objective_outputs.items()
    }


def scalarize_objective_outputs(
    objective_outputs: Mapping[Any, ObjectiveOutput],
    preferences: Mapping[Any, Sequence[float]],
) -> Dict[Any, float]:
    scalarized: Dict[Any, float] = {}
    for agent_id, objective_output in objective_outputs.items():
        preference = preferences.get(agent_id)
        if preference is None:
            continue
        scalarized[agent_id] = objective_output.scalarize(preference)
    return scalarized


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
