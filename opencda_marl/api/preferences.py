"""Per-agent preference assignment for the PCMA learners.

Who holds which preference vector: a preference supplied in the vehicle's
spawn metadata (the traffic manager's per-vehicle sampler) wins; otherwise
a named profile assigns one by entry direction. See docs/marl/pcma.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np


BUILTIN_PREFERENCE_PROFILES: Dict[str, Dict[str, Sequence[float] | Sequence[str]]] = {
    "balanced": {
        "default": [0.5, 0.5],
    },
    "asymmetric_split": {
        "group_a": [0.8, 0.2],
        "group_b": [0.2, 0.8],
        "group_a_directions": ("north", "east"),
        "group_b_directions": ("south", "west"),
    },
    "reversed_asymmetric_split": {
        "group_a": [0.2, 0.8],
        "group_b": [0.8, 0.2],
        "group_a_directions": ("north", "east"),
        "group_b_directions": ("south", "west"),
    },
    "mild_asymmetric_split": {
        "group_a": [0.6, 0.4],
        "group_b": [0.4, 0.6],
        "group_a_directions": ("north", "east"),
        "group_b_directions": ("south", "west"),
    },
}


def register_preference_profile(name: str, profile: Mapping[str, Any], *,
                                override: bool = False) -> None:
    """Register a preference profile under ``name`` (``MARL.preference.profile``).

    ``profile`` has the built-in shape: either a ``default`` vector, or
    ``group_a`` / ``group_b`` vectors with their ``*_directions`` tuples.
    Import-time registration — the same plugin door as algorithms and
    components.
    """
    key = str(name)
    if key in BUILTIN_PREFERENCE_PROFILES and not override:
        raise ValueError(
            f"preference profile {key!r} is already registered; pass "
            "override=True to replace it deliberately.")
    from opencda_marl.plugins import registry as _plugin_registry

    _plugin_registry.guard("preference_profile", key)
    BUILTIN_PREFERENCE_PROFILES[key] = dict(profile)


def _kind_adapter() -> dict:
    return {
        "describe": lambda: [{"key": k, "source": "builtin"}
                             for k in BUILTIN_PREFERENCE_PROFILES],
        "remove": lambda key: BUILTIN_PREFERENCE_PROFILES.pop(key, None),
    }


@dataclass(frozen=True)
class PreferenceConfig:
    profile: str
    assignment: str = "entry_direction"
    injection: str = "state_augmentation"
    dim: int = 2


class PreferenceRegistry:
    def __init__(self, config: Mapping[str, Any] | None = None):
        config = dict(config or {})
        profile_name = str(config.get("profile", "balanced")).strip() or "balanced"
        if profile_name not in BUILTIN_PREFERENCE_PROFILES:
            available = ", ".join(sorted(BUILTIN_PREFERENCE_PROFILES))
            raise ValueError(
                f"Unknown preference profile '{profile_name}'. Available profiles: {available}"
            )

        dim = int(config.get("dim", 2))
        if dim != 2:
            raise ValueError("The preference registry only supports 2D preferences.")

        self.config = PreferenceConfig(
            profile=profile_name,
            assignment=str(config.get("assignment", "entry_direction")),
            injection=str(config.get("injection", "state_augmentation")),
            dim=dim,
        )
        self._profile = BUILTIN_PREFERENCE_PROFILES[profile_name]
        self._preferences: Dict[Any, np.ndarray] = {}

    @property
    def profile_name(self) -> str:
        return self.config.profile

    @property
    def pref_dim(self) -> int:
        return self.config.dim

    def clear(self) -> None:
        self._preferences.clear()

    def ensure(self, agent_id: Any, metadata: Mapping[str, Any] | None = None) -> np.ndarray:
        key = str(agent_id)
        if key not in self._preferences:
            self._preferences[key] = self._assign_preference(key, metadata)
        return self._preferences[key].copy()

    def ensure_many(self, observations: Mapping[Any, Mapping[str, Any]] | None) -> Dict[str, list[float]]:
        for agent_id, observation in dict(observations or {}).items():
            metadata = observation if isinstance(observation, Mapping) else {}
            self.ensure(agent_id, metadata)
        return self.to_dict(observations.keys() if observations else None)

    def get(self, agent_id: Any) -> np.ndarray | None:
        preference = self._preferences.get(str(agent_id))
        return preference.copy() if preference is not None else None

    def get_many(self, agent_ids: Sequence[Any] | None = None) -> Dict[str, np.ndarray]:
        if agent_ids is None:
            return {agent_id: pref.copy() for agent_id, pref in self._preferences.items()}
        return {
            str(agent_id): self._preferences[str(agent_id)].copy()
            for agent_id in agent_ids
            if str(agent_id) in self._preferences
        }

    def to_dict(self, agent_ids: Sequence[Any] | None = None) -> Dict[str, list[float]]:
        preferences = self.get_many(agent_ids)
        return {
            agent_id: [float(value) for value in preference]
            for agent_id, preference in preferences.items()
        }

    def _assign_preference(self, agent_id: Any, metadata: Mapping[str, Any] | None = None) -> np.ndarray:
        metadata = dict(metadata or {})
        if "preference" in metadata and metadata.get("preference") is not None:
            preference = np.asarray(metadata["preference"], dtype=np.float32)
            if preference.ndim != 1 or preference.shape[0] != self.config.dim:
                raise ValueError(
                    f"Metadata preference for {agent_id!r} has shape {preference.shape}; "
                    f"expected ({self.config.dim},)."
                )
            return _normalize_preference(preference)

        direction = str(metadata.get("entry_direction", "") or "").strip().lower()

        if "default" in self._profile:
            return _normalize_preference(self._profile["default"])

        group_a = _normalize_preference(self._profile["group_a"])
        group_b = _normalize_preference(self._profile["group_b"])
        group_a_directions = tuple(str(value) for value in self._profile.get("group_a_directions", ()))
        group_b_directions = tuple(str(value) for value in self._profile.get("group_b_directions", ()))

        if direction in group_a_directions:
            return group_a
        if direction in group_b_directions:
            return group_b

        return group_a if _stable_bucket(agent_id) == 0 else group_b


def _normalize_preference(values: Sequence[float]) -> np.ndarray:
    preference = np.asarray(list(values), dtype=np.float32)
    if preference.ndim != 1 or preference.shape[0] != 2:
        raise ValueError("Preferences must be 2D vectors.")

    preference = np.clip(preference, 1e-6, None)
    total = float(preference.sum())
    if total <= 0.0:
        raise ValueError("Preference vectors must have positive mass.")
    return (preference / total).astype(np.float32)


def _stable_bucket(agent_id: Any) -> int:
    text = str(agent_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits) % 2
    return sum(ord(ch) for ch in text) % 2
