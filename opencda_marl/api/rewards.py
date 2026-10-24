from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from opencda_marl.api.contracts import RewardSpec


DEFAULT_REWARD_PARAMS: Dict[str, float] = {
    "collision": -120.0,
    "success": 120.0,
    "step_penalty": -0.5,
    "speed_bonus": 0.0,
    "speed_threshold": 40.0,
    "ttc_safe_threshold": 4.0,
    "ttc_caution_threshold": 2.0,
    "ttc_danger_threshold": 1.0,
    "ttc_caution_penalty": -0.5,
    "ttc_danger_penalty": -2.0,
    "ttc_critical_penalty": -5.0,
    "ttc_max_penalty": 0.0,
    "ttc_decay_rate": 1.5,
    "ttc_near_miss_threshold": 2.0,
    "clearance_speed_bonus": 0.0,
    "clearance_threshold": 30.0,
    "progress_scale": 0.5,
    "junction_threshold": 5.0,
    "stop_threshold": 5.0,
    "stop_penalty": -3.0,
    "yielding_bonus": 0.0,
    "yielding_ttc_threshold": 3.0,
    "yielding_speed_drop": 5.0,
}


BUILTIN_REWARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "simple_progress_v1": {
        "reward_spec": {
            "scalarizer": "weighted_sum",
            "enabled": {
                "collision": "learn",
                "success": "learn",
                "step_penalty": "learn",
                "speed": "disabled",
                "stop": "disabled",
                "progress": "learn",
                "ttc": "disabled",
                "yielding": "disabled",
                "clearance": "disabled",
            },
        },
        "rewards": {
            "collision": -130.0,
            "success": 130.0,
            "step_penalty": -0.5,
            "progress_scale": 0.5,
            "junction_threshold": 5.0,
        },
    },
    "dense_intersection_v1": {
        "reward_spec": {
            "scalarizer": "weighted_sum",
            "enabled": {
                "collision": "learn",
                "success": "learn",
                "step_penalty": "learn",
                "speed": "learn",
                "stop": "learn",
                "progress": "learn",
                "ttc": "metric_only",
                "yielding": "learn",
                "clearance": "learn",
            },
        },
        "rewards": {
            "collision": -500.0,
            "success": 400.0,
            "step_penalty": -0.8,
            "speed_bonus": 0.3,
            "speed_threshold": 45.0,
            "ttc_safe_threshold": 4.0,
            "ttc_max_penalty": 0.0,
            "ttc_decay_rate": 1.5,
            "ttc_near_miss_threshold": 2.0,
            "clearance_speed_bonus": 0.3,
            "clearance_threshold": 30.0,
            "progress_scale": 0.4,
            "junction_threshold": 5.0,
            "stop_threshold": 3.0,
            "stop_penalty": -1.5,
            "yielding_bonus": 0.5,
            "yielding_ttc_threshold": 3.0,
            "yielding_speed_drop": 5.0,
        },
    },
    "cooperative_intersection_v1": {
        "reward_spec": {
            "scalarizer": "weighted_sum",
            "enabled": {
                "collision": "learn",
                "success": "learn",
                "step_penalty": "learn",
                "speed": "disabled",
                "stop": "disabled",
                "progress": "learn",
                "ttc": "disabled",
                "yielding": "disabled",
                "clearance": "disabled",
            },
        },
        "rewards": {
            "collision": -400.0,
            "success": 300.0,
            "step_penalty": -0.8,
            "progress_scale": 0.4,
            "junction_threshold": 5.0,
        },
    },
    "competitive_intersection_v1": {
        "reward_spec": {
            "scalarizer": "weighted_sum",
            "enabled": {
                "collision": "learn",
                "success": "learn",
                "step_penalty": "learn",
                "speed": "disabled",
                "stop": "disabled",
                "progress": "learn",
                "ttc": "metric_only",
                "yielding": "metric_only",
                "clearance": "metric_only",
            },
        },
        "rewards": {
            "collision": -400.0,
            "success": 300.0,
            "step_penalty": -0.8,
            "progress_scale": 0.4,
            "junction_threshold": 5.0,
            "ttc_max_penalty": 0.0,
            "yielding_bonus": 0.5,
            "yielding_ttc_threshold": 3.0,
            "yielding_speed_drop": 5.0,
            "clearance_speed_bonus": 0.3,
            "clearance_threshold": 30.0,
        },
    },
    # v2 is identical to v1 except that the success reward is +5000 (v1:
    # +300), so a successful trip outweighs the accumulated step_penalty
    # (-1920 per agent over 2400 steps) and the collision penalty (-400).
    # Per-agent arithmetic under v2:
    #   success @ step 1200 + 100m progress  =  +5000 + 40 - 960  = +4080
    #   stand-still timeout                  =  0     +  0 - 1920 = -1920
    #   crash @ step 500 + 20m progress      = -400  +  8 - 400  = -792
    # Order: success +4080 >>> crash -792 > freeze -1920. Freezing is the
    # worst option, so a policy gains nothing by standing still.
    "competitive_intersection_v2": {
        "reward_spec": {
            "scalarizer": "weighted_sum",
            "enabled": {
                "collision": "learn",
                "success": "learn",
                "step_penalty": "learn",
                "speed": "disabled",
                "stop": "disabled",
                "progress": "learn",
                "ttc": "metric_only",
                "yielding": "metric_only",
                "clearance": "metric_only",
            },
        },
        "rewards": {
            "collision": -400.0,
            "success": 5000.0,
            "step_penalty": -0.8,
            "progress_scale": 0.4,
            "junction_threshold": 5.0,
            "ttc_max_penalty": 0.0,
            "yielding_bonus": 0.5,
            "yielding_ttc_threshold": 3.0,
            "yielding_speed_drop": 5.0,
            "clearance_speed_bonus": 0.3,
            "clearance_threshold": 30.0,
        },
    },
}


def register_reward_profile(name: str, profile: Mapping[str, Any], *,
                            override: bool = False) -> None:
    """Register a reward profile under ``name`` (``MARL.reward_profile``).

    ``profile`` has the same shape as the built-in rows: an optional
    ``reward_spec`` block (enabled components + scalarizer) and a
    ``rewards`` block of parameter overrides. Registration happens at
    import time, so an external package adds profiles through the same
    plugin door as algorithms and components.
    """
    key = str(name)
    if key in BUILTIN_REWARD_PROFILES and not override:
        raise ValueError(
            f"reward profile {key!r} is already registered; pass "
            "override=True to replace it deliberately.")
    from opencda_marl.plugins import registry as _plugin_registry

    _plugin_registry.guard("reward_profile", key)
    BUILTIN_REWARD_PROFILES[key] = deepcopy(dict(profile))


def _kind_adapter() -> dict:
    return {
        "describe": lambda: [{"key": k, "source": "builtin"}
                             for k in BUILTIN_REWARD_PROFILES],
        "remove": lambda key: BUILTIN_REWARD_PROFILES.pop(key, None),
    }


@dataclass(frozen=True)
class ResolvedRewardConfig:
    profile_name: str
    reward_params: Dict[str, float]
    reward_spec_config: Dict[str, Any]
    reward_spec: RewardSpec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "reward_params": dict(self.reward_params),
            "reward_spec": dict(self.reward_spec_config),
            "resolved_spec": self.reward_spec.to_dict(),
        }


def resolve_reward_configuration(config: Mapping[str, Any] | None = None) -> ResolvedRewardConfig:
    config = dict(config or {})
    reward_profile_name = str(config.get("reward_profile") or "").strip()

    reward_params = dict(DEFAULT_REWARD_PARAMS)
    reward_spec_config: Dict[str, Any] = {}

    if reward_profile_name:
        profile = BUILTIN_REWARD_PROFILES.get(reward_profile_name)
        if profile is None:
            available = ", ".join(sorted(BUILTIN_REWARD_PROFILES))
            raise ValueError(
                f"Unknown reward profile '{reward_profile_name}'. Available profiles: {available}"
            )
        reward_params.update(profile.get("rewards", {}))
        reward_spec_config = _deep_merge_dicts(reward_spec_config, profile.get("reward_spec", {}))
    else:
        reward_profile_name = "custom"

    reward_params.update(dict(config.get("rewards", {}) or {}))
    reward_spec_config = _deep_merge_dicts(
        reward_spec_config,
        dict(config.get("reward_spec", {}) or {}),
    )
    reward_spec = RewardSpec.from_config(reward_spec_config, reward_params)

    return ResolvedRewardConfig(
        profile_name=reward_profile_name,
        reward_params=reward_params,
        reward_spec_config=reward_spec_config,
        reward_spec=reward_spec,
    )


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
