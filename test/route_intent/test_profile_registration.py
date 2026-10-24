"""Reward / objective / preference profiles register through the plugin door.

An external module calls the register function at import time and the name
works everywhere the config reads it (``MARL.reward_profile``,
``MARL.objective_mode``, ``MARL.preference.profile``); duplicates need
``override=True``.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def clean_tables():
    from opencda_marl.api import objectives, preferences as preference_registry
    from opencda_marl.api import rewards as profiles

    snap = (dict(profiles.BUILTIN_REWARD_PROFILES),
            dict(objectives.BUILTIN_OBJECTIVE_PROFILES),
            dict(preference_registry.BUILTIN_PREFERENCE_PROFILES))
    yield profiles, objectives, preference_registry
    profiles.BUILTIN_REWARD_PROFILES.clear()
    profiles.BUILTIN_REWARD_PROFILES.update(snap[0])
    objectives.BUILTIN_OBJECTIVE_PROFILES.clear()
    objectives.BUILTIN_OBJECTIVE_PROFILES.update(snap[1])
    preference_registry.BUILTIN_PREFERENCE_PROFILES.clear()
    preference_registry.BUILTIN_PREFERENCE_PROFILES.update(snap[2])


def test_reward_profile_registers_and_resolves(clean_tables):
    profiles, _, _ = clean_tables
    profiles.register_reward_profile("lab_rewards", {
        "reward_spec": {"scalarizer": "weighted_sum",
                        "enabled": {"collision": "learn", "success": "learn",
                                    "step_penalty": "learn", "progress": "learn"}},
        "rewards": {"collision": -50.0, "success": 500.0},
    })
    resolved = profiles.resolve_reward_configuration({"reward_profile": "lab_rewards"})
    assert resolved.profile_name == "lab_rewards"
    assert resolved.reward_params["collision"] == -50.0
    with pytest.raises(ValueError, match="already registered"):
        profiles.register_reward_profile("lab_rewards", {})


def test_objective_profile_registers_and_resolves(clean_tables):
    _, objectives, _ = clean_tables
    objectives.register_objective_profile("lab_objectives", {
        "objective_spec": {
            "objectives": ["efficiency", "comfort"],
            "groups": {"efficiency": {"progress": 1.0, "success": 1.0},
                       "comfort": {"step_penalty": 1.0}},
        },
    })
    resolved = objectives.resolve_objective_configuration(
        {"objective_mode": "lab_objectives"})
    assert resolved.objective_spec.objectives == ("efficiency", "comfort")
    with pytest.raises(ValueError, match="already registered"):
        objectives.register_objective_profile("lab_objectives", {})


def test_preference_profile_registers_and_assigns(clean_tables):
    _, _, preference_registry = clean_tables
    preference_registry.register_preference_profile("lab_split", {
        "group_a": [0.9, 0.1],
        "group_b": [0.1, 0.9],
        "group_a_directions": ("north",),
        "group_b_directions": ("south",),
    })
    reg = preference_registry.PreferenceRegistry({"profile": "lab_split"})
    north = reg.ensure("v1", {"entry_direction": "north"})
    assert north[0] > north[1]
    with pytest.raises(ValueError, match="already registered"):
        preference_registry.register_preference_profile("lab_split", {})


def test_unknown_profiles_stay_loud(clean_tables):
    profiles, objectives, preference_registry = clean_tables
    with pytest.raises(ValueError, match="Unknown reward profile"):
        profiles.resolve_reward_configuration({"reward_profile": "nope"})
    with pytest.raises(ValueError, match="Unknown objective mode"):
        objectives.resolve_objective_configuration({"objective_mode": "nope"})
    with pytest.raises(ValueError, match="Unknown preference profile"):
        preference_registry.PreferenceRegistry({"profile": "nope"})
