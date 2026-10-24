"""Reward calculator and reward profile resolution tests."""

from pathlib import Path
import sys

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from test_helpers import merged_scenario_config

from opencda_marl.api.events import StepEvent
from opencda_marl.api.rewards import (
    BUILTIN_REWARD_PROFILES,
    resolve_reward_configuration,
)
from opencda_marl.api.safety import TtcMetricState
from opencda_marl.builtin.rewards.calculator import RewardCalculator


# ═══════════════════════════════════════════════════════════════════════════
# Tests: reward calculator (from test_reward_calculator.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_simple_progress_profile_disables_dense_components_but_keeps_progress():
    resolved = resolve_reward_configuration({"reward_profile": "simple_progress_v1"})
    calculator = RewardCalculator(resolved.reward_params, resolved.reward_spec)
    ttc_metrics = TtcMetricState()

    observations = {
        1: {
            "physics_speed_planar": 50.0,
            "speed": 50.0,
            "distance_to_front_vehicle": 100.0,
            "distance_to_intersection": 15.0,
            "distance_to_destination": 60.0,
            "min_ttc": 1.5,
        }
    }
    previous_observations = {
        1: {
            "physics_speed_planar": 55.0,
            "speed": 55.0,
            "distance_to_front_vehicle": 100.0,
            "distance_to_intersection": 20.0,
            "distance_to_destination": 65.0,
            "min_ttc": 2.5,
        }
    }

    reward_outputs = calculator.calculate_carla(
        [1],
        observations,
        [],
        previous_observations=previous_observations,
        terminal_agents=set(),
        ttc_metrics=ttc_metrics,
    )
    components = reward_outputs[1].components

    assert set(components) == {
        "collision",
        "success",
        "step_penalty",
        "speed",
        "stop",
        "progress",
        "ttc",
        "yielding",
        "clearance",
    }
    assert components["step_penalty"] == -0.5
    assert components["progress"] == 0.5
    assert components["ttc"] == 0.0
    assert components["speed"] == 0.0
    assert components["stop"] == 0.0
    assert components["yielding"] == 0.0
    assert components["clearance"] == 0.0
    assert len(reward_outputs[1].vector) == 9
    assert reward_outputs[1].scalar == 0.0
    assert ttc_metrics.check_count == 1
    assert ttc_metrics.violation_count == 1
    assert ttc_metrics.near_miss_count == 1


def test_terminal_success_overrides_non_terminal_shaping():
    resolved = resolve_reward_configuration({"reward_profile": "simple_progress_v1"})
    calculator = RewardCalculator(resolved.reward_params, resolved.reward_spec)
    ttc_metrics = TtcMetricState()
    terminal_agents = set()

    reward_outputs = calculator.calculate_carla(
        [7],
        {
            7: {
                "speed": 40.0,
                "distance_to_intersection": 5.0,
                "distance_to_destination": 10.0,
                "min_ttc": float("inf"),
            }
        },
        [
            StepEvent(
                step=10,
                event_id="success_7_10",
                vehicle_id=7,
                event_type="success",
            )
        ],
        previous_observations={
            7: {
                "speed": 45.0,
                "distance_to_intersection": 8.0,
                "distance_to_destination": 15.0,
                "min_ttc": float("inf"),
            }
        },
        terminal_agents=terminal_agents,
        ttc_metrics=ttc_metrics,
    )

    components = reward_outputs[7].components
    assert set(components) == {
        "collision",
        "success",
        "step_penalty",
        "speed",
        "stop",
        "progress",
        "ttc",
        "yielding",
        "clearance",
    }
    assert components["success"] == 130.0
    assert components["step_penalty"] == 0.0
    assert components["progress"] == 0.0
    assert components["ttc"] == 0.0
    assert 7 in terminal_agents


def test_terminal_success_is_preserved_even_when_env_already_marked_agent_terminal():
    resolved = resolve_reward_configuration({"reward_profile": "simple_progress_v1"})
    calculator = RewardCalculator(resolved.reward_params, resolved.reward_spec)
    ttc_metrics = TtcMetricState()
    terminal_agents = {9}

    reward_outputs = calculator.calculate_sumo(
        {},
        [
            StepEvent(
                step=5,
                event_id="success_9_5",
                vehicle_id=9,
                event_type="success",
            )
        ],
        previous_observations={},
        terminal_agents=terminal_agents,
        ttc_metrics=ttc_metrics,
    )

    components = reward_outputs[9].components
    assert components["success"] == 130.0
    assert components["step_penalty"] == 0.0
    assert 9 in terminal_agents


def test_sumo_progress_falls_back_to_intersection_distance():
    resolved = resolve_reward_configuration({"reward_profile": "simple_progress_v1"})
    calculator = RewardCalculator(resolved.reward_params, resolved.reward_spec)
    ttc_metrics = TtcMetricState()

    reward_outputs = calculator.calculate_sumo(
        {
            "veh_1": {
                "speed": 35.0,
                "distance_to_intersection": 25.0,
            }
        },
        [],
        previous_observations={
            "veh_1": {
                "speed": 32.0,
                "distance_to_intersection": 30.0,
            }
        },
        terminal_agents=set(),
        ttc_metrics=ttc_metrics,
    )

    components = reward_outputs["veh_1"].components
    assert set(components) == {
        "collision",
        "success",
        "step_penalty",
        "speed",
        "stop",
        "progress",
        "ttc",
        "yielding",
        "clearance",
    }
    assert components["progress"] == 0.5
    assert components["ttc"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: reward profiles (from test_reward_profiles.py)
# ═══════════════════════════════════════════════════════════════════════════

def test_ttc_component_still_works_when_enabled_by_reward_spec():
    resolved = resolve_reward_configuration(
        {
            "reward_spec": {
                "enabled": {
                    "ttc": "learn",
                },
            },
            "rewards": {
                "ttc_max_penalty": 5.0,
                "ttc_decay_rate": 1.0,
                "ttc_safe_threshold": 4.0,
                "ttc_near_miss_threshold": 2.0,
            },
        }
    )
    calculator = RewardCalculator(resolved.reward_params, resolved.reward_spec)
    ttc_metrics = TtcMetricState()

    reward_outputs = calculator.calculate_carla(
        [11],
        {
            11: {
                "speed": 35.0,
                "distance_to_intersection": 10.0,
                "distance_to_destination": 50.0,
                "min_ttc": 1.0,
            }
        },
        [],
        previous_observations={
            11: {
                "speed": 35.0,
                "distance_to_intersection": 12.0,
                "distance_to_destination": 52.0,
                "min_ttc": 1.5,
            }
        },
        terminal_agents=set(),
        ttc_metrics=ttc_metrics,
    )

    components = reward_outputs[11].components
    assert components["ttc"] < 0.0
    assert ttc_metrics.check_count == 1
    assert ttc_metrics.violation_count == 1
    assert ttc_metrics.near_miss_count == 1
    assert len(reward_outputs[11].vector) == 9


def test_shared_simple_reward_profile_is_used_by_main_baselines():
    scenario_refs = [
        "marl/intersection/mappo",
        "marl/intersection/sac",
        "marl/intersection/td3",
        "cosim-marl/intersection/sac",
    ]

    for scenario_ref in scenario_refs:
        merged = merged_scenario_config(scenario_ref)
        marl_cfg = merged["MARL"]
        resolved = resolve_reward_configuration(marl_cfg)

        assert marl_cfg["reward_profile"] == "simple_progress_v1", scenario_ref
        assert resolved.profile_name == "simple_progress_v1", scenario_ref
        assert resolved.reward_spec.mode_for("collision") == "learn", scenario_ref
        assert resolved.reward_spec.mode_for("success") == "learn", scenario_ref
        assert resolved.reward_spec.mode_for("step_penalty") == "learn", scenario_ref
        assert resolved.reward_spec.mode_for("progress") == "learn", scenario_ref
        assert resolved.reward_spec.components == (
            "collision",
            "success",
            "step_penalty",
            "speed",
            "stop",
            "progress",
            "ttc",
            "yielding",
            "clearance",
        ), scenario_ref
        assert resolved.reward_spec.mode_for("ttc") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("speed") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("stop") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("yielding") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("clearance") == "disabled", scenario_ref


def test_cooperative_reward_profile_is_used_by_experimental_benchmarks():
    scenario_refs = [
        "pcma/intersection/masac",
        "pcma/intersection/smoke",
        "sumo-marl/intersection/sac",
    ]

    for scenario_ref in scenario_refs:
        merged = merged_scenario_config(scenario_ref)
        marl_cfg = merged["MARL"]
        resolved = resolve_reward_configuration(marl_cfg)

        assert marl_cfg["reward_profile"] == "cooperative_intersection_v1", scenario_ref
        assert resolved.profile_name == "cooperative_intersection_v1", scenario_ref
        assert resolved.reward_spec.mode_for("collision") == "learn", scenario_ref
        assert resolved.reward_spec.mode_for("progress") == "learn", scenario_ref
        assert resolved.reward_spec.mode_for("ttc") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("yielding") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("clearance") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("speed") == "disabled", scenario_ref
        assert resolved.reward_spec.mode_for("stop") == "disabled", scenario_ref
        assert resolved.reward_params["collision"] == -400.0, scenario_ref
        assert resolved.reward_params["success"] == 300.0, scenario_ref


def test_reward_profile_overrides_apply_after_profile_defaults():
    resolved = resolve_reward_configuration(
        {
            "reward_profile": "simple_progress_v1",
            "rewards": {
                "collision": -250.0,
                "progress_scale": 0.8,
            },
        }
    )

    assert resolved.reward_params["collision"] == -250.0
    assert resolved.reward_params["progress_scale"] == 0.8
    assert resolved.reward_params["step_penalty"] == -0.5
    assert "ttc_safe_threshold" not in BUILTIN_REWARD_PROFILES["simple_progress_v1"]["rewards"]
    assert resolved.reward_spec.mode_for("ttc") == "disabled"
    assert resolved.reward_spec.mode_for("speed") == "disabled"


# ── runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
