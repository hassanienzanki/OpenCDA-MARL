from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opencda_marl.api.events import StepEvent
from opencda_marl.api.contracts import (
    REWARD_MODE_METRIC_ONLY,
    RewardOutput,
    RewardSpec,
    normalize_observation,
)
from opencda_marl.api.objectives import ObjectiveOutput, ObjectiveSpec
from opencda_marl.builtin.extractors import ObservationExtractor
from opencda_marl._kernel.policy.competition import build_competition_summary
from opencda_marl.evaluation.training_metrics import TrainingMetrics
from opencda_marl.api.rewards import resolve_reward_configuration
from opencda_marl.api.safety import TtcMetricState
from opencda_marl.builtin.rewards.calculator import RewardCalculator


def test_reward_spec_defaults_ttc_to_metric_only_when_disabled():
    spec = RewardSpec.from_config({}, {"ttc_max_penalty": 0.0})

    assert spec.mode_for("ttc") == REWARD_MODE_METRIC_ONLY
    assert spec.scalarize({"ttc": -5.0, "speed": 1.0}) == 1.0


def test_reward_output_keeps_ordered_vector_and_scalar():
    spec = RewardSpec.from_config(
        {"components": ["collision", "speed", "progress"]},
        {"collision": -10.0},
    )
    reward_output = RewardOutput.from_components(
        {"speed": 2.5, "progress": 1.0},
        spec,
    )

    assert reward_output.vector == [0.0, 2.5, 1.0]
    assert reward_output.scalar == 3.5


def test_normalize_observation_applies_shared_defaults():
    normalized = normalize_observation(
        {
            "speed": 22.0,
            "location": {"x": 5.0, "y": -2.0},
            "distance_to_intersection": 18.0,
        }
    )

    assert normalized["speed"] == 22.0
    assert normalized["relative_position_to_intersection"] == {"x": 5.0, "y": -2.0}
    assert normalized["distance_to_front_vehicle"] == 999.0
    assert normalized["distance_to_destination"] == 999.0
    assert len(normalized["nearby_vehicles"]) == 35


_RAW_OBSERVATIONS = [
    {},
    {"speed": 22.0, "location": {"x": 5.0, "y": -2.0}, "distance_to_intersection": 18.0},
    {   # SUMO-shaped
        "vehicle_id": "veh_7", "location": {"x": 1.0, "y": 2.0, "z": 0.0}, "speed": 30.0,
        "physics_speed": 30.0, "physics_speed_planar": 30.0, "physics_speed_3d": 30.0,
        "velocity": {"x": 8.3, "y": 0.1, "z": 0.0}, "heading_angle": 0.7,
        "relative_position_to_intersection": {"x": -20.0, "y": 3.0},
        "distance_to_intersection": 20.2, "distance_to_front_vehicle": 12.0,
        "distance_to_destination": 80.0, "lane_position": 40.0, "min_ttc": 3.5,
    },
    {   # CARLA-shaped with a short neighbour list, ints and a status
        "vehicle_id": 42, "speed": 10, "position_x": 3, "position_y": 4, "status": "active",
        "nearby_vehicles": [1.0, 2.0, 3.0], "min_ttc": float("inf"),
        "velocity": {"x": 1, "y": 0}, "waypoint_buffer_size": 5,
    },
]


def _extra_keys_added_after_normalization(obs):
    obs = dict(obs)
    obs["entry_direction"] = "north"
    obs["preference"] = [0.5, 0.5]
    obs["route_intent"] = 1
    return obs


def test_renormalizing_is_idempotent_and_canonical_only():
    """An observation is normalized several times per step (at creation, by
    the extractor for the current and next tables, by the reward calculator).
    Re-normalizing must be a no-op on values, keep the canonical key order,
    hand out fresh nested containers, and drop keys attached afterwards
    (SUMO metadata such as entry_direction / preference / route_intent) —
    consumers of a normalized observation may rely on canonical keys only.
    """
    from opencda_marl.api.contracts import NormalizedObservation

    canonical = set(NormalizedObservation.__annotations__)
    for raw in _RAW_OBSERVATIONS:
        first = normalize_observation(raw)
        assert set(first) == canonical
        again = normalize_observation(first)
        assert list(again) == list(first) and again == first
        for nested in ("relative_position_to_intersection", "location", "velocity", "nearby_vehicles"):
            assert again[nested] is not first[nested], nested
        assert all(isinstance(again[k], float) for k in
                   ("speed", "position_x", "position_y", "waypoint_buffer_size", "min_ttc"))
        with_extras = _extra_keys_added_after_normalization(first)
        assert normalize_observation(with_extras) == first
        assert len(normalize_observation(first, max_nearby_vehicles=2)["nearby_vehicles"]) == 14


def test_observation_extractor_rejects_unknown_features():
    try:
        ObservationExtractor(
            {
                "matd3": {
                    "features": {
                        "speed": 1,
                        "not_a_real_feature": 1,
                    }
                }
            },
            "matd3",
        )
    except ValueError as exc:
        assert "not_a_real_feature" in str(exc)
    else:
        raise AssertionError("Expected invalid feature config to raise ValueError")


def test_training_metrics_tracks_component_totals_and_spec():
    spec = RewardSpec.from_config({"components": ["collision", "speed", "progress"]})
    reward_outputs = {
        "veh_1": RewardOutput.from_components({"speed": 2.0, "progress": 1.0}, spec),
        "veh_2": RewardOutput.from_components({"collision": -5.0}, spec),
    }

    metrics = TrainingMetrics(export_interval=0)
    metrics.update_step(reward_outputs, reward_outputs=reward_outputs)
    current_metrics = metrics.get_current_metrics()

    assert current_metrics["total_reward"] == -2.0
    assert current_metrics["reward_component_totals"] == {
        "collision": -5.0,
        "speed": 2.0,
        "progress": 1.0,
    }
    assert current_metrics["reward_vector_dim"] == 3
    assert current_metrics["reward_component_order"] == ["collision", "speed", "progress"]
    assert current_metrics["reward_spec"]["components"] == ["collision", "speed", "progress"]


def test_training_metrics_tracks_grouped_objectives_and_utilities():
    reward_spec = RewardSpec.from_config({"components": ["collision", "success", "step_penalty", "progress"]})
    objective_spec = ObjectiveSpec.from_config(
        {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {
                    "success": 1.0,
                    "step_penalty": 1.0,
                    "progress": 1.0,
                },
                "safety": {
                    "collision": 1.0,
                },
            },
        }
    )

    reward_outputs = {
        "veh_1": RewardOutput.from_components({"progress": 0.4, "step_penalty": -0.8}, reward_spec),
        "veh_2": RewardOutput.from_components({"success": 300.0, "collision": -400.0}, reward_spec),
    }
    objective_outputs = {
        agent_id: ObjectiveOutput.from_reward_components(reward_output.components, objective_spec)
        for agent_id, reward_output in reward_outputs.items()
    }

    metrics = TrainingMetrics(export_interval=0)
    metrics.update_step(
        {"veh_1": 0.0, "veh_2": 0.0},
        reward_outputs=reward_outputs,
        objective_outputs=objective_outputs,
        scalarized_utilities={"veh_1": -0.2, "veh_2": -50.0},
        agent_preferences={"veh_1": [0.5, 0.5], "veh_2": [0.8, 0.2]},
    )
    current_metrics = metrics.get_current_metrics()

    assert current_metrics["total_utility"] == -50.2
    assert current_metrics["agent_utilities"] == {"veh_1": -0.2, "veh_2": -50.0}
    assert current_metrics["objective_component_totals"] == {
        "efficiency": 299.6,
        "safety": -400.0,
    }
    assert current_metrics["objective_vector_dim"] == 2
    assert current_metrics["objective_component_order"] == ["efficiency", "safety"]
    assert current_metrics["objective_spec"]["objectives"] == ["efficiency", "safety"]

    # Spec metadata is derived once per spec object (not re-serialised per
    # agent per step) and still equals the spec's own serialisation ...
    assert current_metrics["objective_spec"] == objective_spec.to_dict()
    assert current_metrics["reward_spec"] == reward_spec.to_dict()
    metadata_before = metrics.objective_spec_metadata
    metrics.update_step({"veh_1": 0.0}, reward_outputs=reward_outputs, objective_outputs=objective_outputs)
    assert metrics.objective_spec_metadata is metadata_before
    # ... while a different spec object refreshes it.
    other_spec = ObjectiveSpec.from_config({
        "objectives": ["efficiency", "safety"],
        "groups": {"efficiency": {"progress": 2.0}, "safety": {"collision": 1.0}},
    })
    metrics.update_step({"veh_1": 0.0}, objective_outputs={
        "veh_1": ObjectiveOutput.from_reward_components({"progress": 1.0}, other_spec)})
    assert metrics.objective_spec_metadata == other_spec.to_dict()
    assert metrics.objective_spec_metadata is not metadata_before


def test_training_metrics_flags_implausible_speed_summaries():
    metrics = TrainingMetrics(export_interval=0)
    metrics.update_step(
        {"veh_1": 0.0},
        observations={
            "veh_1": {
                "physics_speed": 65.0,
                "physics_speed_planar": 65.0,
                "physics_speed_3d": 420.0,
                "vertical_speed": 415.0,
                "agent_speed": 65.0,
                "speed": 65.0,
                "target_speed": 65.0,
            }
        },
        target_speeds={"veh_1": 65.0},
    )

    current_metrics = metrics.get_current_metrics()

    assert current_metrics["speed_metrics_suspicious"] is True
    assert "vertical_speed_max" in current_metrics["speed_metrics_warning"]
    assert current_metrics["speed_source"] == "carla_planar_physics"
    assert current_metrics["actual_speed_mean"] == 65.0
    assert current_metrics["agent_speed_mean"] == 65.0
    assert current_metrics["speed_3d_max"] == 420.0


def test_training_metrics_uses_physics_speed_as_canonical_actual_speed():
    metrics = TrainingMetrics(export_interval=0)
    metrics.update_step(
        {"veh_1": 0.0, "veh_2": 0.0},
        observations={
            "veh_1": {
                "physics_speed": 40.0,
                "physics_speed_planar": 40.0,
                "physics_speed_3d": 90.0,
                "vertical_speed": 80.0,
                "agent_speed": 35.0,
                "speed": 35.0,
                "target_speed": 42.0,
            },
            "veh_2": {
                "physics_speed": 50.0,
                "physics_speed_planar": 50.0,
                "physics_speed_3d": 50.0,
                "vertical_speed": 0.0,
                "agent_speed": 45.0,
                "speed": 45.0,
                "target_speed": 48.0,
            },
        },
        target_speeds={"veh_1": 42.0, "veh_2": 48.0},
    )

    current_metrics = metrics.get_current_metrics()

    assert current_metrics["actual_speed_mean"] == 45.0
    assert current_metrics["avg_speed"] == 45.0
    assert current_metrics["agent_speed_mean"] == 40.0
    assert current_metrics["target_speed_mean"] == 45.0
    assert current_metrics["speed_3d_mean"] == 70.0
    assert current_metrics["vertical_speed_max"] == 80.0


def test_training_metrics_records_speed_spike_context():
    metrics = TrainingMetrics(export_interval=0)
    metrics.update_step(
        {"veh_99": 0.0},
        observations={
            "veh_99": {
                "vehicle_id": 99,
                "vehicle_blueprint": "vehicle.tesla.model3",
                "physics_speed": 55.0,
                "physics_speed_planar": 55.0,
                "physics_speed_3d": 250.0,
                "vertical_speed": 220.0,
                "agent_speed": 60.0,
                "speed": 60.0,
                "target_speed": 55.0,
                "position_x": 10.0,
                "position_y": -5.0,
                "relative_position_to_intersection": {"x": 3.0, "y": -2.0},
                "location": {"x": 3.0, "y": -2.0},
                "velocity": {"x": 20.0, "y": 0.0},
                "distance_to_intersection": 12.0,
                "distance_to_front_vehicle": 999.0,
                "lane_position": 1,
                "status": True,
            }
        },
        target_speeds={"veh_99": 55.0},
    )

    current_metrics = metrics.get_current_metrics()

    assert current_metrics["speed_spike_count"] == 1
    first_spike = current_metrics["first_speed_spike"]
    assert first_spike["vehicle_id"] == 99
    assert first_spike["physics_speed"] == 55.0
    assert first_spike["agent_speed"] == 60.0
    assert first_spike["target_speed"] == 55.0
    assert first_spike["vehicle_blueprint"] == "vehicle.tesla.model3"
    assert "diverges from physics_speed" not in " ".join(first_spike["reasons"])


def test_training_metrics_records_physics_anomaly_events():
    metrics = TrainingMetrics(export_interval=0)
    anomaly_event = StepEvent(
        step=7,
        event_id="physics_7",
        vehicle_id=123,
        event_type="physics_anomaly",
        metadata={
            "severity": "fatal",
            "policy": "remove_actor",
            "reasons": ["vertical_speed=108.00 km/h exceeds fatal threshold 20.00 km/h"],
            "physics_speed_planar": 18.0,
            "physics_speed_3d": 120.0,
            "vertical_speed": 108.0,
            "speed_3d_gap": 102.0,
            "tilt_pitch": 26.0,
            "tilt_roll": 0.0,
            "height_above_waypoint": 2.5,
            "location_z": 3.0,
            "waypoint_z": 0.5,
            "target_speed": 30.0,
        },
    )

    metrics.record_physics_anomaly_events([anomaly_event])
    current_metrics = metrics.get_current_metrics()

    assert current_metrics["physics_anomaly_count"] == 1
    assert current_metrics["first_physics_anomaly"]["vehicle_id"] == 123
    assert current_metrics["max_physics_anomaly"]["physics_speed_3d"] == 120.0


def test_step_event_accepts_string_vehicle_ids():
    event = StepEvent(step=1, event_id="success_veh_1", vehicle_id="veh_1", event_type="success")

    assert event.vehicle_id == "veh_1"


def test_cooperative_intersection_profile_enables_progress_and_safety_components():
    resolved = resolve_reward_configuration({"reward_profile": "cooperative_intersection_v1"})

    assert resolved.profile_name == "cooperative_intersection_v1"
    assert resolved.reward_spec.mode_for("progress") == "learn"
    assert resolved.reward_spec.mode_for("collision") == "learn"
    assert resolved.reward_spec.mode_for("ttc") == "disabled"
    assert resolved.reward_spec.mode_for("yielding") == "disabled"
    assert resolved.reward_spec.mode_for("clearance") == "disabled"
    assert resolved.reward_spec.mode_for("speed") == "disabled"
    assert resolved.reward_spec.mode_for("stop") == "disabled"
    assert resolved.reward_params["collision"] == -400.0
    assert resolved.reward_params["success"] == 300.0


def test_competitive_intersection_profile_keeps_ttc_and_yielding_metric_only():
    resolved = resolve_reward_configuration({"reward_profile": "competitive_intersection_v1"})

    assert resolved.profile_name == "competitive_intersection_v1"
    assert resolved.reward_spec.mode_for("progress") == "learn"
    assert resolved.reward_spec.mode_for("collision") == "learn"
    assert resolved.reward_spec.mode_for("ttc") == "metric_only"
    assert resolved.reward_spec.mode_for("yielding") == "metric_only"
    assert resolved.reward_spec.mode_for("clearance") == "metric_only"
    assert resolved.reward_spec.mode_for("speed") == "disabled"
    assert resolved.reward_spec.mode_for("stop") == "disabled"


def test_build_competition_summary_tracks_crossing_order_and_group_stats():
    summary = build_competition_summary(
        [
            {
                "agent_id": "veh_a",
                "outcome": "success",
                "completion_step": 10,
                "entry_direction": "north",
                "preference": [0.6, 0.4],
                "preference_group": "group_a",
            },
            {
                "agent_id": "veh_b",
                "outcome": "success",
                "completion_step": 12,
                "entry_direction": "south",
                "preference": [0.4, 0.6],
                "preference_group": "group_b",
            },
            {
                "agent_id": "veh_c",
                "outcome": "collision",
                "completion_step": 11,
                "entry_direction": "east",
                "preference": [0.6, 0.4],
                "preference_group": "group_a",
            },
            {
                "agent_id": "veh_d",
                "outcome": "timeout",
                "completion_step": 30,
                "entry_direction": "west",
                "preference": [0.4, 0.6],
                "preference_group": "group_b",
            },
        ],
        yielding_total=3.0,
    )

    assert summary["crossing_order"] == ["veh_a", "veh_b"]
    assert summary["crossing_order_stats"]["first_mover"] == "north"
    assert summary["crossing_order_stats"]["first_mover_by_entry_direction"] == {"north": 1}
    assert summary["yielding_summary"]["yielding_total"] == 3.0
    assert summary["yielding_summary"]["yielding_per_success"] == 1.5
    assert summary["competition_group_stats"]["group_a"]["collision_count"] == 1
    assert summary["competition_group_stats"]["group_b"]["timeout_count"] == 1


def test_training_metrics_include_competition_summary_when_records_are_present():
    metrics = TrainingMetrics(export_interval=0)
    metrics.current_reward_component_totals["yielding"] = 2.0
    metrics.record_competition_records(
        [
            {
                "agent_id": "veh_a",
                "outcome": "success",
                "completion_step": 8,
                "entry_direction": "north",
                "preference": [0.6, 0.4],
                "preference_group": "group_a",
            },
            {
                "agent_id": "veh_b",
                "outcome": "success",
                "completion_step": 9,
                "entry_direction": "south",
                "preference": [0.4, 0.6],
                "preference_group": "group_b",
            },
        ]
    )
    metrics.agent_rewards = {"veh_a": 0.0}

    current_metrics = metrics.get_current_metrics()

    assert current_metrics["competition_summary"]["crossing_order"] == ["veh_a", "veh_b"]
    assert current_metrics["competition_summary"]["yielding_summary"]["yielding_total"] == 2.0


def test_sumo_reward_calculator_emits_progress_and_safety_proxies_when_enabled():
    resolved = resolve_reward_configuration(
        {
            "reward_profile": "cooperative_intersection_v1",
            "reward_spec": {
                "enabled": {
                    "ttc": "learn",
                    "yielding": "learn",
                    "clearance": "learn",
                }
            },
            "rewards": {
                "ttc_max_penalty": 5.0,
                "clearance_speed_bonus": 0.3,
                "yielding_bonus": 0.5,
                "stop_threshold": 3.0,
                "stop_penalty": -1.5,
            },
        }
    )
    calculator = RewardCalculator(resolved.reward_params, resolved.reward_spec)
    ttc_metrics = TtcMetricState()

    previous_observations = {
        "veh_1": normalize_observation(
            {
                "vehicle_id": "veh_1",
                "speed": 36.0,
                "distance_to_intersection": 20.0,
                "distance_to_destination": 80.0,
                "distance_to_front_vehicle": 20.0,
                "lane_position": 1.0,
                "min_ttc": 4.5,
            }
        )
    }
    observations = {
        "veh_1": {
            "vehicle_id": "veh_1",
            "speed": 24.0,
            "distance_to_intersection": 14.0,
            "distance_to_destination": 72.0,
            "distance_to_front_vehicle": 45.0,
            "lane_position": 1.0,
            "min_ttc": 1.5,
        }
    }

    reward_outputs = calculator.calculate_sumo(
        observations,
        [],
        previous_observations=previous_observations,
        terminal_agents=set(),
        ttc_metrics=ttc_metrics,
    )
    components = reward_outputs["veh_1"].components

    assert components["progress"] > 0.0
    assert components["ttc"] < 0.0
    assert components["yielding"] > 0.0
    assert components["clearance"] > 0.0
    assert ttc_metrics.violation_count == 1
    assert ttc_metrics.near_miss_count == 1
