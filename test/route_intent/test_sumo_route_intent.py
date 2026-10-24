"""SUMO route-intent hook tests without launching SUMO or CARLA."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import numpy as np

from opencda_marl.api.events import StepEvent
from opencda_marl._kernel.traffic.sumo_route_intent import (
    apply_route_decision_to_event,
    make_active_vehicle_proxy,
    make_spawn_event_proxy,
)
from opencda_marl.evaluation.objective_metrics import summarize_intent_decisions
from opencda_marl._kernel.policy.intent_bridge import make_marl_intent_provider
from opencda_marl.builtin.intent_providers.providers import make_rule_based_path_provider

from test_route_intent import _spawn_event_stub, _three_way_planner  # type: ignore


def _tf(x=0.0, y=0.0, yaw=0.0):
    return SimpleNamespace(
        location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
        rotation=SimpleNamespace(yaw=float(yaw)),
    )


def _sumo_event(route_id=11):
    event = _spawn_event_stub(route_id=route_id)
    event.vehicle_id = "sumo_vehicle_0"
    event.transform = _tf(0.0, 0.0, 0.0)
    event.destination = _tf(20.0, 0.0, 0.0)
    event.target_speed = 30.0
    event.metadata.update({
        "route_type": "straight",
        "exit_direction": "south",
        "preference": [0.5, 0.5],
    })
    return event


def test_sumo_route_hook_applies_same_exit_route_change():
    planner = _three_way_planner()
    event = _sumo_event()
    queued_original_lane = make_active_vehicle_proxy(
        vehicle_id="queued",
        spawn_event=SimpleNamespace(
            route_id=11,
            lane_id=0,
            metadata={
                "exit_direction": "south",
                "exit_lane_idx": 0,
                "route_id": 11,
            },
        ),
        position_xy=(1.0, 0.0),
        speed_kmh=0.0,
        angle_deg=0.0,
    )
    provider = make_rule_based_path_provider(planner)
    decision = provider({
        "event": event,
        "adapter": make_spawn_event_proxy(event),
        "active_adapters": [queued_original_lane],
    })

    changed = apply_route_decision_to_event(event, decision)

    assert changed is True
    assert event.route_id == 12
    assert event.destination is decision.selected_candidate.dest_tf
    assert event.metadata["route_intent"]["direction_preserved"] is True
    assert event.metadata["route_intent"]["route_changed"] is True


def test_sumo_learned_provider_uses_proxy_actor_id_and_records_pending():
    class _Algo:
        pref_dim = 2

        def __init__(self):
            self.pending = []

        def select_intent(self, obs, mask, preference, training=True):
            assert mask.tolist() == [False, True, False, True]
            assert obs.shape[0] > 0
            assert np.allclose(preference, np.array([0.5, 0.5], dtype=np.float32))
            return 1, 0.0, 0.0

        def store_pending(self, actor_id, **kwargs):
            self.pending.append((actor_id, kwargs))

    planner = _three_way_planner()
    event = _sumo_event()
    algo = _Algo()
    active = [
        make_active_vehicle_proxy(
            vehicle_id="queued",
            spawn_event=SimpleNamespace(
                route_id=11,
                lane_id=0,
                metadata={
                    "exit_direction": "south",
                    "exit_lane_idx": 0,
                    "route_id": 11,
                },
            ),
            position_xy=(1.0, 0.0),
            speed_kmh=0.0,
            angle_deg=0.0,
        )
    ]
    provider = make_marl_intent_provider(
        algorithm=algo,
        get_active_adapters=lambda: active,
        training=False,
    )

    decision = provider({
        "event": event,
        "adapter": make_spawn_event_proxy(event, actor_id=event.vehicle_id),
        "planner": planner,
    })

    assert decision.selected_candidate.route_id == 12
    assert algo.pending[0][0] == "sumo_vehicle_0"


def test_route_diagnostics_report_change_and_queue_avoidance():
    planner = _three_way_planner()
    event = _sumo_event()
    queued_original_lane = make_active_vehicle_proxy(
        vehicle_id="queued",
        spawn_event=SimpleNamespace(
            route_id=11,
            lane_id=0,
            metadata={
                "exit_direction": "south",
                "exit_lane_idx": 0,
                "route_id": 11,
            },
        ),
        position_xy=(1.0, 0.0),
        speed_kmh=0.0,
        angle_deg=0.0,
    )
    provider = make_rule_based_path_provider(planner)
    decision = provider({
        "event": event,
        "adapter": make_spawn_event_proxy(event),
        "active_adapters": [queued_original_lane],
    })

    summary = summarize_intent_decisions([decision.as_metadata()])

    assert summary["route_change_count"] == 1
    assert summary["queued_lane_avoidance_opportunities"] == 1
    assert summary["queued_lane_avoidance_count"] == 1
    assert summary["direction_preserved_violations"] == 0


def test_sumo_terminal_intent_flushes_after_learning_update(monkeypatch):
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as sumo_env_mod
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")

    from opencda_marl._kernel.envs.episode_state import SumoEpisodeState

    env = object.__new__(SumoMARLEnv)
    env.current_step = 0
    env.total_steps = 0
    env.use_traffic_manager = False
    env.traffic_manager = None
    env.is_training_mode = True
    env.episode = SumoEpisodeState()
    env.previous_observations = {}
    env.reward_spec = SimpleNamespace(to_dict=lambda: {})
    env.objective_spec = None

    calls = []

    class _FakeManager:
        def compute_actions(self, observations, training=True):
            return {}

        def get_agent_preferences(self, agent_ids):
            return {}

        def ensure_agent_preferences(self, observations):
            return None

        def scalarize_objective_outputs(self, objective_outputs):
            return {}

        def update(self, rewards, observations, next_observations, objective_outputs=None):
            calls.append("update")
            assert env.episode.route_intent_terminal_queue == ["veh"]
            assert env.episode.terminal_intent_notified == set()

        def notify_vehicle_terminal(self, vehicle_id):
            calls.append(("notify", vehicle_id))

    env.marl_manager = _FakeManager()
    observations = [
        {"veh": {"vehicle_id": "veh", "speed": 0.0}},
        {},
    ]
    env._get_observations = lambda: observations.pop(0)
    env._apply_actions = lambda target_speeds: None
    env._update_competition_metadata = lambda observations, preferences: None
    env._record_competition_events = lambda events: None
    env._build_objective_outputs = lambda reward_outputs: {}
    env._update_metrics = lambda *args, **kwargs: None
    env._is_episode_done = lambda next_observations: True
    env._calculate_rewards = lambda events, next_observations: {
        "veh": SimpleNamespace(
            scalar=100.0,
            components={"success": 100.0},
            vector=[100.0],
        )
    }

    def _fake_check_events():
        env.episode.terminal_agents.add("veh")
        env._queue_route_intent_terminal("veh")
        return [
            StepEvent(
                step=env.current_step,
                event_id="success_veh_0",
                vehicle_id="veh",
                event_type="success",
            )
        ]

    env._check_events = _fake_check_events
    monkeypatch.setattr(sumo_env_mod.traci, "simulationStep", lambda: None)

    result = SumoMARLEnv.step(env)

    assert result["done"] is True
    assert calls == ["update", ("notify", "veh")]
