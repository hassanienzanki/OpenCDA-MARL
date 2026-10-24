from types import SimpleNamespace

import numpy as np

from opencda_marl._kernel.envs.sumo_runtime import (
    build_sumo_episode_states,
    has_pending_sumo_traffic,
    should_end_sumo_episode,
)
from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
from opencda_marl.evaluation.training_metrics import TrainingMetrics
from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl.api.objectives import ObjectiveOutput, ObjectiveSpec
from opencda_marl._kernel.traffic.sumo_spawner import SumoVehicleSpawner


def test_sumo_runtime_keeps_episode_alive_when_future_traffic_exists():
    assert has_pending_sumo_traffic(min_expected_number=1, pending_events=0) is True
    assert should_end_sumo_episode(
        current_step=20,
        max_steps=100,
        has_observations=False,
        min_expected_number=1,
        pending_events=0,
    ) is False


def test_sumo_runtime_ends_episode_when_no_agents_and_no_future_traffic():
    assert should_end_sumo_episode(
        current_step=20,
        max_steps=100,
        has_observations=False,
        min_expected_number=0,
        pending_events=0,
    ) is True


def test_sumo_episode_state_builder_uses_runtime_counts():
    episode_states = build_sumo_episode_states(
        current_step=55,
        collision_count=2,
        success_count=5,
        timeout_count=3,
        step_length=0.05,
    )

    assert episode_states == {
        "step": 55,
        "collision": 2,
        "success": 5,
        "active_agents": 3,
        "fixed_dt": 0.05,
    }


def test_training_metrics_do_not_double_count_pretracked_sumo_terminals():
    metrics = TrainingMetrics(export_interval=0)
    metrics.successes = 40
    metrics.collisions = 2

    episode = metrics.finish_episode({
        "step": 2400,
        "success": 40,
        "collision": 2,
        "active_agents": 38,
        "fixed_dt": 0.05,
    })

    assert episode["successes"] == 40
    assert episode["collisions"] == 2
    assert episode["timeouts"] == 38
    assert episode["success_rate"] == 50.0
    assert episode["collision_rate"] == 2.5
    assert episode["timeout_rate"] == 47.5


def test_training_metrics_accept_episode_snapshot_when_not_pretracked():
    metrics = TrainingMetrics(export_interval=0)

    episode = metrics.finish_episode({
        "step": 100,
        "success": 5,
        "collision": 1,
        "active_agents": 4,
        "fixed_dt": 0.05,
    })

    assert episode["successes"] == 5
    assert episode["collisions"] == 1
    assert episode["timeouts"] == 4
    assert episode["success_rate"] == 50.0
    assert episode["collision_rate"] == 10.0
    assert episode["timeout_rate"] == 40.0


def test_sumo_arrival_lane_uses_selected_exit_lane(monkeypatch):
    class EdgeApi:
        @staticmethod
        def getLaneNumber(edge_id):
            assert edge_id == "edge_out"
            return 3

    monkeypatch.setattr("opencda_marl._kernel.traffic.sumo_spawner.traci.edge", EdgeApi)

    assert SumoVehicleSpawner._resolve_arrival_lane(2, "edge_out") == "2"
    assert SumoVehicleSpawner._resolve_arrival_lane(5, "edge_out") == "2"
    assert SumoVehicleSpawner._resolve_arrival_lane(-1, "edge_out") == "current"


def test_manager_scales_route_intent_objective_rewards():
    manager = MARLManager(
        {
            "reward_profile": "cooperative_intersection_v1",
            "objective_mode": "cooperative_pref_v1",
            "state_dim": 5,
            "action_dim": 1,
            "action_space": {"mode": "speed_only"},
            "preference": {"profile": "balanced", "dim": 2},
        },
    )
    manager.intent_algorithm = object()
    manager.intent_reward_accumulator = SimpleNamespace(reward_dim=2)
    manager.intent_reward_scale = 0.001
    spec = ObjectiveSpec.from_config(
        {
            "objectives": ["efficiency", "safety"],
            "groups": {
                "efficiency": {"progress": 1.0},
                "safety": {"collision": 1.0},
            },
        }
    )
    objective_outputs = {
        "veh_1": ObjectiveOutput.from_reward_components(
            {"progress": 1500.0, "collision": -800.0},
            spec,
        )
    }

    intent_rewards = manager._prepare_intent_rewards(
        {"veh_1": -99.0},
        objective_outputs=objective_outputs,
        learning_rewards={"veh_1": -99.0},
    )

    assert np.allclose(
        intent_rewards["veh_1"],
        np.array([1.5, -0.8], dtype=np.float32),
    )


def test_sumo_edge_length_is_cached(monkeypatch):
    calls = {"edge": 0, "lane": 0}

    class EdgeApi:
        @staticmethod
        def getLaneNumber(edge_id):
            calls["edge"] += 1
            assert edge_id == "edge_a"
            return 2

    class LaneApi:
        @staticmethod
        def getLength(lane_id):
            calls["lane"] += 1
            return {"edge_a_0": 10.0, "edge_a_1": 14.0}[lane_id]

    monkeypatch.setattr("opencda_marl._kernel.envs.sumo_marl_env.traci.edge", EdgeApi)
    monkeypatch.setattr("opencda_marl._kernel.envs.sumo_marl_env.traci.lane", LaneApi)

    env = SumoMARLEnv.__new__(SumoMARLEnv)
    env._sumo_lane_length_cache = {}
    env._sumo_edge_length_cache = {}

    assert env._estimate_edge_length("edge_a") == 14.0
    assert env._estimate_edge_length("edge_a") == 14.0
    assert calls == {"edge": 1, "lane": 2}


def test_sumo_current_observation_reuses_cache_without_aliasing():
    from opencda_marl._kernel.envs.episode_state import SumoEpisodeState

    env = SumoMARLEnv.__new__(SumoMARLEnv)
    env.episode = SumoEpisodeState(observation_cache={
        "veh_1": {
            "speed": 12.0,
            "entry_direction": "north",
        }
    })

    observations = env._get_current_observations()
    observations["veh_1"]["speed"] = 0.0

    assert env.episode.observation_cache["veh_1"]["speed"] == 12.0
    assert observations["veh_1"]["entry_direction"] == "north"


def test_manager_respects_route_intent_update_interval():
    class IntentAlgorithm:
        route_intent_buffer = [object()] * 4
        route_intent_batch_size = 4

        def __init__(self):
            self.calls = 0

        def has_pending(self, actor_id):
            return False

        def update_route_intent(self):
            self.calls += 1

    manager = MARLManager(
        {
            "reward_profile": "cooperative_intersection_v1",
            "state_dim": 5,
            "action_dim": 1,
            "action_space": {"mode": "speed_only"},
        },
    )
    intent_algorithm = IntentAlgorithm()
    manager.algorithm = object()
    manager.strategy = None
    manager.intent_algorithm = intent_algorithm
    manager.intent_reward_accumulator = SimpleNamespace(reward_dim=0)
    manager.intent_update_interval_steps = 3

    for _ in range(5):
        manager.update(rewards={}, observations={}, next_observations={})

    assert intent_algorithm.calls == 1
    assert manager.get_intent_diagnostics()["intent_update_skipped"] == 4


def test_sumo_env_hands_the_outputs_block_to_the_marl_manager(tmp_path, monkeypatch):
    """SUMO must configure the learner's TensorBoard the way CARLA does.

    ``SumoMARLEnv`` used to build ``MARLManager(config['MARL'])`` without the
    shared ``outputs`` block, so ``outputs.tensorboard`` (mode, frequency,
    directory, segment rotation, retention) was silently ignored and the
    learner logged every update into ./runs. The CARLA pipeline builder
    injects the block; the SUMO env now does the same.
    """
    from opencda_marl.envs import sumo_marl_env as env_module

    captured = {}

    class _FakeManager:
        def __init__(self, config, algorithm):
            captured["config"] = config
            captured["algorithm"] = algorithm
            self.objective_config = None
            self.objective_spec = None
            self.agent_manager = None

    monkeypatch.setattr(env_module, "MARLManager", _FakeManager)
    monkeypatch.setattr(env_module.SumoMARLEnv, "_start_sumo", lambda self: None)
    monkeypatch.setattr(env_module, "CheckpointManager", lambda *a, **k: None)
    sumo_cfg = tmp_path / "net.sumocfg"
    sumo_cfg.write_text("<configuration/>")

    outputs = {
        "root_dir": str(tmp_path / "artifacts"),
        "tensorboard": {"enabled": True, "dir": "tensorboard", "mode": "episode_first",
                        "log_frequency": 100, "max_episodes_per_run": 100, "keep_last_runs": 5},
        "metrics_archive": {"enabled": False},
    }
    marl = {"algorithm": "masac", "reward_profile": "simple_progress_v1",
            "action_space": {"mode": "speed_only"},
            "state_dim": 8, "action_dim": 1, "training": {"training_mode": True}}
    config = {"runtime": {"sumo_cfg": str(sumo_cfg)}, "world": {}, "scenario": {},
              "MARL": marl, "outputs": outputs, "agents": {}}

    env_module.SumoMARLEnv(config)

    assert captured["algorithm"] == "masac"
    assert captured["config"]["outputs"] is outputs
    # The caller's MARL block itself is left untouched.
    assert "outputs" not in marl

    # And the manager derives the learner's TensorBoard settings from it.
    merged = MARLManager._merge_shared_config(
        SimpleNamespace(config=captured["config"]), {"lstm_hidden_size": 8})
    tb = merged["tensorboard"]
    assert tb["mode"] == "episode_first" and tb["log_frequency"] == 100
    assert tb["log_dir"].replace("\\", "/").endswith("artifacts/tensorboard")
    assert tb["max_episodes_per_run"] == 100 and tb["keep_last_runs"] == 5
