"""MARLManager: warmup fallback must work with CARLA-shaped (int) raw ids.

TD3 returns ``None`` during warmup so the rule-based agent drives; the
manager then records the vehicle's current speed as the stored action so
those transitions reach the replay buffer — which is what ends TD3's
(buffer-size based) warmup. The lookup used the extractor's str agent id
against the RAW observation dict, whose keys are CARLA actor ids (ints), so
on CARLA nothing was ever stored and TD3 never left warmup.
"""
import numpy as np
import pytest

from opencda_marl._kernel.policy.manager import MARLManager


def _manager(algorithm, **algo_overrides):
    algo_cfg = {"features": {"rel_x": 1, "rel_y": 1, "speed": 1, "heading_angle": 1,
                             "dist_to_intersection": 1, "dist_to_front_vehicle": 1,
                             "lane_position": 1, "waypoint_buffer": 1},
                "warmup_steps": 40, "batch_size": 8, "memory_size": 500,
                "lstm_hidden_size": 8, "actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16],
                "tensorboard": {"enabled": False}}
    algo_cfg.update(algo_overrides)
    cfg = {"algorithm": algorithm, "action_space": {"mode": "speed_only"},
           "reward_profile": "simple_progress_v1", "state_dim": 8, "action_dim": 1,
           algorithm: algo_cfg, "training": {"training_mode": True},
           "outputs": {"tensorboard": {"enabled": False}}}
    return MARLManager(cfg, algorithm)


def _raw_obs(rng, ids):
    return {
        vid: {"vehicle_id": vid, "speed": float(rng.uniform(5, 40)),
              "relative_position_to_intersection": {"x": float(rng.normal(0, 20)), "y": float(rng.normal(0, 20))},
              "heading_angle": 0.3, "distance_to_intersection": float(rng.uniform(1, 40)),
              "distance_to_front_vehicle": 30.0, "lane_position": 10.0, "waypoint_buffer_size": 5.0,
              "location": {"x": 0.0, "y": 0.0, "z": 0.0}}
        for vid in ids
    }


@pytest.mark.parametrize("ids", [[101, 102, 103], ["veh_1", "veh_2", "veh_3"]], ids=["carla-int-ids", "sumo-str-ids"])
def test_td3_warmup_transitions_are_stored_and_warmup_ends(ids):
    mgr = _manager("matd3", use_per=False, policy_freq=1)
    algo = mgr.algorithm
    rng = np.random.default_rng(0)
    prev = _raw_obs(rng, ids)
    acted = 0
    for _ in range(30):
        actions = mgr.compute_actions(prev, training=True)
        acted += len(actions)
        nxt = _raw_obs(rng, ids)
        mgr.update({str(v): 1.0 for v in ids}, prev, nxt)
        prev = nxt
    assert len(algo.memory) > 0, "warmup transitions must reach the buffer"
    assert not algo._in_action_warmup(True), "40 stored transitions end a warmup of 40"
    assert acted > 0, "after warmup the learner acts"
    assert algo.training_step > 0
