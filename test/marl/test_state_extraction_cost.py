"""Feature extraction reads no config at step time and runs once per observation dict.

The extractor used to walk the configured feature mapping -- an OmegaConf
``DictConfig`` -- for every feature of every agent, paying the config
resolver on each read; and extraction ran three times per step on
identical inputs, because the manager's identity memo keyed on the
*filtered* learning view, which a mixed population rebuilt on every call,
and because the SUMO step handed the learner a copy of the read it had
just extracted from.

Now the feature list is resolved once into a plan over plain values, the
learning view is stable across the calls of a step (and into the next,
where next_obs becomes obs), and the SUMO step reuses its post-step read as
the next pre-step observation while outside readers still get a copy.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import opencda_marl._kernel.envs.sumo_marl_env as sumo_env_mod
from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv
from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl.builtin.extractors import ObservationExtractor

FEATURES = {"rel_x": 1, "rel_y": 1, "position_x": 1, "position_y": 1, "lane_position": 1,
            "lane_position_onehot": 4, "speed": 1, "heading_angle": 1, "dist_to_intersection": 1,
            "dist_to_front_vehicle": 1, "waypoint_buffer": 1, "committed_exit_lanes": 4,
            "nearby_vehicles": 35}

FULL_OBS = {
    "relative_position_to_intersection": {"x": 3.0, "y": -4.0}, "location": {"x": 30.0, "y": 40.0},
    "lane_position": 1, "speed": 12.5, "heading_angle": 0.7, "distance_to_intersection": 25.0,
    "distance_to_front_vehicle": 8.0, "waypoint_buffer_size": 9, "committed_exit_lanes": [1.0, 0.0],
    "nearby_vehicles": [0.1] * 35,
}


# ─────────────────────────────────────────────────────────────────────────────
# The extractor
# ─────────────────────────────────────────────────────────────────────────────

def test_a_dictconfig_and_a_plain_dict_build_the_same_extractor_and_no_config_survives_construction():
    plain = ObservationExtractor({"pcmappo": {"features": dict(FEATURES)}}, "pcmappo")
    resolved = ObservationExtractor(OmegaConf.create({"pcmappo": {"features": dict(FEATURES)}}), "pcmappo")
    assert type(resolved.custom_features) is dict and resolved.custom_features == FEATURES
    assert resolved.state_dim == plain.state_dim == sum(FEATURES.values())
    assert [(n, w) for n, w, _ in resolved._feature_plan] == list(FEATURES.items())
    a = plain.extract({"7": dict(FULL_OBS)})["7"]["ego_state"]
    b = resolved.extract({"7": dict(FULL_OBS)})["7"]["ego_state"]
    np.testing.assert_array_equal(a, b)
    assert a.shape == (sum(FEATURES.values()),) and a.dtype == np.float32


def test_every_feature_reads_its_value_and_its_documented_default():
    extractor = ObservationExtractor({"pcmappo": {"features": dict(FEATURES)}}, "pcmappo")
    present = extractor._extract_custom_features({"v": FULL_OBS})["v"].tolist()
    assert present[:5] == [3.0, -4.0, 30.0, 40.0, 1.0]
    assert present[5:9] == [0.0, 1.0, 0.0, 0.0]
    assert present[9:14] == pytest.approx([12.5, 0.7, 25.0, 8.0, 9.0])
    assert present[14:18] == [1.0, 0.0, 0.0, 0.0]
    assert present[18:] == pytest.approx([0.1] * 35)
    absent = extractor._extract_custom_features({"v": {}})["v"].tolist()
    assert absent[:5] == [0.0, 0.0, 0.0, 0.0, 2.0]             # middle lane by default
    assert absent[5:9] == [0.0, 0.0, 1.0, 0.0]
    assert absent[9:14] == [0.0, 0.0, 100.0, 999.0, 50.0]       # 100 m away -> 50 estimated waypoints
    assert absent[14:18] == [0.0] * 4
    assert absent[18:] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 5
    assert len(present) == len(absent) == extractor.state_dim


def test_the_committed_exit_lane_block_is_padded_or_cut_to_its_configured_width():
    extractor = ObservationExtractor({"pcmappo": {"features": {"committed_exit_lanes": 3}}}, "pcmappo")
    assert extractor._extract_custom_features({"v": {"committed_exit_lanes": [1, 0, 1, 1, 0]}})["v"].tolist() == [1.0, 0.0, 1.0]
    assert extractor._extract_custom_features({"v": {"committed_exit_lanes": [1]}})["v"].tolist() == [1.0, 0.0, 0.0]


# ─────────────────────────────────────────────────────────────────────────────
# The manager: one extraction per unique observation dict
# ─────────────────────────────────────────────────────────────────────────────

class _CountingExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, observations):
        self.calls.append(sorted(observations))
        return {k: {"ego_state": np.zeros(1), "all_states": {}} for k in observations}


def _manager():
    mgr = object.__new__(MARLManager)
    mgr._state_cache = {}
    mgr._learning_cache = {}
    mgr.preference_enabled = False
    mgr.observation_extractor = _CountingExtractor()
    return mgr


def test_a_mixed_population_is_extracted_once_across_action_selection_update_and_the_next_step():
    mgr = _manager()
    obs_t = {"cav": {"learning": True, "speed": 1.0}, "hv": {"learning": False, "speed": 2.0}}
    next_t = {"cav": {"learning": True, "speed": 1.5}, "hv": {"learning": False, "speed": 2.5}}
    # step t: compute_actions(obs_t), then update(obs_t, next_t)
    mgr._extract_states(mgr._learning_view(obs_t))
    mgr._extract_states(mgr._learning_view(obs_t))
    mgr._extract_states(mgr._learning_view(next_t))
    # step t+1: compute_actions(next_t) -- the same object the env read after its step
    mgr._extract_states(mgr._learning_view(next_t))
    assert mgr.observation_extractor.calls == [["cav"], ["cav"]], "one extraction per unique dict, learners only"


def test_the_learning_view_is_the_same_object_for_the_same_dict_and_pure_for_a_homogeneous_one():
    mgr = _manager()
    mixed = {"a": {"learning": True}, "b": {"learning": False}}
    assert mgr._learning_view(mixed) is mgr._learning_view(mixed)
    assert set(mgr._learning_view(mixed)) == {"a"}
    homogeneous = {"a": {}, "b": {}}
    assert mgr._learning_view(homogeneous) is homogeneous
    assert MARLManager._learning_observations(mixed) is not MARLManager._learning_observations(mixed)


def test_the_view_cache_keeps_two_dicts_and_never_aliases_a_recycled_id():
    mgr = _manager()
    first = {"a": {"learning": True}, "b": {"learning": False}}
    view_first = mgr._learning_view(first)
    mgr._learning_view({"c": {"learning": True}})
    mgr._learning_view({"d": {"learning": True}})          # evicts `first`
    assert len(mgr._learning_cache) == 2
    assert mgr._learning_view(first) is not view_first      # recomputed, equal, not the evicted object
    assert mgr._learning_view(first) == view_first


# ─────────────────────────────────────────────────────────────────────────────
# The SUMO step
# ─────────────────────────────────────────────────────────────────────────────

def _env():
    env = object.__new__(SumoMARLEnv)
    env.episode = SimpleNamespace(observation_cache={}, terminal_agents=set(), active_vehicles={})
    env.current_step = 0
    env.marl_config = {"algorithm": "pcmappo"}
    return env


def test_the_step_reuses_the_post_step_read_itself_while_outside_readers_get_a_copy():
    env = _env()
    read = {"veh_1": {"speed": 12.0}}
    env.episode.observation_cache = read
    assert env._pending_observations() is read
    outside = env._get_current_observations()
    assert outside == read and outside is not read and outside["veh_1"] is not read["veh_1"]


def test_neighbour_settings_are_resolved_once_per_config_object():
    env = _env()
    first = env._nearby_settings()
    assert env._nearby_settings() is first
    env.marl_config = {"algorithm": "pcmappo", "pcmappo": {"nearby_vehicle_config": {"detection_radius": 25.0}}}
    assert env._nearby_settings()["detection_radius"] == 25.0


def test_exit_lane_rows_are_built_once_per_step_and_vehicle_set(monkeypatch):
    class _Spawner:
        def __init__(self):
            self.lookups = 0

        def get_spawn_event(self, vid):
            self.lookups += 1
            return SimpleNamespace(metadata={"junction_id": "j1", "exit_lane_idx": int(vid[-1])}, junction_id="j1")

    ids = ("v0", "v1", "v2", "v3")
    monkeypatch.setattr(sumo_env_mod.traci, "vehicle", SimpleNamespace(getIDList=lambda: ids))
    env = _env()
    env.vehicle_spawner = _Spawner()
    for vid in ids:
        env._committed_exit_lanes(vid, {"junction_id": "j1"})
    assert env.vehicle_spawner.lookups == len(ids), "one row per vehicle per step, not per (vehicle, other) pair"
    env.current_step += 1
    env._committed_exit_lanes("v0", {"junction_id": "j1"})
    assert env.vehicle_spawner.lookups == 2 * len(ids)
    rows = env._exit_lane_rows(env.vehicle_spawner)
    assert set(rows) == set(ids) and rows["v2"]["exit_lane_idx"] == 2
