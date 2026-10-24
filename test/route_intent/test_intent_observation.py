"""build_spawn_observation tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from opencda_marl.api.intent_observation import (
    CANDIDATE_FEATURES_PER_INTENT,
    CANDIDATE_TRAFFIC_SLICE,
    SPAWN_OBS_DIM_NO_PREF,
    build_spawn_observation,
)


def _event(route_id=11, entry_dir="north", entry_lane_idx=0,
           route_type="straight", junction_id=100, preference=None):
    metadata = {
        "entry_direction": entry_dir,
        "entry_lane_idx": entry_lane_idx,
        "route_type": route_type,
    }
    if preference is not None:
        metadata["preference"] = preference
    return SimpleNamespace(
        route_id=route_id,
        junction_id=junction_id,
        metadata=metadata,
    )


def _replay_event(route_id=11, entry_dir="north", lane_id=0,
                  route_type="straight", junction_id=100):
    return SimpleNamespace(
        route_id=route_id,
        junction_id=junction_id,
        metadata={
            "entry_direction": entry_dir,
            "lane_id": lane_id,
            "route_type": route_type,
        },
    )


class _PlannerStub:
    def __init__(self, routes):
        self._routes = routes
    def get_routes(self, j):
        return self._routes


def _three_way_routes():
    return {
        10: {"route_type": "left",     "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "west",  "exit_lane_idx": 0, "dest_tf": None},
        11: {"route_type": "straight", "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "south", "exit_lane_idx": 0, "dest_tf": None},
        13: {"route_type": "straight", "entry_direction": "north", "entry_lane_idx": 0, "exit_direction": "south", "exit_lane_idx": 1, "dest_tf": None},
        12: {"route_type": "right",    "entry_direction": "north", "entry_lane_idx": 1, "exit_direction": "east",  "exit_lane_idx": 0, "dest_tf": None},
    }


def test_observation_shape_no_preference():
    planner = _PlannerStub(_three_way_routes())
    obs = build_spawn_observation(_event(), planner, active_adapters=[],
                                  preference=np.zeros(0, dtype=np.float32))
    assert obs.shape == (SPAWN_OBS_DIM_NO_PREF,)
    assert obs.dtype == np.float32


def test_observation_includes_preference_when_provided():
    planner = _PlannerStub(_three_way_routes())
    pref = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    obs = build_spawn_observation(_event(preference=pref), planner,
                                  active_adapters=[], preference=pref)
    assert obs.shape == (SPAWN_OBS_DIM_NO_PREF + pref.size,)
    assert np.allclose(obs[-pref.size:], pref)


def test_observation_legal_mask_block_matches_planner():
    planner = _PlannerStub(_three_way_routes())
    # entry lane 0 -> same-exit-direction straight alternative and keep are legal.
    # Left changes the exit direction, right is from lane 1 only.
    obs = build_spawn_observation(_event(entry_lane_idx=0), planner, [],
                                  preference=np.zeros(0, dtype=np.float32))
    # The legal_mask block occupies a fixed window inside obs; the helper
    # exposes its slice for test access.
    from opencda_marl.api.intent_observation import LEGAL_MASK_SLICE
    mask_block = obs[LEGAL_MASK_SLICE]
    # order matches RouteIntent: LEFT, STRAIGHT, RIGHT, KEEP
    assert mask_block.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_observation_uses_lane_id_when_replay_lacks_entry_lane_idx():
    from opencda_marl.api.intent_observation import LEGAL_MASK_SLICE
    planner = _PlannerStub(_three_way_routes())
    obs = build_spawn_observation(_replay_event(lane_id=0), planner, [],
                                  preference=np.zeros(0, dtype=np.float32))
    assert obs[LEGAL_MASK_SLICE].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_observation_is_deterministic_for_same_inputs():
    planner = _PlannerStub(_three_way_routes())
    pref = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    a = build_spawn_observation(_event(preference=pref), planner, [],
                                preference=pref)
    b = build_spawn_observation(_event(preference=pref), planner, [],
                                preference=pref)
    assert np.array_equal(a, b)


def test_outgoing_traffic_count_increments_per_exit_direction():
    from opencda_marl.api.intent_observation import TRAFFIC_COUNT_SLICE
    planner = _PlannerStub(_three_way_routes())
    # Two adapters heading south, one east; queue threshold not crossed
    adapters = [
        SimpleNamespace(spawn_metadata={"exit_direction": "south"}, current_speed=30.0),
        SimpleNamespace(spawn_metadata={"exit_direction": "south"}, current_speed=25.0),
        SimpleNamespace(spawn_metadata={"exit_direction": "east"},  current_speed=20.0),
    ]
    obs = build_spawn_observation(_event(), planner, adapters,
                                  preference=np.zeros(0, dtype=np.float32))
    # _DIRECTIONS = (north, south, east, west)
    assert obs[TRAFFIC_COUNT_SLICE].tolist() == [0.0, 2.0, 1.0, 0.0]


def test_queue_estimate_uses_low_speed_threshold():
    from opencda_marl.api.intent_observation import (
        QUEUE_SLICE, TRAFFIC_COUNT_SLICE,
    )
    planner = _PlannerStub(_three_way_routes())
    adapters = [
        SimpleNamespace(spawn_metadata={"exit_direction": "south"}, current_speed=2.0),
        SimpleNamespace(spawn_metadata={"exit_direction": "south"}, current_speed=4.99),
        SimpleNamespace(spawn_metadata={"exit_direction": "south"}, current_speed=10.0),
        # Unknown exit direction: must NOT count anywhere.
        SimpleNamespace(spawn_metadata={"exit_direction": "garbage"}, current_speed=0.0),
        # No spawn_metadata at all: must NOT count anywhere.
        SimpleNamespace(spawn_metadata=None, current_speed=0.0),
    ]
    obs = build_spawn_observation(_event(), planner, adapters,
                                  preference=np.zeros(0, dtype=np.float32))
    assert obs[TRAFFIC_COUNT_SLICE].tolist() == [0.0, 3.0, 0.0, 0.0]
    assert obs[QUEUE_SLICE].tolist() == [0.0, 2.0, 0.0, 0.0]


def test_candidate_features_expose_same_lane_queue_pressure():
    planner = _PlannerStub(_three_way_routes())
    adapters = [
        SimpleNamespace(
            spawn_metadata={"exit_direction": "south", "exit_lane_idx": 1, "route_id": 13},
            current_speed=0.0,
        )
    ]
    obs = build_spawn_observation(_event(), planner, adapters,
                                  preference=np.zeros(0, dtype=np.float32))
    straight_offset = CANDIDATE_TRAFFIC_SLICE.start + CANDIDATE_FEATURES_PER_INTENT
    straight_features = obs[
        straight_offset:straight_offset + CANDIDATE_FEATURES_PER_INTENT
    ]
    assert straight_features[0] > 0.0  # same exit direction
    assert straight_features[1] > 0.0  # same exit lane
    assert straight_features[2] > 0.0  # queued same exit lane
    assert straight_features[3] > 0.0  # same route id
    assert straight_features[6] > 0.0  # pressure score
