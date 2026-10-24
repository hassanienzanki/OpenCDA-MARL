"""Single-agent learners (DQN, tabular Q-learning) through the manager.

Two contracts, pinned end to end through ``MARLManager``:

- What ``select_action`` returns is declared by the policy, never inferred
  from its attributes. DQN owns a ``speed_actions`` table but returns the
  target speed itself; a policy with ``returns_action_index = True`` returns
  an index into its table. Reading DQN's speed as an index clamped every
  speed to the last table entry, so the learner drove one constant speed.
- Training goes through the shared strategy, which hands a single-agent
  learner its own per-agent state and the five-argument
  ``store_transition(state, action, reward, next_state, done)``. The
  multi-agent flow indexed ``state['all_states']``, which neither a feature
  vector nor a discrete state index has, so no transition was ever stored.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from opencda_marl._bundled import BUNDLED_CONFIG_ROOT
from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl._kernel.policy.strategy import (
    _SingleAgentStrategy,
    select_strategy,
    strategy_from_capabilities,
)
from opencda_marl.api.policy import PolicyCapabilities
from opencda_marl.tools.config_lint import merged_config

REPO = Path(__file__).resolve().parents[2]
_NO_OUTPUTS = {"metrics_store": {"enabled": False}, "tensorboard": {"enabled": False}}


def _dqn_manager() -> MARLManager:
    """The bundled DQN baseline's own profile, built by the manager."""
    cfg = dict(merged_config(REPO, "marl/intersection/dqn", BUNDLED_CONFIG_ROOT)["MARL"])
    cfg["outputs"] = dict(_NO_OUTPUTS)
    cfg.setdefault("training", {})["save_checkpoints"] = False
    return MARLManager(cfg, cfg["algorithm"])


def _q_manager() -> MARLManager:
    cfg = {
        "algorithm": "q_learning",
        "action_space": {"mode": "speed_only"},
        "reward_profile": "simple_progress_v1",
        "action_dim": 3,
        "q_learning": {
            "speed_actions": [10, 20, 30],
            "state_features": {"speed": {"bins": [15.0, 30.0]}},
            "epsilon": 0.0,
            "learning_rate": 0.5,
        },
        "training": {"training_mode": True},
        "outputs": dict(_NO_OUTPUTS),
    }
    return MARLManager(cfg, "q_learning")


def _raw_obs(rng, ids):
    return {
        vid: {"vehicle_id": vid, "speed": float(rng.uniform(5, 40)),
              "relative_position_to_intersection": {"x": float(rng.normal(0, 20)),
                                                    "y": float(rng.normal(0, 20))},
              "heading_angle": 0.3, "distance_to_intersection": float(rng.uniform(1, 40)),
              "distance_to_front_vehicle": 30.0, "lane_position": 10.0,
              "waypoint_buffer_size": 5.0, "location": {"x": 0.0, "y": 0.0, "z": 0.0}}
        for vid in ids
    }


# --- the action contract -------------------------------------------------------

def test_dqn_declares_that_it_returns_speeds():
    manager = _dqn_manager()
    assert manager.algorithm.returns_action_index is False
    assert manager.algorithm.multi_agent is False
    assert list(manager.algorithm.speed_actions) == [15, 30, 45, 60]


@pytest.mark.parametrize("speed", [15.0, 30.0, 45.0, 60.0])
def test_bundled_dqn_drives_the_speed_it_selected(speed):
    manager = _dqn_manager()
    manager.algorithm.select_action = lambda state, training=True: speed
    obs = _raw_obs(np.random.default_rng(0), ["veh_1", "veh_2"])

    targets = manager.compute_actions(obs, training=False)

    assert targets == {"veh_1": speed, "veh_2": speed}
    assert manager.last_actions == {"veh_1": speed, "veh_2": speed}


def test_bundled_dqn_targets_equal_its_own_choices_while_exploring():
    """The real epsilon-greedy path: every slot is visited, none is rewritten."""
    manager = _dqn_manager()
    algo = manager.algorithm
    algo.epsilon = 1.0
    random.seed(0)
    chosen = {}
    real_select = algo.select_action

    def recording_select(state, training=True):
        speed = real_select(state, training=training)
        chosen.setdefault("all", []).append(speed)
        return speed

    algo.select_action = recording_select
    rng = np.random.default_rng(1)
    targets = []
    for _ in range(10):
        targets.extend(manager.compute_actions(_raw_obs(rng, ["a", "b", "c"]), training=True).values())

    assert targets == chosen["all"]
    assert set(targets) == {15.0, 30.0, 45.0, 60.0}


def test_tabular_q_declares_that_it_returns_indices():
    from opencda_marl.builtin.algorithms.q_learning import QLearningAlgorithm

    assert QLearningAlgorithm.returns_action_index is True


def test_an_index_policy_is_resolved_against_its_own_table():
    """``returns_action_index = True``: the index drives its table entry and
    is what the manager records for store_transition."""
    manager = _q_manager()
    manager.algorithm.select_action = lambda state, training=True: np.int64(1)

    targets = manager.compute_actions(_raw_obs(np.random.default_rng(0), ["a"]), training=False)

    assert targets == {"a": 20.0}
    assert manager.last_actions == {"a": 1}


def test_a_speed_table_held_as_an_array_still_bounds_the_speed():
    """A single-agent policy may keep ``speed_actions`` as a numpy array; the
    clamp reads the table's length, not its truth value, so the step is not
    counted as a failure."""
    manager = _dqn_manager()
    manager.algorithm.speed_actions = np.array([15.0, 30.0, 45.0, 60.0])
    manager.algorithm.select_action = lambda state, training=True: 45.0

    targets = manager.compute_actions(_raw_obs(np.random.default_rng(0), ["veh_1", "veh_2"]),
                                      training=False)

    assert targets == {"veh_1": 45.0, "veh_2": 45.0}
    assert manager._failure_streaks.get("compute_actions", 0) == 0


# --- training through the shared strategy --------------------------------------

def test_single_agent_families_use_the_single_agent_flow():
    assert isinstance(select_strategy("dqn"), _SingleAgentStrategy)
    assert isinstance(select_strategy("q_learning"), _SingleAgentStrategy)
    registered = strategy_from_capabilities("my_single", PolicyCapabilities(multi_agent=False))
    assert isinstance(registered, _SingleAgentStrategy)
    assert not isinstance(strategy_from_capabilities("my_multi", PolicyCapabilities()),
                          _SingleAgentStrategy)


def test_bundled_dqn_stores_transitions_including_a_terminal_one():
    manager = _dqn_manager()
    algo = manager.algorithm
    rng = np.random.default_rng(2)
    obs = _raw_obs(rng, ["a", "b", "c"])
    manager.compute_actions(obs, training=True)
    chosen = dict(manager.last_actions)
    next_obs = _raw_obs(rng, ["a", "b"])          # "c" left this step

    manager.update({"a": 1.0, "b": 2.0, "c": -5.0}, obs, next_obs)

    assert manager._failure_streaks.get("update", 0) == 0
    assert len(algo.memory) == 3
    stored = {round(float(t[2]), 3): t for t in algo.memory.memory}
    terminal = stored[-5.0]
    assert terminal[4] is True
    assert float(terminal[3].abs().sum()) == 0.0, "a terminal next state is masked, zero-filled"
    assert stored[1.0][4] is False
    # The stored action is the index of the speed DQN chose for that agent.
    assert stored[1.0][1] == algo._speed_to_action_idx(chosen["a"])
    assert stored[2.0][1] == algo._speed_to_action_idx(chosen["b"])


def test_q_learning_update_moves_the_q_table():
    manager = _q_manager()
    algo = manager.algorithm
    rng = np.random.default_rng(3)
    obs = _raw_obs(rng, ["a", "b"])
    manager.compute_actions(obs, training=True)
    assert all(isinstance(a, int) for a in manager.last_actions.values())
    before = algo.get_q_table()

    manager.update({"a": 4.0, "b": 1.0}, obs, _raw_obs(rng, ["a"]))    # "b" left

    assert manager._failure_streaks.get("update", 0) == 0
    assert not np.array_equal(algo.get_q_table(), before)
    assert int(np.count_nonzero(algo.get_q_table())) >= 1
