"""A step reward reaches the pending decision it belongs to, whatever id type each side used.

A path provider stores the pending decision under the adapter's actor id,
which on CARLA is the actor's integer. The CARLA environment stringifies
reward keys before they reach the manager (since the sealed-kernel
refactor), so the manager looked up ``has_pending('948')`` against a store
keyed by ``948`` and never matched: on CARLA every decision finalized with
zero reward, the head's value loss fitted zeros, and nothing failed. Seen
live as ``intent_reward_steps = 0`` on CARLA, while the same profile on SUMO,
whose proxies use strings on both sides, recorded reward steps normally.
"""
import numpy as np

from opencda_marl._kernel.policy.accumulator import PerActorRewardAccumulator
from opencda_marl._kernel.policy.manager import MARLManager


class _Head:
    """The pending store as the route head exposes it, keyed as the provider stored it."""

    def __init__(self, pending):
        self.pending = set(pending)
        self.finalized = []

    def has_pending(self, actor_id):
        return actor_id in self.pending

    def finalize_terminal(self, *, actor_id, terminal_reward):
        self.pending.discard(actor_id)
        self.finalized.append((actor_id, np.asarray(terminal_reward, dtype=np.float32)))


def _manager(pending, reward_dim=2):
    manager = object.__new__(MARLManager)
    manager.intent_algorithm = _Head(pending)
    manager.intent_reward_accumulator = PerActorRewardAccumulator(reward_dim=reward_dim)
    manager._intent_episode_diagnostics = {
        "intent_reward_steps": 0, "intent_nonzero_reward_steps": 0,
        "intent_terminal_finalized": 0, "intent_nonzero_terminal_rewards": 0,
        "intent_terminal_without_pending": 0,
    }
    return manager


def test_carla_string_rewards_reach_an_integer_pending_decision():
    manager = _manager(pending={948})                     # stored under the adapter's int id
    manager._notify_intent_step({"948": np.array([1.0, 0.5])})
    manager._notify_intent_step({"948": np.array([2.0, 0.0])})
    d = manager._intent_episode_diagnostics
    assert d["intent_reward_steps"] == 2 and d["intent_nonzero_reward_steps"] == 2

    manager.notify_vehicle_terminal(948)                 # the adapter's id, as the agent manager sends it
    (actor, reward), = manager.intent_algorithm.finalized
    assert actor == 948
    np.testing.assert_allclose(reward, [3.0, 0.5])
    assert d["intent_terminal_finalized"] == 1 and d["intent_nonzero_terminal_rewards"] == 1
    assert d["intent_terminal_without_pending"] == 0


def test_sumo_string_ids_on_both_sides_still_work():
    manager = _manager(pending={"veh_7"})
    manager._notify_intent_step({"veh_7": np.array([0.0, 4.0])})
    manager.notify_vehicle_terminal("veh_7")
    (actor, reward), = manager.intent_algorithm.finalized
    assert actor == "veh_7"
    np.testing.assert_allclose(reward, [0.0, 4.0])


def test_a_terminal_that_arrives_as_a_string_for_an_integer_decision_is_matched():
    manager = _manager(pending={12})
    manager._notify_intent_step({12: np.array([1.0, 1.0])})
    manager.notify_vehicle_terminal("12")
    (actor, reward), = manager.intent_algorithm.finalized
    assert actor == 12
    np.testing.assert_allclose(reward, [1.0, 1.0])


def test_a_reward_for_a_vehicle_with_no_decision_is_ignored_and_a_terminal_counted():
    manager = _manager(pending={1})
    manager._notify_intent_step({"999": np.array([5.0, 5.0])})
    assert manager._intent_episode_diagnostics["intent_reward_steps"] == 0
    manager.notify_vehicle_terminal(999)
    assert manager._intent_episode_diagnostics["intent_terminal_without_pending"] == 1
    assert manager.intent_algorithm.finalized == []
