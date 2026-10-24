"""A replay vehicle CARLA never put on the road is an unfinished route.

SUMO already counts its insertion queue this way (``_pending_insertions``):
a vehicle whose departure lane has no room never entered, so it is an
unfinished route of the episode that scheduled it. CARLA defers a spawn
whose point is occupied and abandons it after the retry budget, and counted
neither -- so under congestion the vehicles it had stopped simply left the
accounting, and every rate looked best exactly where the junction was worst.
The invariant pinned here: every scheduled vehicle ends the episode as a
success, a collision or an unfinished route, whether or not it ever entered.
"""
from collections import deque
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.agent_manager import MARLAgentManager


def _manager():
    """An agent manager with only the spawn-deferral bookkeeping alive."""
    am = object.__new__(MARLAgentManager)
    am._failure_events = deque()
    am._spawn_retry_counts = {}
    am._dropped_spawns = 0
    return am


def _event(vid):
    return SimpleNamespace(vehicle_id=vid)


def _retry(am, event, times):
    """Defer `event` `times` steps, the way `_spawn_vehicles` really does it.

    Each step drains the whole failure queue and re-appends whatever fails
    again, so a pending event is in the deque exactly once -- never once per
    attempt.
    """
    for _ in range(times):
        am._failure_events.clear()
        am._requeue_or_drop(event, reason="spawn point occupied")


def test_an_empty_queue_reports_nothing_unentered():
    assert _manager().get_unentered_count() == 0


def test_a_deferred_spawn_counts_as_unentered():
    am = _manager()
    am._requeue_or_drop(_event("v1"), reason="spawn point occupied")
    am._requeue_or_drop(_event("v2"), reason="collision at spawn")
    assert am.get_queue_count() == 2
    assert am.get_unentered_count() == 2


def test_a_retried_spawn_is_counted_once_not_once_per_attempt():
    am = _manager()
    _retry(am, _event("v1"), 5)
    assert am.get_unentered_count() == 1


def test_a_spawn_abandoned_after_the_budget_still_counts():
    # The vehicle leaves the retry queue but it never entered, so dropping it
    # from the queue must not drop it from the episode's accounting.
    am = _manager()
    _retry(am, _event("v1"), MARLAgentManager.MAX_SPAWN_RETRIES)
    assert am.get_queue_count() == 0, "the abandoned event leaves the retry queue"
    assert am._dropped_spawns == 1
    assert am.get_unentered_count() == 1, "but it is still an unfinished route"


def test_deferred_and_dropped_are_both_counted():
    am = _manager()
    _retry(am, _event("stuck"), MARLAgentManager.MAX_SPAWN_RETRIES)
    am._requeue_or_drop(_event("waiting"), reason="spawn point occupied")
    assert am.get_unentered_count() == 2


def test_the_episode_reset_clears_the_unentered_tally():
    # Otherwise one blocked spawn point would inflate every later episode.
    am = _manager()
    _retry(am, _event("v1"), MARLAgentManager.MAX_SPAWN_RETRIES)
    am._requeue_or_drop(_event("v2"), reason="spawn point occupied")
    assert am.get_unentered_count() == 2

    am._failure_events.clear()
    am._spawn_retry_counts.clear()
    am._dropped_spawns = 0
    assert am.get_unentered_count() == 0


def test_the_environment_reports_unentered_vehicles_as_unfinished():
    # marl_env folds never_entered into 'active_agents', and the metrics sink
    # reads that key as the episode's timeouts. Both halves of that contract
    # have to hold or the count lands nowhere.
    import inspect
    from opencda_marl._kernel.envs import marl_env
    from opencda_marl.evaluation.training_metrics import TrainingMetrics

    env_src = inspect.getsource(marl_env.MARLEnv)
    assert "get_unentered_count" in env_src
    assert "'active_agents': self.sm.states['active_agents'] + never_entered," in env_src

    sink_src = inspect.getsource(TrainingMetrics.finish_episode)
    assert "episode_timeouts = states.get('active_agents'" in sink_src
