"""Episode-boundary behavior: done handling, terminal sweep, reset hygiene."""
from collections import deque
from types import SimpleNamespace

import pytest


# --------------------------------------------------------------------- #
# Coordinator honors the environment's done signal
# --------------------------------------------------------------------- #
def test_coordinator_run_stops_episode_on_done():
    from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator

    coord = object.__new__(MARLCoordinator)
    coord.config = {"runtime": {"backend": "sumo"}}
    coord.pre_step_callbacks = []
    coord.post_step_callbacks = []
    coord.episode_callbacks = []
    coord.world_manager = None
    coord.trace_writer = None
    coord.scenario_manager = None
    coord.start_episode = 0
    coord.states = {"max_steps": 10, "max_episodes": 1}

    steps = {"n": 0}

    class _Env:
        def step(self):
            steps["n"] += 1
            return {"done": steps["n"] >= 3}

        def get_episode_metrics(self):
            return {}

        def get_current_step_rewards(self):
            return {}

        def reset_episode(self):
            return {}

    coord.marl_env = _Env()
    coord.evaluation_manager = SimpleNamespace(update_step=lambda m, r: None)

    coord.run()
    # done fired at step 3; the remaining 7 steps must not run.
    assert steps["n"] == 3


def test_coordinator_step_without_done_key_returns_false():
    from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator

    coord = object.__new__(MARLCoordinator)
    coord.config = {"runtime": {"backend": "sumo"}}
    coord.pre_step_callbacks = []
    coord.post_step_callbacks = []
    coord.world_manager = None
    coord.marl_env = SimpleNamespace(
        step=lambda: {"rewards": {}},  # CARLA-style result: no done key
        get_episode_metrics=lambda: {},
        get_current_step_rewards=lambda: {},
    )
    coord.evaluation_manager = SimpleNamespace(update_step=lambda m, r: None)
    assert coord.step() is False


# --------------------------------------------------------------------- #
# CARLA cleanup sweeps terminal notifications (timeout parity with SUMO)
# --------------------------------------------------------------------- #
def test_cleanup_notifies_terminals_for_surviving_vehicles():
    from opencda_marl._kernel.agent_manager import MARLAgentManager
    from opencda_marl._kernel.lifecycle.actor_registry import ActorRegistry

    class _World:
        def get_actors(self, ids=None):
            return []

        def tick(self):
            return None

    mgr = object.__new__(MARLAgentManager)
    mgr.world = _World()
    mgr.client = None
    mgr.cav_world = None
    mgr.registry = ActorRegistry(mgr.world, None)
    notified = []
    mgr.marl_manager = SimpleNamespace(
        notify_vehicle_terminal=lambda actor_id: notified.append(actor_id))
    mgr._vehicle_adapters = [
        SimpleNamespace(actor_id=7, vm=None),
        SimpleNamespace(actor_id=9, vm=None),
    ]
    mgr._spawn_metadata_by_actor_id = {7: {}, 9: {}}

    mgr.cleanup()

    assert notified == [7, 9]
    assert mgr._vehicle_adapters == []
    assert mgr._spawn_metadata_by_actor_id == {}


# --------------------------------------------------------------------- #
# SUMO spawner: route dedup and per-episode registry reset
# --------------------------------------------------------------------- #
def _bare_spawner():
    import opencda_marl._kernel.traffic.sumo_spawner as sp

    spawner = sp.SumoVehicleSpawner.__new__(sp.SumoVehicleSpawner)
    spawner.spawned_vehicles = {}
    spawner.failed_spawns = []
    spawner.spawn_counter = 5
    spawner._route_id_by_edges = {}
    return sp, spawner


def test_spawner_reuses_route_id_for_identical_edge_sequence(monkeypatch):
    sp, spawner = _bare_spawner()
    added = []
    monkeypatch.setattr(
        sp.traci.route, "add",
        lambda route_id, edges: added.append((route_id, tuple(edges))))

    first = spawner._ensure_route(["a", "b"])
    second = spawner._ensure_route(["a", "b"])
    third = spawner._ensure_route(["a", "c"])

    assert first == second
    assert first != third
    # Only two distinct routes ever registered with SUMO.
    assert len(added) == 2


def test_spawner_reset_clears_registries_but_keeps_route_cache():
    sp, spawner = _bare_spawner()
    spawner.spawned_vehicles = {"veh": object()}
    spawner.failed_spawns = [object()]
    spawner._route_id_by_edges = {("a", "b"): "route_0"}

    spawner.reset_episode_registries()

    assert spawner.spawned_vehicles == {}
    assert spawner.failed_spawns == []
    assert spawner.spawn_counter == 5  # monotonic, never reset
    assert spawner._route_id_by_edges == {("a", "b"): "route_0"}


# --------------------------------------------------------------------- #
# Traffic manager reset propagates to the preference sampler
# --------------------------------------------------------------------- #
def test_traffic_manager_reset_resets_preference_sampler():
    from opencda_marl._kernel.traffic.traffic_manager import MARLTrafficManager

    tm = object.__new__(MARLTrafficManager)
    tm._events = []
    tm._event_traces = []
    calls = []
    tm.preference_sampler = SimpleNamespace(
        reset_episode=lambda: calls.append("reset"))
    tm._initialize_events = lambda: None

    tm.reset()
    assert calls == ["reset"]
