"""Silent-failure guards on the training data path."""
import pytest

from opencda_marl._kernel.policy.manager import MARLManager


def _manager():
    return MARLManager(
        {
            "algorithm": "masac",
            "action_space": {"mode": "speed_only"},
            "reward_profile": "cooperative_intersection_v1",
            "state_dim": 3,
            "action_dim": 1,
            "masac": {
                "features": {"rel_x": 1, "rel_y": 1, "speed": 1},
                "warmup_steps": 0,
                "memory_size": 16,
                "batch_size": 4,
                "actor_hidden_dims": [8, 8],
                "critic_hidden_dims": [8, 8],
                "lstm_hidden_size": 4,
                "tensorboard": {"enabled": False},
            },
        },
        "masac",
    )


def test_persistent_compute_actions_failure_aborts_the_run(monkeypatch):
    mgr = _manager()
    mgr.MAX_CONSECUTIVE_FAILURES = 3

    def boom(*a, **k):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(mgr, "_extract_states", boom)
    obs = {"veh_a": {"relative_position_to_intersection": {"x": 1.0, "y": 2.0}, "speed": 10.0}}
    # First failures degrade gracefully (empty action dict = rule-based fallback)...
    assert mgr.compute_actions(obs) == {}
    assert mgr.compute_actions(obs) == {}
    # ...the streak limit turns a silent non-training run into a hard error.
    with pytest.raises(RuntimeError, match="not training"):
        mgr.compute_actions(obs)


def test_success_resets_the_failure_streak(monkeypatch):
    mgr = _manager()
    mgr.MAX_CONSECUTIVE_FAILURES = 3
    real = mgr._extract_states
    state = {"fail": True}

    def flaky(observations):
        if state["fail"]:
            raise RuntimeError("transient")
        return real(observations)

    monkeypatch.setattr(mgr, "_extract_states", flaky)
    obs = {"veh_a": {"relative_position_to_intersection": {"x": 1.0, "y": 2.0}, "speed": 10.0}}
    mgr.compute_actions(obs)
    mgr.compute_actions(obs)          # streak = 2
    state["fail"] = False
    mgr.compute_actions(obs)          # success -> streak reset
    state["fail"] = True
    mgr.compute_actions(obs)          # streak = 1 again, no raise
    mgr.compute_actions(obs)          # streak = 2, no raise
    assert mgr._failure_streaks["compute_actions"] == 2


def test_failed_observation_is_excluded_not_fabricated():
    """A vehicle whose observation failed must not appear in the batch the
    extractor turns into training state."""
    from types import SimpleNamespace
    from opencda_marl._kernel.agent_manager import MARLAgentManager
    from opencda_marl._kernel.lifecycle.actor_registry import ActorRegistry

    class _World:
        def get_actors(self, ids=None):
            return []

    mgr = object.__new__(MARLAgentManager)
    mgr.world = _World()
    mgr.registry = ActorRegistry(mgr.world, None)
    mgr._spawn_metadata_by_actor_id = {}
    mgr._current_snapshot = None
    good = SimpleNamespace(actor_id=1, vehicle=SimpleNamespace(type_id="vehicle.a"),
                           get_observation=lambda snapshot=None: {"vehicle_id": 1, "speed": 5.0})
    bad = SimpleNamespace(actor_id=2, vehicle=SimpleNamespace(type_id="vehicle.b"),
                          get_observation=lambda snapshot=None: {"vehicle_id": 2, "error": "boom"})
    mgr._vehicle_adapters = [good, bad]

    observations = mgr.get_all_observations()
    assert set(observations) == {1}
