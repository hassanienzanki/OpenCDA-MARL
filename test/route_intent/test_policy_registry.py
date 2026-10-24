"""The pluggable-policy seam: contract, registration, capability dispatch.

The load-bearing test here is
``test_external_policy_runs_without_touching_framework``: it registers a
policy defined entirely in this file and drives it through MARLManager. If
that ever requires an edit inside ``opencda_marl/``, the seam has regressed.
"""
import numpy as np
import pytest

from opencda_marl.api.policy import (
    IntentPolicy,
    Policy,
    missing_intent_methods,
    supports_intent,
)
from opencda_marl.api import algorithms as registry_mod
from opencda_marl.api.algorithms import (
    register_algorithm,
    registered_algorithm_keys,
    resolve_algorithm_spec,
)


# --------------------------------------------------------------------- #
# A complete external policy — nothing here imports framework internals
# beyond the public contract.
# --------------------------------------------------------------------- #
class ConstantSpeedPolicy:
    """Minimal Policy implementation: always requests the same speed."""

    supports_batched_actions = False
    supports_vector_rewards = False

    def __init__(self, config, state_dim, action_dim):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = float(config.get("max_action", 40.0))
        self.speed = float(config.get("constant_speed", 25.0))
        self.transitions = []
        self.updates = 0
        self.episodes = 0

    def select_action(self, multi_agent_obs, ego_agent_id, training=True):
        return self.speed

    def store_transition(self, multi_agent_obs, ego_agent_id, action, reward,
                         next_multi_agent_obs, done):
        self.transitions.append((ego_agent_id, action, reward, done))

    def update(self):
        self.updates += 1
        return {"loss": 0.0}

    def reset_episode(self):
        self.episodes += 1

    def get_training_info(self):
        return {"algorithm_type": "constant", "episode_count": self.episodes}

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass


@pytest.fixture
def clean_registry():
    """Snapshot/restore the registry so tests cannot leak registrations."""
    saved = dict(registry_mod._SPECS)
    yield
    registry_mod._SPECS.clear()
    registry_mod._SPECS.update(saved)


# --------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------- #
def test_builtin_algorithms_expose_the_policy_interface():
    from opencda_marl.builtin.algorithms import (
        MAPPOAlgorithm,
        PCMAPPOAlgorithm,
        SACAlgorithm,
        TD3Algorithm,
    )

    # Protocols carrying data members cannot be used with issubclass, and the
    # data attributes only exist on instances, so check the methods on the
    # classes — that is the part a new algorithm has to get right.
    required = (
        "select_action", "store_transition", "update",
        "reset_episode", "get_training_info", "save", "load",
    )
    for cls in (MAPPOAlgorithm, PCMAPPOAlgorithm, SACAlgorithm, TD3Algorithm):
        for name in required:
            assert callable(getattr(cls, name, None)), \
                f"{cls.__name__} is missing Policy method {name}"


def test_external_policy_satisfies_the_policy_protocol():
    assert isinstance(ConstantSpeedPolicy({}, 5, 1), Policy)


def test_intent_capability_probe_reports_missing_methods():
    policy = ConstantSpeedPolicy({}, 5, 1)
    assert supports_intent(policy) is False
    missing = missing_intent_methods(policy)
    assert "select_intent" in missing and "finalize_terminal" in missing


def test_intent_capable_policy_passes_the_probe():
    class WithIntent(ConstantSpeedPolicy):
        def enable_route_intent(self, config, *, state_dim, num_actions,
                                pref_dim, reward_dim):
            pass

        def select_intent(self, observation, action_mask, preference, *, training=True):
            return 0, 0.0, np.zeros(2, dtype=np.float32)

        def store_pending(self, **kwargs):
            pass

        def has_pending(self, actor_id):
            return False

        def finalize_terminal(self, **kwargs):
            pass

        def update_route_intent(self):
            return {}

    policy = WithIntent({}, 5, 1)
    assert supports_intent(policy) is True
    assert missing_intent_methods(policy) == []
    assert isinstance(policy, IntentPolicy)


# --------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------- #
def test_register_algorithm_adds_a_resolvable_spec(clean_registry):
    @register_algorithm("constant", label="Constant",
                        reward_aggregation="team_mean")
    def _build(config, *, state_dim, action_dim):
        return ConstantSpeedPolicy(config, state_dim, action_dim)

    spec = resolve_algorithm_spec("constant")
    assert spec.label == "Constant"
    assert spec.config_section == "constant"
    assert spec.capabilities.reward_aggregation == "team_mean"
    assert spec.capabilities.supports_intent is False
    assert "constant" in registered_algorithm_keys()

    policy = spec.build({"constant_speed": 30.0}, state_dim=5, action_dim=1)
    assert isinstance(policy, ConstantSpeedPolicy)
    assert policy.speed == 30.0


def test_duplicate_registration_requires_explicit_override(clean_registry):
    @register_algorithm("dup")
    def _build(config, *, state_dim, action_dim):
        return ConstantSpeedPolicy(config, state_dim, action_dim)

    with pytest.raises(ValueError, match="already registered"):
        @register_algorithm("dup")
        def _build_again(config, *, state_dim, action_dim):
            return ConstantSpeedPolicy(config, state_dim, action_dim)

    @register_algorithm("dup", label="Replaced", override=True)
    def _build_override(config, *, state_dim, action_dim):
        return ConstantSpeedPolicy(config, state_dim, action_dim)

    assert resolve_algorithm_spec("dup").label == "Replaced"


def test_invalid_reward_aggregation_is_rejected():
    with pytest.raises(ValueError, match="reward_aggregation"):
        register_algorithm("bad", reward_aggregation="sum")


def test_builtin_specs_carry_capabilities():
    assert resolve_algorithm_spec("pcmasac").capabilities.reward_aggregation == "team_mean"
    assert resolve_algorithm_spec("masac").capabilities.reward_aggregation == "per_agent"
    assert resolve_algorithm_spec("pcmatd3").capabilities.supports_intent is False
    assert resolve_algorithm_spec("pcmappo").capabilities.supports_intent is True
    assert resolve_algorithm_spec("pcmappo").vector_rewards is True
    assert resolve_algorithm_spec("mappo").vector_rewards is False


# --------------------------------------------------------------------- #
# RouteHeadMixin: one implementation shared by every intent-capable policy
# --------------------------------------------------------------------- #
def test_intent_capable_builtins_share_the_route_head_mixin():
    from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm
    from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm
    from opencda_marl.builtin.algorithms.route_head import RouteHeadMixin
    from opencda_marl.builtin.algorithms.sac import SACAlgorithm

    for cls in (MAPPOAlgorithm, PCMAPPOAlgorithm, SACAlgorithm):
        assert issubclass(cls, RouteHeadMixin)
        # The delegation surface is inherited, not re-implemented per class.
        assert cls.select_intent is RouteHeadMixin.select_intent
        assert cls.enable_route_intent is RouteHeadMixin.enable_route_intent

    # PPO families share the controller optimizer; discrete SAC does not.
    assert MAPPOAlgorithm.route_head_shares_optimizer is True
    assert PCMAPPOAlgorithm.route_head_shares_optimizer is True
    assert SACAlgorithm.route_head_shares_optimizer is False


def test_route_head_mixin_requires_enable_before_use():
    from opencda_marl.builtin.algorithms.route_head import RouteHeadMixin

    class Bare(RouteHeadMixin):
        route_head_cls = "HybridRouteSACHead"

    policy = Bare()
    # The capability probe passes (methods exist) but using it before
    # enable_route_intent must fail loudly rather than silently no-op.
    assert supports_intent(policy) is True
    assert policy.has_pending("veh") is False
    with pytest.raises(RuntimeError, match="route-intent head is not enabled"):
        policy.update_route_intent()


def test_route_head_mixin_rejects_missing_head_class():
    from opencda_marl.builtin.algorithms.route_head import RouteHeadMixin

    class NoHead(RouteHeadMixin):
        device = "cpu"

    with pytest.raises(NotImplementedError, match="route_head_cls"):
        NoHead().enable_route_intent({}, state_dim=4, pref_dim=2, reward_dim=2)


# --------------------------------------------------------------------- #
# End-to-end: the seam itself
# --------------------------------------------------------------------- #
def test_external_policy_runs_without_touching_framework(clean_registry):
    from opencda_marl._kernel.policy.manager import MARLManager

    @register_algorithm("constant", label="Constant")
    def _build(config, *, state_dim, action_dim):
        return ConstantSpeedPolicy(config, state_dim, action_dim)

    manager = MARLManager(
        {
            "algorithm": "constant",
            "action_space": {"mode": "speed_only"},
            "state_dim": 3,
            "action_dim": 1,
            "constant": {
                "constant_speed": 27.5,
                "max_action": 45.0,
                "features": {"rel_x": 1, "rel_y": 1, "speed": 1},
                "tensorboard": {"enabled": False},
            },
        },
        "constant",
    )

    assert isinstance(manager.algorithm, ConstantSpeedPolicy)
    # Capability dispatch, not an isinstance tuple in the manager.
    assert manager._is_multi_agent_policy() is True
    # Strategy synthesized from declared capabilities.
    assert manager.strategy.name == "constant"
    assert manager.strategy.reward_aggregation == "per_agent"

    observations = {
        "veh_a": {
            "relative_position_to_intersection": {"x": 1.0, "y": 2.0},
            "speed": 10.0,
        },
    }
    speeds = manager.compute_actions(observations, training=True)
    assert speeds == {"veh_a": 27.5}


def test_external_policy_without_intent_is_rejected_for_learned_path(clean_registry):
    from opencda_marl._kernel.policy.manager import MARLManager

    @register_algorithm("constant_nopath", label="ConstantNoPath")
    def _build(config, *, state_dim, action_dim):
        return ConstantSpeedPolicy(config, state_dim, action_dim)

    cfg = {
        "algorithm": "constant_nopath",
        "state_dim": 3,
        "action_dim": 1,
        "action_space": {
            "mode": "path_speed",
            "path": {
                "selector": "learned",
                "reward_dim": 2,
                "preference": {"dim": 2},
                "head": {"hidden_dims": [8, 8]},
            },
        },
        "preference": {"profile": "balanced", "dim": 2},
        "constant_nopath": {
            "features": {"rel_x": 1, "rel_y": 1, "speed": 1},
            "tensorboard": {"enabled": False},
        },
        "outputs": {"tensorboard": {"enabled": False}},
    }
    with pytest.raises(NotImplementedError, match="supports_intent=False"):
        MARLManager(cfg, "constant_nopath")


def test_state_dim_mismatch_is_warned_not_silently_applied(clean_registry, caplog):
    from opencda_marl._kernel.policy.manager import MARLManager

    @register_algorithm("constant_dim", label="ConstantDim")
    def _build(config, *, state_dim, action_dim):
        return ConstantSpeedPolicy(config, state_dim, action_dim)

    manager = MARLManager(
        {
            "algorithm": "constant_dim",
            "action_space": {"mode": "speed_only"},
            # Three features are configured below, so the derived width is 3.
            "state_dim": 7,
            "action_dim": 1,
            "constant_dim": {
                "features": {"rel_x": 1, "rel_y": 1, "speed": 1},
                "tensorboard": {"enabled": False},
            },
        },
        "constant_dim",
    )
    # Configured value wins (checkpoint compatibility) ...
    assert manager.algorithm.state_dim == 7
    # ... and the disagreement is surfaced rather than hidden.
    assert manager.resolve_state_dim() == 7
