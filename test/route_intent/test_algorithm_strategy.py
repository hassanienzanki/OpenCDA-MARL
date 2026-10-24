"""AlgorithmStrategy contract + PCMASAC team-mean bug fix."""
from __future__ import annotations
import inspect
import numpy as np
import pytest


def test_strategy_protocol_exists():
    from opencda_marl._kernel.policy.strategy import AlgorithmStrategy
    assert AlgorithmStrategy is not None


def test_pcmasac_strategy_has_team_mean_reward_aggregation():
    """Regression test for the previously inaccessible PCMASAC team-mean path."""
    from opencda_marl._kernel.policy.strategy import PCMASACStrategy
    s = PCMASACStrategy()
    assert s.name == "pcmasac"
    assert s.reward_aggregation == "team_mean"


def test_pcmappo_strategy_has_team_mean_reward_aggregation():
    from opencda_marl._kernel.policy.strategy import PCMAPPOStrategy
    s = PCMAPPOStrategy()
    assert s.name == "pcmappo"
    assert s.reward_aggregation == "team_mean"


def test_masac_and_mappo_use_per_agent_aggregation():
    """Non-PC variants stay on per-agent rewards."""
    from opencda_marl._kernel.policy.strategy import SACStrategy, MAPPOStrategy
    s = SACStrategy()
    assert s.name == "sac"
    assert s.reward_aggregation == "per_agent"
    m = MAPPOStrategy()
    assert m.name == "mappo"
    assert m.reward_aggregation == "per_agent"


def test_all_strategies_implement_protocol():
    from opencda_marl._kernel.policy.strategy import (
        AlgorithmStrategy, TD3Strategy, MAPPOStrategy, PCMAPPOStrategy,
        SACStrategy, PCMASACStrategy,
    )
    for cls in (TD3Strategy, MAPPOStrategy, PCMAPPOStrategy,
                SACStrategy, PCMASACStrategy):
        s = cls()
        assert isinstance(s.name, str)
        assert isinstance(s.reward_aggregation, str)
        assert isinstance(s.supports_path_head, bool)
        assert callable(getattr(s, "action_to_speed", None))
        assert callable(getattr(s, "store_transition", None))
        assert callable(getattr(s, "update", None))


def test_select_strategy_routes_pcmasac_to_pcmasac_strategy():
    from opencda_marl._kernel.policy.strategy import select_strategy, PCMASACStrategy
    s = select_strategy("pcmasac")
    assert isinstance(s, PCMASACStrategy)
    assert s.reward_aggregation == "team_mean"


def test_marl_manager_dispatches_via_strategy():
    """MARLManager has a .strategy attribute and no _update_<algo> dispatcher."""
    from opencda_marl._kernel.policy import MARLManager
    forbidden = ["_update_q_learning", "_update_dqn", "_update_td3",
                 "_update_mappo", "_update_pcmappo", "_update_sac"]
    for name in forbidden:
        assert not hasattr(MARLManager, name), (
            f"MARLManager still has {name}; expected strategy dispatch"
        )


def test_strategy_stores_terminal_transition_when_agent_leaves_next_state():
    """Terminal rewards must reach the speed policy even after the ego agent exits."""
    from opencda_marl._kernel.policy.strategy import _BaseStrategy

    class RecordingAlgorithm:
        def __init__(self):
            self.updated = False

        def update(self):
            self.updated = True

    class RecordingStrategy(_BaseStrategy):
        def __init__(self):
            self.calls = []

        def store_transition(self, **kwargs):
            self.calls.append(kwargs)

    class DummyManager:
        def __init__(self):
            self.algorithm = RecordingAlgorithm()
            self.last_actions = {"veh": 12.0}

        def _extract_states(self, observations):
            return observations

    strategy = RecordingStrategy()
    mgr = DummyManager()
    current_state = np.array([1.0, 2.0], dtype=np.float32)

    strategy.update(
        mgr=mgr,
        rewards={"veh": 5.0},
        observations={"veh": {"all_states": {"veh": current_state}}},
        next_observations={},
    )

    assert mgr.algorithm.updated is True
    assert len(strategy.calls) == 1
    call = strategy.calls[0]
    assert call["done"] is True
    assert call["reward"] == 5.0
    assert np.allclose(call["next_multi_agent_obs"]["veh"], np.zeros_like(current_state))
