"""Strategy classes for MARLManager's per-algorithm-family dispatch.

Replaces the legacy _update_<algo> hand-rolled methods. Each strategy adapts
an Algorithm-family-specific action-to-speed conversion, transition storage,
and update path to a uniform protocol.

The name *Strategy* is chosen deliberately to avoid collision with
opencda_marl/_kernel/adapter/vehicle_adapter.py (genuine GOF Adapter
pattern). The adapter package translates CARLA actor APIs into the MARL
contract; this file dispatches between algorithm families with a uniform
interface, which is the Strategy pattern, not the Adapter pattern.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Protocol

import numpy as np
from loguru import logger


class AlgorithmStrategy(Protocol):
    """Uniform interface over algorithm families."""
    name: str
    supports_path_head: bool
    reward_aggregation: str  # "per_agent" | "team_mean"

    def action_to_speed(self, action: Any, agent_id: str) -> Optional[float]: ...
    def store_transition(self, *, mgr, agent_id, multi_agent_obs, action,
                         reward, next_multi_agent_obs, done) -> None: ...
    def update(self, *, mgr, rewards: Dict, observations: Dict,
               next_observations: Dict,
               objective_outputs: Optional[Dict] = None) -> None: ...


# --- Base strategy with shared transition/update flow -----------------------


class _BaseStrategy:
    name = "base"
    supports_path_head = False
    reward_aggregation = "per_agent"

    def action_to_speed(self, action, agent_id):
        if action is None:
            return None
        return float(action)

    def store_transition(self, *, mgr, agent_id, multi_agent_obs, action,
                         reward, next_multi_agent_obs, done):
        mgr.algorithm.store_transition(
            multi_agent_obs, agent_id, action, reward,
            next_multi_agent_obs, done,
        )

    def _aggregate_team_reward(self, rewards: Dict):
        if not rewards:
            return 0.0
        values = list(rewards.values())
        sample = values[0]
        if isinstance(sample, np.ndarray):
            stacked = np.stack(
                [np.asarray(v, dtype=np.float32) for v in values], axis=0,
            )
            return stacked.mean(axis=0)
        return float(np.mean([float(v) for v in values]))

    def _terminal_next_multi_obs(self, *, next_states, agent_id, multi_obs):
        """Build a shaped next-state context for terminal agents.

        PPO-family algorithms ignore this value, while off-policy algorithms
        still need an ego key to prepare tensors before the done mask removes
        bootstrapping from the target.
        """
        if agent_id in next_states:
            return next_states[agent_id]['all_states']

        if next_states:
            first_live = next(iter(next_states.values()))
            next_multi_obs = dict(first_live.get('all_states', {}))
        else:
            next_multi_obs = {}

        if agent_id in multi_obs:
            next_multi_obs[agent_id] = np.zeros_like(multi_obs[agent_id])
        return next_multi_obs

    def update(self, *, mgr, rewards, observations, next_observations,
               objective_outputs=None):
        states = mgr._extract_states(observations)
        next_states = mgr._extract_states(next_observations)
        team_reward = None
        if self.reward_aggregation == "team_mean":
            team_reward = self._aggregate_team_reward(rewards)

        stored = 0
        for agent_id_str in list(states.keys()):
            agent_id = agent_id_str
            if agent_id not in mgr.last_actions:
                continue
            if agent_id not in rewards:
                continue
            multi_obs = states[agent_id]['all_states']
            next_multi_obs = self._terminal_next_multi_obs(
                next_states=next_states,
                agent_id=agent_id,
                multi_obs=multi_obs,
            )
            action = mgr.last_actions[agent_id]
            reward = team_reward if team_reward is not None else rewards[agent_id]
            done = agent_id not in next_states
            self.store_transition(
                mgr=mgr, agent_id=agent_id,
                multi_agent_obs=multi_obs, action=action,
                reward=reward, next_multi_agent_obs=next_multi_obs, done=done,
            )
            stored += 1
        # Only trigger an algorithm update when this step actually contributed
        # new agent experience. On idle steps (no agents present, or all agents
        # filtered out by the key-match checks above) an unconditional call
        # would waste compute and -- worse -- could let off-policy algos
        # over-train on stale buffered transitions while no new signal was
        # flowing in.
        if stored > 0:
            metrics = mgr.algorithm.update()
            record = getattr(mgr, "record_update_metrics", None)
            if record is not None:
                record(metrics)


# --- Concrete strategies -----------------------------------------------------

class TD3Strategy(_BaseStrategy):
    name = "td3"
    supports_path_head = False
    reward_aggregation = "per_agent"

    def action_to_speed(self, action, agent_id):
        # TD3 returns None during warmup to delegate to vanilla agent.
        if action is None:
            return None
        return float(action)


class MAPPOStrategy(_BaseStrategy):
    name = "mappo"
    supports_path_head = True
    reward_aggregation = "per_agent"


class PCMAPPOStrategy(MAPPOStrategy):
    name = "pcmappo"
    reward_aggregation = "team_mean"


class SACStrategy(_BaseStrategy):
    name = "sac"
    supports_path_head = True
    reward_aggregation = "per_agent"


class PCMASACStrategy(SACStrategy):
    """PCMASAC strategy: team-mean reward aggregation.

    PCMASAC needs its own strategy: masac and pcmasac share one learner
    class (the registry maps both to builder_name='sac'), so an isinstance
    check cannot tell them apart. Dispatch is by algorithm name instead, and
    PCMASAC gets team-mean reward aggregation like PCMAPPO.
    """
    name = "pcmasac"
    reward_aggregation = "team_mean"


class _SingleAgentStrategy(_BaseStrategy):
    """Transition flow for single-agent policies (``multi_agent=False``).

    Their extractor returns one state per agent -- a feature vector (DQN)
    or a discrete state index (tabular Q) -- not the ``{'all_states': ...}``
    mapping the multi-agent flow indexes, and their ``store_transition``
    takes ``(state, action, reward, next_state, done)``. ``action`` is
    whatever the manager recorded in ``last_actions``: the target speed, or
    the index for a policy that declares ``returns_action_index``.
    """

    def store_transition(self, *, mgr, agent_id, multi_agent_obs, action,
                         reward, next_multi_agent_obs, done):
        # The keyword names are the shared protocol's; here they carry the
        # agent's own state and next state.
        mgr.algorithm.store_transition(
            multi_agent_obs, action, reward, next_multi_agent_obs, done,
        )

    @staticmethod
    def _terminal_next_state(state):
        """The next state of an agent that left this step.

        Same shape as ``state`` so the learner can batch it; ``done`` masks
        it out of the bootstrap target, so its values never matter. A
        discrete state index is reused as is.
        """
        if isinstance(state, np.ndarray):
            return np.zeros_like(state)
        return state

    def update(self, *, mgr, rewards, observations, next_observations,
               objective_outputs=None):
        states = mgr._extract_states(observations)
        next_states = mgr._extract_states(next_observations)
        team_reward = None
        if self.reward_aggregation == "team_mean":
            team_reward = self._aggregate_team_reward(rewards)

        stored = 0
        for agent_id, state in states.items():
            if agent_id not in mgr.last_actions or agent_id not in rewards:
                continue
            done = agent_id not in next_states
            next_state = (self._terminal_next_state(state) if done
                          else next_states[agent_id])
            reward = team_reward if team_reward is not None else rewards[agent_id]
            self.store_transition(
                mgr=mgr, agent_id=agent_id,
                multi_agent_obs=state, action=mgr.last_actions[agent_id],
                reward=reward, next_multi_agent_obs=next_state, done=done,
            )
            stored += 1
        # Same idle-step rule as the multi-agent flow: no new experience,
        # no update.
        if stored > 0:
            metrics = mgr.algorithm.update()
            record = getattr(mgr, "record_update_metrics", None)
            if record is not None:
                record(metrics)


class QLearningStrategy(_SingleAgentStrategy):
    name = "q_learning"
    supports_path_head = False
    reward_aggregation = "per_agent"


class DQNStrategy(_SingleAgentStrategy):
    name = "dqn"
    supports_path_head = False
    reward_aggregation = "per_agent"


# --- Selector ----------------------------------------------------------------

_STRATEGIES = {
    "td3": TD3Strategy,
    "matd3": TD3Strategy,
    "pcmatd3": TD3Strategy,
    "mappo": MAPPOStrategy,
    "pcmappo": PCMAPPOStrategy,
    "sac": SACStrategy,
    "masac": SACStrategy,
    "pcmasac": PCMASACStrategy,
    "q_learning": QLearningStrategy,
    "dqn": DQNStrategy,
}


def strategy_from_capabilities(algorithm_name: str, capabilities) -> AlgorithmStrategy:
    """Synthesize a strategy for an externally registered algorithm.

    The behavioral axes the built-in strategies vary are reward aggregation
    and the observation shape (multi-agent or single-agent), so a
    registered policy needs no hand-written strategy class: its declared
    capabilities are enough.
    """
    if getattr(capabilities, "multi_agent", True):
        strategy = _BaseStrategy()
    else:
        strategy = _SingleAgentStrategy()
    strategy.name = algorithm_name
    strategy.reward_aggregation = getattr(
        capabilities, "reward_aggregation", "per_agent")
    strategy.supports_path_head = bool(
        getattr(capabilities, "supports_intent", False))
    return strategy


def select_strategy(algorithm_name: str, spec=None) -> AlgorithmStrategy:
    """Map an algorithm name to its Strategy instance.

    Built-in families keep their dedicated classes; anything registered
    through ``registry.register_algorithm`` is served from its declared
    capabilities, so adding an algorithm needs no edit here.
    """
    if algorithm_name in _STRATEGIES:
        return _STRATEGIES[algorithm_name]()
    capabilities = getattr(spec, "capabilities", None)
    if capabilities is not None:
        return strategy_from_capabilities(algorithm_name, capabilities)
    raise NotImplementedError(
        f"No strategy for algorithm '{algorithm_name}'. "
        f"Known: {sorted(_STRATEGIES)}"
    )


__all__ = [
    "AlgorithmStrategy",
    "TD3Strategy", "MAPPOStrategy", "PCMAPPOStrategy",
    "SACStrategy", "PCMASACStrategy",
    "QLearningStrategy", "DQNStrategy",
    "select_strategy", "strategy_from_capabilities",
]
