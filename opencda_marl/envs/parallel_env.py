"""PettingZoo ParallelEnv facade: bring your own training loop.

This is the outermost integration boundary. The framework emits per-agent
observation vectors and accepts per-agent target speeds; everything else —
observation extraction, reward computation, planner/controller execution,
route legality, episode lifecycle, deterministic replays — stays inside and
identical for every algorithm, which is what makes results comparable as a
benchmark::

    from opencda_marl.envs.parallel_env import OpenCDAMARLParallelEnv

    env = OpenCDAMARLParallelEnv.from_scenario(
        "sumo-marl/intersection/...",
        overrides={"MARL": {"external": {"features": {"rel_x": 1, "rel_y": 1}}}})
    observations, infos = env.reset()
    while not env.episode_over:
        actions = {agent: my_policy(observations[agent]) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

Loop on ``episode_over``, not on ``env.agents``: vehicles spawn over time, so
the live set is empty at the start of a replay and between waves of traffic.

Design notes:

- **Action channel is target speed (km/h)** — the matched-protocol contract:
  the planner and low-level controller are held fixed so cross-algorithm
  comparisons measure the decision, not the stack. Path actions stay
  framework-side (``MARL.action_space.path``: selector / provider) and are
  not exposed through this port.
- **Reward is framework-computed** (benchmark standard). The per-component
  breakdown, objective vectors, and preference vectors ride in ``infos`` so
  external loops can re-shape without forking the pipeline.
- **Agents join and leave mid-episode** (vehicles spawn and terminate).
  ``env.agents`` reflects the live set after each step; a terminating agent
  appears one final time in the step returns with its last observation.
- Implemented by registering an ``external`` mailbox policy through the
  standard algorithm registry: the inner environments run unmodified. An
  agent you stop sending actions for is handed back to SUMO's own
  car-following model (under the scenario's ``world.sumo_rl_runtime_mode``);
  an agent you have not commanded yet keeps the departure speed it spawned
  with. To control every agent, send an action for every agent in
  ``env.agents``.

``pettingzoo``/``gymnasium`` are soft dependencies: without them the class
still runs (same API by duck typing); with them it is a genuine
``pettingzoo.ParallelEnv`` subclass with real Gymnasium spaces
(``pixi add pettingzoo gymnasium``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from loguru import logger

from opencda_marl.api.algorithms import (
    register_algorithm,
    registered_algorithm_keys,
)

try:  # soft dependency — see module docstring
    from pettingzoo.utils.env import ParallelEnv as _ParallelEnvBase
except ImportError:  # pragma: no cover - exercised in envs without pettingzoo
    _ParallelEnvBase = object

try:  # soft dependency
    from gymnasium import spaces as _gym_spaces
except ImportError:  # pragma: no cover
    _gym_spaces = None


class ExternalPolicy:
    """Mailbox policy: actions arrive from an external loop, not a learner.

    Registered as algorithm ``external``. ``select_action`` returns whatever
    the ParallelEnv facade last received for that agent; ``None`` (no command)
    hands an agent commanded earlier back to the simulator's own driver; one
    never commanded keeps its departure speed. Learning methods are no-ops —
    training happens outside.
    """

    supports_batched_actions = False
    supports_vector_rewards = False

    def __init__(self, config: Mapping, state_dim: int, action_dim: int):
        self.config = dict(config or {})
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(self.config.get("max_action", 45.0))
        self.min_action = float(self.config.get("min_action", 0.0))
        self._actions: Dict[str, Optional[float]] = {}

    # -- mailbox ------------------------------------------------------- #
    def set_actions(self, actions: Mapping[Any, Any]) -> None:
        self._actions = {
            str(agent_id): _coerce_speed(value)
            for agent_id, value in (actions or {}).items()
        }

    def clear_actions(self) -> None:
        self._actions = {}

    # -- Policy contract ----------------------------------------------- #
    def select_action(self, multi_agent_obs, ego_agent_id, training: bool = True):
        return self._actions.get(str(ego_agent_id))

    def store_transition(self, multi_agent_obs, ego_agent_id, action, reward,
                         next_multi_agent_obs, done) -> None:
        return None

    def update(self) -> Dict[str, float]:
        return {}

    def reset_episode(self) -> None:
        self.clear_actions()

    def get_training_info(self) -> Dict[str, Any]:
        return {"algorithm_type": "external", "commanded_agents": len(self._actions)}

    def save(self, filepath: str) -> None:
        return None

    def load(self, filepath: str) -> None:
        return None


def _coerce_speed(value: Any) -> Optional[float]:
    """Accept the shapes external loops actually produce.

    Scalars, 0-d/1-element arrays (Gymnasium ``Box`` samples), and
    ``{"target_speed": x}`` dicts (forward-compatible with a future Dict
    action space) all normalize to a float; ``None`` passes through as the
    explicit "no command" signal.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _coerce_speed(value.get("target_speed"))
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 1:
        raise ValueError(
            f"Expected a scalar target speed, got shape {np.asarray(value).shape}")
    return float(array[0])


if "external" not in registered_algorithm_keys():  # idempotent module import
    @register_algorithm("external", label="External",
                        reward_aggregation="per_agent")
    def _build_external(config, *, state_dim, action_dim):
        return ExternalPolicy(config, state_dim, action_dim)


class OpenCDAMARLParallelEnv(_ParallelEnvBase):
    """PettingZoo-style parallel environment over an OpenCDA-MARL scenario."""

    metadata = {"name": "opencda_marl_v0", "render_modes": []}

    def __init__(self, inner_env, marl_manager, *, render_mode: Optional[str] = None):
        """Wrap an already-constructed backend environment.

        ``inner_env`` needs ``step() -> dict`` (the standard result dict with
        ``observations``/``rewards``/``events``/``done``) and a reset method;
        ``marl_manager`` must be driving an :class:`ExternalPolicy` (use
        :meth:`from_scenario` to get this wiring from a config).
        """
        self._inner = inner_env
        self._manager = marl_manager
        self.render_mode = render_mode

        policy = getattr(marl_manager, "algorithm", None)
        if not isinstance(policy, ExternalPolicy):
            raise TypeError(
                "OpenCDAMARLParallelEnv requires MARL.algorithm='external' "
                f"(got {type(policy).__name__}). Use from_scenario(), or set "
                "the algorithm to 'external' in the scenario config.")
        self._policy: ExternalPolicy = policy

        self.agents: List[str] = []
        self._seen_agents: List[str] = []
        self._last_obs: Dict[str, np.ndarray] = {}
        self._spaces_cache: Dict[str, Any] = {}
        self._width: Optional[int] = None
        self._episode_over = False

    @property
    def episode_over(self) -> bool:
        """True once the inner environment reported the episode done.

        The episode-level signal ``env.agents`` cannot give: the live set is
        also empty before the first vehicle spawns and between waves of
        traffic. Cleared by :meth:`reset`.
        """
        return self._episode_over

    # ------------------------------------------------------------------ #
    # Construction from a scenario config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_scenario(cls, scenario_ref: str, *, overrides: Optional[Dict] = None,
                      render_mode: Optional[str] = None) -> "OpenCDAMARLParallelEnv":
        """Build the env from a config reference like ``sumo-marl/intersection/x``.

        The MARL algorithm is forced to ``external`` and the scenario's
        observation features (``MARL.external.features``) define the vector
        every algorithm sees — the benchmark declares the observation space,
        not the algorithm. The features can live in the profile or arrive
        through ``overrides``. SUMO backend only for now: the CARLA backend
        needs live-server orchestration and lands with the parallel-training
        work.

        The reference resolves exactly as on the command line (a ``configs/``
        root under the working directory, ``OPENCDA_MARL_CONFIG_ROOTS``, then
        the bundled configs), and a profile's ``include:`` list is composed.
        """
        from omegaconf import OmegaConf

        from opencda_marl.utils import config_locator, config_merger

        from pathlib import Path

        resolved = config_locator.resolve_marl_config_paths(Path.cwd(), scenario_ref)
        merged = config_merger.smart_merge(
            OmegaConf.load(resolved["default_yaml"]),
            OmegaConf.load(resolved["scenario_base_yaml"]),
        )
        merged = config_merger.smart_merge(
            merged, config_locator.load_profile_with_includes(resolved["scenario_yaml"]))
        if overrides:
            merged = config_merger.smart_merge(merged, OmegaConf.create(overrides))
        config = OmegaConf.to_container(merged, resolve=True)
        # The env is built here, not by the coordinator, so the run's
        # sections are finalised here too: one left empty is a mapping.
        config_merger.normalize_run_config(config)

        backend = str(config.get("runtime", {}).get("backend", "carla")).lower()
        if backend != "sumo":
            raise NotImplementedError(
                "from_scenario currently supports the SUMO backend; the CARLA "
                "backend needs a live server and is planned alongside the "
                "parallel-training work.")

        marl_cfg = config.setdefault("MARL", {})
        marl_cfg["algorithm"] = "external"
        # The width follows the declared features: a profile's MARL.state_dim
        # belongs to its own learner (and its checkpoints), and kept here it
        # would make observation_space() disagree with the vectors actually
        # returned.
        marl_cfg.pop("state_dim", None)
        features = (marl_cfg.get("external") or {}).get("features")
        if not features:
            raise ValueError(
                "Scenario config must declare MARL.external.features — the "
                "benchmark's observation definition (feature name -> width), "
                "e.g. {rel_x: 1, rel_y: 1, heading_angle: 1, "
                "dist_to_intersection: 1}. Put it in the profile or pass it "
                "through overrides={'MARL': {'external': {'features': ...}}}.")

        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv  # lazy: needs SUMO_HOME

        inner = SumoMARLEnv(config)
        return cls(inner, inner.marl_manager, render_mode=render_mode)

    # ------------------------------------------------------------------ #
    # PettingZoo ParallelEnv API
    # ------------------------------------------------------------------ #
    @property
    def possible_agents(self) -> List[str]:
        """Agents observed so far. Vehicles are created dynamically, so this
        grows during an episode; spaces are identical for every agent."""
        return list(self._seen_agents)

    def observation_space(self, agent: Optional[str] = None):
        return self._space("observation")

    def action_space(self, agent: Optional[str] = None):
        return self._space("action")

    def _state_width(self) -> int:
        """The observation width, resolved once: it is fixed for the run."""
        if self._width is None:
            self._width = int(self._manager.resolve_state_dim())
        return self._width

    def _space(self, kind: str):
        if _gym_spaces is None:
            raise ImportError(
                "gymnasium is required for space objects: "
                "`pixi add gymnasium pettingzoo`. The env itself runs without "
                "it — observations are float32 vectors of width "
                f"{self._state_width()}, actions are target "
                "speeds in km/h.")
        if kind not in self._spaces_cache:
            if kind == "observation":
                dim = self._state_width()
                self._spaces_cache[kind] = _gym_spaces.Box(
                    low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
            else:
                self._spaces_cache[kind] = _gym_spaces.Box(
                    low=np.float32(self._policy.min_action),
                    high=np.float32(self._policy.max_action),
                    shape=(), dtype=np.float32)
        return self._spaces_cache[kind]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            logger.warning(
                "OpenCDAMARLParallelEnv ignores reset(seed=...): determinism "
                "comes from recorded traffic replays declared in the scenario "
                "config, not per-reset seeding.")
        self._policy.clear_actions()
        self._last_obs = {}
        self._seen_agents = []
        self._episode_over = False

        if hasattr(self._inner, "reset"):
            self._inner.reset()
        elif hasattr(self._inner, "reset_episode"):
            self._inner.reset_episode()

        raw = {}
        getter = getattr(self._inner, "_get_current_observations", None)
        if callable(getter):
            raw = getter() or {}
        observations = self._vectorize(raw)
        infos = {agent: {"raw_observation": raw.get(agent)} for agent in observations}
        self._remember(observations)
        self.agents = list(observations)
        return observations, infos

    def step(self, actions: Mapping[Any, Any]):
        self._policy.set_actions(actions)
        result = self._inner.step() or {}

        raw = result.get("observations") or {}
        raw = {str(k): v for k, v in raw.items()}
        observations = self._vectorize(raw)

        rewards = {
            str(k): float(v)
            for k, v in (result.get("rewards") or {}).items()
        }

        terminations: Dict[str, bool] = {}
        agent_events: Dict[str, str] = {}
        for event in result.get("events") or []:
            event_type = getattr(event, "event_type", None)
            vehicle_id = str(getattr(event, "vehicle_id", ""))
            if event_type in ("collision", "success"):
                terminations[vehicle_id] = True
                agent_events[vehicle_id] = event_type

        # Agents that vanished without a terminal event (e.g. removed by the
        # simulator) still need a final, terminated appearance.
        for agent in self._last_obs:
            if agent not in observations and agent not in terminations:
                terminations[agent] = True
                agent_events.setdefault(agent, "removed")

        episode_done = bool(result.get("done", False))
        self._episode_over = self._episode_over or episode_done
        truncations = {
            agent: episode_done and not terminations.get(agent, False)
            for agent in set(observations) | set(terminations)
        }

        # PettingZoo contract: a terminating agent appears once more in every
        # return dict. Its observation is its last known vector.
        for agent in terminations:
            if agent not in observations:
                observations[agent] = self._last_obs.get(
                    agent,
                    np.zeros(self._state_width(), dtype=np.float32),
                )

        all_agents = set(observations)
        for agent in all_agents:
            rewards.setdefault(agent, 0.0)
            terminations.setdefault(agent, False)
            truncations.setdefault(agent, False)

        infos: Dict[str, Dict[str, Any]] = {}
        components = result.get("reward_components") or {}
        vectors = result.get("reward_vectors") or {}
        objective_vectors = result.get("objective_vectors") or {}
        preferences = result.get("agent_preferences") or {}
        for agent in all_agents:
            info: Dict[str, Any] = {"raw_observation": raw.get(agent)}
            if agent in components:
                info["reward_components"] = components[agent]
            if agent in vectors:
                info["reward_vector"] = vectors[agent]
            if agent in objective_vectors:
                info["objective_vector"] = objective_vectors[agent]
            if agent in preferences:
                info["preference"] = preferences[agent]
            if agent in agent_events:
                info["event"] = agent_events[agent]
            infos[agent] = info

        self._remember({a: observations[a] for a in observations
                        if not terminations[a]})
        self.agents = [a for a in observations
                       if not terminations[a] and not truncations[a]]
        return observations, rewards, terminations, truncations, infos

    def render(self):  # pragma: no cover - GUI handled by the simulators
        return None

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()

    # ------------------------------------------------------------------ #
    def _vectorize(self, raw: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """Raw simulator observations -> the benchmark's feature vectors."""
        if not raw:
            return {}
        states = self._manager._extract_states(raw)
        return {
            str(agent): np.asarray(data["ego_state"], dtype=np.float32)
            for agent, data in states.items()
        }

    def _remember(self, observations: Mapping[str, np.ndarray]) -> None:
        for agent, vector in observations.items():
            if agent not in self._seen_agents:
                self._seen_agents.append(agent)
            self._last_obs[agent] = vector


__all__ = ["OpenCDAMARLParallelEnv", "ExternalPolicy"]
