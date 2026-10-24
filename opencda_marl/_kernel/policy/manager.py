'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 17:58:12
FilePath     : /OpenCDA-MARL/opencda_marl/_kernel/policy/manager.py
Description  : MARL Manager for speed control of autonomous vehicles
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import functools
import os
from collections.abc import Mapping as MappingABC
from typing import Dict, Any, Mapping
import numpy as np
from loguru import logger

from opencda_marl.api.contracts import RewardOutput
# Only the legacy single-agent families are named here, and only so that
# hand-constructed managers (tests, baselines) still dispatch correctly.
# Registered policies are dispatched by declared capability instead —
# see MARLManager._is_multi_agent_policy.
# NOTE: rewards.objectives and rewards.preference_registry are imported lazily
# inside the methods that need them, to keep import order free of cycles.
from opencda_marl.api.rewards import resolve_reward_configuration
from .action_space import resolve_action_space_config, with_baseline_default
from .factory import build_algorithm as build_registered_algorithm
from .path_lifecycle import configure_path_action
from opencda_marl.api.algorithms import resolve_algorithm_spec


class MARLManager:
    """Manages RL algorithms and speed control decisions"""

    def __init__(self, config: Dict, algorithm: str | None = None):
        if algorithm is None:
            config = with_baseline_default(config)
        self.config = config
        self.algorithm_spec = None
        self.algorithm_name = None
        self.algorithm_label = "NONE"
        self.action_space_config = resolve_action_space_config(config)
        # `MARL.plugins` names modules (or .py files) that register
        # algorithms, components or profiles; installed packages advertise
        # an entry point instead and need nothing here. Loaded for every
        # run, learner or not: a run without a learner still selects its
        # planner, agent types and path provider through the registry.
        # Environments built without the coordinator (ParallelEnv, tests)
        # rely on this call.
        from opencda_marl.api.algorithms import load_plugins
        load_plugins(config.get("plugins"))
        if algorithm is not None:
            self.algorithm_spec = resolve_algorithm_spec(algorithm)
            self.algorithm_name = self.algorithm_spec.key
            self.algorithm_label = self.algorithm_spec.label
        self.reward_config = resolve_reward_configuration(config)
        self.reward_spec = self.reward_config.reward_spec
        from opencda_marl.api.objectives import resolve_objective_configuration
        from opencda_marl.api.preferences import PreferenceRegistry
        self.objective_config = resolve_objective_configuration(config)
        self.objective_spec = self.objective_config.objective_spec
        self.preference_registry = PreferenceRegistry(config.get('preference', {}))
        self.preference_enabled = self.objective_spec is not None
        self.reward_mode = str(config.get('reward_mode', 'scalarized_reward'))
        # Two-slot identity cache for _extract_states (see its docstring).
        self._state_cache: Dict[int, tuple] = {}
        self._learning_cache: Dict[int, tuple] = {}
        # Consecutive-failure counters for the hot path (see _note_failure).
        self._failure_streaks: Dict[str, int] = {}
        # The extractor is built first: it computes the true feature width,
        # which resolve_state_dim() cross-checks against the configured
        # MARL.state_dim before the networks are sized.
        from opencda_marl.api.components import component_key, resolve_component
        self.observation_extractor = (
            resolve_component(
                "observation_extractor",
                component_key(config, "observation_extractor"),
                config=config,
            )(
                config,
                self.algorithm_name,
                multi_agent=(
                    self.algorithm_spec.capabilities.multi_agent
                    if self.algorithm_spec is not None else None
                ),
            )
            if self.algorithm_name is not None else None
        )
        self.algorithm = self.build_algorithm(algorithm)
        self._external_store = None
        self._update_calls = 0
        self._episodes_logged = 0
        self._init_external_metrics_store()
        from opencda_marl._kernel.policy.strategy import select_strategy
        self.strategy = (
            select_strategy(self.algorithm_name, spec=self.algorithm_spec)
            if self.algorithm_name else None
        )

        # The action each agent last took, in the form the policy's
        # store_transition takes it back: a target speed, or an index into
        # ``speed_actions`` for a policy with ``returns_action_index``.
        self.last_actions = {}  # Dict[agent_id, speed or action index]
        self._reset_intent_episode_diagnostics()

        self.intent_algorithm = None
        self.intent_reward_accumulator = None
        self.intent_reward_scale = 1.0
        self.intent_update_interval_steps = 1
        self._intent_update_tick = 0
        self._configure_path_action()

        logger.success(f"MARLManager initialized with {self.algorithm_label}")

    def _configure_path_action(self) -> None:
        """Attach learned one-shot path heads for path_speed configs."""
        configure_path_action(self)

    def _merge_shared_config(self, algo_config: Dict) -> Dict:
        """Merge shared config (tensorboard, training mode, etc.) with algorithm-specific config."""
        import copy

        # Deep copy: a shallow .copy() shared nested dicts with the live
        # config object, so downstream writes (e.g. disabling tensorboard in
        # eval mode below) could silently mutate the caller's configuration.
        merged = copy.deepcopy(dict(algo_config))

        # Pass training_mode to algorithm (affects TensorBoard and learning)
        training_config = self.config.get('training', {})
        merged['training_mode'] = training_config.get('training_mode', True)

        # Merge tensorboard config (algorithm can override). On either side,
        # a block left empty in YAML (None) counts as absent and a bare
        # ``true`` / ``false`` is the ``enabled`` switch.
        outputs_cfg = self.config.get('outputs', {})
        root_dir = outputs_cfg.get('root_dir')
        tb_config = outputs_cfg.get('tensorboard')
        if isinstance(tb_config, bool):
            tb_config = {'enabled': tb_config}
        tb_config = dict(tb_config) if isinstance(tb_config, MappingABC) else {}
        tb_dir = tb_config.pop('dir', None) or 'runs'
        if not tb_config.get('log_dir'):
            tb_config['log_dir'] = os.path.join(root_dir, tb_dir) if root_dir else tb_dir
        if merged.get('tensorboard') is None:
            merged['tensorboard'] = tb_config
        elif isinstance(merged['tensorboard'], MappingABC):
            # Algorithm-specific overrides shared config
            base_tb = dict(tb_config)
            base_tb.update(merged['tensorboard'])
            merged['tensorboard'] = base_tb
        elif isinstance(merged['tensorboard'], bool):
            # A bare switch keeps the shared settings (log_dir, mode,
            # rotation, retention) and only sets ``enabled``.
            merged['tensorboard'] = {**tb_config, 'enabled': merged['tensorboard']}

        # Metrics store (the queryable TensorBoard replacement): same merge
        # rules, rooted at outputs.root_dir/<dir>, with the scenario name and
        # the run name (the launcher patches outputs.metrics_store.run_name to
        # the cell slug) so the monitor can label runs.
        ms_config = outputs_cfg.get('metrics_store', {})
        if isinstance(ms_config, bool):
            ms_config = {'enabled': ms_config}
        ms_config = dict(ms_config) if isinstance(ms_config, MappingABC) else {}
        ms_dir = ms_config.pop('dir', None) or 'metrics'
        if not ms_config.get('log_dir'):
            ms_config['log_dir'] = os.path.join(root_dir, ms_dir) if root_dir else ms_dir
        ms_config.setdefault('scenario', self.config.get('scenario_name'))
        # The registered key names the run directory and the run's
        # ``algorithm`` label: masac and pcmasac share one learner class, so
        # a name derived from the class stored both under ``sac``.
        algorithm_key = getattr(self, 'algorithm_name', None)
        if algorithm_key:
            ms_config.setdefault('algorithm', str(algorithm_key))
        if merged.get('metrics_store') is None:
            merged['metrics_store'] = ms_config
        elif isinstance(merged['metrics_store'], MappingABC):
            base_ms = dict(ms_config)
            base_ms.update(merged['metrics_store'])
            merged['metrics_store'] = base_ms
        elif isinstance(merged['metrics_store'], bool):
            # A bare switch in the algorithm block keeps the shared settings,
            # the registered key with them: it also names the TensorBoard run.
            merged['metrics_store'] = {**ms_config, 'enabled': merged['metrics_store']}

        # Disable TensorBoard in evaluation mode (training_mode: false)
        # This prevents recording non-training metrics during checkpoint testing
        if not merged['training_mode']:
            if isinstance(merged.get('tensorboard'), dict):
                merged['tensorboard']['enabled'] = False
            else:
                merged['tensorboard'] = {'enabled': False}
            logger.info("Evaluation mode: TensorBoard disabled (training_mode: false)")

        return merged

    def build_algorithm(self, algorithm: str):
        """Build algorithm instance based on configuration"""
        if algorithm is None:
            logger.info("No MARL algorithm initialized (baseline agent mode)")
            return None
        return build_registered_algorithm(self, algorithm)

    #: Consecutive failures of one hot-path method before we stop the run.
    MAX_CONSECUTIVE_FAILURES = 25

    #: Episode outcome columns every learner's store summary carries, beside
    #: ``success_rate`` / ``collision_rate`` (rates in percent, counts in
    #: vehicles), whichever path logs the episode.
    OUTCOME_SUMMARY_KEYS = ('timeout_rate', 'successes', 'collisions', 'timeouts')

    def _note_failure(self, where: str, exc: BaseException) -> None:
        """Count consecutive hot-path failures; raise once they persist.

        A one-off exception in compute_actions/update degrades one step (the
        rule-based driver takes over / no learning that step) and is logged.
        The same failure every step means the run is silently producing
        nothing — surface that as a hard error instead of a scrolling log.
        """
        counts = self._failure_streaks
        counts[where] = counts.get(where, 0) + 1
        n = counts[where]
        logger.error(f"Error in MARL {where} ({n} consecutive): {exc}")
        if n >= self.MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"MARL {where} failed {n} times in a row; last error: {exc!r}. "
                "The run is not training — aborting.") from exc

    def _note_success(self, where: str) -> None:
        if self._failure_streaks.get(where):
            self._failure_streaks[where] = 0

    def _is_multi_agent_policy(self) -> bool:
        """Whether the policy takes ``(multi_agent_obs, ego_id)`` actions.

        Declared by the registry spec. The fallback, the policy's own
        ``multi_agent`` attribute, only matters for hand-constructed managers
        in tests.
        """
        if self.algorithm_spec is not None:
            return bool(self.algorithm_spec.capabilities.multi_agent)
        return bool(getattr(self.algorithm, "multi_agent", True))

    def preference_dim(self) -> int:
        """Width of the preference vector appended to every observation.

        Augmentation is gated by the objective spec (``MARL.objective_mode``),
        not by whether ``MARL.preference`` is configured: with an objective
        spec present the env always assigns a preference (default assignment
        when none is configured) and appends it, so non-preference-conditioned
        learners still see those extra dims. Width = ``MARL.preference.dim``
        when given, else the number of objectives.
        """
        if not self.preference_enabled:
            return 0
        pref_cfg = self.config.get("preference", {}) or {}
        configured = pref_cfg.get("dim")
        if configured:
            return int(configured)
        return len(self.objective_spec.objectives)

    def resolve_state_dim(self) -> int:
        """Return the observation width the networks should be built with.

        ``MARL.state_dim`` stays authoritative — it is baked into every saved
        checkpoint, so deriving a different value here would silently
        invalidate them. But the extractor already knows the true feature
        width, so we cross-check and warn loudly on disagreement: a mismatch
        means the policy is training on a differently-shaped vector than the
        config claims, which previously surfaced only as an opaque shape
        error deep in a forward pass.
        """
        configured = self.config.get("state_dim")
        derived_features = getattr(self.observation_extractor, "state_dim", None)
        pref_dim = self.preference_dim()

        if derived_features is not None:
            derived_total = int(derived_features) + pref_dim
            if configured is None:
                logger.info(
                    f"MARL.state_dim not set; using derived width {derived_total} "
                    f"({derived_features} features + {pref_dim} preference)")
                return derived_total
            if int(configured) != derived_total:
                logger.warning(
                    f"MARL.state_dim={configured} disagrees with the extractor's "
                    f"derived width {derived_total} ({derived_features} features "
                    f"+ {pref_dim} preference). Using the configured value for "
                    "checkpoint compatibility — fix the config if this is not "
                    "deliberate.")
        if configured is None:
            raise ValueError(
                "MARL.state_dim is required when no observation extractor is "
                "available to derive it.")
        return int(configured)

    # --------------------------------------------------------------------- #
    # Core MARL methods
    # --------------------------------------------------------------------- #
    @staticmethod
    def _learning_observations(observations: Dict) -> Dict:
        """Observations of the learning population only.

        Homogeneous runs carry no ``learning`` flag and pass through
        unchanged, so single-population behaviour is bit-identical.
        """
        if not observations:
            return observations
        if not any('learning' in (obs or {}) for obs in observations.values()):
            return observations
        return {agent_id: obs for agent_id, obs in observations.items()
                if (obs or {}).get('learning', True)}

    def _learning_view(self, observations: Dict) -> Dict:
        """The learning population of ``observations``, the same object for
        the same input dict.

        The state memo keys on dict identity, but a mixed population is
        filtered into a fresh dict on every call, so ``compute_actions(obs)``
        and ``update(obs, next_obs)`` each extracted from their own copy and
        the memo never hit: three extractions per step. Two slots -- the
        current and the previous dict -- make the filtered view stable across
        the calls of one step and into the next, where next_obs becomes obs.
        """
        cache = self._learning_cache
        key = id(observations)
        hit = cache.get(key)
        if hit is not None and hit[0] is observations:
            return hit[1]
        view = self._learning_observations(observations)
        if len(cache) >= 2:
            cache.pop(next(iter(cache)))
        cache[key] = (observations, view)
        return view

    def compute_actions(self, observations: Dict, training: bool = True) -> Dict[int, float]:
        """
        Extract features and compute target speeds for agents.

        Args:
            observations: Dict from scenario_manager.get_observations()
            training: Whether in training mode (affects exploration)

        Returns:
            Dict[agent_id, target_speed] for speed control
        """
        try:
            # Handle baseline agents (no MARL algorithm)
            if self.algorithm is None:
                # Return empty dict - baseline agents control their own speed
                return {}

            # Mixed autonomy: the learner commands only its own population.
            # Background vehicles are absent from the returned dict, so both
            # backends leave them under their own driver model.
            observations = self._learning_view(observations)

            # Extract MARL-relevant features
            states = self._extract_states(observations)

            # Compute actions (speeds) for each agent
            target_speeds = {}
            
            # Multi-agent dispatch is a declared capability, not a class
            # identity: an externally registered policy takes this branch
            # without being added to any tuple here.
            if self._is_multi_agent_policy():
                # Batched-action dispatch: algorithms that opt in via
                # ``supports_batched_actions = True`` (see
                # opencda_marl/builtin/algorithms/base_algorithm.py) get one
                # actor+critic forward per simulation step instead of N
                # per-agent calls. The shared dispatcher lives in
                # ``BaseAlgorithm.select_actions``; each subclass
                # implements only the algorithm-specific
                # ``_batched_forward`` hook. Non-opt-in algorithms keep
                # ``supports_batched_actions = False`` (zero overhead)
                # and the per-agent loop below handles them.
                batched_actions = None
                raw_by_str = None  # str-keyed view of the raw dict, built on demand
                if getattr(self.algorithm, "supports_batched_actions", False) and states:
                    first_multi_agent_data = next(iter(states.values()))
                    batched_actions = self.algorithm.select_actions(
                        first_multi_agent_data['all_states'],
                        list(states.keys()),
                        training=training,
                    )

                for agent_id_str, multi_agent_data in states.items():
                    # Keep agent_id as-is (can be string in SUMO or int in CARLA)
                    agent_id = agent_id_str

                    # Extract all agent states for multi-agent algorithms
                    all_agent_states = multi_agent_data['all_states']
                    if batched_actions is not None:
                        action = batched_actions.get(str(agent_id))
                    else:
                        action = self.algorithm.select_action(all_agent_states, agent_id, training=training)

                    # Compute speed via strategy (handles None for warmup)
                    if self.strategy is None:
                        speed = float(action) if isinstance(action, (int, float)) else 8.0
                    else:
                        speed = self.strategy.action_to_speed(action, agent_id)

                    # Store action for later update (strategy.action_to_speed returns
                    # float or None; store the float value here, not inside the strategy,
                    # because warmup-skip still needs the vanilla speed stored below)
                    if speed is not None:
                        self.last_actions[agent_id] = speed

                    # If speed is None (warmup phase), skip this agent to use vanilla agent
                    if speed is None:
                        algo_name = type(self.algorithm).__name__.replace('Algorithm', '')
                        logger.debug(f"{algo_name}: Agent {agent_id} in warmup phase, using vanilla agent")
                        # Track the vanilla agent's current speed as the stored
                        # action, so warmup transitions reach the buffer and the
                        # warmup (buffer-size based for TD3) can actually end.
                        # ``observations`` is the RAW dict — its keys are CARLA
                        # actor ids (int) on CARLA, strings on SUMO — while
                        # agent_id is the extractor's str key, so look it up in
                        # a str-keyed view (built once per step, warmup only).
                        if raw_by_str is None:
                            raw_by_str = {str(k): v for k, v in observations.items()}
                        raw_obs = raw_by_str.get(agent_id)
                        if raw_obs is not None:
                            self.last_actions[agent_id] = raw_obs.get('speed', 45.0)
                        continue  # Don't add to target_speeds, let vanilla agent handle it

                    # Clamp speed to algorithm action bounds
                    max_action = self.algorithm.max_action
                    clamped_speed = max(0.0, min(max_action, speed))
                    target_speeds[agent_id] = clamped_speed

                    # Log the target speed assignment
                    algo_name = type(self.algorithm).__name__.replace('Algorithm', '')
                    logger.debug(f"{algo_name}: Agent {agent_id} target speed set to {clamped_speed:.2f} km/h")
            else:
                # Single-agent policies (DQN, tabular Q-learning) act on one
                # agent's own state. What select_action returns is declared
                # by the policy, never inferred from its attributes: with
                # ``returns_action_index = True`` it is an index into the
                # policy's ``speed_actions`` (tabular Q), and the index is
                # what its store_transition takes back; otherwise it is the
                # target speed itself (DQN, which owns a speed table too).
                returns_index = bool(getattr(self.algorithm, 'returns_action_index', False))
                max_speed = self._single_agent_max_speed()
                for agent_id, state in states.items():
                    action = self.algorithm.select_action(state, training=training)

                    if returns_index:
                        speed_table = self.algorithm.speed_actions
                        action_idx = max(0, min(int(action), len(speed_table) - 1))
                        speed = float(speed_table[action_idx])
                        self.last_actions[agent_id] = action_idx
                    elif isinstance(action, (int, float, np.integer, np.floating)):
                        speed = float(action)
                        self.last_actions[agent_id] = speed
                    else:
                        speed = 8.0
                        self.last_actions[agent_id] = speed

                    target_speeds[agent_id] = max(0.0, min(max_speed, speed))

            self._note_success("compute_actions")
            return target_speeds

        except Exception as e:
            # One failure is tolerable (agents fall back to the rule-based
            # driver for a step); a *persistent* failure means the run trains
            # nothing while looking alive, so escalate after a short streak.
            self._note_failure("compute_actions", e)
            # Return empty dict to use default agent speeds
            return {}

    def update(
        self,
        rewards: Dict,
        observations: Dict,
        next_observations: Dict,
        objective_outputs: Dict | None = None,
    ):
        """
        Update RL algorithm with experience.

        Args:
            rewards: Dict[agent_id, reward]
            observations: Current observations
            next_observations: Next step observations
        """
        try:
            if self.algorithm is None:
                return
            # Only the learner's own population produces transitions: a
            # background vehicle's experience is not the policy's to learn
            # from (it never chose those actions).
            observations = self._learning_view(observations)
            next_observations = self._learning_view(next_observations)
            scalar_rewards = self._prepare_learning_rewards(
                rewards, objective_outputs=objective_outputs
            )
            intent_rewards = self._prepare_intent_rewards(
                rewards,
                objective_outputs=objective_outputs,
                learning_rewards=scalar_rewards,
            )
            self._notify_intent_step(intent_rewards)
            intent_buffer = self._get_intent_buffer()
            intent_batch_size = self._get_intent_batch_size()
            intent_update_threshold = max(2, intent_batch_size // 2)
            if (
                self.intent_algorithm is not None
                and intent_buffer is not None
                and len(intent_buffer) >= intent_update_threshold
            ):
                self._intent_update_tick += 1
                update_interval = max(1, int(self.intent_update_interval_steps))
                if self._intent_update_tick % update_interval == 0:
                    if hasattr(self.intent_algorithm, "update_route_intent"):
                        self.intent_algorithm.update_route_intent()
                    else:
                        self.intent_algorithm.update()
                else:
                    self._intent_episode_diagnostics["intent_update_skipped"] += 1

            if self.strategy is not None:
                self.strategy.update(
                    mgr=self, rewards=scalar_rewards,
                    observations=observations, next_observations=next_observations,
                    objective_outputs=objective_outputs,
                )
            self._note_success("update")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._note_failure("update", e)

    def _coerce_scalar_rewards(self, rewards: Dict) -> Dict:
        if not rewards:
            return {}

        sample = next(iter(rewards.values()))
        if isinstance(sample, RewardOutput):
            return {
                agent_id: float(reward_output.scalar)
                for agent_id, reward_output in rewards.items()
            }
        return rewards

    def _prepare_learning_rewards(
        self,
        rewards: Dict,
        objective_outputs: Dict | None = None,
    ) -> Dict:
        if (
            self.reward_mode == 'objective_vector'
            and objective_outputs
            and getattr(self.algorithm, 'supports_vector_rewards', False)
            and getattr(self.algorithm, 'critic_mode', 'scalar') == 'vector'
        ):
            return {
                agent_id: np.asarray(objective_output.vector, dtype=np.float32)
                for agent_id, objective_output in objective_outputs.items()
            }
        if objective_outputs and self.preference_enabled:
            return self.scalarize_objective_outputs(objective_outputs)
        return self._coerce_scalar_rewards(rewards)

    def _prepare_intent_rewards(
        self,
        rewards: Dict,
        objective_outputs: Dict | None = None,
        learning_rewards: Dict | None = None,
    ) -> Dict:
        if self.intent_algorithm is None or self.intent_reward_accumulator is None:
            return {}
        reward_dim = int(getattr(self.intent_reward_accumulator, "reward_dim", 0))
        if reward_dim > 0:
            if objective_outputs is None:
                raise ValueError(
                    "Route-intent vector rewards require objective_outputs; "
                    "check the objective configuration before launching runs."
                )
            if not objective_outputs:
                return {}
            intent_rewards = {
                agent_id: np.asarray(objective_output.vector, dtype=np.float32)
                for agent_id, objective_output in objective_outputs.items()
            }
            scale = float(getattr(self, "intent_reward_scale", 1.0) or 1.0)
            if scale != 1.0:
                intent_rewards = {
                    agent_id: reward * scale
                    for agent_id, reward in intent_rewards.items()
                }
            return intent_rewards
        if learning_rewards is not None:
            intent_rewards = learning_rewards
        else:
            intent_rewards = self._coerce_scalar_rewards(rewards)
        scale = float(getattr(self, "intent_reward_scale", 1.0) or 1.0)
        if scale == 1.0:
            return intent_rewards
        return {
            agent_id: float(reward) * scale
            for agent_id, reward in intent_rewards.items()
        }

    def _ensure_agent_preferences(self, observations: Mapping | None) -> Dict:
        if not self.preference_enabled:
            return {}
        return self.preference_registry.ensure_many(observations)

    def ensure_agent_preferences(self, observations: Mapping | None) -> Dict:
        return self._ensure_agent_preferences(observations)

    def _extract_states(self, observations: Dict) -> Dict:
        """Feature-extract an observation dict, memoized on dict identity.

        The same dict object flows compute_actions(obs) -> update(obs,
        next_obs), and next_obs is the following step's obs, so extraction
        ran three times per step on identical inputs. Extraction is a pure
        function of the dict plus per-agent preferences (assigned once and
        stable across a step), so a two-slot identity cache — current and
        previous dict — makes it once per unique dict. Strong references to
        the keyed dicts prevent id() reuse from aliasing a new dict.
        """
        cache = self._state_cache
        key = id(observations)
        hit = cache.get(key)
        if hit is not None and hit[0] is observations:
            return hit[1]

        self._ensure_agent_preferences(observations)
        states = self.observation_extractor.extract(observations)
        if self.preference_enabled:
            preferences = self.preference_registry.get_many(list((observations or {}).keys()))
            states = self.observation_extractor.augment_with_preferences(states, preferences)

        if len(cache) >= 2:
            # Evict the oldest entry (insertion order); the previous step's
            # obs is the only other dict still legitimately in flight.
            cache.pop(next(iter(cache)))
        cache[key] = (observations, states)
        return states

    def get_agent_preferences(self, agent_ids=None) -> Dict:
        return self.preference_registry.to_dict(agent_ids)

    def scalarize_objective_outputs(self, objective_outputs: Dict) -> Dict:
        if not objective_outputs or not self.preference_enabled:
            return {}
        from opencda_marl.api.objectives import scalarize_objective_outputs
        preferences = self.preference_registry.get_many(list(objective_outputs.keys()))
        return scalarize_objective_outputs(objective_outputs, preferences)

    def _reset_intent_episode_diagnostics(self) -> None:
        self._intent_episode_diagnostics = {
            "intent_reward_steps": 0,
            "intent_nonzero_reward_steps": 0,
            "intent_terminal_finalized": 0,
            "intent_nonzero_terminal_rewards": 0,
            "intent_terminal_without_pending": 0,
            "intent_update_skipped": 0,
        }

    def get_intent_diagnostics(self) -> Dict[str, float]:
        diagnostics = dict(getattr(self, "_intent_episode_diagnostics", {}) or {})
        if self.intent_algorithm is not None:
            algo_diag = {}
            if self.intent_algorithm is self.algorithm and hasattr(self.algorithm, "route_intent_head"):
                route_head = getattr(self.algorithm, "route_intent_head", None)
                if route_head is not None and hasattr(route_head, "get_training_info"):
                    try:
                        info = route_head.get_training_info()
                        algo_diag.update(info if isinstance(info, dict) else {})
                    except Exception:
                        algo_diag = {}
            elif hasattr(self.intent_algorithm, "get_training_info"):
                try:
                    info = self.intent_algorithm.get_training_info()
                    algo_diag.update(info if isinstance(info, dict) else {})
                except Exception:
                    algo_diag = {}
            if hasattr(self.intent_algorithm, "diagnostics"):
                algo_diag.update(getattr(self.intent_algorithm, "diagnostics", {}) or {})
            intent_buffer = self._get_intent_buffer()
            if intent_buffer is not None:
                algo_diag["intent_buffer_size"] = len(intent_buffer)
            for key, value in algo_diag.items():
                metric_key = key if str(key).startswith("intent_") else f"intent_{key}"
                if isinstance(value, (int, float, np.integer, np.floating)):
                    diagnostics[metric_key] = float(value)
        return diagnostics

    def _get_intent_buffer(self):
        if self.intent_algorithm is None:
            return None
        if hasattr(self.intent_algorithm, "route_intent_buffer"):
            return getattr(self.intent_algorithm, "route_intent_buffer")
        return getattr(self.intent_algorithm, "buffer", None)

    def _get_intent_batch_size(self) -> int:
        if self.intent_algorithm is None:
            return 1
        if hasattr(self.intent_algorithm, "route_intent_batch_size"):
            return int(getattr(self.intent_algorithm, "route_intent_batch_size"))
        return int(getattr(self.intent_algorithm, "batch_size", 1))

    # --------------------------------------------------------------------- #
    # Route-intent hooks (no-op when a learned path head is not configured)
    # --------------------------------------------------------------------- #
    def get_intent_provider(self, training: bool = True):
        """Return the intent provider callable (or None if disabled).

        ``force_intent`` (pins the feasibility mask to a single intent, for
        diagnostics and what-if evaluation) is read from
        ``MARL.action_space.path.force_intent`` since ``self.config`` is the
        MARL block at this point.

        ``training`` selects between stochastic sampling (training=True) and
        greedy argmax (training=False) at intent-decision time. Evaluation
        callers pass ``training=False`` for deterministic decisions.
        """
        if self.intent_algorithm is None:
            return None
        from opencda_marl._kernel.policy.intent_bridge import make_marl_intent_provider
        return make_marl_intent_provider(
            algorithm=self.intent_algorithm,
            get_active_adapters=self._get_active_adapters_for_intent,
            training=training,
            force_intent=self.action_space_config.path.force_intent,
        )

    @functools.cached_property
    def intent_provider(self):
        """Lazy single-source intent provider for the agent manager.

        Built once per MARLManager. Callers needing a different training mode
        (e.g., eval pass) should call get_intent_provider(training=False)
        directly to get a fresh callable bound to the same algorithm.
        """
        training_mode = self.config.get('training', {}).get('training_mode', True)
        return self.get_intent_provider(training=training_mode)

    def _get_active_adapters_for_intent(self):
        agent_manager = getattr(self, "agent_manager", None)
        if agent_manager is None:
            return []
        return list(getattr(agent_manager, "_vehicle_adapters", []))

    def notify_vehicle_terminal(self, actor_id) -> None:
        """Finalize the pending intent transition for a removed vehicle."""
        if self.intent_algorithm is None:
            return
        actor_id = self._pending_key(actor_id)
        if actor_id is None:
            self._intent_episode_diagnostics["intent_terminal_without_pending"] += 1
            return
        reward = (self.intent_reward_accumulator.finalize(actor_id)
                  if self.intent_reward_accumulator is not None else 0.0)
        self._intent_episode_diagnostics["intent_terminal_finalized"] += 1
        if float(np.linalg.norm(np.asarray(reward, dtype=np.float32))) > 0.0:
            self._intent_episode_diagnostics["intent_nonzero_terminal_rewards"] += 1
        try:
            self.intent_algorithm.finalize_terminal(
                actor_id=actor_id, terminal_reward=reward,
            )
        except Exception as exc:
            logger.warning(f"Intent finalize failed for actor {actor_id}: {exc}")

    def _pending_key(self, actor_id):
        """The key under which ``actor_id`` has a pending decision, or None.

        A provider stores the pending decision under the adapter's actor id,
        which is CARLA's integer; the CARLA environment hands rewards back
        keyed by the same id as a string, while SUMO proxies use strings on
        both sides. Match either key form; the pending store's own key wins.
        """
        if self.intent_algorithm.has_pending(actor_id):
            return actor_id
        text = str(actor_id)
        if self.intent_algorithm.has_pending(text):
            return text
        if text.lstrip("-").isdigit() and self.intent_algorithm.has_pending(int(text)):
            return int(text)
        return None

    def _notify_intent_step(self, rewards) -> None:
        if self.intent_algorithm is None or self.intent_reward_accumulator is None:
            return
        for actor_id, value in (rewards or {}).items():
            key = self._pending_key(actor_id)
            if key is not None:
                self._intent_episode_diagnostics["intent_reward_steps"] += 1
                if float(np.linalg.norm(np.asarray(value, dtype=np.float32))) > 0.0:
                    self._intent_episode_diagnostics["intent_nonzero_reward_steps"] += 1
                self.intent_reward_accumulator.add(key, value)

    # --------------------------------------------------------------------- #
    # Q-Learning specific methods
    # --------------------------------------------------------------------- #
    def _calculate_state_dim(self, state_features: Dict) -> int:
        """Calculate total discrete state space size from feature bins"""
        if not state_features:
            return 1  # Fallback to single state

        state_dim = 1
        for feature, settings in state_features.items():
            bins = settings.get('bins', [])
            # Number of bins is len(bins) + 1 (for values beyond last bin)
            num_bins = len(bins) + 1 if bins else 2
            state_dim *= num_bins

        logger.info(
            f"Calculated state_dim: {state_dim} from features: {list(state_features.keys())}")
        return state_dim

    # --------------------------------------------------------------------- #
    # DQN specific methods
    # --------------------------------------------------------------------- #
    # --------------------------------------------------------------------- #
    # Algorithm interface methods (strict implementation required)
    # --------------------------------------------------------------------- #
    def reset_episode(self, episode_metrics: Dict = None):
        """Reset algorithm for new episode."""
        if self.algorithm is None:
            logger.info("Episode reset skipped for baseline agent")
            return

        # Clear last_actions to prevent stale data accumulation
        self.last_actions.clear()
        self.preference_registry.clear()
        # Preferences are re-assigned next episode, so cached
        # preference-augmented states must not outlive the registry.
        self._state_cache.clear()
        self._learning_cache.clear()
        if episode_metrics is not None:
            episode_metrics.update(self.get_intent_diagnostics())

        # Log episode metrics to TensorBoard if available
        if episode_metrics and hasattr(self.algorithm, 'log_episode_metrics'):
            # Extract traffic metrics for separate logging
            traffic_metrics = {
                'avg_speed': episode_metrics.get('avg_speed', 0.0),
                'speed_std': episode_metrics.get('speed_std', 0.0),
                'speed_variance': episode_metrics.get('speed_variance', 0.0),
                'min_speed': episode_metrics.get('min_speed', 0.0),
                'max_speed': episode_metrics.get('max_speed', 0.0),
                'speed_3d_mean': episode_metrics.get('speed_3d_mean', 0.0),
                'speed_3d_max': episode_metrics.get('speed_3d_max', 0.0),
                'vertical_speed_mean': episode_metrics.get('vertical_speed_mean', 0.0),
                'vertical_speed_max': episode_metrics.get('vertical_speed_max', 0.0),
                'speed_smoothness': episode_metrics.get('speed_smoothness', 0.0),
                'avg_step_speed': episode_metrics.get('avg_step_speed', 0.0),
                'avg_agent_speed_var': episode_metrics.get('avg_agent_speed_var', 0.0),
                # Target speed metrics (RL-commanded speeds)
                'target_speed_mean': episode_metrics.get('target_speed_mean', 0.0),
                'target_speed_max': episode_metrics.get('target_speed_max', 0.0),
                'target_speed_min': episode_metrics.get('target_speed_min', 0.0),
                # Throughput metric (vehicles per hour)
                'throughput': episode_metrics.get('throughput', 0.0),
            }

            additional_metrics = {
                'reward_vector_dim': float(episode_metrics.get('reward_vector_dim', 0)),
                'objective_vector_dim': float(episode_metrics.get('objective_vector_dim', 0)),
                'physics_anomaly_count': float(episode_metrics.get('physics_anomaly_count', 0)),
                'physics_anomaly_rate': float(episode_metrics.get('physics_anomaly_rate', 0.0)),
                'total_utility': float(episode_metrics.get('total_utility', 0.0)),
            }
            # The outcome counts and the timeout rate, as the external-policy
            # path logs them (_record_external_episode): the episode table's
            # columns must not depend on the learner's base class.
            for key in self.OUTCOME_SUMMARY_KEYS:
                value = episode_metrics.get(key)
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    additional_metrics[key] = float(value)
            component_totals = episode_metrics.get('reward_component_totals', {})
            for component, value in component_totals.items():
                additional_metrics[f'reward_component_{component}'] = float(value)
            objective_totals = episode_metrics.get('objective_component_totals', {})
            for objective_name, value in objective_totals.items():
                additional_metrics[f'objective_component_{objective_name}'] = float(value)
            for key, value in self.get_intent_diagnostics().items():
                additional_metrics[key] = float(value)

            self.algorithm.log_episode_metrics(
                episode_reward=episode_metrics.get('total_reward', 0.0),
                episode_length=episode_metrics.get('episode_length', episode_metrics.get('step_length', 0)),
                success_rate=episode_metrics.get('success_rate', 0.0),
                collision_rate=episode_metrics.get('collision_rate', 0.0),
                near_miss_count=episode_metrics.get('near_miss_count', 0),
                ttc_violation_rate=episode_metrics.get('ttc_violation_rate', 0.0),
                additional_metrics=additional_metrics,
                traffic_metrics=traffic_metrics
            )

        if episode_metrics and self._external_store is not None:
            self._record_external_episode(episode_metrics)

        self.algorithm.reset_episode()
        self._reset_intent_episode_diagnostics()
        logger.info(f"Episode reset for {self.algorithm_name}")

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics from algorithm."""
        if self.algorithm is None:
            return {"algorithm_type": "none", "epsilon": "N/A", "training_mode": False}
        metrics = dict(self.algorithm.get_training_info())
        metrics.update(self.get_intent_diagnostics())
        return metrics

    # ------------------------------------------------------------------ #
    # Metrics for external policies
    # ------------------------------------------------------------------ #
    @property
    def run_store(self):
        """The run's open metrics store, or None: the learner's own
        (TrainingLogger), else the one this manager keeps for an external
        policy. Other writers of the same run (the evaluation manager, the
        traffic view) attach to it so one run is one monitor entry."""
        store = getattr(getattr(self, 'algorithm', None), 'metrics_store', None)
        if store is None:
            store = getattr(self, '_external_store', None)
        return store

    def _init_external_metrics_store(self) -> None:
        """Open a metrics store on behalf of a policy that has none.

        Built-in learners inherit the store from TrainingLogger. An external
        ``Policy`` that does not subclass BaseAlgorithm would otherwise be
        invisible to the monitor; the manager sees every ``update()`` result
        and every episode summary, so it keeps the store for such policies.
        """
        if self.algorithm is None or getattr(self.algorithm, 'metrics_store', None) is not None:
            return
        if hasattr(self.algorithm, '_init_metrics_store'):
            return                                   # a TrainingLogger with the store disabled
        merged = self._merge_shared_config({})
        ms_config = merged.get('metrics_store') or {}
        if isinstance(ms_config, bool):
            ms_config = {'enabled': ms_config}
        if not ms_config.get('enabled', True):
            return
        try:
            from datetime import datetime

            from opencda_marl.telemetry import MetricsStore
            run_name = ms_config.get('run_name') or datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(ms_config.get('log_dir', 'metrics'), str(self.algorithm_name), str(run_name))
            meta = {'algorithm': self.algorithm_name, 'run_name': str(run_name),
                    'mode': 'train' if merged.get('training_mode', True) else 'eval',
                    'scenario': ms_config.get('scenario'), 'external_policy': True}
            self._external_store = MetricsStore(run_dir, meta={k: v for k, v in meta.items() if v is not None})
            self._external_store_stride = max(1, int(ms_config.get('log_frequency', 10)))
            self._continue_external_counters()
            logger.info(f"Metrics store (manager-owned, external policy): {run_dir}")
        except Exception as exc:  # noqa: BLE001 - telemetry never blocks training
            logger.warning(f"metrics store for external policy disabled: {exc}")
            self._external_store = None

    def _continue_external_counters(self) -> None:
        """Resume the step and episode counters of a store that has rows.

        A resumed run reopens the store of the same run name. Counting from
        0 again would overwrite its episode rows and interleave its update
        scalars with the earlier ones, and a reader that polls for steps
        past the last one it saw would show nothing new until the counters
        passed the old maximum. Built-in learners get this from their
        checkpoint (``training_step`` / ``episode_count``); an external
        policy's counters live here, so they continue from the store.
        """
        store = self._external_store
        try:
            last_episode = store.episodes(limit=1)
            update_steps = [int(row['last_step']) for row in store.tags()
                            if not str(row['tag']).startswith('Episode/')]
        except Exception as exc:  # noqa: BLE001 - telemetry never blocks training
            logger.warning(f"metrics store counters not resumed: {exc}")
            return
        if last_episode:
            self._episodes_logged = int(last_episode[-1]['episode']) + 1
        if update_steps:
            self._update_calls = max(update_steps)

    def record_update_metrics(self, metrics) -> None:
        """Called by the learning strategy with ``algorithm.update()``'s result."""
        self._update_calls += 1
        store = self._external_store
        if store is None or not isinstance(metrics, dict):
            return
        if self._update_calls % self._external_store_stride:
            return
        for key, value in metrics.items():
            try:
                store.log_scalar(str(key), float(value), self._update_calls)
            except (TypeError, ValueError):
                continue

    def _record_external_episode(self, episode_metrics: Dict) -> None:
        store = self._external_store
        if store is None or not episode_metrics:
            return
        summary = {}
        for key in ('total_reward', 'avg_reward', 'success_rate', 'collision_rate', 'timeout_rate',
                    'successes', 'collisions', 'timeouts', 'step_length', 'episode_length',
                    'throughput', 'near_miss_count', 'ttc_violation_rate'):
            value = episode_metrics.get(key)
            if isinstance(value, (int, float)):
                summary[key] = float(value)
        summary['reward'] = summary.get('total_reward', 0.0)
        summary['length'] = summary.get('episode_length', summary.get('step_length', 0.0))
        summary['training_step'] = self._update_calls
        store.log_episode(self._episodes_logged, summary)
        for key in ('reward', 'success_rate', 'collision_rate', 'length'):
            if key in summary:
                store.log_scalar(f'Episode/{key}', summary[key], self._episodes_logged)
        store.flush()
        self._episodes_logged += 1

    def close(self):
        """Close algorithm resources (e.g., TensorBoard writer)."""
        if self.algorithm is not None and hasattr(self.algorithm, 'close'):
            self.algorithm.close()
        if self._external_store is not None:
            try:
                self._external_store.close()
            finally:
                self._external_store = None

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint.

        Saves the base controller weights to ``filepath``. Legacy separate
        intent policies are saved to a sibling ``<filepath>.intent`` file;
        hybrid MOMARL route heads are owned by ``self.algorithm`` and are
        already included in the base checkpoint.
        """
        if self.algorithm is None:
            logger.warning("Cannot save checkpoint for baseline agent")
            return
        self.algorithm.save(filepath)
        if self.intent_algorithm is not None and self.intent_algorithm is not self.algorithm:
            intent_path = filepath + ".intent"
            try:
                self.intent_algorithm.save(intent_path)
            except Exception as e:
                logger.warning(f"Failed to save intent checkpoint to {intent_path}: {e}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint.

        Loads the base controller from ``filepath``. Legacy separate intent
        policies are restored from a sibling ``<filepath>.intent`` file when
        present; hybrid MOMARL route heads are restored from the base
        checkpoint itself.
        """
        if self.algorithm is None:
            logger.warning("Cannot load checkpoint for baseline agent")
            return
        self.algorithm.load(filepath)
        # A loader that swallows its own failure leaves ``_pretrained`` False;
        # the run must not continue on freshly initialised weights it believes
        # are the configured checkpoint. Learners without the flag (external
        # plugins) are trusted to raise for themselves.
        if getattr(self.algorithm, "_pretrained", None) is False:
            raise RuntimeError(
                f"checkpoint {filepath} did not load into "
                f"{type(self.algorithm).__name__} (see the loader's error above)")
        if self.intent_algorithm is not None and self.intent_algorithm is not self.algorithm:
            from pathlib import Path as _Path
            intent_path = filepath + ".intent"
            if _Path(intent_path).exists():
                try:
                    self.intent_algorithm.load(intent_path)
                    logger.info(f"Loaded intent checkpoint from {intent_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load intent checkpoint from {intent_path}: {e}. "
                        "Intent head will use freshly initialized weights."
                    )
            else:
                logger.warning(
                    f"Intent checkpoint sibling {intent_path} not found; "
                    "intent head will use freshly initialized weights."
                )

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _get_speed_actions(self) -> list:
        """Get speed actions based on algorithm type."""
        if self.algorithm_name == 'q_learning':
            return self.config.get('q_learning', {}).get(
                'speed_actions', [0, 5, 10, 15])
        elif self.algorithm_name == 'dqn':
            return self.config.get('dqn', {}).get(
                'speed_actions', [0, 5, 10, 15])
        else:
            return [0, 5, 10, 15]

    def _single_agent_max_speed(self) -> float:
        """Upper clamp for a single-agent policy's target speed.

        The policy's own ``speed_actions`` table is the authority (the one
        an index resolves against); the configured list is the fallback.
        The table may be any sequence, a numpy array included, so its
        emptiness is tested by length, never by truth value.
        """
        table = getattr(self.algorithm, 'speed_actions', None)
        if table is None or len(table) == 0:
            table = self._get_speed_actions()
        return float(max(table)) if len(table) else 15.0

    def _action_to_speed(self, action) -> float:
        """Convert algorithm action to target speed"""
        # For now, assume action is already a speed value
        return float(action)
