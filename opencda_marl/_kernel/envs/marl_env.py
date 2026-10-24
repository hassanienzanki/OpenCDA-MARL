'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 12:49:47
FilePath     : /OpenCDA-MARL/opencda_marl/envs/marl_env.py
Description  : MARL environment with modular evaluation system
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import os
from typing import Dict, List, Any
from loguru import logger

from opencda_marl._kernel.lifecycle.scenario_manager import MARLScenarioManager
from opencda_marl.api.events import StepEvent
from opencda_marl.api.rewards import resolve_reward_configuration
from opencda_marl.api.safety import TtcMetricState
from opencda_marl.api.contracts import (
    normalize_observations,
    reward_outputs_to_components,
    reward_outputs_to_scalars,
    reward_outputs_to_vectors,
)
from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl._kernel.policy.competition import preference_group_from_values
from opencda_marl.evaluation.training_metrics import TrainingMetrics as Metrics
from opencda_marl._kernel.policy.checkpoint import CheckpointManager
from opencda_marl.api.objectives import (
    build_objective_outputs,
    objective_outputs_to_components,
    objective_outputs_to_vectors,
)


class MARLEnv:

    def __init__(self, scenario_manager: MARLScenarioManager, config: Dict = {}, marl_manager=None):
        """
        Initialize MARL environment with reward calculation capabilities.
        """
        self.sm = scenario_manager
        self.config = config
        self.world = scenario_manager.world

        # Extract MARL specific config
        algorithm = self.config.get('algorithm')

        # Check agent type first
        from opencda_marl.api.components import BASELINE_AGENT_TYPES
        agent_type = self.sm.agent_manager.agent_type

        if marl_manager is not None:
            # Builder path: manager already constructed and wired externally
            self.marl_manager = marl_manager
        else:
            # Standalone / legacy path: build the manager here
            # Use 'none' algorithm for baseline agents to skip MARL processing
            if agent_type in BASELINE_AGENT_TYPES:
                algorithm = None
            elif algorithm is None:
                raise ValueError("MARL.algorithm is required for MARL agents")

            self.marl_manager = MARLManager(self.config, algorithm)

        # RL-specific tracking
        self.events = None
        self.episode_events = []  # Track events for reward calculation
        self.previous_observations = {}  # Store for algorithm updates
        self.terminal_agents = set()  # Track agents that received terminal rewards
        self.current_step_rewards = {}  # Store current step rewards for evaluation
        self.current_step_reward_components = {}
        self.current_step_reward_vectors = {}
        self.current_step_objective_components = {}
        self.current_step_objective_vectors = {}
        self.current_step_agent_preferences = {}

        # Reward parameters and runtime tracking
        self.reward_config = resolve_reward_configuration(self.config)
        self.reward_params = self.reward_config.reward_params
        self.reward_spec = self.reward_config.reward_spec
        self.objective_config = self.marl_manager.objective_config
        self.objective_spec = self.marl_manager.objective_spec
        from opencda_marl.api.objectives import competition_metrics_enabled
        self.competition_enabled = competition_metrics_enabled(self.reward_config, self.objective_config)
        self.competition_agent_metadata = {}
        self.competition_terminal_records = []
        from opencda_marl.api.components import component_key, resolve_component
        self.reward_calculator = resolve_component(
            "reward_calculator", component_key(self.config, "reward_calculator"),
            config=self.config,
        )(self.reward_params, self.reward_spec)
        self.ttc_metrics = TtcMetricState()

        # Training configuration
        self.training_config = self.config.get('training', {})
        self.save_checkpoints = self.training_config.get('save_checkpoints', True)

        # Training metrics (export history every N episodes)
        outputs_cfg = self.config.get('outputs', {})
        archive_cfg = outputs_cfg.get('metrics_archive', {})
        archive_enabled = archive_cfg.get('enabled', True)
        metrics_export_interval = archive_cfg.get('export_interval', 10) if archive_enabled else 0
        archive_root_dir = outputs_cfg.get('root_dir')
        archive_dir = archive_cfg.get('dir', 'metrics_history')
        metrics_export_dir = os.path.join(archive_root_dir, archive_dir) if archive_root_dir else archive_dir
        # Mid-cell resume: when launcher injects MARL.training.start_episode,
        # offset the pkl-export counter so we write ep{N+1}, ep{N+2}, ... and
        # don't overwrite pkls from the interrupted earlier run.
        start_episode_counter = int(self.training_config.get('start_episode', 0))
        # The same offset applies to the episode number itself: checkpoints
        # are named episode_{N:04d} from it, so a resumed run must continue
        # the numbering; restarting at 0 would overwrite (and then, via
        # cleanup_old_checkpoints, delete) the files it had just written.
        if start_episode_counter > 0:
            self.sm.states['episode'] = start_episode_counter
        self.metrics = Metrics(
            export_interval=metrics_export_interval,
            export_dir=metrics_export_dir,
            start_episode_counter=start_episode_counter,
        )
        self.is_training_mode = self.training_config.get('training_mode', True)
        
        # Override training mode for baseline agents
        if agent_type in BASELINE_AGENT_TYPES:
            self.is_training_mode = False
            logger.info(f"Baseline agent '{agent_type}' detected - disabling training mode")

        # Initialize checkpoint manager
        # Note: We need checkpoint manager even in evaluation mode to load trained weights
        load_checkpoint = self.training_config.get('load_checkpoint')
        checkpoint_dir = self.training_config.get(
            'checkpoint_dir', f"checkpoints/{algorithm or 'baseline'}")
        should_create_checkpoint_manager = self.is_training_mode and (
            self.save_checkpoints or bool(load_checkpoint)
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, algorithm or "baseline") if should_create_checkpoint_manager else None

        if self.is_training_mode and not self.save_checkpoints:
            logger.info("Checkpoint saving disabled for this training run")

        # Load checkpoint if specified - works in BOTH training and evaluation modes
        # In evaluation mode (training_mode: false), this loads trained weights to test performance
        if load_checkpoint:
            self._load_checkpoint_from_config(load_checkpoint)
            if not self.is_training_mode:
                logger.info(f"Evaluation mode: Loaded checkpoint '{load_checkpoint}' for testing")

    def step(self):
        # Get current observations from scenario manager
        observations = self.sm.get_observations()

        # MARL computes target speeds based on observations (pass training mode)
        target_speeds = self.marl_manager.compute_actions(
            observations, training=self.is_training_mode)
        pre_step_preferences = self.marl_manager.get_agent_preferences(observations.keys())
        self._update_competition_metadata(observations, pre_step_preferences)

        # Execute scenario manager step with target speeds
        out = self.sm.step(target_speeds)

        # Get new observations for reward calculation (includes min_ttc, distance_to_destination)
        next_observations = self.sm.get_observations()

        # Calculate rewards from events AND observations (Phase 3: includes TTC and progress rewards)
        self.events = out.get('event', [])
        self.metrics.record_physics_anomaly_events(
            [event for event in self.events if event.event_type == "physics_anomaly"]
        )
        reward_outputs = self._calculate_rewards(self.events, observations=next_observations)
        reward_outputs = {str(k): v for k, v in reward_outputs.items()} if reward_outputs else {}
        rewards = reward_outputs_to_scalars(reward_outputs)
        objective_outputs = self._build_objective_outputs(reward_outputs)
        self.marl_manager.ensure_agent_preferences(next_observations)
        agent_preferences = self.marl_manager.get_agent_preferences(next_observations.keys())
        self._update_competition_metadata(next_observations, agent_preferences)
        self._record_competition_events(self.events)
        learning_rewards = self.marl_manager.scalarize_objective_outputs(objective_outputs)

        # Update MARL algorithm with CURRENT step's transition
        # Transition: (observations, action, reward, next_observations)
        # - observations: state before action (S_t)
        # - action: stored in last_actions during compute_actions
        # - reward: calculated for taking action from observations
        # - next_observations: state after action (S_t+1)
        if self.is_training_mode:
            self.marl_manager.update(
                reward_outputs,
                observations,
                next_observations,
                objective_outputs=objective_outputs,
            )  # Use current observations!

        # Store current observations as previous for next step's progress calculation
        self.previous_observations = normalize_observations(next_observations)

        # Store current step rewards for evaluation
        self.current_step_rewards = rewards.copy()
        self.current_step_reward_components = reward_outputs_to_components(reward_outputs)
        self.current_step_reward_vectors = reward_outputs_to_vectors(reward_outputs)
        self.current_step_objective_components = objective_outputs_to_components(objective_outputs)
        self.current_step_objective_vectors = objective_outputs_to_vectors(objective_outputs)
        self.current_step_agent_preferences = agent_preferences

        self.episode_events.extend(self.events)

        # Count successes this step for accurate episode_length tracking
        step_successes = sum(1 for e in self.events if e.event_type == "success")

        # Update training metrics with observations and RL-commanded target speeds
        # Pass target_speeds directly so we track what RL commanded (not adapter cached values)
        self.metrics.update_step(
            rewards,
            next_observations,
            target_speeds,
            step_successes,
            reward_outputs=reward_outputs,
            objective_outputs=objective_outputs,
            scalarized_utilities=learning_rewards,
            agent_preferences=agent_preferences,
        )

        return {
            'rewards': rewards,
            'reward_components': self.current_step_reward_components.copy(),
            'reward_vectors': self.current_step_reward_vectors.copy(),
            'reward_spec': self.reward_spec.to_dict(),
            'objective_components': self.current_step_objective_components.copy(),
            'objective_vectors': self.current_step_objective_vectors.copy(),
            'objective_spec': self.objective_spec.to_dict() if self.objective_spec else None,
            'agent_preferences': dict(self.current_step_agent_preferences),
            'events': list(self.events),
            'observations': next_observations,
        }

    # --------------------------------------------------------------------- #
    # Public Methods
    # --------------------------------------------------------------------- #
    def get_current_events(self):
        return self.events.copy()

    def get_episode_metrics(self):
        return self.metrics.get_current_metrics()

    def get_reward_params(self):
        return self.reward_params

    def get_current_step_rewards(self):
        """Get rewards from the current step for evaluation."""
        return self.current_step_rewards.copy()

    def get_current_step_reward_components(self):
        """Get reward components from the current step for evaluation."""
        return self.current_step_reward_components.copy()

    def get_current_step_reward_vectors(self):
        """Get reward vectors from the current step for evaluation."""
        return self.current_step_reward_vectors.copy()

    def get_current_step_objective_components(self):
        """Get grouped objective components from the current step for evaluation."""
        return self.current_step_objective_components.copy()

    def get_current_step_objective_vectors(self):
        """Get grouped objective vectors from the current step for evaluation."""
        return self.current_step_objective_vectors.copy()

    def get_current_agent_preferences(self):
        """Get per-agent preferences for the current step."""
        return dict(self.current_step_agent_preferences)

    def get_reward_spec(self):
        """Get reward specification metadata."""
        return self.reward_spec.to_dict()

    def get_objective_spec(self):
        """Get grouped objective specification metadata."""
        return self.objective_spec.to_dict() if self.objective_spec else None

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information from MARL algorithm."""
        info = self.marl_manager.get_training_metrics()
        current_episode = self.sm.states.get('episode', 0)
        max_episodes = self.sm.states.get('max_episodes', 1)
        info.update({
            'episode_count': current_episode,
            'max_episodes': max_episodes,
            'current_metrics': self.metrics.get_current_metrics(),
            'training_mode': self.is_training_mode
        })
        return info

    # --------------------------------------------------------------------- #
    # Training control methods (centralized here)
    # --------------------------------------------------------------------- #
    def get_observations(self):
        """
        Get observations from scenario manager with RL enhancements.
        """
        try:
            base_observations = self.sm.get_observations()

            # Enhance observations with RL-specific data
            observations = {}

            for agent_id, obs in base_observations.items():
                enhanced_obs = obs.copy()

                # Add reward information
                enhanced_obs['current_reward'] = self.metrics.get_agent_reward(
                    agent_id)

                observations[agent_id] = enhanced_obs

            return observations

        except Exception as e:
            logger.error(f"Error getting enhanced observations: {e}")
            return self.sm.get_observations()  # Fallback to base observations

    # --------------------------------------------------------------------- #
    # Checkpoint methods
    # --------------------------------------------------------------------- #
    def _handle_training_episode_end(self, episode_metrics: Dict[str, Any], current_episode: int):
        """Handle end-of-episode training logic."""
        if not self.save_checkpoints or not self.checkpoint_manager:
            return

        save_freq = self.training_config.get('save_freq', 10)

        # Always save checkpoint at episode end for training progress
        self._save_training_checkpoint(episode_metrics, current_episode)

        # Save best checkpoint if this episode has best reward
        episode_reward = episode_metrics.get('total_reward', 0.0)
        if hasattr(self.checkpoint_manager, 'best_reward'):
            if episode_reward > self.checkpoint_manager.best_reward:
                self.checkpoint_manager.best_reward = episode_reward
                self.checkpoint_manager.best_episode = current_episode
                self._save_best_checkpoint(episode_metrics, current_episode)

        # Cleanup old checkpoints periodically
        if current_episode % save_freq == 0:
            keep_checkpoints = self.training_config.get('keep_checkpoints', 5)
            self.checkpoint_manager.cleanup_old_checkpoints(keep_checkpoints)

    def _save_training_checkpoint(self, episode_metrics: Dict[str, Any], current_episode: int):
        """Save training checkpoint."""
        try:
            self.checkpoint_manager.save_checkpoint(
                self.marl_manager.algorithm,
                current_episode,
                episode_metrics,
                intent_algorithm=getattr(self.marl_manager, "intent_algorithm", None),
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _save_best_checkpoint(self, episode_metrics: Dict[str, Any], current_episode: int):
        """Save best checkpoint."""
        try:
            self.checkpoint_manager.save_checkpoint(
                self.marl_manager.algorithm,
                current_episode,
                episode_metrics,
                checkpoint_type="best",
                intent_algorithm=getattr(self.marl_manager, "intent_algorithm", None),
            )
            logger.info(
                f"New best checkpoint saved at episode {current_episode} with reward {episode_metrics.get('total_reward', 0.0)}")
        except Exception as e:
            logger.error(f"Failed to save best checkpoint: {e}")

    def _load_checkpoint_from_config(self, checkpoint_path: str):
        """Load the checkpoint the config asked for, or fail loudly.

        A configured checkpoint that cannot be loaded must fail: continuing
        would score freshly initialised weights or restart a resumed run.
        Asking for a checkpoint and not getting it is a hard error; per-piece
        degradation for warm starts across model variants still happens
        inside ``algorithm.load``.
        """
        if not checkpoint_path:
            return
        logger.info(f"Loading checkpoint from config: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            # A kit-relative name (``<set>/<run>``, e.g. the bundled SAC
            # baseline) resolves inside the wheel's bundled baselines before
            # it is treated as missing.
            from opencda_marl._bundled import bundled_checkpoint
            packaged = bundled_checkpoint(checkpoint_path)
            if packaged is None:
                raise FileNotFoundError(
                    f"Configured checkpoint not found: {checkpoint_path} "
                    "(MARL.training.load_checkpoint)")
            logger.info(f"Bundled checkpoint {checkpoint_path} -> {packaged}")
            checkpoint_path = str(packaged)
        try:
            self.marl_manager.load_checkpoint(checkpoint_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load configured checkpoint {checkpoint_path}: {e}") from e

    # --------------------------------------------------------------------- #
    # Private Methods
    # --------------------------------------------------------------------- #

    def _calculate_rewards(self, events: List[StepEvent], observations: Dict = None):
        """Calculate rewards via the shared reward calculator."""
        agent_ids = [agent.actor_id for agent in self.sm.agents]
        return self.reward_calculator.calculate_carla(
            agent_ids,
            observations,
            events,
            previous_observations=self.previous_observations,
            terminal_agents=self.terminal_agents,
            ttc_metrics=self.ttc_metrics,
        )
    # --------------------------------------------------------------------- #
    # Clean Up
    # --------------------------------------------------------------------- #
    def reset_episode(self):
        """
        Reset environment for new episode and handle training logic.
        """
        # Create episode-specific state snapshot BEFORE any resets
        # This captures the actual current episode stats
        current_episode = self.sm.states.get('episode', 0)

        # Calculate TTC violation rate (% of TTC checks that were violations)
        ttc_violation_rate = 0.0
        if self.ttc_metrics.check_count > 0:
            ttc_violation_rate = (self.ttc_metrics.violation_count / self.ttc_metrics.check_count) * 100.0

        # Vehicles the replay scheduled that never reached the road — a spawn
        # point held by a queue, or a spawn abandoned after the retry budget.
        # SUMO counts its own insertion queue as unfinished routes of the
        # episode that scheduled them (see `_pending_insertions`); CARLA does
        # the same here, so every scheduled vehicle is accounted for,
        # including those that congestion kept off the road.
        never_entered = 0
        _am = getattr(self.sm, 'agent_manager', None)
        if _am is not None and hasattr(_am, 'get_unentered_count'):
            never_entered = int(_am.get_unentered_count())
        if never_entered:
            logger.warning(
                f"{never_entered} vehicles never entered in episode {current_episode}: "
                "counted as unfinished routes")

        episode_states = {
            'max_steps': self.sm.states['max_steps'],
            'max_episodes': self.sm.states['max_episodes'],
            'step': self.sm.states['step'],
            'episode': current_episode,  # Current episode, not incremented yet
            'collision': self.sm.states['collision'],  # Current episode stats
            'planner_failure': self.sm.states.get('planner_failure', 0),
            'success': self.sm.states['success'],  # Current episode stats
            'physics_anomaly_count': self.sm.states.get('physics_anomaly', 0),
            'active_agents': self.sm.states['active_agents'] + never_entered,
            'fixed_dt': self.sm.states.get('fixed_dt', 0.05),
            # Near-miss tracking for learning analysis
            'near_miss_count': self.ttc_metrics.near_miss_count,
            # TTC violation rate (% of checks with TTC < safe threshold)
            'ttc_violation_rate': ttc_violation_rate,
            'competition_records': self._build_timeout_competition_records(
                list(self.previous_observations.keys()),
                self.sm.states.get('step', 0),
            ),
        }

        # Finish current episode metrics with clean episode-specific snapshot
        episode_metrics = self.metrics.finish_episode(episode_states)
        if self.marl_manager is not None:
            episode_metrics.update(self.marl_manager.get_intent_diagnostics())

        # Handle training-specific logic with correct episode number
        if self.is_training_mode:
            self._handle_training_episode_end(episode_metrics, current_episode)

        # Reset MARL algorithm and log episode metrics to TensorBoard
        self.marl_manager.reset_episode(episode_metrics=episode_metrics)

        # Reset environment state
        self.events = None
        self.episode_events.clear()
        self.previous_observations.clear()
        self.terminal_agents = set()
        self.current_step_rewards.clear()
        self.current_step_reward_components.clear()
        self.current_step_reward_vectors.clear()
        self.current_step_objective_components.clear()
        self.current_step_objective_vectors.clear()
        self.current_step_agent_preferences.clear()
        self.competition_agent_metadata.clear()
        self.competition_terminal_records.clear()

        self.ttc_metrics.reset()

        logger.info(f"Episode {current_episode} reset completed")
        return episode_metrics

    def _build_objective_outputs(self, reward_outputs):
        if self.objective_spec is None:
            return {}
        return build_objective_outputs(reward_outputs, self.objective_spec)

    def _update_competition_metadata(self, observations: Dict, agent_preferences: Dict[str, list[float]]):
        if not self.competition_enabled:
            return
        for agent_id, observation in dict(observations or {}).items():
            preference = agent_preferences.get(agent_id)
            if preference is None:
                preference = agent_preferences.get(str(agent_id))
            self.competition_agent_metadata[str(agent_id)] = {
                "entry_direction": observation.get("entry_direction", "unknown"),
                "route_type": observation.get("route_type", "unknown"),
                "lane_id": observation.get("lane_id"),
                "junction_id": observation.get("junction_id"),
                "preference": list(preference or []),
                "preference_group": preference_group_from_values(preference),
                "route_intent": observation.get("route_intent"),
            }

    def _record_competition_events(self, events: List[StepEvent]):
        if not self.competition_enabled:
            return
        for event in events:
            if event.event_type not in {"success", "collision"}:
                continue
            self.competition_terminal_records.append(
                self._build_competition_record(
                    agent_id=event.vehicle_id,
                    outcome=event.event_type,
                    completion_step=event.step,
                )
            )

    def _build_timeout_competition_records(self, active_agent_ids: List[Any], completion_step: int):
        if not self.competition_enabled:
            return []
        records = list(self.competition_terminal_records)
        recorded_ids = {record.get("agent_id") for record in records}
        for agent_id in active_agent_ids:
            agent_key = str(agent_id)
            if agent_key in recorded_ids:
                continue
            records.append(
                self._build_competition_record(
                    agent_id=agent_key,
                    outcome="timeout",
                    completion_step=completion_step,
                )
            )
        return records

    def _build_competition_record(self, agent_id: Any, outcome: str, completion_step: int):
        metadata = dict(self.competition_agent_metadata.get(str(agent_id), {}) or {})
        preference = list(metadata.get("preference") or [])
        return {
            "agent_id": str(agent_id),
            "outcome": outcome,
            "completion_step": int(completion_step),
            "entry_direction": metadata.get("entry_direction", "unknown"),
            "route_type": metadata.get("route_type", "unknown"),
            "lane_id": metadata.get("lane_id"),
            "junction_id": metadata.get("junction_id"),
            "preference": preference,
            "preference_group": metadata.get("preference_group", preference_group_from_values(preference)),
            "route_intent": metadata.get("route_intent"),
        }

    # ------------------------------------------------------------------ #
    # Schematic traffic view (telemetry/traffic_view.py)
    # ------------------------------------------------------------------ #

    def collect_traffic_geometry(self):
        """Lane polylines from the map topology — one-time cost per world."""
        lanes = []
        try:
            carla_map = self.sm.world.get_map()
            for start_wp, _end_wp in carla_map.get_topology():
                points = [start_wp.transform.location]
                try:
                    points += [w.transform.location
                               for w in start_wp.next_until_lane_end(2.0)]
                except RuntimeError:
                    pass
                if len(points) >= 2:
                    lanes.append([[float(p.x), float(p.y)] for p in points])
        except Exception as exc:  # noqa: BLE001 - the view must never kill a run
            from loguru import logger
            logger.debug(f"traffic view geometry unavailable: {exc}")
            return None
        # CARLA (UE) is left-handed: y grows to the "south" on the map.
        return {"lanes": lanes, "junction": None, "y_up": False}

    def collect_traffic_view(self):
        """Vehicle poses from the step's retained world snapshot: no RPC.

        ``role`` is ``agent`` only for vehicles of the learning population;
        background vehicles of a population mix are ``traffic`` like any
        other actor, although they have adapters too.
        """
        from opencda_marl._kernel.traffic.population import is_learning_population

        vehicles = []
        agent_manager = getattr(self.sm, 'agent_manager', None)
        snapshot = getattr(agent_manager, '_current_snapshot', None)
        agent_ids = set()
        if agent_manager is not None:
            agent_type = getattr(agent_manager, 'agent_type', None)
            agent_ids = {int(a.actor_id)
                         for a in getattr(agent_manager, '_vehicle_adapters', [])
                         if is_learning_population(
                             getattr(a, 'spawn_metadata', None), agent_type)}
        if snapshot is not None:
            for v in snapshot.vehicles:
                speed = float((v.vx ** 2 + v.vy ** 2) ** 0.5)
                vehicles.append({
                    "id": str(v.actor_id),
                    "x": float(v.x), "y": float(v.y),
                    "yaw": float(v.yaw_rad),
                    "speed": speed,
                    "role": "agent" if int(v.actor_id) in agent_ids else "traffic",
                })
        episode = int(self.sm.states.get('episode', 0)) if hasattr(self.sm, 'states') else 0
        step = int(self.sm.states.get('step', 0)) if hasattr(self.sm, 'states') else 0
        return vehicles, episode, step
