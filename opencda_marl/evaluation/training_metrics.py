from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import copy
import numpy as np
import pickle
import os
from loguru import logger

from opencda_marl.api.contracts import RewardOutput, RewardSpec
from opencda_marl._kernel.policy.competition import build_competition_summary
from opencda_marl.api.objectives import ObjectiveOutput


class TrainingMetrics:
    """Track and compute training metrics including traffic performance.

    Uses rolling windows for real-time stats and periodic file export for full history.
    """

    HARD_SPEED_PLAUSIBILITY_KMH = 180.0
    MAX_SPIKE_RECORDS = 50

    def __init__(self, export_interval: int = 100, export_dir: str = "metrics_history",
                 start_episode_counter: int = 0):
        """
        Initialize metrics tracker.

        Args:
            export_interval: Export history to file every N episodes (0 = disabled)
            export_dir: Directory to save history files
        """
        # Rolling windows for real-time stats (bounded memory)
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_states: deque = deque(maxlen=100)

        # Traffic performance history (bounded)
        self.episode_avg_speeds: deque = deque(maxlen=100)
        self.episode_speed_variances: deque = deque(maxlen=100)

        # Full history for file export (cleared after export)
        self._full_history = {
            'rewards': [],
            'reward_component_totals': [],
            'reward_specs': [],
            'utilities': [],
            'objective_component_totals': [],
            'objective_specs': [],
            'agent_utilities': [],
            'agent_preferences': [],
            'actual_speed_means': [],
            'actual_speed_variances': [],
            'speed_3d_means': [],
            'speed_3d_variances': [],
            'vertical_speed_means': [],
            'vertical_speed_variances': [],
            'agent_speed_means': [],
            'agent_speed_variances': [],
            'target_speed_means': [],
            'speed_spike_records': [],
            'max_speed_spike_records': [],
            'physics_anomaly_counts': [],
            'physics_anomaly_records': [],
            'episode_lengths': [],
            'success_counts': [],
            'collision_counts': [],
            'physics_anomaly_event_counts': [],
            'timeout_counts': [],  # Track timeout vehicles
            'near_miss_counts': [],  # Track near-miss events for learning analysis
            'competition_summaries': [],
        }

        # Export configuration
        self.export_interval = export_interval
        self.export_dir = export_dir
        # When resuming a partially-trained cell, the launcher passes the
        # count of already-exported pkls so finish_episode -> _episode_counter
        # increments cleanly past existing ep1..epN files and writes
        # ep{N+1}, ep{N+2}, ... without overwriting earlier-window pkls.
        self._episode_counter = int(start_episode_counter)

        # Create export directory if needed
        if export_interval > 0:
            os.makedirs(export_dir, exist_ok=True)

        self.reset()

    def reset(self):
        """Reset current episode metrics."""
        self.current_reward = 0.0
        self.current_length = 0
        self.current_total_reward = 0.0
        self.agent_rewards = {}
        self.current_total_utility = 0.0
        self.agent_utilities = {}
        self.current_reward_component_totals = defaultdict(float)
        self.reward_vector_dim = 0
        self.reward_component_order: List[str] = []
        self.reward_spec_metadata = None
        self.current_objective_component_totals = defaultdict(float)
        self.objective_vector_dim = 0
        self.objective_component_order: List[str] = []
        self.objective_spec_metadata = None
        self.current_agent_preferences: Dict[Any, List[float]] = {}
        self.current_competition_records: List[Dict[str, Any]] = []
        self.collisions = 0
        self.successes = 0
        self.physics_anomaly_count = 0
        self.timeouts = 0  # Track timeout vehicles

        # Track step when LAST vehicle completes (for accurate episode length)
        self.last_completion_step = 0

        # Traffic performance tracking (per episode)
        self.step_actual_speeds: List[float] = []  # Canonical actual speeds from CARLA physics
        self.step_speed_3d: List[float] = []  # 3D CARLA speed for anomaly auditing
        self.step_vertical_speeds: List[float] = []  # Signed vertical speed (km/h)
        self.step_agent_speeds: List[float] = []  # Agent/localizer reported speeds
        self.step_actual_avg_speeds: List[float] = []  # Average actual speed per step
        self.step_agent_avg_speeds: List[float] = []  # Average agent speed per step
        self.agent_actual_speeds: Dict[int, List[float]] = {}  # Actual speed history per agent
        self.agent_reported_speeds: Dict[int, List[float]] = {}  # Agent speed history per agent
        self.step_target_speeds: List[float] = []  # Target (commanded) speeds for comparison
        self.speed_spike_records: List[Dict[str, Any]] = []
        self.max_speed_spike_record: Optional[Dict[str, Any]] = None
        self.physics_anomaly_records: List[Dict[str, Any]] = []
        self.max_physics_anomaly_record: Optional[Dict[str, Any]] = None
        self._speed_spike_keys = set()
    # --------------------------------------------------------------------- #
    # Main steps for updating metrics
    # --------------------------------------------------------------------- #

    def update_step(self, rewards: Dict[int, float], observations: Optional[Dict] = None,
                    target_speeds: Optional[Dict[int, float]] = None,
                    step_successes: int = 0,
                    reward_outputs: Optional[Dict[Any, RewardOutput]] = None,
                    objective_outputs: Optional[Dict[Any, ObjectiveOutput]] = None,
                    scalarized_utilities: Optional[Dict[Any, float]] = None,
                    agent_preferences: Optional[Dict[Any, list[float]]] = None):
        """
        Update metrics for current step.

        Args:
            rewards: Dictionary of agent_id -> reward
            observations: Optional dictionary of agent observations (containing speed data)
            target_speeds: Optional dictionary of agent_id -> RL-commanded target speed (km/h)
            step_successes: Number of vehicles that succeeded this step (for tracking last completion)
        """
        if reward_outputs is not None:
            rewards = {
                agent_id: float(reward_output.scalar)
                for agent_id, reward_output in reward_outputs.items()
            }
        elif rewards:
            sample = next(iter(rewards.values()))
            if isinstance(sample, RewardOutput):
                reward_outputs = rewards
                rewards = {
                    agent_id: float(reward_output.scalar)
                    for agent_id, reward_output in reward_outputs.items()
                }

        self.current_reward = sum(rewards.values()) if rewards else 0.0
        self.current_total_reward += self.current_reward
        self.current_length += 1
        for agent_id, reward in rewards.items():
            if agent_id not in self.agent_rewards:
                self.agent_rewards[agent_id] = 0.0
            self.agent_rewards[agent_id] += reward

        if reward_outputs:
            self._update_reward_component_metrics(reward_outputs)
        if objective_outputs:
            self._update_objective_metrics(objective_outputs)

        if scalarized_utilities:
            step_utility = sum(scalarized_utilities.values())
            self.current_total_utility += step_utility
            for agent_id, utility in scalarized_utilities.items():
                if agent_id not in self.agent_utilities:
                    self.agent_utilities[agent_id] = 0.0
                self.agent_utilities[agent_id] += utility

        if agent_preferences:
            self.current_agent_preferences = {
                agent_id: [float(value) for value in preference]
                for agent_id, preference in agent_preferences.items()
            }

        # Track step when vehicles complete (for accurate episode length)
        if step_successes > 0:
            self.last_completion_step = self.current_length

        # Track traffic performance metrics if observations provided
        if observations:
            self._update_traffic_metrics(observations)

        # Track RL-commanded target speeds directly (more reliable than observations)
        # This captures what the RL algorithm actually commanded, not adapter cached values
        if target_speeds:
            for agent_id, ts in target_speeds.items():
                if ts is not None and ts > 0:
                    self.step_target_speeds.append(ts)

    def _update_reward_component_metrics(self, reward_outputs: Dict[Any, RewardOutput]):
        for reward_output in reward_outputs.values():
            for component, value in reward_output.components.items():
                self.current_reward_component_totals[component] += float(value)

            self.reward_vector_dim = len(reward_output.vector)
            # The spec is one shared (frozen) object for the whole run: derive
            # its metadata once per distinct spec instead of re-serialising it
            # for every agent on every step.
            spec = reward_output.spec
            if spec is not getattr(self, "_reward_spec_seen", None):
                self._reward_spec_seen = spec
                self.reward_component_order = list(spec.components)
                self.reward_spec_metadata = spec.to_dict()

    def _update_objective_metrics(self, objective_outputs: Dict[Any, ObjectiveOutput]):
        for objective_output in objective_outputs.values():
            for component, value in objective_output.components.items():
                self.current_objective_component_totals[component] += float(value)

            self.objective_vector_dim = len(objective_output.vector)
            spec = objective_output.spec
            if spec is not getattr(self, "_objective_spec_seen", None):
                self._objective_spec_seen = spec
                self.objective_component_order = list(spec.objectives)
                self.objective_spec_metadata = spec.to_dict()

    def _update_traffic_metrics(self, observations: Dict):
        """
        Update traffic performance metrics from observations.

        Args:
            observations: Dictionary of agent observations
        """
        step_actual_speeds_list = []
        step_speed_3d_list = []
        step_vertical_speed_list = []
        step_agent_speeds_list = []
        step_target_speeds_list = []

        for agent_id, obs in observations.items():
            # Extract canonical actual speed and secondary agent speed.
            physics_speed = None
            physics_speed_3d = None
            vertical_speed = None
            agent_speed = None
            target_speed = None
            if isinstance(obs, dict):
                physics_speed = obs.get('physics_speed_planar', obs.get('physics_speed'))
                physics_speed_3d = obs.get('physics_speed_3d', physics_speed)
                vertical_speed = abs(obs.get('vertical_speed', 0.0))
                agent_speed = obs.get('agent_speed', obs.get('speed'))
                target_speed = obs.get('target_speed')  # Track commanded speed
            elif hasattr(obs, '__getitem__'):
                # Numpy array or list - speed is typically index 2 in 7D features
                try:
                    physics_speed = float(obs[2]) if len(obs) > 2 else None
                    physics_speed_3d = physics_speed
                    vertical_speed = 0.0
                except (IndexError, TypeError):
                    pass

            if physics_speed is not None:
                step_actual_speeds_list.append(physics_speed)

                # Track per-agent speed history
                if agent_id not in self.agent_actual_speeds:
                    self.agent_actual_speeds[agent_id] = []
                self.agent_actual_speeds[agent_id].append(physics_speed)

            if physics_speed_3d is not None:
                step_speed_3d_list.append(physics_speed_3d)

            if vertical_speed is not None:
                step_vertical_speed_list.append(vertical_speed)

            if agent_speed is not None:
                step_agent_speeds_list.append(agent_speed)
                if agent_id not in self.agent_reported_speeds:
                    self.agent_reported_speeds[agent_id] = []
                self.agent_reported_speeds[agent_id].append(agent_speed)

            # Filter out 0.0 values (uninitialized or stopped vehicles)
            if target_speed is not None and target_speed > 0:
                step_target_speeds_list.append(target_speed)

            self._maybe_record_speed_spike(
                agent_id=agent_id,
                observation=obs if isinstance(obs, dict) else {},
                physics_speed=physics_speed,
                physics_speed_3d=physics_speed_3d,
                vertical_speed=vertical_speed,
                agent_speed=agent_speed,
                target_speed=target_speed,
            )

        # Store step-level speed data
        if step_actual_speeds_list:
            self.step_actual_speeds.extend(step_actual_speeds_list)
            self.step_actual_avg_speeds.append(float(np.mean(step_actual_speeds_list)))

        if step_speed_3d_list:
            self.step_speed_3d.extend(step_speed_3d_list)

        if step_vertical_speed_list:
            self.step_vertical_speeds.extend(step_vertical_speed_list)

        if step_agent_speeds_list:
            self.step_agent_speeds.extend(step_agent_speeds_list)
            self.step_agent_avg_speeds.append(float(np.mean(step_agent_speeds_list)))

        # Store target speeds
        if step_target_speeds_list:
            self.step_target_speeds.extend(step_target_speeds_list)

    # --------------------------------------------------------------------- #
    # Public Methods
    # --------------------------------------------------------------------- #

    def get_agent_reward(self, agent_id: int) -> float:
        return self.agent_rewards.get(agent_id, 0.0)

    def record_physics_anomaly_events(self, events: List[Any]):
        for event in events:
            metadata = dict(getattr(event, 'metadata', {}) or {})
            record = {
                'step': getattr(event, 'step', self.current_length),
                'vehicle_id': getattr(event, 'vehicle_id', None),
                'reasons': metadata.get('reasons', []),
                'severity': metadata.get('severity', 'fatal'),
                'policy': metadata.get('policy'),
                'physics_speed_planar': metadata.get('physics_speed_planar'),
                'physics_speed_3d': metadata.get('physics_speed_3d'),
                'vertical_speed': metadata.get('vertical_speed'),
                'speed_3d_gap': metadata.get('speed_3d_gap'),
                'tilt_pitch': metadata.get('tilt_pitch'),
                'tilt_roll': metadata.get('tilt_roll'),
                'height_above_waypoint': metadata.get('height_above_waypoint'),
                'location_z': metadata.get('location_z'),
                'waypoint_z': metadata.get('waypoint_z'),
                'target_speed': metadata.get('target_speed'),
            }
            self.physics_anomaly_records.append(record)
            self.physics_anomaly_count += 1
            if (
                self.max_physics_anomaly_record is None or
                float(record.get('physics_speed_3d') or 0.0) > float(self.max_physics_anomaly_record.get('physics_speed_3d') or 0.0)
            ):
                self.max_physics_anomaly_record = record

    def record_competition_records(self, records: List[Dict[str, Any]]):
        if not records:
            return
        self.current_competition_records.extend(records)

    def _route_intent_episode_metrics(self, competition_summary: Dict[str, Any]) -> Dict[str, float]:
        summary = (competition_summary or {}).get('route_intent_summary') or {}
        if not summary:
            return {}
        distribution = summary.get('intent_distribution') or {}
        decision_count = sum(int(v) for v in distribution.values())
        direction_violations = float(
            summary.get(
                'direction_preserved_violations',
                summary.get(
                    'same_exit_direction_violations',
                    summary.get('same_destination_violations', 0.0),
                ),
            )
        )
        direction_violation_rate = float(
            summary.get(
                'direction_preserved_violation_rate',
                summary.get(
                    'same_exit_direction_violation_rate',
                    summary.get('same_destination_violation_rate', 0.0),
                ),
            )
        )
        same_exit_candidates = float(
            summary.get(
                'mean_candidate_count_same_exit_direction',
                summary.get('mean_candidate_count_same_destination', 0.0),
            )
        )
        return {
            'route_intent_decision_count': float(decision_count),
            'route_intent_keep_current_rate': float(summary.get('keep_current_rate', 0.0)),
            'route_intent_mask_legal_fraction': float(summary.get('mask_legal_fraction', 0.0)),
            'route_intent_mean_mask_sparsity': float(summary.get('mean_mask_sparsity', 0.0)),
            'route_intent_keep_only_mask_count': float(summary.get('keep_only_mask_count', 0.0)),
            'route_intent_keep_only_mask_rate': float(summary.get('keep_only_mask_rate', 0.0)),
            'route_intent_direction_preserved_violations': direction_violations,
            'route_intent_direction_preserved_violation_rate': direction_violation_rate,
            'route_intent_same_exit_direction_violations': direction_violations,
            'route_intent_same_exit_direction_violation_rate': direction_violation_rate,
            'route_intent_same_destination_violations': direction_violations,
            'route_intent_same_destination_violation_rate': direction_violation_rate,
            'route_intent_fallback_count': float(summary.get('fallback_count', 0.0)),
            'route_intent_fallback_rate': float(summary.get('fallback_rate', 0.0)),
            'route_intent_route_change_count': float(summary.get('route_change_count', 0.0)),
            'route_intent_route_change_rate': float(summary.get('route_change_rate', 0.0)),
            'route_intent_queued_lane_avoidance_count': float(
                summary.get('queued_lane_avoidance_count', 0.0)
            ),
            'route_intent_queued_lane_avoidance_opportunities': float(
                summary.get('queued_lane_avoidance_opportunities', 0.0)
            ),
            'route_intent_queued_lane_avoidance_rate': float(
                summary.get('queued_lane_avoidance_rate', 0.0)
            ),
            'route_intent_mean_selected_queued_same_exit_lane_count': float(
                summary.get('mean_selected_queued_same_exit_lane_count', 0.0)
            ),
            'route_intent_mean_original_queued_same_exit_lane_count': float(
                summary.get('mean_original_queued_same_exit_lane_count', 0.0)
            ),
            'route_intent_mean_candidate_count_total': float(
                summary.get('mean_candidate_count_total', 0.0)
            ),
            'route_intent_mean_candidate_count_same_exit_direction': same_exit_candidates,
            'route_intent_mean_candidate_count_same_destination': same_exit_candidates,
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get metrics for current episode including traffic performance."""
        if len(self.agent_rewards) == 0:
            avg_reward = 0.0
        else:
            avg_reward = self.current_total_reward / \
                len(self.agent_rewards)
        if len(self.agent_utilities) == 0:
            avg_utility = 0.0
        else:
            avg_utility = self.current_total_utility / len(self.agent_utilities)

        # Compute traffic performance metrics for current episode
        traffic_metrics = self._compute_traffic_metrics()
        competition_summary = build_competition_summary(
            self.current_competition_records,
            yielding_total=float(self.current_reward_component_totals.get('yielding', 0.0)),
        )

        # Compute success/collision/timeout rates from tracked counts
        # Include timeout vehicles in total to properly account for all spawned vehicles
        total_vehicles = self.successes + self.collisions + self.physics_anomaly_count + self.timeouts
        success_rate = (self.successes / total_vehicles * 100) if total_vehicles > 0 else 0.0
        collision_rate = (self.collisions / total_vehicles * 100) if total_vehicles > 0 else 0.0
        physics_anomaly_rate = (self.physics_anomaly_count / total_vehicles * 100) if total_vehicles > 0 else 0.0
        timeout_rate = (self.timeouts / total_vehicles * 100) if total_vehicles > 0 else 0.0

        if not self.episode_rewards:
            metrics = {
                'step_reward': self.current_reward,
                'step_length': self.current_length,
                'total_reward': self.current_total_reward,
                'avg_reward': avg_reward,
                'total_utility': self.current_total_utility,
                'avg_utility': avg_utility,
                'agent_utilities': dict(self.agent_utilities),
                'objective_component_totals': dict(self.current_objective_component_totals),
                'objective_vector_dim': self.objective_vector_dim,
                'objective_component_order': list(self.objective_component_order),
                'objective_spec': self.objective_spec_metadata,
                'agent_preferences': dict(self.current_agent_preferences),
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'physics_anomaly_rate': physics_anomaly_rate,
                'timeout_rate': timeout_rate,
                'successes': self.successes,
                'collisions': self.collisions,
                'physics_anomalies': self.physics_anomaly_count,
                'timeouts': self.timeouts,
                'reward_component_totals': dict(self.current_reward_component_totals),
                'reward_vector_dim': self.reward_vector_dim,
                'reward_component_order': list(self.reward_component_order),
                'reward_spec': self.reward_spec_metadata,
                'speed_spike_count': len(self.speed_spike_records),
                'physics_anomaly_count': self.physics_anomaly_count,
                'competition_summary': competition_summary,
            }
            if self.speed_spike_records:
                metrics['first_speed_spike'] = self.speed_spike_records[0]
                metrics['latest_speed_spike'] = self.speed_spike_records[-1]
            if self.max_speed_spike_record:
                metrics['max_speed_spike'] = self.max_speed_spike_record
            if self.physics_anomaly_records:
                metrics['first_physics_anomaly'] = self.physics_anomaly_records[0]
                metrics['latest_physics_anomaly'] = self.physics_anomaly_records[-1]
            if self.max_physics_anomaly_record:
                metrics['max_physics_anomaly'] = self.max_physics_anomaly_record
            metrics.update(self._route_intent_episode_metrics(competition_summary))
            metrics.update(traffic_metrics)
            return metrics

        # Compute running averages (convert deque to list for slicing)
        rewards_list = list(self.episode_rewards)
        window_size = min(10, len(rewards_list))
        recent_rewards = rewards_list[-window_size:]

        metrics = {
            'step_reward': self.current_reward,
            'step_length': self.current_length,
            'total_reward': self.current_total_reward,
            'avg_reward': avg_reward,
            'total_utility': self.current_total_utility,
            'avg_utility': avg_utility,
            'agent_utilities': dict(self.agent_utilities),
            'objective_component_totals': dict(self.current_objective_component_totals),
            'objective_vector_dim': self.objective_vector_dim,
            'objective_component_order': list(self.objective_component_order),
            'objective_spec': self.objective_spec_metadata,
            'agent_preferences': dict(self.current_agent_preferences),
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'physics_anomaly_rate': physics_anomaly_rate,
            'timeout_rate': timeout_rate,
            'successes': self.successes,
            'collisions': self.collisions,
            'physics_anomalies': self.physics_anomaly_count,
            'timeouts': self.timeouts,
            'reward_component_totals': dict(self.current_reward_component_totals),
            'reward_vector_dim': self.reward_vector_dim,
            'reward_component_order': list(self.reward_component_order),
            'reward_spec': self.reward_spec_metadata,
            'speed_spike_count': len(self.speed_spike_records),
            'physics_anomaly_count': self.physics_anomaly_count,
            'competition_summary': competition_summary,
            f'mean_reward_episode_{window_size}': float(np.mean(recent_rewards)),
            f'std_reward_episode_{window_size}': float(np.std(recent_rewards)),
            'max_reward_episode': max(rewards_list),
            'total_episodes': len(rewards_list),
        }
        if self.speed_spike_records:
            metrics['first_speed_spike'] = self.speed_spike_records[0]
            metrics['latest_speed_spike'] = self.speed_spike_records[-1]
        if self.max_speed_spike_record:
            metrics['max_speed_spike'] = self.max_speed_spike_record
        if self.physics_anomaly_records:
            metrics['first_physics_anomaly'] = self.physics_anomaly_records[0]
            metrics['latest_physics_anomaly'] = self.physics_anomaly_records[-1]
        if self.max_physics_anomaly_record:
            metrics['max_physics_anomaly'] = self.max_physics_anomaly_record
        metrics.update(self._route_intent_episode_metrics(competition_summary))
        metrics.update(traffic_metrics)

        # Add multi-episode traffic trends if available
        if len(self.episode_avg_speeds) > 0:
            metrics['mean_episode_speed'] = float(np.mean(self.episode_avg_speeds))
            metrics['mean_episode_speed_var'] = float(np.mean(self.episode_speed_variances))

        return metrics

    def _compute_traffic_metrics(self) -> Dict[str, float]:
        """
        Compute traffic performance metrics for current episode.

        Returns:
            Dictionary with traffic performance metrics
        """
        metrics = {}

        # Canonical actual speed across all agents and steps, from CARLA physics.
        if self.step_actual_speeds:
            actual_speed_mean = float(np.mean(self.step_actual_speeds))
            actual_speed_std = float(np.std(self.step_actual_speeds))
            actual_speed_variance = float(np.var(self.step_actual_speeds))
            actual_speed_min = float(np.min(self.step_actual_speeds))
            actual_speed_max = float(np.max(self.step_actual_speeds))
        else:
            actual_speed_mean = 0.0
            actual_speed_std = 0.0
            actual_speed_variance = 0.0
            actual_speed_min = 0.0
            actual_speed_max = 0.0

        metrics['actual_speed_mean'] = actual_speed_mean
        metrics['actual_speed_std'] = actual_speed_std
        metrics['actual_speed_variance'] = actual_speed_variance
        metrics['actual_speed_min'] = actual_speed_min
        metrics['actual_speed_max'] = actual_speed_max
        metrics['speed_source'] = 'carla_planar_physics'

        # Backward-compatible aliases now point to the canonical actual speed source.
        metrics['avg_speed'] = actual_speed_mean
        metrics['speed_std'] = actual_speed_std
        metrics['speed_variance'] = actual_speed_variance
        metrics['min_speed'] = actual_speed_min
        metrics['max_speed'] = actual_speed_max

        # Secondary agent/localizer-reported speed summaries.
        if self.step_agent_speeds:
            metrics['agent_speed_mean'] = float(np.mean(self.step_agent_speeds))
            metrics['agent_speed_std'] = float(np.std(self.step_agent_speeds))
            metrics['agent_speed_variance'] = float(np.var(self.step_agent_speeds))
            metrics['agent_speed_min'] = float(np.min(self.step_agent_speeds))
            metrics['agent_speed_max'] = float(np.max(self.step_agent_speeds))
        else:
            metrics['agent_speed_mean'] = 0.0
            metrics['agent_speed_std'] = 0.0
            metrics['agent_speed_variance'] = 0.0
            metrics['agent_speed_min'] = 0.0
            metrics['agent_speed_max'] = 0.0

        if self.step_speed_3d:
            metrics['speed_3d_mean'] = float(np.mean(self.step_speed_3d))
            metrics['speed_3d_std'] = float(np.std(self.step_speed_3d))
            metrics['speed_3d_variance'] = float(np.var(self.step_speed_3d))
            metrics['speed_3d_max'] = float(np.max(self.step_speed_3d))
        else:
            metrics['speed_3d_mean'] = 0.0
            metrics['speed_3d_std'] = 0.0
            metrics['speed_3d_variance'] = 0.0
            metrics['speed_3d_max'] = 0.0

        if self.step_vertical_speeds:
            metrics['vertical_speed_mean'] = float(np.mean(self.step_vertical_speeds))
            metrics['vertical_speed_std'] = float(np.std(self.step_vertical_speeds))
            metrics['vertical_speed_variance'] = float(np.var(self.step_vertical_speeds))
            metrics['vertical_speed_max'] = float(np.max(self.step_vertical_speeds))
        else:
            metrics['vertical_speed_mean'] = 0.0
            metrics['vertical_speed_std'] = 0.0
            metrics['vertical_speed_variance'] = 0.0
            metrics['vertical_speed_max'] = 0.0

        # Target (commanded) speed metrics - for comparing RL output vs actual vehicle speed
        if self.step_target_speeds:
            metrics['target_speed_mean'] = float(np.mean(self.step_target_speeds))
            metrics['target_speed_max'] = float(np.max(self.step_target_speeds))
            metrics['target_speed_min'] = float(np.min(self.step_target_speeds))
        else:
            metrics['target_speed_mean'] = 0.0
            metrics['target_speed_max'] = 0.0
            metrics['target_speed_min'] = 0.0

        # Speed smoothness (variance of step-average speeds = how stable traffic flow is)
        if self.step_actual_avg_speeds:
            metrics['speed_smoothness'] = float(np.var(self.step_actual_avg_speeds))
            # Lower variance = smoother flow
            metrics['avg_step_speed'] = float(np.mean(self.step_actual_avg_speeds))
        else:
            metrics['speed_smoothness'] = 0.0
            metrics['avg_step_speed'] = 0.0

        # Per-agent actual speed consistency (how consistent each agent maintains speed)
        if self.agent_actual_speeds:
            agent_speed_vars = []
            for agent_id, speeds in self.agent_actual_speeds.items():
                if len(speeds) > 1:
                    agent_speed_vars.append(np.var(speeds))
            if agent_speed_vars:
                metrics['avg_agent_speed_var'] = float(np.mean(agent_speed_vars))
            else:
                metrics['avg_agent_speed_var'] = 0.0
        else:
            metrics['avg_agent_speed_var'] = 0.0

        # Per-agent reported speed consistency for debugging.
        if self.agent_reported_speeds:
            reported_speed_vars = []
            for agent_id, speeds in self.agent_reported_speeds.items():
                if len(speeds) > 1:
                    reported_speed_vars.append(np.var(speeds))
            metrics['avg_agent_reported_speed_var'] = float(np.mean(reported_speed_vars)) if reported_speed_vars else 0.0
        else:
            metrics['avg_agent_reported_speed_var'] = 0.0

        metrics['speed_command_gap_mean'] = metrics['target_speed_mean'] - metrics['actual_speed_mean']
        metrics['agent_vs_physics_speed_gap_mean'] = metrics['agent_speed_mean'] - metrics['actual_speed_mean']
        metrics['speed_3d_gap_mean'] = metrics['speed_3d_mean'] - metrics['actual_speed_mean']

        suspicious, warning = self._validate_speed_metrics(metrics)
        metrics['speed_metrics_suspicious'] = suspicious
        if warning:
            metrics['speed_metrics_warning'] = warning

        return metrics

    def _maybe_record_speed_spike(
        self,
        agent_id: Any,
        observation: Dict[str, Any],
        physics_speed: Optional[float],
        physics_speed_3d: Optional[float],
        vertical_speed: Optional[float],
        agent_speed: Optional[float],
        target_speed: Optional[float],
    ):
        """Capture concrete context for suspicious speed samples."""
        if physics_speed is None:
            return

        reasons = self._speed_spike_reasons(
            physics_speed,
            physics_speed_3d,
            vertical_speed,
            agent_speed,
            target_speed,
        )
        if not reasons:
            return

        vehicle_id = observation.get('vehicle_id', agent_id)
        spike_key = (self.current_length, str(vehicle_id))
        if spike_key in self._speed_spike_keys:
            return
        self._speed_spike_keys.add(spike_key)

        relative_position = observation.get('relative_position_to_intersection', {})
        location = observation.get('location', {})
        velocity = observation.get('velocity', {})
        spike_record = {
            'step': self.current_length,
            'agent_id': str(agent_id),
            'vehicle_id': vehicle_id,
            'vehicle_blueprint': observation.get('vehicle_blueprint'),
            'physics_speed': float(physics_speed),
            'agent_speed': float(agent_speed) if agent_speed is not None else None,
            'target_speed': float(target_speed) if target_speed is not None else None,
            'physics_speed_3d': float(physics_speed_3d) if physics_speed_3d is not None else None,
            'vertical_speed': float(vertical_speed) if vertical_speed is not None else None,
            'position_x': observation.get('position_x'),
            'position_y': observation.get('position_y'),
            'relative_position_to_intersection': {
                'x': relative_position.get('x'),
                'y': relative_position.get('y'),
            },
            'location': {
                'x': location.get('x'),
                'y': location.get('y'),
            },
            'velocity': {
                'x': velocity.get('x'),
                'y': velocity.get('y'),
                'z': velocity.get('z'),
            },
            'distance_to_intersection': observation.get('distance_to_intersection'),
            'distance_to_front_vehicle': observation.get('distance_to_front_vehicle'),
            'lane_position': observation.get('lane_position'),
            'status': observation.get('status'),
            'reasons': reasons,
        }
        if (
            self.max_speed_spike_record is None or
            spike_record['physics_speed'] > self.max_speed_spike_record['physics_speed']
        ):
            self.max_speed_spike_record = spike_record

        if len(self.speed_spike_records) >= self.MAX_SPIKE_RECORDS:
            return

        self.speed_spike_records.append(spike_record)
        logger.debug(f"Recorded suspicious speed spike: {spike_record}")

    def _speed_spike_reasons(
        self,
        physics_speed: float,
        physics_speed_3d: Optional[float],
        vertical_speed: Optional[float],
        agent_speed: Optional[float],
        target_speed: Optional[float],
    ) -> List[str]:
        reasons: List[str] = []
        if physics_speed > self.HARD_SPEED_PLAUSIBILITY_KMH:
            reasons.append(
                f"physics_speed={physics_speed:.2f} km/h exceeds plausibility threshold {self.HARD_SPEED_PLAUSIBILITY_KMH:.2f}"
            )

        if target_speed is not None and target_speed > 0.0:
            if physics_speed > max(target_speed * 2.0, target_speed + 30.0):
                reasons.append(
                    f"physics_speed={physics_speed:.2f} km/h inconsistent with target_speed={target_speed:.2f} km/h"
                )

        if agent_speed is not None and abs(agent_speed - physics_speed) > 30.0:
            reasons.append(
                f"agent_speed={agent_speed:.2f} km/h diverges from physics_speed={physics_speed:.2f} km/h"
            )

        if physics_speed_3d is not None and physics_speed_3d - physics_speed > 15.0:
            reasons.append(
                f"physics_speed_3d={physics_speed_3d:.2f} km/h diverges from planar physics_speed={physics_speed:.2f} km/h"
            )

        if vertical_speed is not None and vertical_speed > 15.0:
            reasons.append(
                f"vertical_speed={vertical_speed:.2f} km/h indicates CARLA vertical-physics anomaly"
            )

        return reasons

    def _validate_speed_metrics(self, metrics: Dict[str, float]) -> tuple[bool, str]:
        """
        Flag clearly implausible speed summaries without breaking the current run.

        This is a guardrail for CARLA metric auditing, not a physics model.
        """
        avg_speed = float(metrics.get('actual_speed_mean', 0.0))
        max_speed = float(metrics.get('actual_speed_max', 0.0))
        agent_speed_mean = float(metrics.get('agent_speed_mean', 0.0))
        vertical_speed_max = float(metrics.get('vertical_speed_max', 0.0))
        speed_3d_gap_mean = float(metrics.get('speed_3d_gap_mean', 0.0))
        target_speed_mean = float(metrics.get('target_speed_mean', 0.0))
        target_speed_max = float(metrics.get('target_speed_max', 0.0))

        suspicious_reasons = []
        if max_speed > 180.0:
            suspicious_reasons.append(f"actual_speed_max={max_speed:.2f} km/h exceeds hard plausibility threshold")

        if target_speed_max > 0.0 and max_speed > max(target_speed_max * 2.0, target_speed_max + 30.0):
            suspicious_reasons.append(
                f"actual_speed_max={max_speed:.2f} km/h is inconsistent with target_speed_max={target_speed_max:.2f} km/h"
            )

        if target_speed_mean > 0.0 and avg_speed > max(target_speed_mean * 2.0, target_speed_mean + 20.0):
            suspicious_reasons.append(
                f"actual_speed_mean={avg_speed:.2f} km/h is inconsistent with target_speed_mean={target_speed_mean:.2f} km/h"
            )

        if agent_speed_mean > 0.0 and abs(agent_speed_mean - avg_speed) > 30.0:
            suspicious_reasons.append(
                f"agent_speed_mean={agent_speed_mean:.2f} km/h diverges from actual_speed_mean={avg_speed:.2f} km/h"
            )

        if vertical_speed_max > 15.0:
            suspicious_reasons.append(
                f"vertical_speed_max={vertical_speed_max:.2f} km/h suggests CARLA vertical-physics anomaly"
            )

        if speed_3d_gap_mean > 10.0:
            suspicious_reasons.append(
                f"speed_3d_gap_mean={speed_3d_gap_mean:.2f} km/h indicates 3D speed divergence from planar road speed"
            )

        if suspicious_reasons:
            warning = "; ".join(suspicious_reasons)
            logger.debug(f"Traffic speed metrics look suspicious: {warning}")
            return True, warning

        return False, ""

    def finish_episode(self, states: Dict[str, Any]):
        """Finish current episode and compute metrics."""
        self._episode_counter += 1

        # Reconcile terminal counters from the episode snapshot. CARLA-style
        # episodes provide these counts only at finish time, while SUMO-style
        # episodes may already have incremented success/collision counters as
        # events happened. Use the snapshot as a floor instead of adding it
        # blindly, otherwise SUMO terminal outcomes are double-counted.
        episode_successes = states.get('success', 0)
        episode_collisions = states.get('collision', 0)
        episode_physics_anomalies = states.get('physics_anomaly_count', 0)
        episode_timeouts = states.get('active_agents', 0)  # Active agents at episode end = timeouts
        episode_near_misses = states.get('near_miss_count', 0)
        self.successes = max(int(self.successes), int(episode_successes))
        self.collisions = max(int(self.collisions), int(episode_collisions))
        self.physics_anomaly_count = max(int(self.physics_anomaly_count), int(episode_physics_anomalies))
        self.timeouts = max(int(self.timeouts), int(episode_timeouts))

        # Compute traffic metrics first
        traffic_metrics = self._compute_traffic_metrics()
        avg_speed = traffic_metrics.get('actual_speed_mean', 0.0)
        speed_variance = traffic_metrics.get('actual_speed_variance', 0.0)

        # Calculate actual episode length: step when LAST vehicle reaches destination
        # If no vehicles completed, use total simulation steps as fallback
        effective_episode_length = self.last_completion_step if self.last_completion_step > 0 else self.current_length

        # Calculate throughput: successful vehicles per hour
        # Formula: (successes / effective_steps / fixed_dt) * 3600
        fixed_dt = states.get('fixed_dt', 0.05)  # 20 FPS = 0.05s per step
        if effective_episode_length > 0 and fixed_dt > 0:
            vehicles_per_second = episode_successes / effective_episode_length / fixed_dt
            throughput = vehicles_per_second * 3600  # Convert to vehicles per hour
        else:
            throughput = 0.0

        # Add to rolling windows (bounded memory)
        self.episode_rewards.append(self.current_total_reward)
        self.episode_states.append(states)
        self.episode_avg_speeds.append(avg_speed)
        self.episode_speed_variances.append(speed_variance)
        competition_summary = build_competition_summary(
            states.get('competition_records', self.current_competition_records),
            yielding_total=float(self.current_reward_component_totals.get('yielding', 0.0)),
        )

        # Add to full history for export (use effective length, not total steps)
        self._full_history['rewards'].append(self.current_total_reward)
        self._full_history['reward_component_totals'].append(dict(self.current_reward_component_totals))
        # The spec metadata dicts are cached per spec (see
        # _update_reward_component_metrics); the history keeps its own copy
        # per episode so entries never alias each other.
        self._full_history['reward_specs'].append(copy.deepcopy(self.reward_spec_metadata))
        self._full_history['utilities'].append(self.current_total_utility)
        self._full_history['objective_component_totals'].append(dict(self.current_objective_component_totals))
        self._full_history['objective_specs'].append(copy.deepcopy(self.objective_spec_metadata))
        self._full_history['agent_utilities'].append(dict(self.agent_utilities))
        self._full_history['agent_preferences'].append(dict(self.current_agent_preferences))
        self._full_history['actual_speed_means'].append(avg_speed)
        self._full_history['actual_speed_variances'].append(speed_variance)
        self._full_history['speed_3d_means'].append(traffic_metrics.get('speed_3d_mean', 0.0))
        self._full_history['speed_3d_variances'].append(traffic_metrics.get('speed_3d_variance', 0.0))
        self._full_history['vertical_speed_means'].append(traffic_metrics.get('vertical_speed_mean', 0.0))
        self._full_history['vertical_speed_variances'].append(traffic_metrics.get('vertical_speed_variance', 0.0))
        self._full_history['agent_speed_means'].append(traffic_metrics.get('agent_speed_mean', 0.0))
        self._full_history['agent_speed_variances'].append(traffic_metrics.get('agent_speed_variance', 0.0))
        self._full_history['target_speed_means'].append(traffic_metrics.get('target_speed_mean', 0.0))
        self._full_history['speed_spike_records'].append(list(self.speed_spike_records))
        self._full_history.setdefault('max_speed_spike_records', []).append(self.max_speed_spike_record)
        self._full_history['physics_anomaly_counts'].append(self.physics_anomaly_count)
        self._full_history['physics_anomaly_records'].append(list(self.physics_anomaly_records))
        self._full_history['episode_lengths'].append(effective_episode_length)
        self._full_history['success_counts'].append(episode_successes)
        self._full_history['collision_counts'].append(episode_collisions)
        self._full_history['physics_anomaly_event_counts'].append(episode_physics_anomalies)
        self._full_history['timeout_counts'].append(episode_timeouts)
        self._full_history['near_miss_counts'].append(episode_near_misses)
        self._full_history['competition_summaries'].append(competition_summary)

        # Check if it's time to export
        if self.export_interval > 0 and self._episode_counter % self.export_interval == 0:
            self.export_history()

        metrics = self.get_current_metrics()
        metrics.update({
            'episode_states': states,
            'near_miss_count': episode_near_misses,  # Add for TensorBoard logging
            'ttc_violation_rate': states.get('ttc_violation_rate', 0.0),  # % of TTC checks with violations
            'throughput': throughput,  # Vehicles per hour
            'episode_length': effective_episode_length,  # Step when LAST vehicle completed
            'total_simulation_steps': self.current_length,  # Total steps for reference
            'competition_summary': competition_summary,
        })
        metrics.update(self._route_intent_episode_metrics(competition_summary))
        self.reset()
        return metrics

    def export_history(self, filepath: str = None):
        """
        Export full history to file and clear memory.

        Args:
            filepath: Optional custom filepath. If None, auto-generates based on episode count.
        """
        if not self._full_history['rewards']:
            logger.debug("No history to export")
            return

        if filepath is None:
            filepath = os.path.join(
                self.export_dir,
                f"metrics_history_ep{self._episode_counter}.pkl"
            )

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self._full_history, f)

            exported_count = len(self._full_history['rewards'])
            logger.info(f"Exported {exported_count} episodes to {filepath}")

            # Clear full history to free memory
            self._full_history = {
                'rewards': [],
                'reward_component_totals': [],
                'reward_specs': [],
                'utilities': [],
                'objective_component_totals': [],
                'objective_specs': [],
                'agent_utilities': [],
                'agent_preferences': [],
                'actual_speed_means': [],
                'actual_speed_variances': [],
                'speed_3d_means': [],
                'speed_3d_variances': [],
                'vertical_speed_means': [],
                'vertical_speed_variances': [],
                'agent_speed_means': [],
                'agent_speed_variances': [],
                'target_speed_means': [],
                'speed_spike_records': [],
                'max_speed_spike_records': [],
                'physics_anomaly_counts': [],
                'physics_anomaly_records': [],
                'episode_lengths': [],
                'success_counts': [],
                'collision_counts': [],
                'physics_anomaly_event_counts': [],
                'timeout_counts': [],
                'near_miss_counts': [],
                'competition_summaries': [],
            }
        except Exception as e:
            logger.error(f"Failed to export metrics history: {e}")

    def get_traffic_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive traffic performance statistics for reporting.

        Returns:
            Dictionary with traffic performance metrics across all episodes
        """
        stats = {
            'total_episodes': len(self.episode_avg_speeds),
        }

        if len(self.episode_avg_speeds) > 0:
            stats.update({
                'mean_avg_speed': float(np.mean(self.episode_avg_speeds)),
                'std_avg_speed': float(np.std(self.episode_avg_speeds)),
                'mean_speed_variance': float(np.mean(self.episode_speed_variances)),
                'std_speed_variance': float(np.std(self.episode_speed_variances)),
                'episode_avg_speeds': list(self.episode_avg_speeds),
                'episode_speed_variances': list(self.episode_speed_variances),
            })

            # Trend analysis
            if len(self.episode_avg_speeds) > 1:
                speed_trend = np.polyfit(
                    range(len(self.episode_avg_speeds)),
                    self.episode_avg_speeds, 1
                )[0]
                var_trend = np.polyfit(
                    range(len(self.episode_speed_variances)),
                    self.episode_speed_variances, 1
                )[0]
                stats['speed_trend'] = float(speed_trend)  # km/h per episode
                stats['variance_trend'] = float(var_trend)  # Decreasing = improving smoothness

        return stats
