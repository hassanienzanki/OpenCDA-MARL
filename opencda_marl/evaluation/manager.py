"""Engine-side evaluation: episode summaries, mirrored into the metrics store.

Two modes:

- ``summary`` — per-episode traffic statistics (success / collision /
  timeout rates, throughput, rewards) kept in memory, appended to a CSV,
  written as ``final_summary.json`` at the end of the run, and mirrored
  into a :class:`~opencda_marl.telemetry.store.MetricsStore` so the web
  monitor shows them (``Eval/*`` tags and the final summary).
- ``disabled`` — no tracking at all.

Where the mirror goes: one run is one monitor entry. When the run has a
metrics store of its own (the learner's, handed over with
:meth:`EvaluationManager.attach_run_store`), the ``Eval/*`` scalars, one
``evaluation_episode`` event per episode and the ``final_summary`` meta key
are written into it; its episode rows stay the learner's. Only a run without
one (a baseline agent, or a learner whose store is disabled) gets a separate
run under ``<outputs.root_dir>/metrics/evaluation/``, with episode rows of
its own, labelled with the learner key when there is one and with the run's
mode: ``train`` while a learner trains, ``eval`` otherwise.

The legacy ``full`` mode (per-step ``.npz`` archives and matplotlib PNG
plots) is retired; configs that still ask for it are mapped to ``summary``
with a warning. The monitor is the way to look at metrics.
"""
import csv
import json
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

_LEGACY_MODES = {"full": "summary", "lightweight": "summary"}


class EvaluationManager:
    """Track per-episode traffic statistics for one run."""

    def __init__(self, config: Optional[Dict] = None,
                 scenario_name: Optional[str] = None,
                 agent_name: Optional[str] = None,
                 algorithm_name: Optional[str] = None):
        self.config = config or {}
        self.scenario_name = scenario_name
        # A missing label must not crash a run whose evaluation is
        # disabled anyway; directories just get a neutral name.
        self.agent_name = agent_name or 'agent'
        # The registered learner key, when the run has one; a separate
        # evaluation run is labelled with the agent type otherwise.
        self.algorithm_name = algorithm_name
        # Whether that learner trains in this run (attach_run_store sets it):
        # a separate evaluation run is then labelled ``train``, not ``eval``.
        self.learner_training = False
        self.fixed_dt = None

        self.mode = self._resolve_mode(self.config.get('mode', 'summary'))

        self._stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        # The lifecycle resolves ``outputs.root_dir/outputs.evaluation.dir``
        # into ``root_dir``; a bare config lands under the documented default
        # root rather than a directory of its own at the repository root.
        self.root_dir = Path(self.config.get('root_dir', 'artifacts/evaluation'))
        self.output_dir = self.root_dir / (self.scenario_name or 'scenario') / \
            datetime.now().strftime("%Y_%m_%d_%H")
        self.data_dir = self.output_dir / self.agent_name / 'data'

        self.keep_episode_summaries = self.config.get(
            'keep_episode_summaries', 10)
        self.last_saved_episode_count = 0

        # The metrics-store mirror: the run's own store when one is attached
        # (borrowed, never closed here), else a separate one created lazily
        # on the first episode so a run that never completes an episode
        # leaves no empty store behind.
        self._store = None
        self._store_borrowed = False
        self._store_unavailable = False

        self._init_tracking()

    @staticmethod
    def _resolve_mode(mode: str) -> str:
        mode = str(mode or 'summary')
        if mode in _LEGACY_MODES:
            mapped = _LEGACY_MODES[mode]
            logger.warning(
                f"Evaluation mode '{mode}' was retired (per-step archives and "
                f"PNG plots); using '{mapped}'. Metrics are in the web monitor.")
            return mapped
        return mode

    def _init_tracking(self):
        self.episode_history = []
        self.cumulative_reward = 0.0
        self.agent_cumulative_rewards = {}
        # Track when all vehicles complete for accurate throughput calculation
        self.last_completion_step = None

        if not self.is_enabled():
            logger.info("Evaluation disabled - no history tracking")
            return
        logger.info("Evaluation summary mode - per-episode statistics, "
                    "mirrored to the metrics store")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    # The metrics-store mirror
    # --------------------------------------------------------------------- #

    def attach_run_store(self, store, *, algorithm_name: Optional[str] = None,
                         training: Optional[bool] = None) -> bool:
        """Mirror into ``store``, the run's own metrics store, instead of a
        separate evaluation run.

        Without it a run with a learner is listed twice in the monitor:
        once by the learner's store and once by this mirror. Attached, the
        ``Eval/*`` scalars, an ``evaluation_episode`` event per episode and
        the ``final_summary`` meta key go into ``store``. Its episode rows
        stay the learner's: both writers key rows by episode number, so one
        would overwrite the other. ``store`` stays its owner's and is never
        closed here. Call before the first episode ends; True when attached.

        ``algorithm_name`` (the registered learner key) and ``training`` (the
        run's ``training_mode``) are recorded even when ``store`` is None:
        they label the separate run that then follows, ``train`` only when a
        learner is named and trains.
        """
        if algorithm_name:
            self.algorithm_name = str(algorithm_name)
        if training is not None:
            self.learner_training = bool(training)
        if store is None or not self.is_enabled():
            return False
        if self._store is not None and not self._store_borrowed:
            # A separate run already holds episodes: keep writing there
            # rather than split one evaluation across two runs.
            logger.debug("evaluation: separate store already open, not attaching")
            return False
        self._store = store
        self._store_borrowed = True
        self._store_unavailable = False
        path = getattr(store, 'path', None)
        logger.info(f"Evaluation metrics mirrored into the run's store "
                    f"{path.parent if path is not None else store}")
        return True

    def _ensure_store(self):
        if self._store is not None or self._store_unavailable:
            return self._store
        metrics_root = self.config.get('metrics_root')
        if not metrics_root:
            self._store_unavailable = True
            logger.debug("evaluation: no metrics_root configured, "
                         "no monitor mirror for this run")
            return None
        try:
            from opencda_marl.telemetry.store import MetricsStore

            run_dir = (Path(metrics_root) / 'evaluation' /
                       f"{self.scenario_name or 'scenario'}_{self.agent_name}_{self._stamp}")
            meta = {
                'kind': 'evaluation',
                # The monitor's run list shows algorithm and mode: a learner
                # whose own store is disabled still trains, a baseline agent
                # or a frozen learner does not.
                'mode': 'train' if (self.learner_training and self.algorithm_name) else 'eval',
                'algorithm': self.algorithm_name or self.agent_name,
                'scenario': self.scenario_name,
                'agent': self.agent_name,
            }
            self._store = MetricsStore(run_dir, meta={k: v for k, v in meta.items() if v is not None})
            self._store_borrowed = False
            logger.info(f"Evaluation metrics mirrored to {run_dir}")
        except Exception as exc:  # noqa: BLE001 - the mirror must never kill a run
            self._store_unavailable = True
            logger.warning(f"evaluation metrics store unavailable: {exc}")
        return self._store

    @property
    def store_dir(self) -> Optional[Path]:
        """The metrics-store run directory, when the mirror is available."""
        path = getattr(self._ensure_store(), 'path', None)
        return path.parent if path is not None else None

    def _mirror_episode(self, episode: int, stat: Dict[str, Any]):
        store = self._ensure_store()
        if store is None:
            return
        summary = {k: v for k, v in stat.items()
                   if k not in ('rewards_by_agent', 'agent_cumulative_rewards')}
        try:
            for tag, key in (
                ('Eval/success_rate', 'success_rate'),
                ('Eval/collision_rate', 'collision_rate'),
                ('Eval/timeout_rate', 'timeout_rate'),
                ('Eval/throughput', 'throughput'),
                ('Eval/cumulative_reward', 'cumulative_reward'),
                ('Eval/avg_reward_per_agent', 'avg_reward_per_agent'),
                ('Eval/episode_length', 'episode_length'),
            ):
                value = stat.get(key)
                if value is not None:
                    store.log_scalar(tag, float(value), step=episode)
            if self._store_borrowed:
                store.log_event('evaluation_episode', dict(summary, episode=episode))
            else:
                store.log_episode(episode, summary)
            store.flush()
        except Exception as exc:  # noqa: BLE001 - the mirror must never kill a run
            # A borrowed store its owner already closed, a full disk: the CSV
            # and final_summary.json still get everything.
            logger.warning(f"evaluation metrics mirror stopped: {exc}")
            self._close_store()

    # --------------------------------------------------------------------- #
    # Core update methods
    # --------------------------------------------------------------------- #

    def update_step(self, metrics: Dict[str, Any], rewards: Dict[int, float] = None):
        """Remember the latest step snapshot and accumulate rewards."""
        if not self.is_enabled():
            return

        if self.fixed_dt is None:
            self.fixed_dt = metrics.get('fixed_dt')

        assert self.fixed_dt is not None, "fixed_dt is not set"

        # Remember the latest snapshot: end_episode() closes the episode from
        # it when the coordinator says the episode is over.
        self._last_metrics = metrics
        self._last_rewards = rewards

    def end_episode(self, metrics: Dict[str, Any] = None, rewards: Dict[int, float] = None,
                    *, final: bool = False):
        """Close the current episode: summary row, CSV, store mirror.

        Called by the coordinator when the episode actually ends. The manager
        does not infer the end from ``step == max_steps - 1``, which is wrong
        on both backends: CARLA's step counter is incremented before the
        metrics are assembled (so it would fire one step early, and the real
        last step would leak into the next episode's history), and the SUMO
        metrics carry no ``step``/``max_steps`` at all (so it would never
        fire). It also cannot see an episode the environment ended early.

        ``final`` additionally writes the whole-run summary and closes the
        store mirror.
        """
        if not self.is_enabled():
            return
        metrics = metrics if metrics is not None else getattr(self, '_last_metrics', None)
        rewards = rewards if rewards is not None else getattr(self, '_last_rewards', None)
        if metrics is None:
            if final:
                self.summary_evaluation()
                self._close_store()
            return
        if self.fixed_dt is None:
            self.fixed_dt = metrics.get('fixed_dt')
        current_episode = metrics.get('episode', len(self.episode_history))

        stat = self._step_statistics(metrics, rewards)
        self.episode_history.append(stat)
        self._save_episode_summary()
        self._mirror_episode(int(current_episode), stat)

        # Bound the in-memory history; the CSV and the store keep everything.
        if len(self.episode_history) > self.keep_episode_summaries:
            excess = len(self.episode_history) - self.keep_episode_summaries
            self.episode_history = self.episode_history[excess:]

        # Reset reward tracking for the next episode
        self.cumulative_reward = 0.0
        self.agent_cumulative_rewards.clear()
        self.last_completion_step = None
        self._last_metrics = None
        self._last_rewards = None

        if final:
            self.summary_evaluation()
            self._close_store()

    # --------------------------------------------------------------------- #
    # Private methods
    # --------------------------------------------------------------------- #

    def _step_statistics(self, metrics: Dict[str, Any], rewards: Dict[int, float] = None):
        # get current step metrics
        step = metrics.get('step', 0)
        success = metrics.get('success', 0)
        collision = metrics.get('collision', 0)
        active_agents = metrics.get('active_agents', 0)
        pending_spawns = metrics.get('pending_spawns', 0)
        total_vehicles = success + collision + active_agents

        # Track the step when ALL vehicles complete:
        # - No pending spawns (all vehicles have been spawned)
        # - No active agents (all spawned vehicles have finished)
        if pending_spawns == 0 and active_agents == 0 and (success > 0 or collision > 0):
            if self.last_completion_step is None:
                self.last_completion_step = step

        # Calculate rates including timeout vehicles for complete accountability
        success_rate = (success / total_vehicles *
                        100) if total_vehicles > 0 else 0
        collision_rate = (collision / total_vehicles *
                          100) if total_vehicles > 0 else 0
        timeout_rate = (active_agents / total_vehicles *
                       100) if total_vehicles > 0 else 0

        # throughput: how many vehicles have completed per hour
        # Use completion step if all vehicles finished, otherwise current step
        effective_step = self.last_completion_step if self.last_completion_step else step
        if effective_step > 0:
            vps = success / effective_step / self.fixed_dt
            throughput = vps * 3600
        else:
            throughput = 0.0

        # Calculate reward metrics
        step_reward = 0.0
        avg_reward_per_agent = 0.0
        rewards_by_agent = {}

        if rewards:
            step_reward = sum(rewards.values())
            self.cumulative_reward += step_reward

            # Update agent cumulative rewards
            for agent_id, reward in rewards.items():
                if agent_id not in self.agent_cumulative_rewards:
                    self.agent_cumulative_rewards[agent_id] = 0.0
                self.agent_cumulative_rewards[agent_id] += reward
                rewards_by_agent[agent_id] = reward

            # Calculate average reward per agent
            if len(rewards) > 0:
                avg_reward_per_agent = step_reward / len(rewards)

        stat = {
            'step': step,
            'success': success,
            'collision': collision,
            'active_agents': active_agents,
            'total_vehicles': total_vehicles,
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'timeout_rate': timeout_rate,
            'throughput': throughput,
            # Episode length: step when ALL vehicles completed (for consistent reporting)
            'episode_length': effective_step,  # Same as used in throughput calc
            'total_simulation_steps': step,    # Total steps for reference
            'step_reward': step_reward,
            'cumulative_reward': self.cumulative_reward,
            'avg_reward_per_agent': avg_reward_per_agent,
            'rewards_by_agent': rewards_by_agent,
            'agent_cumulative_rewards': dict(self.agent_cumulative_rewards)  # Copy for safety
        }

        return stat

    def _save_episode_summary(self):
        """Append the latest episode summary to the consolidated CSV."""
        if not self.episode_history:
            return

        csv_path = self.data_dir / "episodes_summary.csv"
        file_exists = csv_path.exists()

        with open(csv_path, 'a', newline='') as f:
            latest_episode = self.episode_history[-1]
            writer = csv.DictWriter(f, fieldnames=latest_episode.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(latest_episode)

        logger.debug(f"Latest episode summary saved: {csv_path}")

    # --------------------------------------------------------------------- #
    # Analysis and summary methods
    # --------------------------------------------------------------------- #

    def summary_evaluation(self):
        """Generate the final whole-run evaluation summary."""
        if not self.is_enabled():
            logger.info("Evaluation disabled - no summary generated")
            return

        if not self.episode_history:
            logger.warning("No episode data available for summary evaluation")
            return

        try:
            logger.info("=== FINAL EVALUATION SUMMARY ===")

            num_episodes = len(self.episode_history)
            success_rates = [ep['success_rate'] for ep in self.episode_history]
            collision_rates = [ep['collision_rate'] for ep in self.episode_history]
            timeout_rates = [ep.get('timeout_rate', 0.0) for ep in self.episode_history]
            throughputs = [ep['throughput'] for ep in self.episode_history]
            total_successes = [ep['success'] for ep in self.episode_history]
            total_collisions = [ep['collision'] for ep in self.episode_history]
            total_timeouts = [ep.get('active_agents', 0) for ep in self.episode_history]

            final_stats = {
                'scenario_name': self.scenario_name,
                'agent_name': self.agent_name,
                'total_episodes': num_episodes,
                'avg_success_rate': np.mean(success_rates),
                'std_success_rate': np.std(success_rates),
                'best_success_rate': np.max(success_rates),
                'worst_success_rate': np.min(success_rates),
                'avg_collision_rate': np.mean(collision_rates),
                'std_collision_rate': np.std(collision_rates),
                'best_collision_rate': np.min(collision_rates),  # Lower is better
                'worst_collision_rate': np.max(collision_rates),
                'avg_timeout_rate': np.mean(timeout_rates),
                'std_timeout_rate': np.std(timeout_rates),
                'best_timeout_rate': np.min(timeout_rates),  # Lower is better
                'worst_timeout_rate': np.max(timeout_rates),
                'avg_throughput': np.mean(throughputs),
                'std_throughput': np.std(throughputs),
                'best_throughput': np.max(throughputs),
                'worst_throughput': np.min(throughputs),
                'total_success': sum(total_successes),
                'total_collision': sum(total_collisions),
                'total_timeout': sum(total_timeouts),
                'total_vehicles': sum(total_successes) + sum(total_collisions) + sum(total_timeouts),
                'final_episode_success_rate': success_rates[-1],
                'final_episode_collision_rate': collision_rates[-1],
                'final_episode_timeout_rate': timeout_rates[-1],
                'final_episode_throughput': throughputs[-1]
            }

            if num_episodes > 1:
                success_trend = np.polyfit(range(num_episodes), success_rates, 1)[0]
                collision_trend = np.polyfit(range(num_episodes), collision_rates, 1)[0]
                throughput_trend = np.polyfit(range(num_episodes), throughputs, 1)[0]

                final_stats.update({
                    'success_rate_trend': success_trend,  # % per episode
                    'collision_rate_trend': collision_trend,  # % per episode
                    'throughput_trend': throughput_trend,  # vph per episode
                    'improvement_factor': success_rates[-1] / max(success_rates[0], 0.1)  # Avoid div by 0
                })

            logger.info(f"Scenario: {self.scenario_name} | Agent: {self.agent_name}")
            logger.info(f"Episodes Completed: {num_episodes}")
            logger.info(f"Success Rate: {final_stats['avg_success_rate']:.1f}% +/- {final_stats['std_success_rate']:.1f}% "
                       f"(Best: {final_stats['best_success_rate']:.1f}%)")
            logger.info(f"Collision Rate: {final_stats['avg_collision_rate']:.1f}% +/- {final_stats['std_collision_rate']:.1f}% "
                       f"(Best: {final_stats['best_collision_rate']:.1f}%)")
            logger.info(f"Timeout Rate: {final_stats['avg_timeout_rate']:.1f}% +/- {final_stats['std_timeout_rate']:.1f}% "
                       f"(Best: {final_stats['best_timeout_rate']:.1f}%)")
            logger.info(f"Throughput: {final_stats['avg_throughput']:.1f} +/- {final_stats['std_throughput']:.1f} vph "
                       f"(Best: {final_stats['best_throughput']:.1f} vph)")
            logger.info(f"Total Vehicles: {final_stats['total_vehicles']} "
                       f"({final_stats['total_success']} success, {final_stats['total_collision']} collision, "
                       f"{final_stats['total_timeout']} timeout)")

            if num_episodes > 1:
                logger.info(f"Trends: Success {final_stats['success_rate_trend']:+.2f}%/ep, "
                           f"Collision {final_stats['collision_rate_trend']:+.2f}%/ep, "
                           f"Throughput {final_stats['throughput_trend']:+.1f} vph/ep")
                logger.info(f"Overall Improvement: {final_stats['improvement_factor']:.2f}x")

            self._save_final_summary(final_stats)

            store = self._ensure_store()
            if store is not None:
                try:
                    store.set_meta(final_summary=json.loads(
                        json.dumps(final_stats, default=float)))
                    store.flush()
                except Exception as exc:  # noqa: BLE001 - the file above has it
                    logger.warning(f"evaluation final summary not mirrored: {exc}")

            logger.info("=== EVALUATION COMPLETE ===")

        except Exception as e:
            logger.error(f"Failed to generate summary evaluation: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _save_final_summary(self, summary_stats: dict):
        """Save final summary statistics to file"""
        try:
            summary_stats['generated_at'] = datetime.now().isoformat()
            json_path = self.data_dir / "final_summary.json"
            with open(json_path, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=float)
            logger.info(f"Final summary saved: {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save final summary: {e}")

    def _close_store(self):
        if self._store is not None and not self._store_borrowed:
            try:
                self._store.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"evaluation store close: {exc}")
        self._release_store()

    def _release_store(self):
        """Drop the mirror for the rest of the run (a borrowed store is
        left open: its owner closes it)."""
        self._store = None
        self._store_borrowed = False
        self._store_unavailable = True

    # --------------------------------------------------------------------- #
    # Utility methods
    # --------------------------------------------------------------------- #

    def is_enabled(self) -> bool:
        """Check if evaluation is enabled"""
        return self.get_mode() != 'disabled'

    def get_mode(self) -> str:
        """Get current evaluation mode"""
        return self.mode

    def reset_config(self, new_config: Dict):
        """Update configuration and reinitialize if needed"""
        self.config = new_config
        old_mode = self.mode
        self.mode = self._resolve_mode(self.config.get('mode', 'summary'))

        if old_mode != self.mode:
            logger.info(
                f"Evaluation mode changed from {old_mode} to {self.mode}")
            self._init_tracking()
