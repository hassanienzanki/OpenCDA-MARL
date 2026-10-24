"""Training-time logging and convergence tracking for algorithms.

Extracted from ``BaseAlgorithm`` so the algorithm contract stays small: this
mixin owns the TensorBoard writer lifecycle (segment rotation, retention),
category-gated scalar/histogram logging, per-episode metric logging, and the
reward/convergence bookkeeping. Every attribute keeps its historical name
(``writer``, ``tb_*``, ``reward_history`` ...) so checkpoints, the world-reload
snapshot, and existing tests are unaffected. Since it owns the run-output
config blocks, it also owns the copy of the config a checkpoint stores
without them (:func:`portable_config`).

Any registered policy that subclasses ``BaseAlgorithm`` gets all of this for
free; a policy that does not can still use :class:`TrainingLogger` directly.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from datetime import datetime
import os
from pathlib import PureWindowsPath
import shutil
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

# TensorBoard for training visualization
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
    TENSORBOARD_AVAILABLE = False
    logger.warning(f"TensorBoard not available. Install with: pip install tensorboard. Details: {exc}")


#: The config blocks that locate one run's outputs. MARLManager merges them
#: into every learner's config from ``outputs.*``: directories under
#: ``outputs.root_dir``, the run name, the scenario label. They say where a
#: run wrote, not how the learner learns, so a checkpoint leaves them out.
RUN_OUTPUT_BLOCKS = ('tensorboard', 'metrics_store')

#: What a checkpoint's config holds in place of an absolute filesystem path.
ABSOLUTE_PATH_PLACEHOLDER = '<absolute path removed>'


def _plain(node: Any) -> Any:
    """``node`` as plain dicts and lists when it is an OmegaConf container.

    Interpolations are kept as their ``${...}`` text, as the profile wrote
    them: resolving one would bring in values from the rest of the run
    config or from the machine (``oc.env``), which may be paths or names
    inside a longer string that no path check recognises.
    """
    try:
        from omegaconf import Container, OmegaConf
    except ImportError:  # pragma: no cover - omegaconf ships with the framework
        return node
    if not isinstance(node, Container):
        return node
    return OmegaConf.to_container(node, resolve=False)


def _is_absolute_path(value: Any) -> bool:
    """Whether ``value`` is a rooted filesystem path in either OS's syntax:
    ``/x``, ``\\x``, ``C:/x``, ``//host/share``, ``\\\\host\\share``, or a
    ``file:`` URL of one. Relative and ``~`` paths and network URLs are not."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return False
    if value[:5].lower() == 'file:':
        value = value[5:]
    # Windows parsing reads both separators; a bare root names nothing.
    return bool(PureWindowsPath(value).root) and bool(value.strip('/\\'))


def _scrub_paths(value: Any) -> Any:
    # Every node, not just the top: the manager hands a learner a plain dict
    # whose list and dict values are still OmegaConf nodes, and each one
    # keeps a reference to the whole run config and would pickle all of it.
    value = _plain(value)
    if isinstance(value, Mapping):
        return {key: _scrub_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_scrub_paths(item) for item in value]
        return items if isinstance(value, list) else tuple(items)
    if _is_absolute_path(value):
        return ABSOLUTE_PATH_PLACEHOLDER
    return value


def portable_config(config: Any) -> Any:
    """A copy of a learner's config that is safe to ship inside a checkpoint.

    A checkpoint pickles the config it was trained with, and that config
    carries the run's output locations, so a shared checkpoint would
    disclose the directory tree of the machine that trained it. The copy
    drops the :data:`RUN_OUTPUT_BLOCKS` and replaces every remaining
    string or path value that is an absolute filesystem path with
    :data:`ABSOLUTE_PATH_PLACEHOLDER`; every other key, hyperparameters
    included, is kept as configured. A path is replaced rather than dropped
    so the copy keeps the shape the learner was built from (every key and
    list position the learner read is still there, and a reader can see
    that a path was configured), and code that reads the value as a
    location fails on the placeholder instead of silently falling back to a
    default.

    OmegaConf containers, nested ones included, come back as plain dicts
    and lists with their interpolations unresolved; the live config is
    never modified.
    """
    config = _plain(config)
    if isinstance(config, Mapping):
        config = {key: item for key, item in config.items()
                  if key not in RUN_OUTPUT_BLOCKS}
    return _scrub_paths(config)


class TrainingLogger:
    """Mixin: TensorBoard logging + convergence tracking for a learner.

    Expects the host to define ``config``, ``training_step`` and
    ``episode_count`` before :meth:`_init_training_logger` is called.
    """

    _TB_MODE_DEFAULTS = {
        'episode_first': {
            'losses': 500,
            'q_values': None,
            'buffer': 500,
            'rewards': 500,
            'episode': 1,
            'histograms': None,
        },
        'balanced': {
            'losses': 100,
            'q_values': 250,
            'buffer': 250,
            'rewards': 100,
            'episode': 1,
            'histograms': None,
        },
        'full': {
            'losses': 1,
            'q_values': 1,
            'buffer': 1,
            'rewards': 1,
            'episode': 1,
            'histograms': 1,
        },
    }

    def _init_training_logger(self, config: Dict[str, Any]) -> None:
        """Set up reward/convergence tracking and the TensorBoard writer."""
        # Reward tracking for convergence analysis
        self.reward_window_size = config.get('reward_window_size', 10)
        self.reward_history: deque = deque(maxlen=100)  # Full history (last 100 episodes)
        self.episode_length_history: deque = deque(maxlen=100)

        # Success/collision tracking for MARL traffic convergence
        self.success_rate_history: deque = deque(maxlen=100)
        self.collision_rate_history: deque = deque(maxlen=100)

        # Convergence detection parameters (more lenient for MARL traffic scenarios)
        self.convergence_threshold = config.get('convergence_threshold', 0.15)  # 15% CV threshold
        self.convergence_window = config.get('convergence_window', 10)  # Episodes to check
        self.min_episodes_for_convergence = config.get('min_episodes_for_convergence', 20)
        self.is_converged = False
        self.convergence_episode = None  # Episode when convergence was detected

        self._init_tensorboard(config)
        self._init_metrics_store(config)

    def checkpoint_config(self) -> Any:
        """The config this learner's checkpoints store: ``self.config``
        without its run-output blocks and absolute paths (see
        :func:`portable_config`). Every built-in learner's ``save`` writes
        this instead of ``self.config``; no loader reads it back."""
        return portable_config(self.config)

    def _run_algorithm_name(self, config: Dict[str, Any]) -> str:
        """The name that keys this learner's TensorBoard and metrics-store
        run directories.

        It is the registered algorithm key (``metrics_store.algorithm``,
        which MARLManager sets): ``masac`` and ``pcmasac`` share one
        learner class but are different runs. A learner built without the
        manager falls back to its class name (``SACAlgorithm`` -> ``sac``).
        Both writers resolve the name here so their directories cannot
        disagree.
        """
        ms_config = config.get('metrics_store')
        if isinstance(ms_config, Mapping) and ms_config.get('algorithm'):
            return str(ms_config['algorithm'])
        return self.__class__.__name__.lower().replace('algorithm', '')

    def _init_metrics_store(self, config: Dict[str, Any]):
        """Open the run's SQLite metrics store (the TensorBoard replacement).

        Config key ``metrics_store`` (merged by MARLManager from
        ``outputs.metrics_store``): ``enabled`` (default True), ``log_dir``
        (default ``<outputs.root_dir>/metrics``), ``run_name`` (the launcher
        sets the cell slug; default a timestamp). The file lands at
        ``<log_dir>/<algo>/<run_name>/metrics.sqlite``, where ``<algo>`` is
        :meth:`_run_algorithm_name`. Stays enabled in evaluation mode
        -- evaluation episodes are exactly what the monitor is for -- with
        ``mode`` recorded in the run metadata.
        """
        self.metrics_store = None
        self.ms_log_frequency = 10
        ms_config = config.get('metrics_store', {})
        if isinstance(ms_config, bool):
            ms_config = {'enabled': ms_config}
        if not ms_config.get('enabled', True):
            return
        self.ms_log_frequency = max(1, int(ms_config.get('log_frequency', 10)))
        try:
            from opencda_marl.telemetry import MetricsStore
            base_dir = ms_config.get('log_dir', 'metrics')
            algo_name = self._run_algorithm_name(config)
            run_name = ms_config.get('run_name') or datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(base_dir, algo_name, str(run_name))
            meta = {
                'algorithm': algo_name,
                'run_name': str(run_name),
                'mode': 'train' if config.get('training_mode', True) else 'eval',
                'scenario': ms_config.get('scenario'),
                'config': ms_config.get('config_snapshot'),
            }
            self.metrics_store = MetricsStore(run_dir, meta={k: v for k, v in meta.items() if v is not None})
            self.metrics_store_dir = run_dir
            logger.info(f"Metrics store: {os.path.join(run_dir, 'metrics.sqlite')} "
                        f"(view: python -m opencda_marl.telemetry.monitor --root {base_dir})")
        except Exception as exc:  # noqa: BLE001 - telemetry must never block training
            logger.warning(f"Metrics store disabled: {exc}")
            self.metrics_store = None

    def _init_tensorboard(self, config: Dict[str, Any]):
        """Initialize TensorBoard logging based on configuration.

        Event files land under ``<log_dir>/<algo>/<run_name>/``, where
        ``<algo>`` is :meth:`_run_algorithm_name`, the same name that keys
        the metrics store.
        """
        # Get tensorboard config (can be nested under algorithm or at root level)
        tb_config = config.get('tensorboard', {})
        if isinstance(tb_config, bool):
            # Handle simple boolean config
            tb_config = {'enabled': tb_config}

        self.tb_mode = tb_config.get('mode', 'full')
        if self.tb_mode not in self._TB_MODE_DEFAULTS:
            logger.warning(f"Unknown TensorBoard mode '{self.tb_mode}', falling back to 'full'")
            self.tb_mode = 'full'

        self.tb_enabled = tb_config.get('enabled', True) and TENSORBOARD_AVAILABLE
        self.writer: Optional[SummaryWriter] = None
        self.tensorboard_dir: Optional[str] = None
        self.tensorboard_root_dir: Optional[str] = None

        default_metrics = {
            'losses': True,
            'q_values': True,
            'buffer': True,
            'episode': True,
            'rewards': True,
            'histograms': False,
        }
        configured_metrics = tb_config.get('metrics', {})
        if isinstance(configured_metrics, dict):
            default_metrics.update(configured_metrics)
        self.tb_metrics = default_metrics
        self.tb_log_frequency = max(1, int(tb_config.get('log_frequency', 1)))
        self.tb_category_frequencies = self._build_tb_category_frequencies(self.tb_mode, self.tb_log_frequency)
        self.tb_histograms_enabled = bool(self.tb_metrics.get('histograms', False)) and \
            self.tb_category_frequencies.get('histograms') is not None
        self.tb_max_episodes_per_run = max(0, int(tb_config.get('max_episodes_per_run', 0)))
        self.tb_keep_last_runs = max(0, int(tb_config.get('keep_last_runs', 0)))
        self.tb_run_index = 0
        self.tb_logged_episodes_in_run = 0
        self.tb_run_name: Optional[str] = None

        if self.tb_enabled:
            # Configurable log directory
            base_dir = tb_config.get('log_dir', 'runs')
            algo_name = self._run_algorithm_name(config)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Allow custom run name
            self.tb_run_name = tb_config.get('run_name', timestamp)
            self.tensorboard_root_dir = os.path.join(base_dir, algo_name, self.tb_run_name)
            self._open_tensorboard_writer()
            logger.info(f"TensorBoard logging enabled: {self.tensorboard_dir} (mode={self.tb_mode})")
            logger.info(f"View with: tensorboard --logdir={base_dir}")

    def _build_tb_category_frequencies(self, mode: str, base_frequency: int) -> Dict[str, Optional[int]]:
        frequencies = {}
        for category, mode_frequency in self._TB_MODE_DEFAULTS[mode].items():
            if mode_frequency is None:
                frequencies[category] = None
            elif category == 'episode':
                # Episode scalars are stepped by episode number, not by training
                # step: the per-step log_frequency would thin Episode/* to one
                # point per log_frequency episodes.
                frequencies[category] = mode_frequency
            else:
                frequencies[category] = max(base_frequency, mode_frequency)
        return frequencies

    def _open_tensorboard_writer(self):
        if not self.tb_enabled or not self.tensorboard_root_dir:
            return

        if self.tb_max_episodes_per_run > 0:
            run_dir = os.path.join(
                self.tensorboard_root_dir,
                f"segment_{self.tb_run_index:04d}",
            )
        else:
            run_dir = self.tensorboard_root_dir

        os.makedirs(run_dir, exist_ok=True)
        self.tensorboard_dir = run_dir
        self.writer = SummaryWriter(log_dir=run_dir)

    def _store_should_log(self, category: Optional[str], step: Optional[int]) -> bool:
        """The store keeps every episode-level scalar and per-step scalars every
        ``metrics_store.log_frequency`` steps (default 10), independent of the
        TensorBoard mode -- the store is cheap and queryable, TensorBoard's
        category policy exists because TensorBoard is not."""
        if getattr(self, 'metrics_store', None) is None:
            return False
        if category in (None, 'episode', 'rewards'):
            return True
        stride = max(1, int(getattr(self, 'ms_log_frequency', 10)))
        if step is None:
            step = self.training_step
        return step % stride == 0

    def _tb_should_log(self, category: Optional[str], step: Optional[int]) -> bool:
        if self.writer is None:
            return False

        if category is None:
            category = 'losses'

        if not self.tb_metrics.get(category, True):
            return False

        frequency = self.tb_category_frequencies.get(category, self.tb_log_frequency)
        if frequency is None:
            return False

        if step is None:
            step = self.training_step

        return step % frequency == 0

    def _should_log_category(self, category: Optional[str], step: Optional[int]) -> bool:
        """True when at least one sink wants this scalar."""
        return self._tb_should_log(category, step) or self._store_should_log(category, step)

    def _rotate_tensorboard_writer(self, reason: str):
        if self.writer is None or not self.tb_enabled or self.tb_max_episodes_per_run <= 0:
            return

        self.writer.flush()
        self.writer.close()
        self.writer = None

        self.tb_run_index += 1
        self.tb_logged_episodes_in_run = 0
        self._open_tensorboard_writer()
        self._cleanup_old_tensorboard_runs()
        logger.info(f"TensorBoard rotated to {self.tensorboard_dir} ({reason})")

    def _cleanup_old_tensorboard_runs(self):
        if not self.tensorboard_root_dir or self.tb_keep_last_runs <= 0:
            return

        try:
            segment_dirs = sorted(
                entry for entry in os.listdir(self.tensorboard_root_dir)
                if entry.startswith('segment_')
                and os.path.isdir(os.path.join(self.tensorboard_root_dir, entry))
            )
            excess = len(segment_dirs) - self.tb_keep_last_runs
            for segment_name in segment_dirs[:max(excess, 0)]:
                shutil.rmtree(os.path.join(self.tensorboard_root_dir, segment_name), ignore_errors=True)
        except OSError as exc:
            logger.warning(f"Failed to cleanup old TensorBoard runs: {exc}")

    def finalize_episode_logging(self):
        if self.writer is None:
            return

        self.tb_logged_episodes_in_run += 1
        if self.tb_max_episodes_per_run > 0 and \
                self.tb_logged_episodes_in_run >= self.tb_max_episodes_per_run:
            self._rotate_tensorboard_writer("episode window reached")

    def log_scalar(self, tag: str, value: float, step: int = None, category: str = None):
        """
        Log a scalar value to TensorBoard.

        Args:
            tag: Metric name
            value: Scalar value
            step: Global step (defaults to training_step)
            category: Optional category to check if logging is enabled
        """
        if not self._should_log_category(category, step):
            return

        if step is None:
            step = self.training_step

        if self._tb_should_log(category, step):
            self.writer.add_scalar(tag, value, step)
        if self._store_should_log(category, step):
            self.metrics_store.log_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float],
                    step: int = None, category: str = None):
        """
        Log multiple scalars under a main tag.

        Args:
            main_tag: Main category tag
            tag_scalar_dict: Dictionary of {tag: value}
            step: Global step
            category: Optional category to check if logging is enabled
        """
        if not self._should_log_category(category, step):
            return

        if step is None:
            step = self.training_step

        if self._tb_should_log(category, step):
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        if self._store_should_log(category, step):
            self.metrics_store.log_scalars(tag_scalar_dict, step, prefix=f"{main_tag}/")

    def log_histogram(self, tag: str, values: np.ndarray, step: int = None):
        """Log a histogram of values."""
        if not self.tb_histograms_enabled:
            return

        if step is None:
            step = self.training_step

        if not self._should_log_category('histograms', step):
            return

        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def _log_episode_scalar(self, tag: str, value: float):
        """Log an episode-scoped scalar using the shared category policy."""
        self.log_scalar(tag, value, step=self.episode_count, category='episode')

    def log_episode_metrics(self, episode_reward: float, episode_length: int,
                           success_rate: float = 0.0, collision_rate: float = 0.0,
                           near_miss_count: int = 0,
                           ttc_violation_rate: float = 0.0,
                           additional_metrics: Dict[str, float] = None,
                           traffic_metrics: Dict[str, float] = None):
        """
        Log episode-level metrics to TensorBoard with learning quality analysis.

        Args:
            episode_reward: Total episode reward
            episode_length: Number of steps in episode
            success_rate: Success rate (0-1)
            collision_rate: Collision rate (0-1)
            near_miss_count: Number of near-miss events (TTC < threshold without collision)
            ttc_violation_rate: Percentage of TTC checks that were below safe threshold
            additional_metrics: Additional custom metrics to log
            traffic_metrics: Traffic performance metrics (avg_speed, speed_variance, etc.)
        """
        # Track reward and episode length history (always track, even without TensorBoard)
        self.reward_history.append(episode_reward)
        self.episode_length_history.append(episode_length)

        # Track success/collision rates for MARL traffic convergence analysis
        self.success_rate_history.append(success_rate)
        self.collision_rate_history.append(collision_rate)

        # Compute learning quality metrics
        reward_ma, reward_var, reward_std = self._compute_reward_statistics()
        #length_ma = self._compute_episode_length_ma()

        # Check for convergence (MARL-aware)
        self._check_convergence()

        if not self._should_log_category('episode', self.episode_count):
            return

        # Core episode metrics
        self._log_episode_scalar('Episode/reward', episode_reward)
        self._log_episode_scalar('Episode/success_rate', success_rate)
        self._log_episode_scalar('Episode/collision_rate', collision_rate)
        self._log_episode_scalar('Episode/length', episode_length)

        # Safety metrics: Near-miss count and TTC violation rate
        # Decreasing near-misses = agent learning to avoid dangerous situations
        self._log_episode_scalar('Safety/near_miss_count', near_miss_count)
        # TTC violation rate: % of TTC checks with TTC < safe threshold (lower = safer)
        self._log_episode_scalar('Safety/ttc_violation_rate', ttc_violation_rate)

        # Learning quality metrics (plot-ready)
        # Note: episode_length_ma removed - not useful for fixed simulation length
        self._log_episode_scalar('Learning/reward_moving_avg', reward_ma)
        self._log_episode_scalar('Learning/reward_variance', reward_var)
        self._log_episode_scalar('Learning/reward_std', reward_std)
        self._log_episode_scalar('Learning/converged', 1.0 if self.is_converged else 0.0)

        # Normalized coefficient of variation (for convergence visualization)
        if reward_ma != 0:
            cv = reward_std / abs(reward_ma)  # Coefficient of variation
            self._log_episode_scalar('Learning/reward_cv', cv)

        # Traffic performance metrics
        if traffic_metrics:
            # ============ KEY METRICS (most important for monitoring) ============
            # avg_speed: Actual average vehicle speed from CARLA (km/h)
            if 'avg_speed' in traffic_metrics:
                self._log_episode_scalar('Traffic/avg_speed', traffic_metrics['avg_speed'])

            # target_speed_mean: What RL algorithm commanded vehicles to do (km/h)
            if 'target_speed_mean' in traffic_metrics:
                self._log_episode_scalar('Traffic/target_speed_mean', traffic_metrics['target_speed_mean'])

            # speed_gap: Difference between commanded and actual speed (target - actual)
            # Positive = vehicles slower than commanded, Negative = vehicles faster
            avg_speed = traffic_metrics.get('avg_speed', 0)
            target_speed = traffic_metrics.get('target_speed_mean', 0)
            if target_speed > 0 and avg_speed > 0:
                speed_gap = target_speed - avg_speed
                self._log_episode_scalar('Traffic/speed_gap', speed_gap)

            # ============ SECONDARY METRICS (for detailed analysis) ============
            # max_speed: Fastest actual vehicle speed in episode (km/h)
            if 'max_speed' in traffic_metrics:
                self._log_episode_scalar('Traffic/max_speed', traffic_metrics['max_speed'])
            # min_speed: Slowest actual vehicle speed in episode (km/h)
            if 'min_speed' in traffic_metrics:
                self._log_episode_scalar('Traffic/min_speed', traffic_metrics['min_speed'])
            # speed_std: Standard deviation of actual speeds (lower = more uniform)
            if 'speed_std' in traffic_metrics:
                self._log_episode_scalar('Traffic/speed_std', traffic_metrics['speed_std'])

            # target_speed_max/min: Range of RL commanded speeds
            if 'target_speed_max' in traffic_metrics:
                self._log_episode_scalar('Traffic/target_speed_max', traffic_metrics['target_speed_max'])
            if 'target_speed_min' in traffic_metrics:
                self._log_episode_scalar('Traffic/target_speed_min', traffic_metrics['target_speed_min'])
            # throughput: Vehicles completing per hour (success rate efficiency)
            if 'throughput' in traffic_metrics:
                self._log_episode_scalar('Traffic/throughput', traffic_metrics['throughput'])

        # Log additional custom metrics
        if additional_metrics:
            for name, value in additional_metrics.items():
                self._log_episode_scalar(f'Episode/{name}', value)

        # ============ IMPROVEMENT METRICS (for plot-ready visualization) ============
        # These show relative improvement over training, better for long training runs
        self._log_improvement_metrics(success_rate, collision_rate, traffic_metrics)

        store = getattr(self, 'metrics_store', None)
        if store is not None:
            summary = {
                'reward': float(episode_reward),
                'length': int(episode_length),
                'success_rate': float(success_rate),
                'collision_rate': float(collision_rate),
                'near_miss_count': int(near_miss_count),
                'ttc_violation_rate': float(ttc_violation_rate),
                'training_step': int(self.training_step),
            }
            if additional_metrics:
                summary.update({str(k): v for k, v in additional_metrics.items()})
            if traffic_metrics:
                summary.update({f'traffic/{k}': v for k, v in traffic_metrics.items()})
            store.log_episode(self.episode_count, summary)
            store.flush()

        # Flush periodically (every 10 episodes) instead of every episode
        # Reduces I/O blocking during training
        if self.writer is not None and self.episode_count % 10 == 0:
            self.writer.flush()

    def flush_tensorboard(self):
        """Flush every metrics sink."""
        if self.writer is not None:
            self.writer.flush()
        store = getattr(self, 'metrics_store', None)
        if store is not None:
            store.flush()

    def close(self):
        """Close the metrics sinks and cleanup resources."""
        if self.writer is not None:
            self.writer.close()
            logger.info("TensorBoard writer closed")
        store = getattr(self, 'metrics_store', None)
        if store is not None:
            try:
                store.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"metrics store close failed: {exc}")
            self.metrics_store = None

    # ------------------------------------------------------------------ #
    # Learning Quality Analysis Methods
    # ------------------------------------------------------------------ #

    def _compute_reward_statistics(self) -> tuple:
        """
        Compute reward moving average, variance, and standard deviation.

        Returns:
            (moving_average, variance, std_deviation)
        """
        if len(self.reward_history) == 0:
            return 0.0, 0.0, 0.0

        # Use last N episodes for moving average
        window = list(self.reward_history)[-self.reward_window_size:]
        reward_ma = float(np.mean(window))
        reward_var = float(np.var(window))
        reward_std = float(np.std(window))

        return reward_ma, reward_var, reward_std

    def _compute_episode_length_ma(self) -> float:
        """Compute moving average of episode lengths."""
        if len(self.episode_length_history) == 0:
            return 0.0

        window = list(self.episode_length_history)[-self.reward_window_size:]
        return float(np.mean(window))

    def _check_convergence(self):
        """
        Check if training has converged based on reward and performance stability.

        MARL Traffic Convergence Criteria:
        1. Minimum episodes reached (default: 20)
        2. Reward CV (coefficient of variation) below threshold (default: 15%)
        3. Success rate stabilized (CV < 20%)
        4. Collision rate decreasing or stable

        This approach is more suitable for MARL traffic scenarios where:
        - Early episodes have high variance due to exploration
        - Success/collision rates are key performance indicators
        """
        if self.is_converged:
            return  # Already converged

        # Need minimum episodes before checking convergence
        if self.episode_count < self.min_episodes_for_convergence:
            return

        if len(self.reward_history) < self.convergence_window:
            return  # Not enough data

        # Get recent metrics
        recent_rewards = list(self.reward_history)[-self.convergence_window:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        # Avoid division by zero
        if abs(mean_reward) < 1e-8:
            return

        # Coefficient of variation for rewards
        reward_cv = std_reward / abs(mean_reward)

        # Check success rate stability (if tracked)
        success_rate_stable = True
        if len(self.success_rate_history) >= self.convergence_window:
            recent_success = list(self.success_rate_history)[-self.convergence_window:]
            mean_success = np.mean(recent_success)
            if mean_success > 0.01:  # Only check if there are meaningful successes
                success_cv = np.std(recent_success) / mean_success
                success_rate_stable = success_cv < 0.20  # 20% CV for success rate

        # Check collision rate trend (should be decreasing or stable)
        collision_improving = True
        if len(self.collision_rate_history) >= self.convergence_window:
            recent_collision = list(self.collision_rate_history)[-self.convergence_window:]
            # Compare first half to second half
            first_half = np.mean(recent_collision[:self.convergence_window//2])
            second_half = np.mean(recent_collision[self.convergence_window//2:])
            collision_improving = second_half <= first_half * 1.1  # Allow 10% tolerance

        # Check convergence condition
        reward_converged = reward_cv < self.convergence_threshold
        if reward_converged and success_rate_stable and collision_improving:
            self.is_converged = True
            self.convergence_episode = self.episode_count
            logger.success(f"Convergence detected at episode {self.episode_count}! "
                         f"Reward CV={reward_cv:.4f}, Success stable={success_rate_stable}, "
                         f"Collision improving={collision_improving}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics for reporting.

        Returns:
            Dictionary with learning quality metrics
        """
        reward_ma, reward_var, reward_std = self._compute_reward_statistics()
        length_ma = self._compute_episode_length_ma()

        stats = {
            'episode_count': self.episode_count,
            'reward_moving_avg': reward_ma,
            'reward_variance': reward_var,
            'reward_std': reward_std,
            'episode_length_ma': length_ma,
            'is_converged': self.is_converged,
            'convergence_episode': self.convergence_episode,
            'reward_history_size': len(self.reward_history),
        }

        # Add coefficient of variation if mean is non-zero
        if reward_ma != 0:
            stats['coefficient_of_variation'] = reward_std / abs(reward_ma)

        # Add full reward history for plotting
        if len(self.reward_history) > 0:
            stats['reward_history'] = list(self.reward_history)
            stats['episode_length_history'] = list(self.episode_length_history)

        return stats

    def _log_improvement_metrics(self, success_rate: float, collision_rate: float,
                                  traffic_metrics: Dict[str, float] = None):
        """
        Log improvement metrics for plot-ready visualization.

        These metrics show relative improvement over training, which is more meaningful
        than raw values for long training runs (100K+ episodes).

        Metrics logged:
        - Windowed moving averages (100-episode window)
        - Delta from baseline (improvement from first 100 episodes)
        - Percentage improvement from baseline
        - Time-to-target tracking (episodes to reach X% success rate)
        """
        if self.writer is None and getattr(self, 'metrics_store', None) is None:
            return

        window_size = 100  # Use 100-episode windows for smoother curves

        # Track success/collision rate moving averages with larger window
        if len(self.success_rate_history) >= window_size:
            # 100-episode moving average (smoother than 10-episode)
            recent_success = list(self.success_rate_history)[-window_size:]
            recent_collision = list(self.collision_rate_history)[-window_size:]

            success_ma_100 = float(np.mean(recent_success))
            collision_ma_100 = float(np.mean(recent_collision))

            self._log_episode_scalar('Improvement/success_rate_ma100', success_ma_100)
            self._log_episode_scalar('Improvement/collision_rate_ma100', collision_ma_100)

            # Calculate baseline from first 100 episodes
            baseline_success = list(self.success_rate_history)[:window_size]
            baseline_collision = list(self.collision_rate_history)[:window_size]

            baseline_success_avg = float(np.mean(baseline_success))
            baseline_collision_avg = float(np.mean(baseline_collision))

            # Absolute improvement (delta from baseline)
            success_delta = success_ma_100 - baseline_success_avg
            collision_delta = collision_ma_100 - baseline_collision_avg

            self._log_episode_scalar('Improvement/success_rate_delta', success_delta)
            self._log_episode_scalar('Improvement/collision_rate_delta', collision_delta)

            # Percentage improvement from baseline
            # Formula: (current - baseline) / baseline * 100
            if baseline_success_avg > 1.0:  # Avoid division by near-zero
                success_improvement_pct = (success_ma_100 - baseline_success_avg) / baseline_success_avg * 100
                self._log_episode_scalar('Improvement/success_improvement_pct', success_improvement_pct)

            if baseline_collision_avg > 1.0:
                # For collision, negative improvement is good (reduction)
                collision_reduction_pct = (baseline_collision_avg - collision_ma_100) / baseline_collision_avg * 100
                self._log_episode_scalar('Improvement/collision_reduction_pct', collision_reduction_pct)

        # Track throughput improvement
        if traffic_metrics and 'throughput' in traffic_metrics:
            # Store throughput history for improvement calculation
            if not hasattr(self, 'throughput_history'):
                self.throughput_history = []
            self.throughput_history.append(traffic_metrics['throughput'])

            if len(self.throughput_history) >= window_size:
                recent_throughput = self.throughput_history[-window_size:]
                baseline_throughput = self.throughput_history[:window_size]

                throughput_ma_100 = float(np.mean(recent_throughput))
                baseline_throughput_avg = float(np.mean(baseline_throughput))

                self._log_episode_scalar('Improvement/throughput_ma100', throughput_ma_100)

                if baseline_throughput_avg > 0:
                    throughput_improvement_pct = (throughput_ma_100 - baseline_throughput_avg) / baseline_throughput_avg * 100
                    self._log_episode_scalar('Improvement/throughput_improvement_pct', throughput_improvement_pct)

        # Time-to-target tracking: log when key milestones are reached
        # Track first episode to reach 70%, 80%, 90% success rate
        if not hasattr(self, '_milestone_episodes'):
            self._milestone_episodes = {70: None, 80: None, 90: None, 95: None}

        for target_pct, first_episode in self._milestone_episodes.items():
            if first_episode is None and success_rate >= target_pct:
                self._milestone_episodes[target_pct] = self.episode_count
                self._log_episode_scalar(
                    f'Milestone/episodes_to_{target_pct}pct_success',
                    self.episode_count,
                )
                logger.info(f"Milestone reached: {target_pct}% success rate at episode {self.episode_count}")


__all__ = ["TrainingLogger", "TENSORBOARD_AVAILABLE", "RUN_OUTPUT_BLOCKS",
           "ABSOLUTE_PATH_PLACEHOLDER", "portable_config"]
