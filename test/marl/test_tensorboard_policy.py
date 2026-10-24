from pathlib import Path

import opencda_marl.builtin.algorithms.training_logger as base_algorithm_module  # writer lives in the TrainingLogger mixin
from opencda_marl.builtin.algorithms.base_algorithm import BaseAlgorithm


class _FakeWriter:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "events.fake").write_text("created", encoding="utf-8")

    def add_scalar(self, tag, value, step):
        return None

    def add_scalars(self, main_tag, tag_scalar_dict, step):
        return None

    def add_histogram(self, tag, values, step):
        return None

    def flush(self):
        return None

    def close(self):
        return None


class _DummyAlgorithm(BaseAlgorithm):
    def select_action(self, state, training=True):
        return 0

    def store_transition(self, state, action, reward, next_state, done):
        return None

    def update(self):
        return {}

    def reset_episode(self):
        self.episode_count += 1
        self.finalize_episode_logging()

    def get_training_info(self):
        return {}

    def save(self, path: str):
        return None

    def load(self, path: str):
        return None


def test_episode_first_rotation_keeps_only_recent_segment_dirs(tmp_path):
    original_writer = base_algorithm_module.SummaryWriter
    original_available = base_algorithm_module.TENSORBOARD_AVAILABLE
    base_algorithm_module.SummaryWriter = _FakeWriter
    base_algorithm_module.TENSORBOARD_AVAILABLE = True

    try:
        algo = _DummyAlgorithm(
            {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": str(tmp_path / "runs"),
                    "mode": "episode_first",
                    "max_episodes_per_run": 2,
                    "keep_last_runs": 2,
                }
            },
            state_dim=4,
            action_dim=1,
        )

        for episode_index in range(5):
            algo.log_episode_metrics(
                episode_reward=float(episode_index),
                episode_length=20,
            )
            algo.reset_episode()

        segment_names = sorted(
            path.name for path in Path(algo.tensorboard_root_dir).iterdir() if path.is_dir()
        )

        assert algo.tb_mode == "episode_first"
        assert algo.tb_category_frequencies["q_values"] is None
        assert segment_names == ["segment_0001", "segment_0002"]
    finally:
        base_algorithm_module.SummaryWriter = original_writer
        base_algorithm_module.TENSORBOARD_AVAILABLE = original_available


def test_episode_first_disables_histograms_even_when_metric_requested(tmp_path):
    original_writer = base_algorithm_module.SummaryWriter
    original_available = base_algorithm_module.TENSORBOARD_AVAILABLE
    base_algorithm_module.SummaryWriter = _FakeWriter
    base_algorithm_module.TENSORBOARD_AVAILABLE = True

    try:
        algo = _DummyAlgorithm(
            {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": str(tmp_path / "runs"),
                    "mode": "episode_first",
                    "metrics": {"histograms": True},
                }
            },
            state_dim=4,
            action_dim=1,
        )

        assert algo.tb_histograms_enabled is False
    finally:
        base_algorithm_module.SummaryWriter = original_writer
        base_algorithm_module.TENSORBOARD_AVAILABLE = original_available


def test_shared_tensorboard_defaults_to_artifacts_runs_when_cell_override_omits_dir():
    from opencda_marl._kernel.policy.manager import MARLManager

    manager = MARLManager.__new__(MARLManager)
    manager.config = {
        "outputs": {
            "root_dir": "artifacts",
            "tensorboard": {
                "enabled": True,
                "mode": "episode_first",
            },
        },
        "training": {"training_mode": True},
    }

    merged = manager._merge_shared_config(
        {
            "tensorboard": {
                "enabled": True,
                "mode": "balanced",
            }
        }
    )

    assert Path(merged["tensorboard"]["log_dir"]) == Path("artifacts") / "runs"
    assert merged["tensorboard"]["mode"] == "balanced"
