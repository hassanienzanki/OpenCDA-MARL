"""TensorBoard run directories are named like the metrics store's.

``masac`` and ``pcmasac`` share one learner class (``SACAlgorithm``); a
directory named after the class would put both under ``sac``. The event
files land under ``<log_dir>/<algo>/<run_name>/`` with ``<algo>`` the
registered key MARLManager hands the learner, the same name the metrics
store uses, for every built-in learner; a learner built without the manager
keeps the class name.

A bare ``true`` / ``false`` switches either writer: in the algorithm block
it keeps the shared ``outputs`` settings, in ``outputs`` it is rooted at
``outputs.root_dir`` like a full block. A block left empty in YAML reads as
absent.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

import opencda_marl.builtin.algorithms.training_logger as training_logger
from opencda_marl._bundled import BUNDLED_CONFIG_ROOT
from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl.tools.config_lint import merged_config

REPO = Path(__file__).resolve().parents[2]

_FEATURES = {"rel_x": 1, "rel_y": 1, "speed": 1, "heading_angle": 1,
             "dist_to_intersection": 1, "dist_to_front_vehicle": 1,
             "lane_position": 1, "waypoint_buffer": 1}

_Q_LEARNING_SECTION = {"speed_actions": [10, 20, 30],
                       "state_features": {"speed": {"bins": [15.0, 30.0]}}}


class _FakeWriter:
    """Stands in for SummaryWriter: records where it was opened, writes nothing."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)

    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


@pytest.fixture(autouse=True)
def _fake_tensorboard(monkeypatch):
    monkeypatch.setattr(training_logger, "SummaryWriter", _FakeWriter)
    monkeypatch.setattr(training_logger, "TENSORBOARD_AVAILABLE", True)


def _outputs(tmp_path):
    return {"tensorboard": {"enabled": True, "log_dir": str(tmp_path / "tb"),
                            "run_name": "r"},
            "metrics_store": {"enabled": True, "log_dir": str(tmp_path / "metrics"),
                              "run_name": "r"}}


def _sac_section(**overrides):
    section = {"features": dict(_FEATURES), "warmup_steps": 0, "batch_size": 8,
               "memory_size": 64, "lstm_hidden_size": 4, "actor_hidden_dims": [8, 8],
               "critic_hidden_dims": [8, 8]}
    section.update(overrides)
    return section


def _config(tmp_path, algorithm, **section):
    return {"algorithm": algorithm, "action_space": {"mode": "speed_only"},
            "reward_profile": "simple_progress_v1", "state_dim": 8, "action_dim": 1,
            algorithm: _sac_section(**section), "training": {"training_mode": True},
            "outputs": _outputs(tmp_path)}


def _manager(tmp_path, algorithm, **section):
    return MARLManager(_config(tmp_path, algorithm, **section), algorithm)


def _tensorboard_root(tmp_path, algorithm):
    manager = _manager(tmp_path, algorithm)
    try:
        algo = manager.algorithm
        assert algo.tb_enabled and algo.writer is not None
        assert Path(algo.tensorboard_dir) == Path(algo.tensorboard_root_dir)
        # One name for both writers of the run.
        assert Path(algo.metrics_store_dir).parent.name == Path(algo.tensorboard_root_dir).parent.name
        return Path(algo.tensorboard_root_dir)
    finally:
        manager.close()


def test_learners_sharing_a_class_get_their_own_tensorboard_directory(tmp_path):
    masac = _tensorboard_root(tmp_path, "masac")
    pcmasac = _tensorboard_root(tmp_path, "pcmasac")
    assert masac == tmp_path / "tb" / "masac" / "r"
    assert pcmasac == tmp_path / "tb" / "pcmasac" / "r"
    assert masac != pcmasac


def _profile_config(profile, section=None, key=None):
    """A bundled profile's ``MARL`` block; with *section*, that learner block
    re-keyed to *key* (two keys that share a class share a profile)."""
    cfg = copy.deepcopy(dict(merged_config(REPO, profile, BUNDLED_CONFIG_ROOT)["MARL"]))
    if section:
        cfg["algorithm"], cfg[key] = key, cfg[section]
    return cfg


def _q_learning_config():
    return {"algorithm": "q_learning", "action_space": {"mode": "speed_only"},
            "reward_profile": "simple_progress_v1", "action_dim": 3,
            "q_learning": copy.deepcopy(_Q_LEARNING_SECTION)}


#: The other built-in keys, built from their bundled profiles (q_learning
#: has none, so an inline one); each builds in well under a second.
_LEARNER_CONFIGS = {
    "matd3": lambda: _profile_config("marl/intersection/td3"),
    "pcmatd3": lambda: _profile_config("marl/intersection/td3", "matd3", "pcmatd3"),
    "mappo": lambda: _profile_config("marl/intersection/mappo"),
    "pcmappo": lambda: _profile_config("momarl/intersection/speed_only_pcmappo"),
    "dqn": lambda: _profile_config("marl/intersection/dqn"),
    "q_learning": _q_learning_config,
}


@pytest.mark.parametrize("key", list(_LEARNER_CONFIGS))
def test_every_learner_names_its_runs_by_the_registered_key(tmp_path, key):
    """``td3`` -> ``matd3`` / ``pcmatd3`` and ``qlearning`` -> ``q_learning``
    change; ``mappo``, ``pcmappo`` and ``dqn`` keep their names."""
    cfg = _LEARNER_CONFIGS[key]()
    assert cfg["algorithm"] == key
    cfg["outputs"] = _outputs(tmp_path)
    cfg["training"] = {**cfg.get("training", {}), "training_mode": True,
                       "save_checkpoints": False}
    manager = MARLManager(cfg, key)
    try:
        algo = manager.algorithm
        assert Path(algo.tensorboard_root_dir) == tmp_path / "tb" / key / "r"
        assert Path(algo.metrics_store_dir) == tmp_path / "metrics" / key / "r"
    finally:
        manager.close()


def test_a_bare_metrics_store_switch_keeps_the_registered_name(tmp_path):
    """``metrics_store: false`` in the algorithm block turns the store off
    without losing the key that names the TensorBoard directory."""
    manager = _manager(tmp_path, "pcmasac", metrics_store=False)
    try:
        algo = manager.algorithm
        assert algo.metrics_store is None
        assert Path(algo.tensorboard_root_dir) == tmp_path / "tb" / "pcmasac" / "r"
    finally:
        manager.close()


def test_a_bare_tensorboard_switch_keeps_the_shared_settings(tmp_path, monkeypatch):
    """``tensorboard: true`` in the algorithm block keeps the shared
    ``log_dir`` and run name instead of replacing the shared block."""
    monkeypatch.chdir(tmp_path)  # a default ``runs/`` must not land in the checkout
    manager = _manager(tmp_path, "pcmasac", tensorboard=True)
    try:
        assert Path(manager.algorithm.tensorboard_root_dir) == tmp_path / "tb" / "pcmasac" / "r"
    finally:
        manager.close()


@pytest.mark.parametrize("block", ["tensorboard", "metrics_store"])
def test_an_empty_algorithm_block_reads_as_absent(tmp_path, block):
    """A YAML ``tensorboard:`` or ``metrics_store:`` with no value in the
    algorithm block (None) takes the shared block."""
    manager = _manager(tmp_path, "pcmasac", **{block: None})
    try:
        algo = manager.algorithm
        assert algo.metrics_store is not None
        assert Path(algo.tensorboard_root_dir) == tmp_path / "tb" / "pcmasac" / "r"
        assert Path(algo.metrics_store_dir) == tmp_path / "metrics" / "pcmasac" / "r"
    finally:
        manager.close()


def test_an_empty_shared_tensorboard_block_reads_as_absent(tmp_path):
    """``outputs.tensorboard:`` with no value takes the defaults, rooted at
    ``outputs.root_dir`` like an absent block."""
    cfg = _config(tmp_path, "pcmasac")
    cfg["outputs"]["tensorboard"] = None
    cfg["outputs"]["root_dir"] = str(tmp_path)
    manager = MARLManager(cfg, "pcmasac")
    try:
        algo = manager.algorithm
        assert algo.tb_enabled
        assert Path(algo.tensorboard_root_dir).parent == tmp_path / "runs" / "pcmasac"
    finally:
        manager.close()


def test_a_shared_tensorboard_false_survives_an_algorithm_block(tmp_path, monkeypatch):
    """``outputs.tensorboard: false`` stays off when the algorithm block
    tunes TensorBoard without naming ``enabled``."""
    monkeypatch.chdir(tmp_path)  # a default ``runs/`` must not land in the checkout
    cfg = _config(tmp_path, "pcmasac", tensorboard={"mode": "balanced"})
    cfg["outputs"]["tensorboard"] = False
    manager = MARLManager(cfg, "pcmasac")
    try:
        algo = manager.algorithm
        assert algo.tb_enabled is False
        assert algo.writer is None
    finally:
        manager.close()


def test_a_bare_shared_tensorboard_switch_is_rooted_at_root_dir(tmp_path, monkeypatch):
    """``outputs.tensorboard: true`` writes under ``outputs.root_dir``, as
    ``outputs.metrics_store: true`` does, not under the working directory."""
    monkeypatch.chdir(tmp_path)  # a default ``runs/`` must not land in the checkout
    cfg = _config(tmp_path, "pcmasac")
    cfg["outputs"]["tensorboard"] = True
    cfg["outputs"]["root_dir"] = str(tmp_path / "out")
    manager = MARLManager(cfg, "pcmasac")
    try:
        algo = manager.algorithm
        assert algo.tb_enabled
        assert Path(algo.tensorboard_root_dir).parent == tmp_path / "out" / "runs" / "pcmasac"
    finally:
        manager.close()


@pytest.mark.parametrize("learner, section, expected", [
    ("SACAlgorithm", _sac_section, "sac"),
    ("TD3Algorithm", lambda: _profile_config("marl/intersection/td3")["matd3"], "td3"),
    ("QLearningAlgorithm", lambda: copy.deepcopy(_Q_LEARNING_SECTION), "qlearning"),
], ids=["sac", "td3", "qlearning"])
def test_a_learner_built_without_the_manager_keeps_the_class_name(tmp_path, learner, section,
                                                                 expected):
    from opencda_marl.builtin import algorithms

    cfg = dict(section())
    cfg.update(tensorboard={"enabled": True, "log_dir": str(tmp_path / "tb"), "run_name": "r"},
               metrics_store={"enabled": False})
    algo = getattr(algorithms, learner)(cfg, 8, 1)
    try:
        assert Path(algo.tensorboard_root_dir) == tmp_path / "tb" / expected / "r"
    finally:
        algo.close()
