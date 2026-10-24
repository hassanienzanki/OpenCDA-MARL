# test/marl/test_checkpoint_load_integrity.py
"""A configured checkpoint either loads or the run stops — no third state.

Two gaps closed together: DQN could not even be constructed from its own
shipped profile (``MARL.action_dim`` defaulted to 1 and the fallback divided
by zero), and the MAPPO/DQN loaders swallowed their failures so a bad file
scored freshly initialised weights with exit code 0. The manager now treats
``_pretrained`` as authoritative, and the loaders accept eval-only files
(no optimizer state) — the shape the bundled baselines ship in.
"""
from pathlib import Path

import pytest
import torch

from opencda_marl._kernel.policy.manager import MARLManager
from opencda_marl.builtin.algorithms.dqn import DQNAlgorithm
from opencda_marl.tools.config_lint import merged_config

REPO = Path(__file__).resolve().parents[2]


def _manager(profile_ref: str) -> MARLManager:
    cfg = dict(merged_config(REPO, profile_ref, REPO / "configs")["MARL"])
    cfg["outputs"] = {"metrics_store": {"enabled": False}, "tensorboard": {"enabled": False}}
    cfg.setdefault("training", {})["save_checkpoints"] = False
    return MARLManager(cfg, cfg["algorithm"])


# --- DQN action space ---------------------------------------------------------

def test_dqn_takes_its_action_space_from_speed_actions():
    algo = DQNAlgorithm({"speed_actions": [15, 30, 45]}, state_dim=8, action_dim=1)
    assert algo.action_dim == 3
    assert algo.q_network.network[-1].out_features == 3


def test_dqn_refuses_a_single_action():
    with pytest.raises(ValueError, match="at least two discrete actions"):
        DQNAlgorithm({}, state_dim=8, action_dim=1)


def test_shipped_dqn_profile_constructs_through_the_manager():
    manager = _manager("marl/intersection/dqn")
    assert manager.algorithm.action_dim == 4        # speed_actions: [15, 30, 45, 60]


# --- the manager stops on a swallowed load failure ------------------------------

class _SwallowingLoader:
    """A learner whose load() hides its failure the way the old ones did."""
    _pretrained = False

    def load(self, path):
        return None


class _PluginWithoutFlag:
    def load(self, path):
        return None


def _bare_manager(algorithm) -> MARLManager:
    manager = object.__new__(MARLManager)
    manager.algorithm = algorithm
    manager.intent_algorithm = None
    return manager


def test_manager_raises_when_the_loader_swallowed_its_failure(tmp_path):
    with pytest.raises(RuntimeError, match="did not load"):
        _bare_manager(_SwallowingLoader()).load_checkpoint(str(tmp_path / "x.pth"))


def test_manager_trusts_learners_without_the_flag(tmp_path):
    _bare_manager(_PluginWithoutFlag()).load_checkpoint(str(tmp_path / "x.pth"))


# --- eval-only files (no optimizer state) load as pretrained ---------------------

@pytest.mark.parametrize("profile_ref", ["marl/intersection/mappo", "marl/intersection/dqn"])
def test_eval_only_checkpoint_loads_as_pretrained(profile_ref, tmp_path):
    manager = _manager(profile_ref)
    algo = manager.algorithm
    full = tmp_path / "full.pth"
    algo.save(str(full))
    saved = torch.load(full, map_location="cpu", weights_only=False)
    stripped = {k: v for k, v in saved.items() if "optimizer" not in k}
    assert len(stripped) < len(saved)
    lean = tmp_path / "eval_only.pth"
    torch.save(stripped, lean)

    algo._pretrained = False
    manager.load_checkpoint(str(lean))
    assert algo._pretrained is True
