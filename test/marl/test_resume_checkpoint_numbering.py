"""Mid-cell resume: episode numbering and checkpoint history continue.

The launcher injects ``MARL.training.start_episode`` when it restarts an
interrupted cell. Before this was wired end to end, the metrics exporter was
offset but the *episode number* was not, so a resumed run wrote
``episode_0000.pth`` again — and ``cleanup_old_checkpoints(keep_last_n)``,
which sorts by file name, then deleted exactly the file it had just written.
The checkpoint manager also started with ``best_reward = -inf``, so the first
resumed episode always claimed ``best_checkpoint.pth`` and the rewritten
``training_metadata.json`` dropped every pre-resume episode.
"""
import json
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.policy.checkpoint import CheckpointManager


class _Algo:
    """Minimal algorithm: save() writes a file, load() reads it."""

    def __init__(self, tag="a"):
        self.tag = tag

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.tag)

    def load(self, path):
        with open(path) as f:
            self.tag = f.read()


def _save(manager, algo, episode, reward):
    manager.save_checkpoint(algo, episode, {"total_reward": reward}, checkpoint_type="latest")


def test_manager_adopts_the_history_already_in_its_directory(tmp_path):
    algo = _Algo()
    first = CheckpointManager(str(tmp_path), "masac")
    _save(first, algo, 0, 10.0)
    _save(first, algo, 1, 99.0)          # the best of the pre-resume run
    _save(first, algo, 2, 20.0)
    assert first.best_reward == pytest.approx(99.0) and first.best_episode == 1

    resumed = CheckpointManager(str(tmp_path), "masac")
    assert resumed.best_reward == pytest.approx(99.0), "best carried over"
    assert resumed.best_episode == 1
    assert len(resumed.training_history) == 3, "pre-resume episodes kept"

    best_before = (tmp_path / "masac" / "best_checkpoint.pth").read_text()
    _save(resumed, _Algo("worse"), 3, 30.0)          # worse than 99 -> not best
    assert (tmp_path / "masac" / "best_checkpoint.pth").read_text() == best_before
    meta = json.loads((tmp_path / "masac" / "training_metadata.json").read_text())
    assert [h["episode"] for h in meta["history"]] == [0, 1, 2, 3]
    assert meta["best_reward"] == pytest.approx(99.0)

    _save(resumed, _Algo("better"), 4, 150.0)        # beats it -> new best
    assert (tmp_path / "masac" / "best_checkpoint.pth").read_text() == "better"
    assert resumed.best_episode == 4


def test_fresh_directory_starts_without_history(tmp_path):
    manager = CheckpointManager(str(tmp_path), "masac")
    assert manager.training_history == [] and manager.best_episode == 0
    assert manager.best_reward == float("-inf")


def test_resumed_checkpoints_do_not_overwrite_or_delete_earlier_episodes(tmp_path):
    """keep_last_n cleanup sorts by name: a resumed run that restarted the
    numbering deleted its own fresh files."""
    algo = _Algo()
    first = CheckpointManager(str(tmp_path), "masac")
    for ep in range(5):
        _save(first, algo, ep, float(ep))
        first.cleanup_old_checkpoints(keep_last_n=2)
    names = sorted(p.name for p in (tmp_path / "masac").glob("episode_*.pth"))
    assert names == ["episode_0003.pth", "episode_0004.pth"]

    resumed = CheckpointManager(str(tmp_path), "masac")
    for ep in (5, 6):                       # numbering continues (start_episode=5)
        _save(resumed, algo, ep, float(ep))
        resumed.cleanup_old_checkpoints(keep_last_n=2)
    names = sorted(p.name for p in (tmp_path / "masac").glob("episode_*.pth"))
    assert names == ["episode_0005.pth", "episode_0006.pth"], names


@pytest.mark.parametrize("backend", ["carla", "sumo"])
def test_env_offsets_the_episode_counter_by_start_episode(backend):
    """Both envs must start numbering at start_episode, not at 0."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    name = "marl_env.py" if backend == "carla" else "sumo_marl_env.py"
    src = (repo / "opencda_marl" / "_kernel" / "envs" / name).read_text(encoding="utf-8")
    marker = ("self.sm.states['episode'] = start_episode_counter" if backend == "carla"
              else "self.current_episode = start_episode_counter")
    assert marker in src, f"{name} must offset its episode counter by start_episode"
    assert "start_episode_counter = int(self.training_config.get('start_episode', 0))" in src
