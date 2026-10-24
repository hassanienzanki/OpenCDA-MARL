"""A checkpoint stores a portable copy of its learner's config.

Every learner pickles its config into its checkpoints, and a run's config
says where that run wrote: MARLManager merges the ``tensorboard`` and
``metrics_store`` blocks from ``outputs.*``, rooted at ``outputs.root_dir``,
with the run name beside them. Worse, the manager hands a learner a plain
dict whose list and dict values are still OmegaConf nodes, each keeping a
reference to the configuration it came from, so pickling one carried the
entire run configuration (every section, ``outputs.root_dir`` included)
into the checkpoint. A shared checkpoint therefore disclosed the directory
tree of the machine that trained it. The copy a checkpoint stores now
holds plain values only, leaves the run-output blocks out, keeps
interpolations unresolved and replaces every other absolute path with a
placeholder, keeping every hyperparameter. No loader reads the saved
config back, so checkpoints written before (which still carry all of it)
load as they did, and training resumes from either.
"""
from __future__ import annotations

import copy
import pickle
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from opencda_marl.builtin.algorithms.dqn import DQNAlgorithm
from opencda_marl.builtin.algorithms.mappo import MAPPOAlgorithm
from opencda_marl.builtin.algorithms.pcmappo import PCMAPPOAlgorithm
from opencda_marl.builtin.algorithms.q_learning import QLearningAlgorithm
from opencda_marl.builtin.algorithms.sac import SACAlgorithm
from opencda_marl.builtin.algorithms.td3 import TD3Algorithm
from opencda_marl.builtin.algorithms.training_logger import (
    ABSOLUTE_PATH_PLACEHOLDER,
    RUN_OUTPUT_BLOCKS,
    portable_config,
)

_POSIX = "/home/example/runs"
_DRIVE = "Z:\\example\\runs"


def _run_config(root, **hyperparameters) -> dict:
    """Hyperparameters plus the blocks MARLManager merges in, under ``root``."""
    return {
        **hyperparameters,
        "training_mode": True,
        "tensorboard": {"enabled": False, "log_dir": f"{root}/runs", "run_name": "cell",
                        "mode": "episode_first"},
        "metrics_store": {"enabled": False, "log_dir": f"{root}/metrics", "run_name": "cell",
                          "scenario": "family/scenario/profile", "algorithm": "masac"},
    }


# --- the helper -------------------------------------------------------------------

def test_the_run_output_blocks_are_dropped_and_hyperparameters_kept():
    hyperparameters = {"learning_rate": 3e-4, "actor_hidden_dims": [16, 16],
                       "features": {"speed": 1, "rel_x": 1}, "pref_mode": "direct",
                       "reward_dim": 3}
    config = _run_config(_POSIX, **hyperparameters)
    portable = portable_config(config)
    assert set(RUN_OUTPUT_BLOCKS) == {"tensorboard", "metrics_store"}
    assert portable == {**hyperparameters, "training_mode": True}


@pytest.mark.parametrize("value", [
    _POSIX,
    _DRIVE,
    "Z:/example/runs",
    "//host/share/runs",
    "\\\\host\\share\\runs",
    "\\example\\runs",
    "file:///example/runs",
    "FILE://host/share/runs",
    PurePosixPath(_POSIX),
    PureWindowsPath(_DRIVE),
])
def test_an_absolute_path_anywhere_else_is_replaced(value):
    config = {"gamma": 0.99, "warm_start": value,
              "nested": {"paths": ["recordings/traffic.json", value, {"deep": value}],
                         "pair": (value, 1)}}
    portable = portable_config(config)
    marker = ABSOLUTE_PATH_PLACEHOLDER
    assert portable == {"gamma": 0.99, "warm_start": marker,
                        "nested": {"paths": ["recordings/traffic.json", marker, {"deep": marker}],
                                   "pair": (marker, 1)}}


@pytest.mark.parametrize("value", [
    "checkpoints/sac/",
    "recordings/traffic.json",
    "./var/run",
    "~/runs",
    "km/h",
    "Loss/actor",
    "http://127.0.0.1:8000/data/x",
    "C:relative",
    "/",
    "",
    "LSTM",
    PurePosixPath("checkpoints", "data"),
    0.5,
    None,
    True,
])
def test_relative_paths_urls_and_other_values_are_kept(value):
    assert portable_config({"value": value}) == {"value": value}


def test_the_live_config_is_not_modified():
    config = _run_config(_POSIX, learning_rate=1e-3, notes={"dir": _DRIVE})
    before = copy.deepcopy(config)
    portable_config(config)
    assert config == before


def test_an_omegaconf_config_comes_back_as_plain_containers():
    config = OmegaConf.create(_run_config(
        _POSIX, learning_rate=1e-3, hidden=[8, 8], width="${hidden}",
        log_root=_POSIX, unresolved="${outputs.root_dir}/x"))
    portable = portable_config(config)
    assert type(portable) is dict and type(portable["hidden"]) is list
    assert not set(RUN_OUTPUT_BLOCKS) & set(portable)
    assert portable["log_root"] == ABSOLUTE_PATH_PLACEHOLDER
    # An interpolation that only the full run config resolves stays as text.
    assert portable["unresolved"] == "${outputs.root_dir}/x"


def test_an_omegaconf_interpolation_is_kept_as_written():
    config = OmegaConf.create({"hidden": [8, 8], "width": "${hidden}"})
    assert portable_config(config) == {"hidden": [8, 8], "width": "${hidden}"}


def _section_as_the_manager_hands_it(run, name) -> dict:
    """A learner section the way MARLManager copies it: a plain dict whose
    list and dict values are still nodes of ``run``."""
    config = copy.deepcopy(dict(run.MARL[name]))
    assert any(OmegaConf.is_config(v) for v in config.values())
    return config


def test_a_nested_node_carries_none_of_the_run_config(monkeypatch):
    """A nested OmegaConf node references the whole run config, and
    resolving its interpolations would pull values from that config or the
    machine into the copy, inside strings no path check recognises."""
    monkeypatch.setenv("OPENCDA_MARL_TEST_USER", "someone")
    run = OmegaConf.create({
        "outputs": {"root_dir": _POSIX},
        "world": {"weather": "world-only-value"},
        "MARL": {"sac": {"gamma": 0.99, "actor_hidden_dims": [16, 16],
                         "notes": {"tag": "run@${outputs.root_dir}",
                                   "who": "by-${oc.env:OPENCDA_MARL_TEST_USER}",
                                   "dir": _DRIVE}}},
    })
    portable = portable_config(_section_as_the_manager_hands_it(run, "sac"))

    assert portable == {"gamma": 0.99, "actor_hidden_dims": [16, 16],
                        "notes": {"tag": "run@${outputs.root_dir}",
                                  "who": "by-${oc.env:OPENCDA_MARL_TEST_USER}",
                                  "dir": ABSOLUTE_PATH_PLACEHOLDER}}
    raw = pickle.dumps(portable)
    for leaked in (b"omegaconf", b"world-only-value", b"someone", _POSIX.encode()):
        assert leaked not in raw, leaked


# --- every learner: save, load, resume ----------------------------------------------

_NETS = {"actor_hidden_dims": [16, 16], "critic_hidden_dims": [16, 16], "lstm_hidden_size": 8}
_PPO = {**_NETS, "warmup_steps": 0, "n_steps": 4, "batch_size": 4, "num_mini_batch": 1,
        "ppo_epochs": 1}
_OFF_POLICY = {**_NETS, "warmup_steps": 0, "memory_size": 256, "batch_size": 8,
               "policy_freq": 1}

#: class, hyperparameters, state_dim, action_dim, the counter training advances.
_LEARNERS = {
    "dqn": (DQNAlgorithm, {"speed_actions": [15.0, 30.0, 45.0], "hidden_dims": [16],
                           "batch_size": 4}, 8, 3, "training_step"),
    "q_learning": (QLearningAlgorithm, {"speed_actions": [10, 20, 30], "learning_rate": 0.5},
                   4, 3, "training_step"),
    "mappo": (MAPPOAlgorithm, dict(_PPO), 5, 1, "updates_performed"),
    "pcmappo": (PCMAPPOAlgorithm, {**_PPO, "pref_dim": 2}, 5, 1, "updates_performed"),
    "sac": (SACAlgorithm, dict(_OFF_POLICY), 5, 1, "training_step"),
    # TD3's batched actor reads the distance feature at index 6.
    "td3": (TD3Algorithm, dict(_OFF_POLICY), 8, 1, "training_step"),
}


def _build(name, root, *, as_a_run_builds_it=False):
    cls, hyperparameters, state_dim, action_dim, _ = _LEARNERS[name]
    if not as_a_run_builds_it:
        return cls(_run_config(root, **hyperparameters), state_dim, action_dim)
    run = OmegaConf.create({"outputs": {"root_dir": str(root)},
                            "MARL": {name: _run_config(root, **hyperparameters)}})
    return cls(_section_as_the_manager_hands_it(run, name), state_dim, action_dim)


def _omegaconf_nodes(value, where="checkpoint") -> list[str]:
    """Where an OmegaConf node sits in a loaded checkpoint."""
    if OmegaConf.is_config(value):
        return [where]
    if isinstance(value, dict):
        return [hit for key, item in value.items()
                for hit in _omegaconf_nodes(item, f"{where}[{key!r}]")]
    if isinstance(value, (list, tuple)):
        return [hit for i, item in enumerate(value)
                for hit in _omegaconf_nodes(item, f"{where}[{i}]")]
    return []


def _train(learner, rng, steps=8):
    """Act, store and update the way the manager's strategy does."""
    if isinstance(learner, QLearningAlgorithm):
        for _ in range(steps):
            state = int(rng.integers(learner.state_dim))
            action = int(learner.select_action(state))
            learner.store_transition(state, action, 1.0, int(rng.integers(learner.state_dim)), False)
            learner.update()
    elif isinstance(learner, DQNAlgorithm):
        for _ in range(steps):
            state = rng.normal(size=learner.state_dim).astype(np.float32)
            nxt = rng.normal(size=learner.state_dim).astype(np.float32)
            learner.store_transition(state, learner.select_action(state), 1.0, nxt, False)
            learner.update()
    else:
        for _ in range(steps):
            obs = {f"v{i}": rng.normal(size=learner.state_dim).astype(np.float32) for i in range(3)}
            nxt = {k: rng.normal(size=learner.state_dim).astype(np.float32) for k in obs}
            actions = learner.select_actions(obs, list(obs), training=True)
            for agent_id in obs:
                learner.store_transition(obs, agent_id, actions[agent_id], 1.0, nxt, False)
            learner.update()


def _weights(learner) -> dict:
    if isinstance(learner, QLearningAlgorithm):
        return {"q_table": torch.as_tensor(learner.q_table.copy())}
    module = learner.q_network if isinstance(learner, DQNAlgorithm) else learner.actor
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _read(path: Path) -> dict:
    if zipfile.is_zipfile(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _write(path: Path, data: dict) -> None:
    if path.suffix == ".pkl":
        with open(path, "wb") as handle:
            pickle.dump(data, handle)
    else:
        torch.save(data, path)


def _target(tmp_path, name, stem="checkpoint") -> Path:
    return tmp_path / f"{stem}{'.pkl' if name == 'q_learning' else '.pth'}"


def _assert_loads_and_resumes(name, src, path, tmp_path):
    counter = _LEARNERS[name][4]
    dst = _build(name, tmp_path)
    dst.load(str(path))
    if hasattr(dst, "_pretrained"):
        assert dst._pretrained is True
    loaded = _weights(dst)
    for key, value in _weights(src).items():
        assert torch.equal(value, loaded[key]), key
    assert dst.episode_count == src.episode_count == 7
    if isinstance(dst, DQNAlgorithm):
        assert list(dst.speed_actions) == list(src.speed_actions)
    before = getattr(dst, counter)
    _train(dst, np.random.default_rng(1))
    assert getattr(dst, counter) > before, f"{name} did not keep training after the load"


@pytest.mark.parametrize("name", sorted(_LEARNERS))
def test_a_saved_checkpoint_holds_the_portable_config_and_resumes(tmp_path, name):
    src = _build(name, tmp_path)
    live = copy.deepcopy(src.config)
    _train(src, np.random.default_rng(0))
    src.episode_count = 7
    path = _target(tmp_path, name)
    src.save(str(path))

    saved = _read(path)["config"]
    assert saved == portable_config(src.config)
    assert not set(RUN_OUTPUT_BLOCKS) & set(saved)
    hyperparameters = _LEARNERS[name][1]
    assert {k: saved[k] for k in hyperparameters} == hyperparameters
    assert src.config == live, "saving must not modify the learner's own config"

    _assert_loads_and_resumes(name, src, path, tmp_path)


@pytest.mark.parametrize("name", sorted(_LEARNERS))
def test_a_learner_configured_by_a_run_saves_plain_values_and_resumes(tmp_path, name):
    """Built from OmegaConf nodes the way a run builds it, a learner saves
    no node anywhere in its checkpoint, and the checkpoint resumes."""
    src = _build(name, tmp_path, as_a_run_builds_it=True)
    _train(src, np.random.default_rng(0))
    src.episode_count = 7
    path = _target(tmp_path, name)
    src.save(str(path))

    data = _read(path)
    assert _omegaconf_nodes(data) == []
    assert data["config"] == portable_config(src.config)
    hyperparameters = _LEARNERS[name][1]
    assert {k: data["config"][k] for k in hyperparameters} == hyperparameters

    _assert_loads_and_resumes(name, src, path, tmp_path)


@pytest.mark.parametrize("name", sorted(_LEARNERS))
def test_a_checkpoint_written_with_the_whole_config_still_loads(tmp_path, name):
    """Checkpoints saved before the portable copy carry the full run config,
    an OmegaConf container with the run-output blocks and absolute paths
    (and DQN's speed list as a node of it); nothing reads the config, so
    they load and resume unchanged."""
    src = _build(name, tmp_path)
    _train(src, np.random.default_rng(0))
    src.episode_count = 7
    path = _target(tmp_path, name)
    src.save(str(path))
    data = _read(path)
    data["config"] = OmegaConf.create(_run_config(_POSIX, **_LEARNERS[name][1]))
    if "speed_actions" in data:
        data["speed_actions"] = data["config"]["speed_actions"]
    old_style = _target(tmp_path, name, stem="old_style")
    _write(old_style, data)

    _assert_loads_and_resumes(name, src, old_style, tmp_path)
