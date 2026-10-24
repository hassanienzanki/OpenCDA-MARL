"""A configured checkpoint must load, and evaluation must run the configured
number of episodes on both backends.

Swallowing a checkpoint-load failure meant an evaluation scored freshly
initialised weights and a "resumed" run trained from scratch — both exiting 0
with plausible metrics. And the SUMO coordinator clamped evaluation to a single
episode while the CARLA path honoured the config, so a SUMO eval asking for N
episodes silently produced one.
"""
from types import SimpleNamespace

import pytest


def _env(cls, manager):
    env = object.__new__(cls)
    env.marl_manager = manager
    return env


@pytest.mark.parametrize("backend", ["carla", "sumo"])
def test_missing_checkpoint_is_fatal(backend, tmp_path):
    if backend == "carla":
        from opencda_marl._kernel.envs.marl_env import MARLEnv as cls
    else:
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv as cls
    env = _env(cls, SimpleNamespace(load_checkpoint=lambda p: None))
    with pytest.raises(FileNotFoundError, match="Configured checkpoint not found"):
        env._load_checkpoint_from_config(str(tmp_path / "nope.pth"))


@pytest.mark.parametrize("backend", ["carla", "sumo"])
def test_broken_checkpoint_is_fatal(backend, tmp_path):
    if backend == "carla":
        from opencda_marl._kernel.envs.marl_env import MARLEnv as cls
    else:
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv as cls
    path = tmp_path / "broken.pth"
    path.write_bytes(b"not a checkpoint")

    def boom(_p):
        raise ValueError("bad magic")

    env = _env(cls, SimpleNamespace(load_checkpoint=boom))
    with pytest.raises(RuntimeError, match="Failed to load configured checkpoint"):
        env._load_checkpoint_from_config(str(path))


@pytest.mark.parametrize("backend", ["carla", "sumo"])
def test_no_checkpoint_configured_is_a_no_op(backend):
    if backend == "carla":
        from opencda_marl._kernel.envs.marl_env import MARLEnv as cls
    else:
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv as cls
    calls = []
    env = _env(cls, SimpleNamespace(load_checkpoint=lambda p: calls.append(p)))
    env._load_checkpoint_from_config("")
    env._load_checkpoint_from_config(None)
    assert calls == []


@pytest.mark.parametrize("backend", ["carla", "sumo"])
def test_good_checkpoint_loads(backend, tmp_path):
    if backend == "carla":
        from opencda_marl._kernel.envs.marl_env import MARLEnv as cls
    else:
        from opencda_marl._kernel.envs.sumo_marl_env import SumoMARLEnv as cls
    path = tmp_path / "ok.pth"
    path.write_bytes(b"x")
    calls = []
    env = _env(cls, SimpleNamespace(load_checkpoint=lambda p: calls.append(p)))
    env._load_checkpoint_from_config(str(path))
    assert calls == [str(path)]


def test_sumo_evaluation_honours_the_configured_episode_count():
    """Both backends read the configured horizon; neither clamps to 1."""
    import inspect

    from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator
    from opencda_marl._kernel.lifecycle.scenario_manager import MARLScenarioManager

    sumo_src = inspect.getsource(MARLCoordinator.initialize)
    assert "self.states['max_episodes'] = 1" not in sumo_src, (
        "SUMO evaluation must not clamp max_episodes to 1")
    assert "max(\n                    1, int(self.states.get('max_episodes', 1)))" in sumo_src \
        or "max(1, int(self.states.get('max_episodes', 1)))" in sumo_src

    carla_src = inspect.getsource(MARLScenarioManager.__init__)
    assert "max(1, int(self.states.get('max_episodes', 1)))" in carla_src
