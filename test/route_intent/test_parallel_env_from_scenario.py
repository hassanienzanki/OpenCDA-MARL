"""The ParallelEnv quick start runs end to end on a bundled SUMO scenario.

``from_scenario`` needs ``MARL.external.features`` (the benchmark's
observation definition); no bundled profile declares it, so the quick start
passes it through ``overrides``. The documented loop runs on
``episode_over``: a replay starts with no vehicle on the road, so a loop on
``env.agents`` stopped before the first step. The observation width follows
the declared features, not the base profile's learner ``MARL.state_dim``.
"""
import numpy as np
import pytest

from opencda_marl.envs.parallel_env import OpenCDAMARLParallelEnv

try:
    import gymnasium  # noqa: F401
    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False

FEATURES = {"rel_x": 1, "rel_y": 1, "heading_angle": 1, "dist_to_intersection": 1,
            "dist_to_front_vehicle": 1}


def _overrides(root, max_steps):
    return {
        "MARL": {"external": {"features": FEATURES},
                 "training": {"training_mode": False, "save_checkpoints": False}},
        "scenario": {"simulation": {"max_steps": max_steps}},
        "outputs": {"root_dir": str(root), "monitor": False, "traffic_view": False,
                    "tensorboard": {"enabled": False},
                    "metrics_archive": {"enabled": False},
                    "evaluation": {"mode": "disabled"}},
    }


def test_a_missing_feature_block_says_how_to_declare_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="MARL.external.features") as err:
        OpenCDAMARLParallelEnv.from_scenario("sumo-marl/intersection/route_speed_only_pcmappo")
    assert "overrides" in str(err.value)


def test_the_quick_start_loop_runs_a_sumo_episode(tmp_path, monkeypatch):
    try:
        import opencda_marl._kernel.envs.sumo_marl_env  # noqa: F401
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    monkeypatch.chdir(tmp_path)          # any relative output lands in the sandbox
    max_steps = 400                      # the replay's first vehicle enters a few hundred steps in
    env = OpenCDAMARLParallelEnv.from_scenario(
        "sumo-marl/intersection/route_speed_only_pcmappo",
        overrides=_overrides(tmp_path / "artifacts", max_steps))
    try:
        observations, _ = env.reset()
        assert env.agents == [] and not env.episode_over, "a replay starts with an empty road"

        width = env.observation_space().shape[0] if _HAS_GYMNASIUM else None
        steps, seen = 0, set()
        while not env.episode_over:
            actions = {agent: 20.0 for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            steps += 1
            seen.update(observations)
            for agent, vector in observations.items():
                assert vector.ndim == 1 and vector.dtype == np.float32
                assert vector.shape[0] >= len(FEATURES)
                if width is not None:
                    assert vector.shape == (width,), "the declared space is the vector's width"
                assert set(infos[agent]) >= {"raw_observation"}
            assert steps <= max_steps + 1, "the episode ends at the configured horizon"

        assert seen, "vehicles joined during the episode"
        assert env.agents == []
        env.reset()
        assert not env.episode_over
    finally:
        env.close()


def test_episode_over_latches_until_reset():
    from opencda_marl.envs.parallel_env import ExternalPolicy

    class _Manager:
        algorithm = ExternalPolicy({}, 1, 1)

        def resolve_state_dim(self):
            return 1

        def _extract_states(self, raw):
            return {str(a): {"ego_state": np.zeros(1, np.float32)} for a in raw}

    class _Inner:
        script = [{"done": False}, {"done": True}, {"done": False}]

        def reset(self):
            pass

        def step(self):
            return dict(self.script.pop(0))

    env = OpenCDAMARLParallelEnv(_Inner(), _Manager())
    env.reset()
    env.step({})
    assert not env.episode_over
    env.step({})
    assert env.episode_over
    env.step({})
    assert env.episode_over, "stays over until reset"
    env.reset()
    assert not env.episode_over
