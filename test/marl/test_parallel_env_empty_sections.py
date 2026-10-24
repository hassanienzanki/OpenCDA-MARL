"""The PettingZoo port finalises the run's config like the coordinator.

``from_scenario`` builds the SUMO env itself, without the coordinator. A
top-level section that no layer defines as a mapping and a profile leaves
empty (``outputs:``) reached the env as None and crashed its first
``config.get('outputs', {}).get(...)``.
"""
import pytest

from opencda_marl.envs.parallel_env import OpenCDAMARLParallelEnv
from opencda_marl.utils.config_locator import MARL_CONFIG_ROOTS_ENV


class _Reached(Exception):
    pass


def test_from_scenario_hands_the_env_a_mapping_for_an_empty_section(tmp_path, monkeypatch):
    try:
        import opencda_marl._kernel.envs.sumo_marl_env as sumo_marl_env
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"SUMO runtime unavailable: {exc}")
    family = tmp_path / "configs" / "sumo-marl"
    (family / "intersection").mkdir(parents=True)
    (family / "default.yaml").write_text(
        "runtime:\n  backend: sumo\nMARL:\n  training: {}\n", encoding="utf-8")
    (family / "intersection" / "base.yaml").write_text(
        "scenario:\n  type: intersection\n", encoding="utf-8")
    (family / "intersection" / "p.yaml").write_text("outputs:\nagents:\n", encoding="utf-8")
    monkeypatch.delenv(MARL_CONFIG_ROOTS_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    captured = {}

    def _env(config):
        captured["config"] = config
        raise _Reached

    monkeypatch.setattr(sumo_marl_env, "SumoMARLEnv", _env)

    with pytest.raises(_Reached):
        OpenCDAMARLParallelEnv.from_scenario(
            "sumo-marl/intersection/p",
            overrides={"MARL": {"external": {"features": {"rel_x": 1}}}})

    config = captured["config"]
    assert config["outputs"] == {} and config["agents"] == {}
    assert config.get("outputs", {}).get("metrics_archive", {}) == {}
