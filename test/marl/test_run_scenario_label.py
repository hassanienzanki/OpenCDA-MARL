"""Both backends label a run with its scenario reference in the metrics store.

The CARLA pipeline builder handed ``scenario_name`` (the ``-t`` reference)
to the MARL manager, but the SUMO env built its own manager without it, so
every SUMO run was stored with ``scenario=None``.
"""
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.lifecycle.pipeline_builder import scenario_name_from_config
from opencda_marl._kernel.policy.manager import MARLManager

REF = "sumo-marl/intersection/route_speed_only_pcmappo"


@pytest.mark.parametrize("config, expected", [
    ({"opt": {"test_scenario": REF}}, REF),
    ({"opt": {}}, None),
    ({"opt": None}, None),
    ({}, None),
])
def test_scenario_name_comes_from_the_cli_reference(config, expected):
    assert scenario_name_from_config(config) == expected


def _build_sumo_env(tmp_path, monkeypatch, **extra):
    from opencda_marl.envs import sumo_marl_env as env_module

    captured = {}

    class _FakeManager:
        def __init__(self, config, algorithm):
            captured["config"] = config
            self.objective_config = None
            self.objective_spec = None
            self.agent_manager = None

    monkeypatch.setattr(env_module, "MARLManager", _FakeManager)
    monkeypatch.setattr(env_module.SumoMARLEnv, "_start_sumo", lambda self: None)
    monkeypatch.setattr(env_module, "CheckpointManager", lambda *a, **k: None)
    sumo_cfg = tmp_path / "net.sumocfg"
    sumo_cfg.write_text("<configuration/>")
    marl = {"algorithm": "masac", "reward_profile": "simple_progress_v1",
            "action_space": {"mode": "speed_only"},
            "state_dim": 8, "action_dim": 1, "training": {"training_mode": True}}
    marl.update(extra.pop("marl", {}))
    config = {"runtime": {"sumo_cfg": str(sumo_cfg)}, "world": {}, "scenario": {},
              "MARL": marl, "outputs": {"root_dir": str(tmp_path / "artifacts")},
              "agents": {}}
    config.update(extra)
    env_module.SumoMARLEnv(config)
    return captured["config"], marl


def test_the_sumo_env_hands_the_scenario_reference_to_the_manager(tmp_path, monkeypatch):
    manager_config, marl = _build_sumo_env(
        tmp_path, monkeypatch, opt={"test_scenario": REF})
    assert manager_config["scenario_name"] == REF
    assert "scenario_name" not in marl, "the caller's MARL block stays untouched"

    # The manager writes it into the metrics store's run meta.
    merged = MARLManager._merge_shared_config(
        SimpleNamespace(config=manager_config), {})
    assert merged["metrics_store"]["scenario"] == REF


def test_an_explicit_scenario_name_wins(tmp_path, monkeypatch):
    manager_config, _ = _build_sumo_env(
        tmp_path, monkeypatch, opt={"test_scenario": REF},
        marl={"scenario_name": "custom-label"})
    assert manager_config["scenario_name"] == "custom-label"


def test_a_run_without_a_reference_has_no_label(tmp_path, monkeypatch):
    manager_config, _ = _build_sumo_env(tmp_path, monkeypatch)
    assert "scenario_name" not in manager_config
