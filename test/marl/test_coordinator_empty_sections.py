"""The coordinator finalises the run's config: no top-level section is None.

The merge keeps a lower layer's section under an empty override, but a
section still reaches the run as None when no layer defines it as a mapping
(or from a config built in code). Every component reads the sections as
``config.get('outputs', {}).get(...)``, whose default only covers an absent
key, so the coordinator normalises them once, before anything is built.
"""
from types import SimpleNamespace

from omegaconf import OmegaConf

from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator
from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder

EMPTY_SECTIONS = "world:\nscenario:\nagents:\noutputs:\nMARL:\n"


def test_null_sections_reach_the_components_as_empty_mappings():
    coord = MARLCoordinator(OmegaConf.create("runtime:\n" + EMPTY_SECTIONS))

    assert coord._build_output_section_config("evaluation") == {}
    builder = PipelineBuilder(coord.config)
    assert builder.build_output_section_config("trace") == {}
    assert builder.build_world_manager(coord) is None      # runtime.reset absent: off


def test_the_config_object_itself_is_kept():
    class _Config(dict):
        opt = SimpleNamespace(apply_ml=False)

    config = _Config({"outputs": None, "MARL": {"training": {"training_mode": False}}})
    coord = MARLCoordinator(config)

    assert coord.config is config and coord.config.opt.apply_ml is False
    assert config["outputs"] == {}
    assert config["MARL"] == {"training": {"training_mode": False}}


def test_a_sumo_run_initialises_with_null_sections(tmp_path, monkeypatch):
    import opencda_marl._kernel.envs.sumo_marl_env as sumo_marl_env
    import opencda_marl.plugins.loader as loader
    import opencda_marl.telemetry.monitor as monitor

    seen = {}

    class _Env:
        def __init__(self, config):
            # what the SUMO env does with the run's outputs section
            seen["root_dir"] = config.get("outputs", {}).get("root_dir", "artifacts")

    monkeypatch.setattr(sumo_marl_env, "SumoMARLEnv", _Env)
    monkeypatch.setattr(loader, "load_plugins", lambda extra=None, **_: seen.update(plugins=extra) or {})
    monkeypatch.setattr(monitor, "ensure_service",
                        lambda root, host, port, allow_hosts=(): seen.update(monitor=root) or ("http://x", True))
    monkeypatch.chdir(tmp_path)

    coord = MARLCoordinator(OmegaConf.create("runtime:\n  backend: sumo\n" + EMPTY_SECTIONS))
    coord.initialize()

    assert seen == {"plugins": None, "root_dir": "artifacts", "monitor": "artifacts"}
    assert coord.states == {"max_steps": 2400, "max_episodes": 1000}
    assert coord.start_episode == 0
