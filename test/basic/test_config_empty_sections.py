"""A section left empty in YAML is no override.

``outputs:`` with nothing under it parses as None. A plain merge replaced
the family's whole ``outputs`` block with that None, and the run then
crashed on the first ``config.get('outputs', {}).get(...)``: the default
only covers an absent key. The merge now keeps a lower layer's mapping under
an empty override, and the run's top-level sections are normalised to
mappings once, where the coordinator takes the config.
"""
import argparse
import importlib.util
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigTypeError

from opencda_marl.utils.config_locator import MARL_CONFIG_ROOTS_ENV
from opencda_marl.utils.config_merger import (
    RUN_CONFIG_SECTIONS,
    normalize_run_config,
    smart_merge,
)

REPO = Path(__file__).resolve().parents[2]

FAMILY_DEFAULT = """\
outputs:
  root_dir: family_artifacts
  evaluation:
    dir: evaluation
  tensorboard:
    enabled: true
MARL:
  training: {}
"""
SCENARIO_BASE = """\
scenario:
  type: intersection
  simulation:
    max_episodes: 3
"""


def _family(tmp_path, profile_text):
    family = tmp_path / "configs" / "sumo-marl"
    (family / "intersection").mkdir(parents=True)
    (family / "default.yaml").write_text(FAMILY_DEFAULT, encoding="utf-8")
    (family / "intersection" / "base.yaml").write_text(SCENARIO_BASE, encoding="utf-8")
    (family / "intersection" / "p.yaml").write_text(profile_text, encoding="utf-8")
    return tmp_path / "configs"


def _load_configuration(config_root, monkeypatch):
    """opencda.py's own resolution (three layers and includes) of ``p``."""
    # opencda.py sets a logging default at import; keep it out of the session.
    monkeypatch.setenv("LOGURU_LEVEL", os.environ.get("LOGURU_LEVEL", "INFO"))
    spec = importlib.util.spec_from_file_location("opencda_entry", REPO / "opencda.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    opt = argparse.Namespace(marl=True, test_scenario="sumo-marl/intersection/p",
                             marl_config_root=[str(config_root)])
    return entry.load_configuration(opt)


def test_an_empty_profile_section_resolves_to_the_family_section(tmp_path, monkeypatch):
    monkeypatch.delenv(MARL_CONFIG_ROOTS_ENV, raising=False)
    root = _family(tmp_path, "outputs:\nMARL:\nscenario:\n  simulation:\n")

    cfg = OmegaConf.to_container(_load_configuration(root, monkeypatch), resolve=True)

    assert cfg["outputs"] == {"root_dir": "family_artifacts",
                              "evaluation": {"dir": "evaluation"},
                              "tensorboard": {"enabled": True}}
    assert cfg["MARL"] == {"training": {}}
    assert cfg["scenario"]["simulation"] == {"max_episodes": 3}


def test_the_wheel_cli_resolves_an_empty_section_the_same_way(tmp_path, monkeypatch):
    from opencda_marl.cli import _build_config

    monkeypatch.delenv(MARL_CONFIG_ROOTS_ENV, raising=False)
    _family(tmp_path, "outputs:\n")
    monkeypatch.chdir(tmp_path)

    cfg = _build_config("sumo-marl/intersection/p", [])

    assert cfg.outputs.root_dir == "family_artifacts"
    assert cfg.get("outputs", {}).get("evaluation", {}).get("dir") == "evaluation"


def test_a_non_empty_override_still_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv(MARL_CONFIG_ROOTS_ENV, raising=False)
    root = _family(tmp_path, "outputs:\n  root_dir: mine\n  tensorboard: false\n")

    cfg = OmegaConf.to_container(_load_configuration(root, monkeypatch), resolve=True)

    assert cfg["outputs"] == {"root_dir": "mine",
                              "evaluation": {"dir": "evaluation"},
                              "tensorboard": False}


def test_none_still_overrides_a_value_that_is_not_a_mapping():
    base = OmegaConf.create({"MARL": {"training": {"load_checkpoint": "ckpt/x"}},
                             "scenario": {"traffic": {"excluded_vehicle_types": ["bike"]}},
                             "outputs": {"root_dir": "a", "evaluation": {"dir": "e"}}})
    override = OmegaConf.create("MARL:\n  training:\n    load_checkpoint: null\n"
                                "scenario:\n  traffic:\n    excluded_vehicle_types:\n"
                                "outputs:\n  root_dir:\n  evaluation:\n"
                                "world:\n")

    merged = OmegaConf.to_container(smart_merge(base, override))

    assert merged["MARL"]["training"]["load_checkpoint"] is None
    assert merged["scenario"]["traffic"]["excluded_vehicle_types"] is None
    assert merged["outputs"] == {"root_dir": None, "evaluation": {"dir": "e"}}
    assert merged["world"] is None          # no lower layer holds a mapping to keep
    assert override.outputs.evaluation is None, "the caller's override is not mutated"


@pytest.mark.parametrize("lock", [OmegaConf.set_struct, OmegaConf.set_readonly],
                         ids=["struct", "readonly"])
def test_a_struct_or_read_only_override_still_merges(lock):
    # Hydra hands out struct configs, and from_scenario's overrides may be
    # one: the flags guard the caller's object, not the merger's own copy.
    base = OmegaConf.create({"outputs": {"root_dir": "a", "evaluation": {"dir": "e"}}})
    override = OmegaConf.create({"outputs": {"root_dir": "b", "evaluation": None}})
    lock(override, True)

    merged = OmegaConf.to_container(smart_merge(base, override))

    assert merged == {"outputs": {"root_dir": "b", "evaluation": {"dir": "e"}}}
    assert override.outputs.evaluation is None


def test_a_struct_override_may_carry_extend_fields():
    base = OmegaConf.create({"scenario": {"traffic": {"excluded_vehicle_types": ["bike"]}}})
    override = OmegaConf.create({
        "_extend_fields": ["scenario.traffic.excluded_vehicle_types"],
        "scenario": {"traffic": {"excluded_vehicle_types": ["truck"]}},
    })
    OmegaConf.set_struct(override, True)

    merged = OmegaConf.to_container(smart_merge(base, override))

    assert merged == {"scenario": {"traffic": {"excluded_vehicle_types": ["bike", "truck"]}}}
    assert "_extend_fields" in override


@pytest.mark.parametrize("base, override", [
    ({"outputs": {"root_dir": "a"}}, [1, 2]),
    ([1, 2], {"outputs": None}),
], ids=["list-override", "list-base"])
def test_a_layer_that_is_not_a_mapping_gets_omegaconfs_own_error(base, override):
    # A profile written as a YAML list by mistake: the merge, not the empty
    # section rule, explains what is wrong.
    with pytest.raises(ConfigTypeError, match="Cannot merge"):
        smart_merge(OmegaConf.create(base), OmegaConf.create(override))


def test_normalize_run_config_gives_null_sections_an_empty_mapping():
    config = {"outputs": None, "MARL": None, "world": {"seed": 7}, "description": None}

    assert normalize_run_config(config) is config
    assert config == {"outputs": {}, "MARL": {}, "world": {"seed": 7}, "description": None}
    assert "outputs" not in normalize_run_config({}), "an absent section stays absent"

    dict_config = OmegaConf.create({section: None for section in RUN_CONFIG_SECTIONS})
    normalize_run_config(dict_config)
    for section in RUN_CONFIG_SECTIONS:
        assert dict_config.get(section, {}).get("anything", "default") == "default", section
