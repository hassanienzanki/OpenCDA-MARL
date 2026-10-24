"""Config locator tests covering family refs and custom config roots."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from opencda_marl.utils.config_locator import resolve_marl_config_paths


def test_three_part_ref_resolves(tmp_path: Path):
    family = tmp_path / "configs" / "pcma"
    family.mkdir(parents=True)
    (family / "default.yaml").write_text("dummy: 1")
    (family / "intersection").mkdir()
    (family / "intersection" / "base.yaml").write_text("dummy: 1")
    (family / "intersection" / "smoke.yaml").write_text("dummy: 1")

    result = resolve_marl_config_paths(tmp_path, "pcma/intersection/smoke")
    assert result["family"] == "pcma"
    assert result["scenario_name"] == "intersection"
    assert result["profile_name"] == "smoke"
    assert result["scenario_yaml"].name == "smoke.yaml"


def test_momarl_route_profile_ref_resolves(tmp_path: Path):
    family = tmp_path / "configs" / "momarl"
    family.mkdir(parents=True)
    (family / "default.yaml").write_text("dummy: 1")
    (family / "intersection").mkdir()
    (family / "intersection" / "base.yaml").write_text("dummy: 1")
    (family / "intersection" / "learn_path_pcmappo.yaml").write_text("dummy: 1")

    result = resolve_marl_config_paths(
        tmp_path, "momarl/intersection/learn_path_pcmappo"
    )
    assert result["profile_name"] == "learn_path_pcmappo"
    assert result["scenario_yaml"].parent.name == "intersection"
    assert result["scenario_base_yaml"] == family / "intersection" / "base.yaml"


def test_custom_config_root_ref_resolves(tmp_path: Path):
    config_root = tmp_path / "custom_configs"
    family = config_root / "momarl"
    family.mkdir(parents=True)
    (family / "default.yaml").write_text("dummy: 1")
    (family / "intersection").mkdir()
    (family / "intersection" / "base.yaml").write_text("dummy: 1")
    (family / "intersection" / "speed_only_pcma.yaml").write_text("dummy: 1")

    result = resolve_marl_config_paths(
        tmp_path,
        "momarl/intersection/speed_only_pcma",
        custom_config_roots=[config_root],
    )
    assert result["family"] == "momarl"
    assert result["family_root"] == family
    assert result["scenario_yaml"] == family / "intersection" / "speed_only_pcma.yaml"


def test_missing_profile_raises(tmp_path: Path):
    family = tmp_path / "configs" / "pcma" / "intersection"
    family.mkdir(parents=True)
    (tmp_path / "configs" / "pcma" / "default.yaml").write_text("dummy: 1")
    (family / "base.yaml").write_text("dummy: 1")
    with pytest.raises(FileNotFoundError):
        resolve_marl_config_paths(tmp_path, "pcma/intersection/missing")


def test_too_short_ref_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_marl_config_paths(tmp_path, "momarl/intersection")


def test_unknown_family_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_marl_config_paths(tmp_path, "fake-family/intersection/smoke")
