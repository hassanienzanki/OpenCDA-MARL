"""The engine wheel's bundled data is the source, not a copy.

Config families and recordings live inside the package
(``opencda_marl/_bundled``), so a pip install and a checkout resolve the
same files by the same rule — explicit roots first, then the bundle — and
nothing at the repository root shadows them. Every replay a shipped profile
names is packaged, so no profile depends on a file outside the wheel.
"""
from __future__ import annotations

import re
from pathlib import Path

from opencda_marl._bundled import BUNDLED_CONFIG_ROOT, bundled_recording

REPO = Path(__file__).resolve().parents[2]
FAMILIES = {"marl", "pcma", "momarl", "sumo-marl", "cosim-marl", "opencda"}
_REPLAY = re.compile(r"replay_file:\s*[\"']?([^\"'\s#]+)")


def test_bundle_is_the_only_config_home():
    assert not (REPO / "configs").exists(), "root configs/ came back"
    assert not (REPO / "recordings").exists(), "root recordings/ came back"
    families = {p.name for p in BUNDLED_CONFIG_ROOT.iterdir() if p.is_dir()}
    assert families == FAMILIES


def test_every_replay_a_profile_names_is_bundled():
    missing = set()
    for yaml_path in BUNDLED_CONFIG_ROOT.rglob("*.yaml"):
        for name in _REPLAY.findall(yaml_path.read_text(encoding="utf-8")):
            if bundled_recording(Path(name).name) is None:
                missing.add(f"{yaml_path.relative_to(REPO).as_posix()} -> {name}")
    assert not missing, sorted(missing)


def test_bundled_scenario_resolves_without_a_checkout_root(tmp_path):
    from opencda_marl.utils.config_locator import resolve_marl_config_paths

    resolved = resolve_marl_config_paths(tmp_path, "sumo-marl/intersection/random_speed")
    for key in ("default_yaml", "scenario_base_yaml", "scenario_yaml"):
        assert Path(resolved[key]).resolve().is_relative_to(BUNDLED_CONFIG_ROOT.resolve()), key
