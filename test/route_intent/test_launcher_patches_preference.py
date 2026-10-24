# test/route_intent/test_launcher_patches_preference.py
"""The matrix launcher's regime axis actually reaches the cell config.

The launcher patches `regime` and `seed` into the profile's preference
block. It addressed that block by its historical path
(`agents.route_intent.preference`), so when the canonical home became
`agents.preference` the patch silently stopped applying: every cell of a
`--regimes conflicting` sweep would have run in the profile's default
regime, and nothing would have failed. The launcher now resolves whichever
location the profile uses.
"""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "experiments"
CONFIGS = REPO / "opencda_marl" / "_bundled" / "configs" / "momarl" / "intersection"


def _launcher():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import importlib
    return importlib.import_module("launch_matrix")


@pytest.mark.parametrize("regime", ["aligned", "conflicting"])
def test_regime_lands_in_a_shipped_cell(regime, tmp_path):
    lm = _launcher()
    src = CONFIGS / "speed_only_pcmappo.yaml"
    dst = tmp_path / "cell.yaml"
    lm._patch_config(src, dst, regime=regime, seed=3, slug="t", condition="speed_only_pcmappo")

    cfg = OmegaConf.load(dst)
    assert cfg.agents.preference.regime == regime
    assert cfg.agents.preference.seed == 103          # seed + 100
    assert cfg.world.seed == 3


def test_the_legacy_location_is_still_patched(tmp_path):
    """An external profile written before the move keeps working."""
    from matrix_common import preference_block_path

    legacy = OmegaConf.create({"agents": {"route_intent": {"preference": {"mode": "per_vehicle"}}}})
    canonical = OmegaConf.create({"agents": {"preference": {"mode": "per_vehicle"}}})
    neither = OmegaConf.create({"agents": {"agent_type": "marl"}})

    assert preference_block_path(legacy) == "agents.route_intent.preference"
    assert preference_block_path(canonical) == "agents.preference"
    assert preference_block_path(neither) is None
