"""Every shipped scenario profile must resolve and validate statically.

Configs can drift from two contracts (canonical algorithm names,
MARL.action_space) and only fail when launched. The lint runs the same
resolvers the manager uses, without a simulator, so drift is caught here
first.
"""
from pathlib import Path

import pytest

from opencda_marl.tools.config_lint import (
    iter_scenario_refs,
    lint_all,
    lint_marl_block,
    lint_scenario,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "opencda_marl" / "_bundled" / "configs"


def test_every_shipped_profile_lints_clean():
    results = lint_all(REPO_ROOT, CONFIG_ROOT)
    assert results, "no scenario profiles discovered under configs/"
    problems = {r.scenario_ref: r.errors for r in results if not r.ok}
    assert not problems, "config drift:\n" + "\n".join(
        f"  {ref}: {errs}" for ref, errs in problems.items())


def test_lint_discovers_all_families():
    refs = list(iter_scenario_refs(CONFIG_ROOT))
    families = {r.split("/")[0] for r in refs}
    assert {"marl", "momarl", "pcma", "sumo-marl"} <= families


# --------------------------------------------------------------------- #
# The lint must catch the known drift classes.
# --------------------------------------------------------------------- #
def _base_marl():
    return {
        "algorithm": "masac",
        "action_space": {"mode": "speed_only"},
        "reward_profile": "cooperative_intersection_v1",
        "state_dim": 3,
        "masac": {"features": {"rel_x": 1, "rel_y": 1, "speed": 1}},
    }


def test_lint_catches_an_unregistered_algorithm_name():
    cfg = _base_marl()
    cfg["algorithm"] = "sac"  # the 1.x spelling; 2.x registers masac
    errors = lint_marl_block(cfg)
    assert any("MARL.algorithm" in e and "Unsupported" in e for e in errors)


def test_lint_catches_missing_action_space():
    cfg = _base_marl()
    del cfg["action_space"]
    errors = lint_marl_block(cfg)
    assert any("MARL.action_space" in e for e in errors)


def test_lint_catches_missing_algorithm_section():
    cfg = _base_marl()
    del cfg["masac"]
    errors = lint_marl_block(cfg)
    assert any("MARL.masac" in e for e in errors)


def test_lint_catches_state_dim_mismatch():
    cfg = _base_marl()
    cfg["state_dim"] = 7  # features sum to 3, no preference
    errors = lint_marl_block(cfg)
    assert any("state_dim=7" in e for e in errors)


def test_lint_accepts_preference_augmented_width():
    """With an objective spec the env appends the preference vector even for
    non-preference-conditioned learners, so features + objectives is right."""
    cfg = _base_marl()
    cfg["objective_mode"] = "competitive_ordering_v1"
    cfg["state_dim"] = 3 + 2
    assert lint_marl_block(cfg) == []


def test_clean_profile_passes():
    result = lint_scenario(REPO_ROOT, "marl/intersection/smoke", CONFIG_ROOT)
    assert result.ok, result.errors
