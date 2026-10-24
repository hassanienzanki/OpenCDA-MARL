"""What the bundled profiles promise, checked without a simulator.

* An evaluation profile (``MARL.training.training_mode: false``) loads the
  checkpoint it names, and that name resolves inside the package -- a
  checkout-relative ``checkpoints/...`` path does not exist in a wheel or
  in a fresh clone, and the loader refuses to evaluate untrained weights.
* Every RAL kit entry's profile names that entry's checkpoint, so running
  the profile evaluates exactly the weights the manifest pairs it with
  (``test_bundled_baselines`` loads those weights into the same profile).
* The rule-based baseline states its lane-hold switches instead of
  inheriting them from code defaults.
* The SUMO family is headless unless a profile asks for the window.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencda_marl._bundled import BUNDLED_CHECKPOINTS, BUNDLED_CONFIG_ROOT, bundled_checkpoint
from opencda_marl.tools.config_lint import iter_scenario_refs, merged_config

REPO = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((BUNDLED_CHECKPOINTS / "ral" / "manifest.json").read_text(encoding="utf-8"))
ENTRIES = MANIFEST["entries"]


def _training(profile_ref: str) -> dict:
    return dict(merged_config(REPO, profile_ref, BUNDLED_CONFIG_ROOT).get("MARL", {}).get("training") or {})


def _eval_profiles() -> list[str]:
    refs = []
    for ref in iter_scenario_refs(BUNDLED_CONFIG_ROOT):
        if _training(ref).get("training_mode", True) is False:
            refs.append(ref)
    return refs


EVAL_PROFILES = _eval_profiles()


def test_the_eval_profiles_are_found():
    # Guards the parametrisation below against silently testing nothing.
    assert {e["profile"] for e in ENTRIES} <= set(EVAL_PROFILES)


@pytest.mark.parametrize("profile_ref", EVAL_PROFILES)
def test_an_eval_profile_names_a_packaged_checkpoint(profile_ref):
    load = str(_training(profile_ref).get("load_checkpoint") or "").strip()
    assert load, f"{profile_ref}: training_mode false with no load_checkpoint"
    assert bundled_checkpoint(load) is not None, (
        f"{profile_ref}: load_checkpoint {load!r} does not resolve inside the package")


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["run"] for e in ENTRIES])
def test_a_kit_profile_loads_its_own_entry(entry):
    load = str(_training(entry["profile"]).get("load_checkpoint") or "")
    assert bundled_checkpoint(load) == bundled_checkpoint(f"ral/{entry['run']}"), (
        f"{entry['profile']} loads {load!r}, not its manifest entry ral/{entry['run']}")


@pytest.mark.parametrize("profile_ref", EVAL_PROFILES)
def test_the_env_loader_accepts_the_profile_value(profile_ref):
    """The value reaches the manager as the packaged file through the CARLA
    environment's own loader, which raises on anything it cannot find."""
    module = importlib.import_module("opencda_marl._kernel.envs.marl_env")
    env = object.__new__(module.MARLEnv)
    calls: list[str] = []
    env.marl_manager = SimpleNamespace(load_checkpoint=lambda p: calls.append(p))
    load = str(_training(profile_ref)["load_checkpoint"])
    env._load_checkpoint_from_config(load)
    assert calls == [str(bundled_checkpoint(load))]


def test_undistributed_weights_ship_as_a_training_profile():
    """td3_44d has no weights in the package: it must train, not evaluate an
    untrained network or fail at start-up on a missing checkpoint."""
    training = _training("marl/intersection/td3_44d")
    assert training.get("training_mode", True) is True
    assert not str(training.get("load_checkpoint") or "").strip()


def test_rule_based_profile_states_its_lane_hold_switches():
    agent = merged_config(REPO, "marl/intersection/rule_based", BUNDLED_CONFIG_ROOT)["agents"]["rule_based"]
    assert agent["lane_hold"] is True
    assert agent["lane_hold_geometry_priority"] is True


def test_a_sumo_profile_that_names_no_window_runs_headless(tmp_path):
    """A new profile inherits the family's GUI setting; the family default is
    headless, so a profile written on a display-less machine runs."""
    from opencda_marl.cli import _build_config

    profile = tmp_path / "sumo-marl" / "intersection" / "my_quick.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("scenario:\n  simulation:\n    max_episodes: 5\n", encoding="utf-8")
    cfg = _build_config("sumo-marl/intersection/my_quick", [str(tmp_path)])
    assert cfg["world"]["sumo_gui"] is False
