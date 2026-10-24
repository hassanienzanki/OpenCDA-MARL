"""A bundled CARLA baseline profile runs one episode.

The CARLA scenario manager honours ``scenario.simulation.max_episodes`` for
a baseline agent (``behavior`` / ``vanilla`` / ``rule_based``) and falls
back to one episode only when the merged config sets none. A family base
that sets a training horizon (``marl/intersection/base.yaml`` sets 500)
would otherwise turn a baseline profile into a 500-episode CARLA run, and
``scripts/verify.py --full`` and the quick-test tasks run these profiles.
Each bundled baseline profile therefore states its own one-episode horizon.
"""
from __future__ import annotations

import pytest

from opencda_marl._bundled import BUNDLED_CONFIG_ROOT
from opencda_marl.api.components import BASELINE_AGENT_TYPES
from opencda_marl.cli import _build_config
from opencda_marl.utils.config_locator import MARL_CONFIG_ROOTS_ENV

_SKIP_FILES = {"base.yaml", "default.yaml"}


def _profile_refs() -> list[str]:
    refs = []
    for profile in sorted(BUNDLED_CONFIG_ROOT.glob("*/*/*.yaml")):
        if profile.name in _SKIP_FILES:
            continue
        family, scenario = profile.parent.parent.name, profile.parent.name
        refs.append(f"{family}/{scenario}/{profile.stem}")
    return refs


@pytest.fixture
def carla_baselines(monkeypatch, tmp_path):
    """Every bundled CARLA baseline profile, resolved as the CLI resolves it,
    against the bundled configs alone (no checkout ``configs/`` under the
    working directory, no extra roots from the environment)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(MARL_CONFIG_ROOTS_ENV, raising=False)
    found = {}
    for ref in _profile_refs():
        merged = _build_config(ref, [])
        backend = str((merged.get("runtime") or {}).get("backend", "carla")).lower()
        agent_type = (merged.get("agents") or {}).get("agent_type")
        if backend == "carla" and agent_type in BASELINE_AGENT_TYPES:
            found[ref] = merged
    return found


def test_the_carla_baseline_profiles_are_found(carla_baselines):
    # Guards the check below against silently testing nothing.
    assert {"marl/intersection/behavior", "marl/intersection/rule_based",
            "marl/intersection/vanilla"} <= set(carla_baselines), sorted(carla_baselines)


def test_every_carla_baseline_profile_runs_one_episode(carla_baselines):
    longer = {}
    for ref, merged in carla_baselines.items():
        simulation = (merged.get("scenario") or {}).get("simulation") or {}
        if simulation.get("max_episodes", 1) != 1:
            longer[ref] = simulation.get("max_episodes")
    assert not longer, (
        f"baseline profiles inherit a multi-episode horizon {longer}; "
        "set scenario.simulation.max_episodes: 1 in each")
