# test/route_intent/test_preference_config_home.py
"""``agents.preference`` is the canonical home of the spawn-time sampler.

The block historically lived under ``agents.route_intent.preference`` (a
misplacement: preferences condition objectives, not routes — the
speed-only profiles need them with no path action at all). The derived
intent config must honour both locations, canonical winning.
"""
from pathlib import Path

import yaml

from opencda_marl._kernel.policy.action_space import derive_route_intent_config

REPO = Path(__file__).resolve().parents[2]

PREF = {"mode": "per_vehicle", "sampler": "dirichlet", "dim": 2,
        "alpha": 1.0, "seed": 7, "regime": "aligned"}


def test_canonical_location_flows_into_intent_cfg():
    cfg = derive_route_intent_config({"preference": dict(PREF)}, {})
    assert cfg["preference"] == PREF


def test_legacy_location_still_works():
    cfg = derive_route_intent_config(
        {"route_intent": {"preference": dict(PREF)}}, {})
    assert cfg["preference"] == PREF


def test_canonical_wins_over_legacy():
    legacy = dict(PREF, regime="conflicting")
    cfg = derive_route_intent_config(
        {"preference": dict(PREF),
         "route_intent": {"preference": legacy}}, {})
    assert cfg["preference"]["regime"] == "aligned"


def test_shipped_profile_uses_canonical_location():
    """A migrated profile resolves end to end — file and shim pinned together."""
    profile = REPO / "opencda_marl/_bundled/configs/sumo-marl/intersection/route_speed_only_pcmappo.yaml"
    raw = yaml.safe_load(profile.read_text(encoding="utf-8"))
    agents = raw.get("agents") or {}
    assert "preference" in agents
    assert "route_intent" not in agents  # nothing left behind
    cfg = derive_route_intent_config(agents, raw.get("MARL") or {})
    assert cfg["preference"]["mode"] == "per_vehicle"
