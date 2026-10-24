"""Worked examples in the docs run as written.

The snippets are read out of the pages themselves, so an edit that breaks
one fails here instead of on a reader's first try.

- Quick Start, "Change one thing": the profile, dropped under an extra
  config root, resolves on its own (an action space, no SUMO window) and
  builds its learner with an observation width that agrees with the
  extractor's features plus the preference entries -- a profile that named
  only the learner used to stop at "MARL.action_space is required".
- An Intent Provider: the example registers a new key (no ``override``),
  and the documented action-space block routes the path decision to it
  through ``path.provider`` with no learned head. Its body runs end to end
  against a planner route table: it turns right when the mask allows a
  right turn that keeps the exit direction, keeps the original route when
  it does not, and abstains when the spawn has no entry direction.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"


def _block(page: str, heading: str, lang: str) -> str:
    """The first ``lang`` fenced block after ``heading`` on ``page``."""
    text = (DOCS / page).read_text(encoding="utf-8")
    start = text.index(heading)
    match = re.compile(rf"^```{lang}\n(.*?)^```", re.M | re.S).search(text, start)
    assert match, f"{page}: no ```{lang} block after {heading!r}"
    return match.group(1)


def test_quick_start_profile_resolves_and_builds(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from opencda_marl._kernel.policy.action_space import resolve_action_space_config
    from opencda_marl._kernel.policy.manager import MARLManager
    from opencda_marl.cli import _build_config

    snippet = _block("quick-start.md", "## 3. Change one thing", "yaml")
    first = snippet.splitlines()[0]
    match = re.match(r"#\s*(\S+)/(sumo-marl/intersection/my_quick\.yaml)$", first)
    assert match, f"the snippet names its own path on its first line: {first!r}"
    root = tmp_path / match.group(1)
    profile = root / match.group(2)
    profile.parent.mkdir(parents=True)
    profile.write_text(snippet, encoding="utf-8")

    monkeypatch.chdir(tmp_path)          # the manager writes run output to the cwd
    cfg = OmegaConf.to_container(
        _build_config("sumo-marl/intersection/my_quick", [str(root)]), resolve=True)

    assert cfg["world"]["sumo_gui"] is False, "the documented run is headless"
    assert cfg["scenario"]["simulation"]["max_episodes"] == 5
    marl = cfg["MARL"]
    assert marl["algorithm"] == "pcmasac"
    assert resolve_action_space_config(marl).mode == "speed_only"

    marl["outputs"] = {"metrics_store": {"enabled": False}, "tensorboard": {"enabled": False}}
    marl.setdefault("training", {})["save_checkpoints"] = False
    manager = MARLManager(marl, marl["algorithm"])
    assert manager.algorithm is not None
    assert manager.preference_enabled, "the objective switches preference conditioning on"
    derived = int(manager.observation_extractor.state_dim) + manager.preference_dim()
    assert derived == int(marl["state_dim"]) == manager.resolve_state_dim(), (
        "MARL.state_dim must equal the declared features plus the preference entries")


@pytest.fixture
def components():
    from opencda_marl.api import components as comp

    before = ({k: dict(v) for k, v in comp._COMPONENTS.items()},
              {k: dict(v) for k, v in comp._COMPONENT_SOURCES.items()})
    yield comp
    comp._COMPONENTS.clear(); comp._COMPONENTS.update(
        {k: dict(v) for k, v in before[0].items()})
    comp._COMPONENT_SOURCES.clear(); comp._COMPONENT_SOURCES.update(
        {k: dict(v) for k, v in before[1].items()})


def test_intent_provider_example_is_selected_by_provider_key(components):
    from opencda_marl._kernel.policy.action_space import (derive_route_intent_config,
                                                          resolve_action_space_config)
    from opencda_marl.api.intent import build_provider_from_config

    source = _block("marl/custom_intent_provider.md", "## Worked example", "python")
    assert "override=True" not in source, "the example adds a key; it replaces none"
    namespace: dict = {}
    exec(compile(source, "custom_intent_provider.md", "exec"), namespace)
    assert "prefer_right" in components.registered_components("intent_provider")

    marl = yaml.safe_load(_block("marl/custom_intent_provider.md",
                                 "Selection runs through the action space", "yaml"))["MARL"]
    action_space = resolve_action_space_config(marl)
    assert action_space.path.provider == "prefer_right"
    assert action_space.path.selector == "rule_based", "a built-in selector: no learned head"

    route_cfg = derive_route_intent_config({}, marl)
    assert route_cfg["mode"] == "prefer_right"
    provider = build_provider_from_config(route_cfg, planner=object())
    assert callable(provider)

    class _Event:                        # no entry direction: the example abstains
        metadata: dict = {}
    assert provider({"event": _Event()}) is None


def _route(route_id, route_type, exit_lane_idx, dest_x):
    return {"route_id": route_id, "route_type": route_type,
            "entry_direction": "north", "exit_direction": "south",
            "entry_lane_idx": 0, "exit_lane_idx": exit_lane_idx,
            "dest_tf": SimpleNamespace(location=SimpleNamespace(x=dest_x, y=0.0, z=0.0))}


class _Planner:
    """The one planner query the example makes: ``get_routes(junction_id)``."""

    def __init__(self, routes):
        self._routes = {1: routes}

    def get_routes(self, junction_id):
        return self._routes.get(int(junction_id), {})


def test_intent_provider_example_turns_right_only_when_the_mask_allows(components):
    from opencda_marl._kernel.policy.action_space import derive_route_intent_config
    from opencda_marl.api.intent import build_provider_from_config
    from opencda_marl.api.route_intent import RouteIntent, RouteIntentDecision

    exec(compile(_block("marl/custom_intent_provider.md", "## Worked example", "python"),
                 "custom_intent_provider.md", "exec"), {})
    marl = yaml.safe_load(_block("marl/custom_intent_provider.md",
                                 "Selection runs through the action space", "yaml"))["MARL"]
    route_cfg = derive_route_intent_config({}, marl)
    event = SimpleNamespace(junction_id=1, route_id=11,
                            metadata={"entry_direction": "north", "entry_lane_idx": 0})

    original = _route(11, "straight", 0, 0.0)
    right = _route(12, "right", 1, 1.0)      # same exit direction: a legal alternative
    with_right = build_provider_from_config(
        route_cfg, planner=_Planner({11: original, 12: right}))
    decision = with_right({"event": event})
    assert isinstance(decision, RouteIntentDecision)
    assert decision.mask[RouteIntent.RIGHT] is True
    assert decision.selected_intent == RouteIntent.RIGHT
    assert decision.selected_candidate.route_id == 12
    assert decision.original_route_id == 11 and decision.direction_preserved

    keep_only = build_provider_from_config(route_cfg, planner=_Planner({11: original}))
    decision = keep_only({"event": event})
    assert isinstance(decision, RouteIntentDecision)
    assert decision.mask[RouteIntent.RIGHT] is False
    assert decision.selected_intent == RouteIntent.KEEP
    assert decision.selected_candidate.route_id == 11 and decision.used_keep_current
