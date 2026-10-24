"""A built-in agent type named only in a population mix still gets a full config.

``agents.population.mix: {marl: 0.5, rule_based: 0.5}`` puts rule-based
drivers on a profile that has no ``agents.rule_based`` section. The vanilla
and rule-based factories read their section alone, and the vanilla
constructor indexes four keys, so the first such spawn raised KeyError and
took the episode down. Every built-in now reads its own section over the
shared ``vehicle.behavior`` block, as the MARL agent always did.
"""
from pathlib import Path

import pytest
import yaml

pytest.importorskip("carla", reason="the agent factories build CARLA agents; no carla package here")

from opencda_marl.builtin.agents import agent_factory  # noqa: E402

#: The keys VanillaAgent (and so RuleBasedAgent) indexes without a default.
REQUIRED = ("ignore_traffic_light", "max_speed", "emergency_param", "collision_time_ahead")

_DEFAULT = Path(agent_factory.__file__).resolve().parents[2] / "_bundled" / "configs" / "marl" / "default.yaml"


def _family_agents_block():
    """The agents block every bundled MARL profile inherits (no per-type sections)."""
    agents = yaml.safe_load(_DEFAULT.read_text(encoding="utf-8"))["agents"]
    assert "rule_based" not in agents and "vanilla" not in agents
    return agents


def _captured(monkeypatch, cls_name, build, config):
    seen = {}
    monkeypatch.setattr(agent_factory, cls_name,
                        lambda vehicle, carla_map, cfg: seen.setdefault("config", cfg))
    build(None, None, config)
    return seen["config"]


@pytest.mark.parametrize("cls_name,build", [
    ("RuleBasedAgent", agent_factory.build_rule_based_agent),
    ("VanillaAgent", agent_factory.build_vanilla_agent),
])
def test_a_type_without_its_own_section_gets_the_shared_behavior_keys(monkeypatch, cls_name, build):
    config = _captured(monkeypatch, cls_name, build, _family_agents_block())
    missing = [key for key in REQUIRED if key not in config]
    assert not missing, f"{cls_name} would raise KeyError on {missing}"


@pytest.mark.parametrize("section,cls_name,build", [
    ("rule_based", "RuleBasedAgent", agent_factory.build_rule_based_agent),
    ("vanilla", "VanillaAgent", agent_factory.build_vanilla_agent),
])
def test_the_types_own_section_still_wins(monkeypatch, section, cls_name, build):
    agents = _family_agents_block()
    agents[section] = {"max_speed": 65, "ignore_traffic_light": True,
                       "local_planner": {"debug": False}}
    config = _captured(monkeypatch, cls_name, build, agents)
    assert config["max_speed"] == 65 and config["ignore_traffic_light"] is True
    assert config["local_planner"]["debug"] is False
    # Shared keys the section does not restate still arrive underneath it.
    assert config["emergency_param"] == agents["vehicle"]["behavior"]["emergency_param"]
