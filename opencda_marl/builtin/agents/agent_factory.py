'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-29 20:36:17
FilePath     : /OpenCDA-MARL/opencda_marl/agents/agent_factory.py
Description  : 
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from typing import Dict, Any
import copy
from collections.abc import Mapping

import carla

from opencda_marl.builtin.agents.marl_behavior_agent import MARLBehaviorAgent
from opencda_marl.builtin.agents.vanilla_agent import VanillaAgent
from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent
from opencda_marl.builtin.agents.marl_agent import MARLAgent


class AgentFactory:
    @staticmethod
    def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in overrides.items():
            if (
                key in merged
                and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = AgentFactory._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def get_available_types() -> list:
        """Every registered agent type (built-ins plus plugin-registered)."""
        from opencda_marl.api.components import registered_components

        return list(registered_components("agent"))

    @staticmethod
    def get_baseline_types() -> list:
        """Get list of baseline agent types for benchmarking."""
        return ["behavior", "vanilla", "rule_based"]
    
    @staticmethod
    def get_rl_types() -> list:
        """Get list of reinforcement learning agent types."""
        return ["marl"]

    @staticmethod
    def validate_type(agent_type: str) -> bool:
        """Check if agent type is registered."""
        return agent_type.lower() in AgentFactory.get_available_types()

    @staticmethod
    def get_agent(agent_type: str, vehicle: carla.Vehicle, carla_map: carla.Map,
                  config: Dict[str, Any]):
        """Build the agent registered under ``agent_type``.

        Agent types are registered components: an external package registers
        a factory with ``register_component("agent", "my_driver")`` taking
        ``(vehicle, carla_map, config)`` (the full agents config — read your
        own section from it) and selects it with ``agents.agent_type``.
        """
        from opencda_marl.api.components import resolve_component

        return resolve_component("agent", agent_type.lower(), config=config)(vehicle, carla_map, config)


def _merged_behavior_config(config, section):
    return AgentFactory._deep_merge(
        config.get('vehicle', {}).get('behavior', {}), config.get(section) or {})


# The engine's default agent factories, wired declaratively into the
# component registry (opencda_marl.api.components._BUILTINS); an external
# package registers its own with register_component("agent", "<key>").
#
# Every built-in reads its own section over the shared vehicle.behavior
# block, so a type a profile only names in agents.population.mix (and gives
# no section of its own) still gets the keys its constructor requires.

def build_behavior_agent(vehicle, carla_map, config):
    return MARLBehaviorAgent(vehicle, carla_map, _merged_behavior_config(config, "behavior"))


def build_vanilla_agent(vehicle, carla_map, config):
    return VanillaAgent(vehicle, carla_map, _merged_behavior_config(config, "vanilla"))


def build_rule_based_agent(vehicle, carla_map, config):
    return RuleBasedAgent(vehicle, carla_map, _merged_behavior_config(config, "rule_based"))


def build_marl_agent(vehicle, carla_map, config):
    return MARLAgent(vehicle, carla_map, _merged_behavior_config(config, "marl"))
