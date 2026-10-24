"""Lifecycle: coordinator, pipeline, world/scenario management.

Exports resolve lazily (PEP 562) to keep the envs<->lifecycle import
cycle broken regardless of which module a caller touches first.
"""

__all__ = ["MARLCoordinator", "PipelineBuilder", "MARLScenarioManager", "WorldManager"]


def __getattr__(name):
    if name == "MARLCoordinator":
        from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator
        return MARLCoordinator
    if name == "PipelineBuilder":
        from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder
        return PipelineBuilder
    if name == "MARLScenarioManager":
        from opencda_marl._kernel.lifecycle.scenario_manager import MARLScenarioManager
        return MARLScenarioManager
    if name == "WorldManager":
        from opencda_marl._kernel.lifecycle.world_manager import WorldManager
        return WorldManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
