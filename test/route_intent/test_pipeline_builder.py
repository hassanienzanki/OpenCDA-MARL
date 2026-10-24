"""PipelineBuilder + MARLManager.intent_provider cached property contract."""
from __future__ import annotations
import functools


def test_pipeline_builder_imports():
    from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder
    assert PipelineBuilder is not None


def test_pipeline_builder_exposes_expected_methods():
    from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder
    expected = {
        "build_scenario", "build_marl_manager", "build_env",
        "build_world_manager", "build_trace_writer", "wire_intent_provider",
        "build_output_section_config",
    }
    for name in expected:
        assert hasattr(PipelineBuilder, name), f"missing method: {name}"


def test_intent_provider_is_cached_property():
    """MARLManager.intent_provider must be a functools.cached_property."""
    from opencda_marl._kernel.policy import MARLManager
    descriptor = MARLManager.__dict__.get("intent_provider")
    assert descriptor is not None, "MARLManager must expose intent_provider"
    assert isinstance(descriptor, functools.cached_property), (
        f"intent_provider must be functools.cached_property, got {type(descriptor)}"
    )


def test_marl_env_no_longer_double_binds():
    """MARLEnv.__init__ no longer calls _bind_marl_manager_to_scenario twice."""
    import inspect
    from opencda_marl.envs import marl_env
    src = inspect.getsource(marl_env)
    # _bind_marl_manager_to_scenario should be gone entirely (was the double-call workaround).
    assert "_bind_marl_manager_to_scenario" not in src, (
        "MARLEnv should no longer use the _bind_marl_manager_to_scenario workaround "
        "after PipelineBuilder centralizes the binding."
    )


def test_build_scenario_propagates_map_config():
    """Regression: build_scenario MUST propagate scenario.map.{town,xodr_path}.

    An earlier implementation called MARLScenarioManager(...) directly without
    passing town and xodr_path, so CARLA stayed on whatever map was last loaded.
    A run then failed at planner.get_random_route_by_direction with
    KeyError: 4 because the active_junctions config said junction 4 but the
    previously loaded map didn't have it.

    Verifies build_scenario either delegates to ScenarioBuilder.build_from_config
    (which extracts and forwards the map config) OR explicitly references both
    town and xodr_path in its body.
    """
    import inspect
    from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder
    src = inspect.getsource(PipelineBuilder.build_scenario)
    delegates_via_scenario_builder = "ScenarioBuilder" in src
    references_map_fields = ("town" in src) and ("xodr_path" in src)
    assert delegates_via_scenario_builder or references_map_fields, (
        "PipelineBuilder.build_scenario must propagate scenario.map.town and "
        "scenario.map.xodr_path; otherwise CARLA stays on the cached map. "
        "Either delegate to ScenarioBuilder.build_from_config or pass town "
        "and xodr_path explicitly."
    )


def test_wire_intent_provider_updates_marl_manager_back_reference():
    """Regression: wire_intent_provider must rebind marl_manager.agent_manager.

    Without this back-reference update, post-world-reset rebuilds leave
    the marl_manager pointing at the destroyed old agent_manager. The
    intent provider's cached closure (_get_active_adapters_for_intent)
    then returns [] permanently, and -- more critically -- the manager
    pair the marl_manager thinks it owns no longer matches what the
    scenario manager actually drives. Symptom: episodes 2+ after a
    periodic WorldManager reset spawn 0 vehicles, while with the reset
    disabled every episode spawns and trains normally.

    The PipelineBuilder rewrite dropped the back-reference assignment
    that the earlier _rebuild_scenario_components had inlined. This test
    asserts the bidirectional binding lives in wire_intent_provider so
    every rebuild path picks it up uniformly.
    """
    import inspect
    from opencda_marl._kernel.lifecycle.pipeline_builder import PipelineBuilder
    src = inspect.getsource(PipelineBuilder.wire_intent_provider)
    assert "marl_manager.agent_manager" in src and "scenario.agent_manager" in src, (
        "wire_intent_provider must update marl_manager.agent_manager to "
        "point at the fresh scenario.agent_manager (bidirectional rebind). "
        "Without it, episodes 2+ after a world reset cannot spawn vehicles."
    )
