"""The coordinator's run-start services: monitor, traffic view, plugins.

- ``outputs.monitor`` / ``outputs.traffic_view`` accept a bare bool, like
  the sibling output sections: ``false`` used to read as an empty dict and
  start the service anyway, and ``true`` crashed initialize() on ``True.get``.
- The traffic view writes the lane geometry before the first vehicle state,
  so a page that finds a state also finds something to draw it on.
- ``MARL.plugins`` is loaded before either backend builds anything: CARLA
  resolves the planner, the intent provider and the agent types while it
  builds the scenario, before the MARL manager (whose own load used to be
  the only one) exists.
"""
import json
from types import SimpleNamespace

import pytest

from opencda_marl._kernel.lifecycle.coordinator import MARLCoordinator


def _coordinator(config, env=None):
    coord = MARLCoordinator(config)
    coord.marl_env = env
    return coord


@pytest.fixture
def ensure_service_calls(monkeypatch):
    import opencda_marl.telemetry.monitor as monitor

    calls = []
    monkeypatch.setattr(monitor, "ensure_service",
                        lambda root, host, port, allow_hosts=(): calls.append((root, host, port)) or ("http://x", True))
    return calls


@pytest.mark.parametrize("value, started", [
    (False, False),
    (True, True),
    ({"enabled": False}, False),
    ({"enabled": True, "port": 9000}, True),
    ({}, True),
])
def test_the_monitor_honours_the_bool_shorthand(ensure_service_calls, value, started):
    coord = _coordinator({"outputs": {"root_dir": "r", "monitor": value}})
    coord._setup_monitor_service()
    assert bool(ensure_service_calls) is started


def test_an_absent_monitor_section_keeps_the_default_on(ensure_service_calls):
    _coordinator({"outputs": {"root_dir": "r"}})._setup_monitor_service()
    assert ensure_service_calls == [("r", "127.0.0.1", 8765)]


class _ViewEnv:
    def __init__(self, geometry=None):
        self.geometry = geometry or {"lanes": [[[0.0, 0.0], [1.0, 0.0]]], "junction": None, "y_up": True}
        self.geometry_calls = 0

    def collect_traffic_view(self):
        return [{"id": "a", "x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 1.0, "role": "agent"}], 0, 3

    def collect_traffic_geometry(self):
        self.geometry_calls += 1
        return self.geometry


@pytest.mark.parametrize("value, registered", [
    (False, False), (True, True), ({"enabled": False}, False), ({"interval": 0.5}, True),
])
def test_the_traffic_view_honours_the_bool_shorthand(value, registered):
    coord = _coordinator({"outputs": {"traffic_view": value}}, env=_ViewEnv())
    coord._setup_traffic_view()
    assert bool(coord.post_step_callbacks) is registered


def test_the_geometry_is_written_before_the_first_state(tmp_path, monkeypatch):
    from opencda_marl.telemetry.traffic_view import GEOMETRY_NAME, STATE_NAME

    env = _ViewEnv()
    coord = _coordinator({"outputs": {"traffic_view": {"interval": 0.05}}}, env=env)
    monkeypatch.setattr(coord, "_resolve_view_run_dir", lambda: tmp_path)
    seen = []
    import opencda_marl.telemetry.traffic_view as tv
    original = tv.TrafficViewPublisher.publish_state

    def publish_state(self, vehicles, *, episode, step):
        seen.append((tmp_path / GEOMETRY_NAME).is_file())
        return original(self, vehicles, episode=episode, step=step)

    monkeypatch.setattr(tv.TrafficViewPublisher, "publish_state", publish_state)
    coord._setup_traffic_view()
    coord.post_step_callbacks[0]()

    assert seen == [True], "the geometry exists by the time the first state is written"
    assert json.loads((tmp_path / STATE_NAME).read_text(encoding="utf-8"))["step"] == 3
    coord.post_step_callbacks[0]()                 # written once, not re-collected
    assert env.geometry_calls == 1


def test_a_failing_geometry_never_holds_back_the_state(tmp_path, monkeypatch):
    from opencda_marl.telemetry.traffic_view import GEOMETRY_NAME, STATE_NAME

    env = _ViewEnv()
    env.collect_traffic_geometry = lambda: (_ for _ in ()).throw(RuntimeError("no map yet"))
    coord = _coordinator({"outputs": {"traffic_view": True}}, env=env)
    monkeypatch.setattr(coord, "_resolve_view_run_dir", lambda: tmp_path)
    coord._setup_traffic_view()
    coord.post_step_callbacks[0]()

    assert (tmp_path / STATE_NAME).is_file()
    assert not (tmp_path / GEOMETRY_NAME).exists()


def test_marl_plugins_load_before_any_backend_is_built(monkeypatch):
    import opencda_marl.plugins.loader as loader

    loaded = []
    monkeypatch.setattr(loader, "load_plugins", lambda extra=None, **_: loaded.append(extra) or {})
    # An unknown backend fails right after the shared prelude: whatever ran
    # before the error ran before either backend could resolve a component.
    coord = _coordinator({"runtime": {"backend": "none"},
                          "MARL": {"plugins": ["my_components"]}})
    with pytest.raises(ValueError, match="Unsupported MARL runtime backend"):
        coord.initialize()
    assert loaded == [["my_components"]]


def test_carla_loads_marl_plugins_before_building_the_scenario(monkeypatch):
    pytest.importorskip("carla", reason="the CARLA branch imports the CARLA stack")
    import opencda_marl.plugins.loader as loader
    from opencda_marl._kernel.lifecycle import pipeline_builder
    import opencda.core.common.cav_world as cav_world

    order = []

    class _Stop(Exception):
        pass

    def build_scenario(self, cav):
        order.append("build_scenario")
        raise _Stop

    monkeypatch.setattr(loader, "load_plugins", lambda extra=None, **_: order.append(("plugins", extra)) or {})
    monkeypatch.setattr(pipeline_builder.PipelineBuilder, "build_scenario", build_scenario)
    monkeypatch.setattr(cav_world, "CavWorld", lambda apply_ml=False: SimpleNamespace())

    class _Config(dict):
        opt = SimpleNamespace(apply_ml=False)

    coord = _coordinator(_Config({"runtime": {"backend": "carla"},
                                  "MARL": {"plugins": ["my_planner"]}}))
    with pytest.raises(_Stop):
        coord.initialize()
    assert order == [("plugins", ["my_planner"]), "build_scenario"]


def test_monitor_allow_hosts_reach_the_service(monkeypatch):
    """outputs.monitor.allow_hosts is handed to ensure_service unchanged."""
    import opencda_marl.telemetry.monitor as monitor

    seen = {}
    monkeypatch.setattr(monitor, "ensure_service",
                        lambda root, host, port, allow_hosts=(): seen.update(allow_hosts=allow_hosts)
                        or ("http://x", True))
    coord = _coordinator({"outputs": {"root_dir": "r", "monitor": {"allow_hosts": ["proxy.test"]}}})
    coord._setup_monitor_service()
    assert seen["allow_hosts"] == ["proxy.test"]
