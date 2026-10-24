"""A CARLA world build that times out is retried once before the run fails.

On a loaded machine generating the OpenDRIVE world can outlast the client
timeout (``world.timeout``); a second attempt usually succeeds.
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("carla", reason="the vendored scenario manager imports carla")

from opencda.scenario_testing.utils import sim_api  # noqa: E402

TIMEOUT = RuntimeError(
    "time-out of 20000ms while waiting for the simulator, make sure the "
    "simulator is ready and connected to 127.0.0.1:2000")


class _Build:
    """A world build that fails with the given errors, then returns a world."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = 0
        self.world = MagicMock(name="world")

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.world


def test_a_timed_out_build_is_retried_once_with_a_longer_timeout():
    client = MagicMock()
    build = _Build(TIMEOUT)
    assert sim_api.build_world_with_retry(build, client, 20.0) is build.world
    assert build.calls == 2
    # The retry waits twice as long; the configured timeout is restored after.
    assert [c.args[0] for c in client.set_timeout.call_args_list] == [40.0, 20.0]


def test_a_build_that_succeeds_first_time_leaves_the_client_alone():
    client = MagicMock()
    build = _Build()
    assert sim_api.build_world_with_retry(build, client, 20.0) is build.world
    assert build.calls == 1
    client.set_timeout.assert_not_called()


def test_a_second_timeout_propagates_and_restores_the_timeout():
    client = MagicMock()
    build = _Build(TIMEOUT, TIMEOUT)
    with pytest.raises(RuntimeError, match="time-out"):
        sim_api.build_world_with_retry(build, client, 20)
    assert build.calls == 2
    assert client.set_timeout.call_args_list[-1].args[0] == 20


def test_other_errors_are_not_retried():
    client = MagicMock()
    build = _Build(RuntimeError("map not found"))
    with pytest.raises(RuntimeError, match="map not found"):
        sim_api.build_world_with_retry(build, client, 20.0)
    assert build.calls == 1
    client.set_timeout.assert_not_called()


@pytest.mark.parametrize("error, expected", [
    (TIMEOUT, True),
    (RuntimeError("rpc call timed out"), True),
    (RuntimeError("map not found"), False),
    (ValueError("time-out"), False),
])
def test_client_timeout_detection(error, expected):
    assert sim_api.is_client_timeout(error) is expected


def _world(timeout=None):
    world = {'sync_mode': True, 'weather': {}}
    if timeout is not None:
        world['timeout'] = timeout
    return world


def test_scenario_manager_retries_the_opendrive_build(monkeypatch, tmp_path):
    client = MagicMock()
    monkeypatch.setattr(sim_api.carla, 'Client', lambda *_: client)
    monkeypatch.setattr(sim_api.ScenarioManager, 'set_weather', lambda *_: object())
    build = _Build(TIMEOUT)
    monkeypatch.setattr(sim_api, 'load_customized_world', lambda *a, **k: build())
    xodr = tmp_path / 'roads.xodr'
    xodr.write_text('<OpenDRIVE/>')

    manager = sim_api.ScenarioManager(
        {'world': _world(timeout=25.0)}, apply_ml=False, xodr_path=str(xodr))

    assert manager.world is build.world
    assert build.calls == 2
    # world.timeout configures the client; the retry doubles it, then restores.
    assert [c.args[0] for c in client.set_timeout.call_args_list] == [25.0, 50.0, 25.0]


def test_scenario_manager_reports_a_town_timeout_as_a_timeout(monkeypatch):
    client = MagicMock()
    client.load_world.side_effect = [TIMEOUT, TIMEOUT]
    monkeypatch.setattr(sim_api.carla, 'Client', lambda *_: client)
    monkeypatch.setattr(sim_api.ScenarioManager, 'set_weather', lambda *_: object())

    with pytest.raises(RuntimeError, match="time-out"):
        sim_api.ScenarioManager({'world': _world()}, apply_ml=False, town='Town05')
    assert client.load_world.call_count == 2
