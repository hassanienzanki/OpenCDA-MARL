"""Vehicle-only OpenDRIVE worlds must not start pedestrian navmesh builds."""
from unittest.mock import MagicMock

import pytest

from opencda.scenario_testing.utils.customized_map_api import load_customized_world


@pytest.mark.parametrize('enabled', [False, True])
def test_customized_world_navigation_is_explicit(tmp_path, enabled):
    xodr = tmp_path / 'roads.xodr'
    xodr.write_text('<OpenDRIVE/>')
    client = MagicMock()

    result = load_customized_world(
        str(xodr), client, enable_pedestrian_navigation=enabled)

    source, parameters = client.generate_opendrive_world.call_args.args
    assert source == '<OpenDRIVE/>'
    assert parameters.enable_pedestrian_navigation is enabled
    assert parameters.smooth_junctions is False
    assert parameters.enable_mesh_visibility is True
    assert parameters.additional_width == pytest.approx(0.6)
    assert result is client.generate_opendrive_world.return_value


def test_customized_world_defaults_to_no_pedestrian_navigation(tmp_path):
    xodr = tmp_path / 'roads.xodr'
    xodr.write_text('<OpenDRIVE/>')
    client = MagicMock()
    load_customized_world(str(xodr), client)
    assert client.generate_opendrive_world.call_args.args[1].enable_pedestrian_navigation is False


@pytest.mark.parametrize('setting', [None, False, True])
def test_scenario_manager_passes_navigation_setting(monkeypatch, tmp_path, setting):
    from opencda.scenario_testing.utils import sim_api

    client = MagicMock()
    monkeypatch.setattr(sim_api.carla, 'Client', lambda *_: client)
    monkeypatch.setattr(sim_api.ScenarioManager, 'set_weather', lambda *_: object())
    xodr = tmp_path / 'roads.xodr'
    xodr.write_text('<OpenDRIVE/>')
    world = {'sync_mode': True, 'weather': {}}
    if setting is not None:
        world['pedestrian_navigation'] = setting

    sim_api.ScenarioManager({'world': world}, apply_ml=False, xodr_path=str(xodr))

    assert client.generate_opendrive_world.call_args.args[1].enable_pedestrian_navigation is (setting is True)
