"""CARLA observations and route triggers use the assigned junction's frame."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opencda_marl._kernel.agent_manager import MARLAgentManager
from opencda_marl._kernel.policy.decision_zone import DecisionZone


def manager_and_adapter(position, *, junction_id=4, centers=None):
    centers = centers or {4: SimpleNamespace(x=2.0, y=40.0)}
    manager = object.__new__(MARLAgentManager)
    manager.planner = SimpleNamespace(get_junctions=lambda: {
        key: {"center": value} for key, value in centers.items()})
    manager.intent_provider = Mock(return_value=None)
    manager._decision_zone = DecisionZone('approach', 30)
    manager._decision_zone.register('ego')
    manager._spawn_event_by_actor_id = {'ego': SimpleNamespace(junction_id=junction_id)}
    manager._spawn_metadata_by_actor_id = {}
    vehicle = SimpleNamespace(type_id='vehicle.test', get_location=lambda: position)
    adapter = SimpleNamespace(actor_id='ego', vehicle=vehicle,
        _calculate_distance_to_intersection=lambda: 10.0,
        get_observation=lambda **kwargs: {
            'position_x': position.x, 'position_y': position.y,
            'relative_position_to_intersection': {'x': -5.0, 'y': 8.0},
            'distance_to_intersection': 10.0,
            'location': {'x': -5.0, 'y': 8.0},
        })
    manager._vehicle_adapters = [adapter]
    return manager, adapter


def test_observation_is_vehicle_minus_assigned_center():
    manager, _ = manager_and_adapter(SimpleNamespace(x=7.0, y=25.0))
    observations = manager._collect_observations({}, None)
    obs = observations['ego']
    assert obs['relative_position_to_intersection'] == pytest.approx({'x': 5, 'y': -15})
    assert obs['distance_to_intersection'] == pytest.approx((5**2 + 15**2)**0.5)
    assert obs['position_x'] == 7.0
    assert obs['location'] == {'x': -5.0, 'y': 8.0}  # legacy alias is preserved


def test_decision_does_not_fire_early_at_thirty_metres_from_entry():
    position = SimpleNamespace(x=2.0, y=80.0)        # 40 m from the centre
    manager, _ = manager_and_adapter(position)
    manager._process_decision_zone()
    assert manager.decision_zone.is_pending('ego')
    manager.intent_provider.assert_not_called()
    position.y = 70.0                                # 30 m: inside the zone
    manager._process_decision_zone()
    assert not manager.decision_zone.is_pending('ego')
    manager.intent_provider.assert_called_once()


def test_assigned_junction_is_used_instead_of_a_nearer_one():
    manager, _ = manager_and_adapter(SimpleNamespace(x=4.0, y=0.0), junction_id=20,
        centers={10: SimpleNamespace(x=0.0, y=0.0), 20: SimpleNamespace(x=50.0, y=0.0)})
    obs = manager._collect_observations({}, None)['ego']
    assert obs['relative_position_to_intersection'] == {'x': -46.0, 'y': 0.0}
    assert obs['distance_to_intersection'] == 46.0
    manager._process_decision_zone()
    assert manager.decision_zone.is_pending('ego')


def test_legacy_planner_without_geometry_keeps_its_existing_distance():
    manager, _ = manager_and_adapter(SimpleNamespace(x=2.0, y=80.0))
    manager.planner = object()
    assert manager._collect_observations({}, None)['ego']['distance_to_intersection'] == 10.0
    manager._process_decision_zone()
    manager.intent_provider.assert_called_once()
