"""The decision zone executes a decision through the seam, whatever its type.

``_process_decision_zone`` asked the provider, recorded the decision, and then
re-routed the vehicle by reaching into ``decision.selected_candidate`` inside
a ``try`` that turns any error into a warning. A provider with its own
decision type therefore *appeared* to work on CARLA: decisions were made and
recorded, and every one of them silently kept the demand route, leaving only
a "decision-zone re-route failed: ... object has no attribute
'selected_candidate'" warning per decision. The apply site now reads
``decision_destination`` like the spawn-time site does.
"""
from types import SimpleNamespace

from opencda_marl._kernel.agent_manager import MARLAgentManager
from opencda_marl._kernel.policy.decision_zone import DecisionZone


class _Vehicle:
    def get_location(self):
        return "here"


class _Adapter:
    def __init__(self, actor_id, distance):
        self.actor_id = actor_id
        self.vehicle = _Vehicle()
        self.spawn_metadata = {}
        self.destinations = []
        self._distance = distance

    def _calculate_distance_to_intersection(self):
        return self._distance

    def set_destination(self, start, end, clean=False):
        self.destinations.append((start, end, clean))


class _PluginDecision:
    """No selected_candidate, no route_id: only what the seam defines."""

    def __init__(self, destination, admissible=True):
        self._destination = destination
        self.admissible = admissible

    @property
    def destination(self):
        return self._destination

    def as_metadata(self):
        return {"interface": "plugin", "admissible": self.admissible}


def _manager(adapters, provider, distance_m=30.0):
    manager = object.__new__(MARLAgentManager)
    zone = DecisionZone(trigger="approach", distance_m=distance_m)
    for adapter in adapters:
        zone.register(adapter.actor_id)
    manager._decision_zone = zone     # decision_zone is a read-only property over this
    manager.intent_provider = provider
    manager.planner = object()
    manager._vehicle_adapters = list(adapters)
    manager._spawn_event_by_actor_id = {a.actor_id: SimpleNamespace(vehicle_id=a.actor_id) for a in adapters}
    manager._spawn_metadata_by_actor_id = {a.actor_id: {"population": "marl"} for a in adapters}
    return manager


def test_a_plugin_decision_is_executed_not_just_recorded():
    target = SimpleNamespace(location="the chosen destination")
    adapter = _Adapter(actor_id=7, distance=10.0)            # inside the 30 m zone
    manager = _manager([adapter], lambda context: _PluginDecision(target))

    assert manager._process_decision_zone() == 1
    assert adapter.destinations == [("here", "the chosen destination", True)], "re-routed to the decision's destination"
    assert adapter.spawn_metadata["route_intent"]["interface"] == "plugin", "and recorded"
    assert not manager.decision_zone.is_pending(7), "one shot"


def test_an_inadmissible_decision_is_recorded_and_not_executed():
    adapter = _Adapter(actor_id=8, distance=5.0)
    manager = _manager([adapter], lambda context: _PluginDecision(SimpleNamespace(location="x"), admissible=False))
    assert manager._process_decision_zone() == 1
    assert adapter.destinations == []
    assert adapter.spawn_metadata["route_intent"]["admissible"] is False


def test_a_decision_without_a_destination_keeps_the_demand_route():
    adapter = _Adapter(actor_id=9, distance=5.0)
    manager = _manager([adapter], lambda context: _PluginDecision(None))
    assert manager._process_decision_zone() == 1
    assert adapter.destinations == []


def test_a_vehicle_outside_the_zone_is_not_asked_yet():
    asked = []
    adapter = _Adapter(actor_id=10, distance=80.0)            # still 80 m out
    manager = _manager([adapter], lambda context: asked.append(context) or None)
    assert manager._process_decision_zone() == 0
    assert asked == [] and manager.decision_zone.is_pending(10)
