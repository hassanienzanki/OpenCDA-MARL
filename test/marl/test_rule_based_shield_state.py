"""The rule-based driver reports its lane hold and merge order to the step trace.

``RuleBasedAgent.shield_state()`` is the duck-typed hook MARLVehicleAdapter
reads on CARLA; the trace writer records the fields it returns. These tests
drive real ``run_step`` calls on a stand-in straight road (the planner's own
step is stubbed, no CARLA server) and carry one step through the adapter's
observation into a written trace record.
"""
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda_marl.builtin.agents import rule_based_agent  # noqa: E402
from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402
from opencda_marl.builtin.agents.vanilla_agent import VanillaAgent  # noqa: E402

#: The planner's speed; the driver cruises at 95 % of it on a free road.
PLANNER_KMH = 40.0
CRUISE_KMH = 38.0
OWN_LANE = (10.0, 0.3)       # a planner target on the ego's lane
LEFT_LANE = (10.0, 3.5)      # a planner target on the next lane to the left


def _loc(x, y):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def _wp(x, y):
    """A lane-centre waypoint heading +x, outside any junction."""
    w = SimpleNamespace(transform=SimpleNamespace(location=_loc(x, y), rotation=SimpleNamespace(yaw=0.0)),
                        is_junction=False)
    w.next = lambda d: [_wp(x + d, y)]
    return w


def _neighbour(actor_id, x, y, speed_kmh):
    """A vehicle heading +x, as perception hands it to the driver."""
    return SimpleNamespace(
        carla_id=actor_id,
        get_location=lambda: _loc(x, y),
        get_transform=lambda: SimpleNamespace(location=_loc(x, y), rotation=SimpleNamespace(yaw=0.0)),
        get_velocity=lambda: SimpleNamespace(x=speed_kmh / 3.6, y=0.0, z=0.0))


def _driver(monkeypatch, cls=RuleBasedAgent, config=None, *, ego_id=5, ego_speed=20.0, neighbours=()):
    """A driver built through its own __init__ (the CARLA-bound base
    constructor stubbed), driving +x from the origin on a straight road. The
    planner's step returns ``PLANNER_KMH`` and ``agent.planner_target``."""
    monkeypatch.setattr(VanillaAgent, '__init__', lambda self, *args, **kwargs: None)
    monkeypatch.setattr(VanillaAgent, 'run_step',
                        lambda self, target_speed=None: (PLANNER_KMH, _loc(*self.planner_target)))
    agent = cls(SimpleNamespace(id=ego_id), None, dict(config or {}))
    agent._vehicle = SimpleNamespace(id=ego_id)
    agent._ego_speed = float(ego_speed)
    agent._ego_pos = SimpleNamespace(location=_loc(0.0, 0.0), rotation=SimpleNamespace(yaw=0.0))
    agent._map = SimpleNamespace(get_waypoint=lambda location: _wp(0.0, 0.0))
    agent.obstacle_vehicles = list(neighbours)
    agent.is_close_to_destination = lambda: False
    agent.planner_target = OWN_LANE
    return agent


def _held(monkeypatch, **kwargs):
    """A driver whose planner steers into the left lane while a vehicle 2 m
    ahead occupies it at 25 km/h: the hold binds, 8 km/h below it."""
    agent = _driver(monkeypatch, neighbours=[_neighbour(9, 2.0, 3.5, 25.0)], **kwargs)
    agent.planner_target = LEFT_LANE
    return agent


# --------------------------------------------------------------------------- #
# The hook itself
# --------------------------------------------------------------------------- #

def test_before_the_first_step_the_hook_reports_no_hold_and_no_speed(monkeypatch):
    idle = {'lane_hold': False, 'merge_priority_enabled': True,
            'merge_priority_release': False, 'merge_priority_override': False}
    assert _driver(monkeypatch).shield_state() == idle
    assert object.__new__(RuleBasedAgent).shield_state() == idle, "safe without __init__ too"
    off = _driver(monkeypatch, config={'lane_hold_geometry_priority': False}).shield_state()
    assert off['merge_priority_enabled'] is False


def test_a_lane_hold_step_then_a_normal_step(monkeypatch):
    agent = _held(monkeypatch)
    speed, location = agent.run_step()
    assert (location.x, location.y) == (8.0, 0.0) and speed == pytest.approx(17.0)
    assert agent.shield_state() == {
        'lane_hold': True, 'merge_priority_enabled': True,
        'merge_priority_release': False, 'merge_priority_override': False,
        'applied_speed': 17.0}
    agent.planner_target = OWN_LANE
    speed, _location = agent.run_step()
    assert speed == pytest.approx(CRUISE_KMH)
    assert agent.shield_state() == {
        'lane_hold': False, 'merge_priority_enabled': True,
        'merge_priority_release': False, 'merge_priority_override': False,
        'applied_speed': CRUISE_KMH}


def test_a_merge_release_is_reported_for_its_own_step_only(monkeypatch):
    # Both standing; the ego (the larger id) is 3 m ahead of the vehicle in
    # its target lane, so the geometry order releases it against the id order.
    agent = _driver(monkeypatch, ego_id=9, ego_speed=0.0, neighbours=[_neighbour(5, -3.0, 3.5, 0.0)])
    agent.planner_target = LEFT_LANE
    agent.run_step()
    state = agent.shield_state()
    assert state['lane_hold'] and state['merge_priority_release'] and state['merge_priority_override']
    agent.planner_target = OWN_LANE
    agent.run_step()
    state = agent.shield_state()
    assert not (state['lane_hold'] or state['merge_priority_release'] or state['merge_priority_override'])


def test_a_step_that_returns_before_the_rules_does_not_repeat_the_last_one(monkeypatch):
    agent = _held(monkeypatch)
    agent.run_step()
    assert agent.shield_state()['lane_hold'] is True
    agent._ego_pos = None                       # not localised this step
    assert agent.run_step() == (0.0, None)
    state = agent.shield_state()
    assert state['lane_hold'] is False and 'applied_speed' not in state


def test_a_fallback_step_does_not_repeat_the_last_one(monkeypatch):
    agent = _held(monkeypatch)
    agent.run_step()

    def planner_fails(self, target_speed=None):
        raise RuntimeError('no route')

    monkeypatch.setattr(VanillaAgent, 'run_step', planner_fails)
    assert agent.run_step()[0] == agent.cautious_speed
    state = agent.shield_state()
    assert state['lane_hold'] is False and 'applied_speed' not in state


def test_a_fallback_after_the_hold_reports_no_hold(monkeypatch):
    # The rules fail after the hold replaced the target (here: the debug log
    # raises), so the step falls back to the planner's own target.
    agent = _held(monkeypatch)
    agent.debug_rules = True

    def info(message, *args, **kwargs):
        raise RuntimeError('log sink down')

    stub = SimpleNamespace(info=info, error=lambda *a, **k: None,
                           warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    monkeypatch.setattr(rule_based_agent, 'logger', stub)
    speed, location = agent.run_step()
    assert speed == PLANNER_KMH and (location.x, location.y) == LEFT_LANE
    state = agent.shield_state()
    assert state['lane_hold'] is False and 'applied_speed' not in state


def test_a_lane_hold_that_fails_midway_reports_no_hold(monkeypatch):
    agent = _held(monkeypatch)
    agent._ego_speed = None                     # a stand-in without a speed: the hold raises after it bound
    target = _loc(*LEFT_LANE)
    assert agent._hold_lane(target) == (target, None)
    assert agent.shield_state()['lane_hold'] is False


# --------------------------------------------------------------------------- #
# Through the adapter's observation into the trace
# --------------------------------------------------------------------------- #

def _adapter(agent, actor_id=5):
    """MARLVehicleAdapter around ``agent``, its simulator reads stubbed."""
    from opencda_marl._kernel.adapter.vehicle_adapter import MARLVehicleAdapter

    adapter = object.__new__(MARLVehicleAdapter)
    adapter.actor_id = actor_id
    adapter.target_speed = 30.0
    adapter.vm = SimpleNamespace(agent=agent, update_info=lambda: None)
    adapter.vehicle = SimpleNamespace(is_alive=True, get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0))
    adapter._get_physics_state = lambda: {
        'planar_speed_kmh': 20.0, 'speed_3d_kmh': 20.0, 'vertical_speed_kmh': 0.0,
        'pitch_deg': 0.0, 'roll_deg': 0.0, 'height_delta_m': 0.0}
    adapter._calculate_distance_to_intersection = lambda: 50.0
    adapter._calculate_distance_to_front_vehicle = lambda: 999.0
    adapter._classify_lane_position = lambda: 1
    adapter._compute_nearby_vehicle_features = lambda: ([], 10.0)
    adapter._calculate_distance_to_destination = lambda: 100.0
    adapter._calculate_relative_position_to_intersection = lambda: (0.0, -50.0)
    adapter._get_vehicle_heading_angle = lambda: 0.0
    return adapter


def test_the_adapter_and_the_trace_carry_a_lane_hold_step(monkeypatch, tmp_path):
    from opencda_marl.telemetry.trace import RouteActionTraceWriter

    agent = _held(monkeypatch)
    agent.run_step()
    adapter = _adapter(agent)
    obs = adapter.get_observation()
    assert 'error' not in obs
    assert obs['lane_hold'] is True and obs['applied_speed'] == 17.0
    assert 'shield_active' not in obs and 'shield_stage' not in obs

    coordinator = SimpleNamespace(
        config={}, states={'episode': 0, 'step': 1}, marl_env=None,
        scenario_manager=SimpleNamespace(get_observations=lambda: {adapter.actor_id: adapter.get_observation()}))
    writer = RouteActionTraceWriter({'enabled': True, 'root_dir': str(tmp_path)}, coordinator)
    writer.record_step(coordinator)
    writer.close()
    [line] = (tmp_path / 'steps.jsonl').read_text(encoding='utf-8').splitlines()
    [record] = json.loads(line)['agents']
    assert record['agent_id'] == '5'
    assert record['lane_hold'] is True
    assert record['merge_priority_enabled'] is True
    assert record['merge_priority_release'] is False
    assert record['merge_priority_override'] is False
    assert record['applied_speed'] == 17.0
    assert record['shield_active'] is None and record['shield_stage'] is None


def test_a_subclass_with_its_own_hook_still_overrides_it(monkeypatch):
    class ShieldedDriver(RuleBasedAgent):
        def shield_state(self):
            return {'shield_active': True, 'shield_stage': 2}

    agent = _driver(monkeypatch, cls=ShieldedDriver)
    agent.run_step()
    obs = _adapter(agent).get_observation()
    assert obs['shield_active'] is True and obs['shield_stage'] == 2
    assert 'lane_hold' not in obs
