"""The junction conflict predictor follows the branch a turning vehicle is actually taking.

The trajectory walk for other vehicles took the first branch at every
fork, so a vehicle turning left inside a junction was predicted straight
on and its crossing path never met the ego's. At a fork the walk now takes
the branch whose heading is closest to the vehicle's heading extrapolated
by its yaw rate over the step.
"""
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402


def _wp(x, y, yaw, nxt=()):
    w = SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=float(x), y=float(y), z=0.0),
                                                  rotation=SimpleNamespace(yaw=float(yaw))))
    w.next = lambda step, _n=list(nxt): list(_n)
    return w


def _fork_map():
    """A straight road that forks: straight on (yaw 0), a gentle left (+6 deg per metre) and a gentle right (-6)."""
    straight = [_wp(1.0 + i, 0.0, 0.0) for i in range(6)]
    left = [_wp(1.0 + i * math.cos(math.radians(6 * (i + 1))), 0.3 * (i + 1) ** 2 / 3.0, 6 * (i + 1)) for i in range(6)]
    right = [_wp(1.0 + i * math.cos(math.radians(6 * (i + 1))), -0.3 * (i + 1) ** 2 / 3.0, -6 * (i + 1)) for i in range(6)]
    for chain in (straight, left, right):
        for a, b in zip(chain, chain[1:]):
            a.next = lambda step, _b=b: [_b]
        chain[-1].next = lambda step: []
    fork = _wp(0.0, 0.0, 0.0, nxt=(straight[0], left[0], right[0]))     # the old walk always took straight[0]
    return fork, straight, left, right


def _driver(fork):
    agent = object.__new__(RuleBasedAgent)
    agent.trajectory_lookahead_time = 3.0
    agent.trajectory_step_size = 1.0
    agent._map = SimpleNamespace(get_waypoint=lambda location: fork)
    agent.debug_rules = False
    return agent


def test_the_branch_follows_the_yaw_rate_and_a_straight_vehicle_goes_straight():
    fork, straight, left, right = _fork_map()
    dt = 1.0 / 5.0                                        # one metre at 5 m/s
    assert RuleBasedAgent._choose_branch(list(fork.next(1.0)), 0.0, 0.0, dt) is straight[0]
    assert RuleBasedAgent._choose_branch(list(fork.next(1.0)), 0.0, 40.0, dt) is left[0]      # +8 deg predicted -> the +6 branch
    assert RuleBasedAgent._choose_branch(list(fork.next(1.0)), 0.0, -40.0, dt) is right[0]
    assert RuleBasedAgent._choose_branch([straight[0]], 0.0, 90.0, dt) is straight[0], "no fork, no choice"


def test_a_turning_vehicle_is_predicted_along_its_turn_and_the_old_walk_was_not():
    fork, straight, left, right = _fork_map()
    agent = _driver(fork)
    transform = SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0), rotation=SimpleNamespace(yaw=0.0))
    turning = agent._predict_vehicle_trajectory(transform, 18.0, is_ego=False, yaw_rate_deg=40.0)
    ahead = agent._predict_vehicle_trajectory(transform, 18.0, is_ego=False, yaw_rate_deg=0.0)
    assert len(turning) >= 4 and len(ahead) >= 4
    assert turning[-1][1] > 1.0, "the turning vehicle's predicted path bends to the left branch"
    assert abs(ahead[-1][1]) < 1e-9, "the straight-going vehicle is predicted straight on"
    # The old walk took the first branch regardless: with the yaw rate ignored both predictions were the same.
    assert agent._predict_vehicle_trajectory(transform, 18.0, is_ego=False)[-1] == ahead[-1]


def test_the_yaw_rate_reads_carlas_angular_velocity_and_tolerates_a_stand_in():
    actor = SimpleNamespace(get_angular_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=-12.5))
    assert RuleBasedAgent._yaw_rate_deg(actor) == -12.5
    assert RuleBasedAgent._yaw_rate_deg(SimpleNamespace()) == 0.0


def _obstacle(actor_id, yaw_deg, x=0.0, y=0.0, speed_kmh=18.0):
    """A neighbour as perception hands it over: an ObstacleVehicle built from
    a snapshot record, with a transform and no angular velocity."""
    carla = pytest.importorskip("carla")
    from opencda.core.sensing.perception.obstacle_vehicle import ObstacleVehicle

    location = carla.Location(x=float(x), y=float(y), z=0.0)
    speed = speed_kmh / 3.6
    yaw = math.radians(yaw_deg)
    return ObstacleVehicle.from_snapshot(SimpleNamespace(
        actor_id=actor_id, type_id="vehicle.test", location=location,
        transform=carla.Transform(location, carla.Rotation(yaw=float(yaw_deg))),
        velocity=carla.Vector3D(speed * math.cos(yaw), speed * math.sin(yaw), 0.0),
        bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(2.4, 0.9, 0.75))))


def _observe(agent, *obstacles):
    agent.obstacle_vehicles = list(obstacles)
    agent._track_obstacle_headings()
    return agent.obstacle_vehicles


def test_an_obstacle_vehicle_gets_its_yaw_rate_from_its_heading_change():
    fork, straight, left, right = _fork_map()
    agent = _driver(fork)
    agent._step_seconds = 0.05
    first, = _observe(agent, _obstacle(7, 0.0))
    assert not hasattr(first, "get_angular_velocity"), "perception carries no angular velocity"
    assert agent._obstacle_yaw_rate_deg(first) == 0.0, "no previous step, no rate yet"
    turning, = _observe(agent, _obstacle(7, 2.0))
    assert agent._obstacle_yaw_rate_deg(turning) == pytest.approx(40.0)    # 2 deg in one 0.05 s step
    path = agent._predict_vehicle_trajectory(turning.get_transform(), 18.0, is_ego=False,
                                             yaw_rate_deg=agent._obstacle_yaw_rate_deg(turning))
    assert path[-1][1] > 1.0, "the turning neighbour is predicted along the left branch"
    # Across the +-180 seam the change is the short way round, and a vehicle
    # missing for a step starts over.
    _observe(agent, _obstacle(7, 179.0))
    wrapped, = _observe(agent, _obstacle(7, -179.0))
    assert agent._obstacle_yaw_rate_deg(wrapped) == pytest.approx(40.0)
    _observe(agent)
    back, = _observe(agent, _obstacle(7, 10.0))
    assert agent._obstacle_yaw_rate_deg(back) == 0.0


def test_the_conflict_check_uses_the_estimated_yaw_rate():
    carla = pytest.importorskip("carla")
    fork, straight, left, right = _fork_map()
    agent = _driver(fork)
    agent._step_seconds = 0.05
    agent._ego_pos = carla.Transform(carla.Location(x=0.0, y=-10.0, z=0.0), carla.Rotation(yaw=90.0))
    agent._ego_speed = 18.0
    agent.junction_conflict_distance = 50.0
    agent.ttc_threshold = 2.0
    _observe(agent, _obstacle(7, 0.0, x=5.0))
    _observe(agent, _obstacle(7, 2.0, x=5.0))
    seen = []

    def predict(transform, speed, is_ego=False, yaw_rate_deg=0.0):
        if is_ego:
            return [(0.0, -10.0, 0.0)]
        seen.append(yaw_rate_deg)
        return []

    agent._predict_vehicle_trajectory = predict
    agent._detect_junction_conflicts(10.0)
    assert seen == [pytest.approx(40.0)]


def test_detections_without_an_actor_id_get_no_yaw_rate():
    # Perception sets carla_id -1 on every detection it could not match to
    # an actor. Keyed by that id, two such vehicles shared one slot, and a
    # swap in detection order read the 90 deg between them as a turn.
    fork, straight, left, right = _fork_map()
    agent = _driver(fork)
    agent._step_seconds = 0.05
    east, north = _obstacle(-1, 0.0, x=5.0), _obstacle(-1, 90.0, x=-5.0)
    _observe(agent, east, north)
    _observe(agent, north, east)
    assert agent._obstacle_yaw_rate_deg(east) == 0.0
    assert agent._obstacle_yaw_rate_deg(north) == 0.0
    assert -1 not in agent._heading_memory


def test_an_id_two_objects_carry_in_one_step_tracks_neither():
    fork, straight, left, right = _fork_map()
    agent = _driver(fork)
    agent._step_seconds = 0.05
    _observe(agent, _obstacle(9, 0.0, x=5.0), _obstacle(9, 90.0, x=-5.0))
    first, second = _observe(agent, _obstacle(9, 90.0, x=-5.0), _obstacle(9, 0.0, x=5.0))
    assert agent._obstacle_yaw_rate_deg(first) == 0.0
    assert agent._obstacle_yaw_rate_deg(second) == 0.0
    # Once the id names one object again, it is tracked from that step on.
    _observe(agent, _obstacle(9, 0.0, x=5.0))
    turning, = _observe(agent, _obstacle(9, 2.0, x=5.0))
    assert agent._obstacle_yaw_rate_deg(turning) == pytest.approx(40.0)
