"""The rule-based follower keeps a bumper-to-bumper gap and never creeps into a stopped leader.

Low-speed contacts in a queue come from measuring the wrong gap: the
follower measured its leader centre to centre and stopped at a 5 m buffer
-- 0.3 m between two 4.7 m cars -- and its closing margin let it approach a
stopped leader at 5 km/h whatever the gap. The leader search now returns
the physical gap, the buffer is that gap, and the margin vanishes at the
safe distance.
"""
import math
import pathlib
from types import SimpleNamespace

import pytest

pytest.importorskip("carla", reason="the rule-based driver is a CARLA agent; no carla package here")

from opencda_marl.builtin.agents.rule_based_agent import RuleBasedAgent  # noqa: E402


class _Location:
    def __init__(self, x, y):
        self.x, self.y, self.z = float(x), float(y), 0.0

    def distance(self, other):
        return math.hypot(other.x - self.x, other.y - self.y)


def _vehicle(vid, x, y, yaw_deg=0.0, speed_kmh=0.0, half_length=2.35):
    yaw = math.radians(yaw_deg)
    v = speed_kmh / 3.6
    transform = SimpleNamespace(location=_Location(x, y), rotation=SimpleNamespace(yaw=yaw_deg))
    return SimpleNamespace(
        id=vid,
        get_location=lambda: transform.location,
        get_transform=lambda: transform,
        get_velocity=lambda: SimpleNamespace(x=v * math.cos(yaw), y=v * math.sin(yaw), z=0.0),
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=half_length, y=0.9, z=0.7)),
    )


def _follower(ego_speed_kmh, others, buffer=3.0, headway=2.0, decel=1.0, gain=0.5):
    agent = object.__new__(RuleBasedAgent)
    ego = _vehicle(1, 0.0, 0.0, speed_kmh=ego_speed_kmh)
    agent._vehicle = ego
    agent._ego_pos = ego.get_transform()
    agent._ego_speed = float(ego_speed_kmh)
    agent.obstacle_vehicles = list(others)
    agent.minimum_distance_buffer = buffer
    agent.time_headway = headway
    agent.closing_decel = decel
    agent.following_gain = gain
    agent.same_lane_tolerance = math.radians(30)
    agent.front_cone_angle = math.radians(45)
    agent.cautious_speed = 15.0
    agent.debug_rules = False
    return agent


def test_the_leader_gap_is_bumper_to_bumper_and_floored_at_zero():
    # Two 4.7 m cars 9.7 m apart centre to centre: a 5 m physical gap.
    leader = _vehicle(2, 9.7, 0.0)
    agent = _follower(0.0, [leader])
    found, gap = agent._get_leading_vehicle()
    assert found is leader and gap == pytest.approx(5.0)
    # Touching bumpers is a zero gap, never a negative one.
    agent = _follower(0.0, [_vehicle(2, 4.0, 0.0)])
    assert agent._get_leading_vehicle()[1] == 0.0
    # A stand-in without a bounding box counts as a passenger car.
    bare = _vehicle(3, 9.8, 0.0)
    del bare.bounding_box
    agent = _follower(0.0, [bare])
    assert agent._get_leading_vehicle()[1] == pytest.approx(9.8 - 2.35 - 2.4)
    # Nothing ahead: nothing to follow.
    assert _follower(0.0, [_vehicle(2, -12.0, 0.0)])._get_leading_vehicle() == (None, math.inf)


def test_a_follower_stops_behind_a_stopped_leader_instead_of_creeping():
    stopped = _vehicle(2, 30.0, 0.0, speed_kmh=0.0)
    # At the safe distance (3 m buffer + 2 s * 5 km/h) the old rule still allowed leader + 5 km/h.
    crawling = _follower(5.0, [stopped])
    safe = 3.0 + (5.0 / 3.6) * 2.0
    assert crawling._calculate_safe_following_speed(stopped, safe, 40.0) == 0.0
    # Inside the buffer, whatever the ego does, the target is a stop.
    assert crawling._calculate_safe_following_speed(stopped, 1.0, 40.0) == 0.0
    assert crawling._calculate_safe_following_speed(stopped, 0.0, 40.0) == 0.0
    # Beyond the safe distance the follower closes in on a 1 m/s^2 braking
    # curve that reaches zero at the safe distance: decisive from afar, a
    # speed the vehicle can execute near the buffer (the old linear term
    # asked for 0.75 km/h here and the queue froze), and never a crawl.
    assert crawling._calculate_safe_following_speed(stopped, safe + 3.0, 40.0) == pytest.approx(math.sqrt(6.0) * 3.6)
    assert crawling._calculate_safe_following_speed(stopped, safe + 1.5, 40.0) == pytest.approx(math.sqrt(3.0) * 3.6)
    assert crawling._calculate_safe_following_speed(stopped, safe + 30.0, 40.0) == pytest.approx(math.sqrt(60.0) * 3.6)
    assert crawling._calculate_safe_following_speed(stopped, safe + 300.0, 40.0) == pytest.approx(40.0)


def test_a_follower_never_gains_on_a_moving_leader_inside_the_safe_distance():
    leader = _vehicle(2, 30.0, 0.0, speed_kmh=20.0)
    agent = _follower(30.0, [leader])
    safe = 3.0 + (30.0 / 3.6) * 2.0
    # Inside the safe distance the cap is the leader's own speed (the old
    # rule allowed 25 km/h here) and the gap term opens the gap below it...
    assert agent._calculate_safe_following_speed(leader, safe - 4.0, 60.0) == pytest.approx(20.0 - 0.5 * 4.0)
    # ...and a large shortfall lets the proportional term pull the target below it.
    assert agent._calculate_safe_following_speed(leader, safe - 25.0, 60.0) == pytest.approx(20.0 - 0.5 * 25.0)
    # Far behind, the follower closes on the braking curve above the leader's speed, bounded by the base speed.
    assert agent._calculate_safe_following_speed(leader, safe + 20.0, 60.0) == pytest.approx(20.0 + math.sqrt(40.0) * 3.6)
    assert agent._calculate_safe_following_speed(leader, safe + 20.0, 22.0) == pytest.approx(22.0)


def test_the_bundled_driver_profile_carries_the_physical_buffer():
    import yaml
    import opencda_marl
    profile = pathlib.Path(opencda_marl.__file__).parent / "_bundled" / "configs" / "marl" / "intersection" / "rule_based.yaml"
    rule_based = yaml.safe_load(profile.read_text(encoding="utf-8"))
    section = next(v for v in _walk(rule_based) if isinstance(v, dict) and "minimum_distance_buffer" in v)
    assert section["minimum_distance_buffer"] == 3.0
    assert section["closing_decel"] == 1.0 and "following_speed_buffer" not in section


def _walk(node):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_a_stopped_follower_moves_off_with_its_leader():
    # Standing queue: the ego is stopped at the buffer behind a leader that
    # has just pulled away to 15 km/h, one metre beyond the safe distance.
    leader = _vehicle(2, 30.0, 0.0, speed_kmh=15.0)
    agent = _follower(0.0, [leader])
    safe = 3.0
    target = agent._calculate_safe_following_speed(leader, safe + 1.0, 40.0)
    # Around the ego's own speed the target was 0.5 km/h; around the leader's it is the leader's speed plus the closing curve.
    assert target == pytest.approx(15.0 + math.sqrt(2.0) * 3.6)
    # Still at the buffer with the leader already moving: match the leader, do not wait for a gap.
    assert agent._calculate_safe_following_speed(leader, safe, 40.0) == pytest.approx(15.0)


def test_a_queue_in_the_adjacent_lane_does_not_block_a_clear_lane():
    # Same heading and inside the 45-degree front cone, but a full lane over.
    adjacent = _vehicle(2, 5.0, 3.5)
    agent = _follower(0.0, [adjacent])
    assert agent._get_leading_vehicle() == (None, math.inf)
    assert agent._calculate_car_following_speed(30.0) is None


def test_an_adjacent_queue_does_not_hide_the_actual_leader():
    adjacent = _vehicle(2, 5.0, -3.5)
    actual = _vehicle(3, 20.0, 0.0, speed_kmh=15.0)
    agent = _follower(0.0, [adjacent, actual])
    leader, gap = agent._get_leading_vehicle()
    assert leader is actual
    assert gap == pytest.approx(15.3)


def test_a_vehicle_overlapping_the_driving_corridor_still_blocks():
    straddling = _vehicle(2, 5.0, 1.5)
    agent = _follower(0.0, [straddling])
    assert agent._get_leading_vehicle()[0] is straddling
    assert agent._calculate_car_following_speed(30.0) == 0.0


def test_a_turning_vehicle_uses_its_projected_body_width():
    merging = _vehicle(2, 6.0, 2.5, yaw_deg=-25.0)
    agent = _follower(0.0, [merging])
    assert agent._get_leading_vehicle()[0] is merging
