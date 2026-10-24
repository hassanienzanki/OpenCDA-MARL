"""The safety API exposes the TTC the framework computes, and the step's kinematics, to agents that act.

An agent that acts on a vehicle's minimum time-to-collision against the
others must read the same number the learner and the metrics see -- the
world snapshot's constant-velocity quadratic -- and it must come from the
kinematics the agent manager already captured for the step, not from a
second round of simulator queries per agent.
"""
import math
from types import SimpleNamespace

import numpy as np

from opencda_marl._kernel.adapter.world_snapshot import VehicleKinematics, WorldSnapshot
from opencda_marl.api.safety import constant_velocity_ttc, min_ttc_to_others, published_kinematics


def _snapshot(rows):
    return WorldSnapshot([VehicleKinematics(actor_id=aid, x=x, y=y, z=0.0, vx=vx, vy=vy, yaw_rad=0.0)
                          for aid, (x, y, vx, vy) in rows.items()])


def test_the_public_quadratic_matches_the_snapshot_on_random_pairs():
    rng = np.random.default_rng(3)
    rows = {i: tuple(float(v) for v in rng.uniform(-30, 30, 4)) for i in range(12)}
    snapshot = _snapshot(rows)
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            a, b = rows[i], rows[j]
            mine = constant_velocity_ttc(b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3])
            assert mine == snapshot.ttc(i, j) or (math.isinf(mine) and math.isinf(snapshot.ttc(i, j)))


def test_min_ttc_to_others_picks_the_closest_conflict_and_is_inf_when_alone():
    records = {1: SimpleNamespace(x=0.0, y=0.0, vx=10.0, vy=0.0),
               2: SimpleNamespace(x=20.0, y=0.0, vx=-10.0, vy=0.0),     # head-on: (20 - 4) / 20 = 0.8 s
               3: SimpleNamespace(x=0.0, y=30.0, vx=0.0, vy=-10.0),     # crossing: reaches the ego later
               4: SimpleNamespace(x=-50.0, y=0.0, vx=-5.0, vy=0.0)}     # receding
    assert min_ttc_to_others(records, 1) == 0.8
    assert min_ttc_to_others(records, "1") == 0.8, "string ids resolve to the integer record"
    assert math.isinf(min_ttc_to_others(records, 99))
    assert math.isinf(min_ttc_to_others({1: records[1]}, 1))


def test_published_kinematics_follow_the_snapshots_publish_and_unpublish():
    if published_kinematics() is None and not _carla_stack_available():
        return                                     # no vendored perception layer here: nothing to publish to
    snapshot = _snapshot({7: (0.0, 0.0, 1.0, 0.0), 8: (5.0, 0.0, 0.0, 0.0)})
    snapshot.publish_to_perception()
    try:
        records = published_kinematics()
        assert records is not None and set(records) == {7, 8}
        assert min_ttc_to_others(records, 7) == (5.0 - 4.0) / 1.0
    finally:
        WorldSnapshot.unpublish_from_perception()
    assert published_kinematics() is None


def _carla_stack_available():
    try:
        from opencda.core.sensing.perception.perception_manager import PerceptionManager  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
