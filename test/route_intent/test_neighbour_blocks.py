"""Every vehicle's neighbour block comes from one vectorised pass per snapshot.

The per-ego neighbour loop did a bounding-box search, a stable sort and up
to five scalar TTC solves for every vehicle on every call. The snapshot now
computes all rows at once -- the k nearest others within the radius,
ascending distance with ties by ascending row, the same normalised
features and the same constant-velocity TTC -- and caches the result per
parameter set, so both backends' per-ego calls index into it. The scalar
implementation stays as the reference these tests compare against.
"""
import numpy as np
import pytest

from opencda_marl._kernel.adapter.world_snapshot import VehicleKinematics, WorldSnapshot

PARAMS = dict(radius=50.0, k=5, max_relative_velocity=40.0, max_ttc=10.0)


def _snapshot(rows):
    return WorldSnapshot([VehicleKinematics(actor_id=aid, x=x, y=y, z=0.0, vx=vx, vy=vy, yaw_rad=yaw)
                          for aid, (x, y, vx, vy, yaw) in rows.items()])


def _random_scene(rng, n, *, spread, speed):
    """Ids are deliberately non-contiguous and unordered, like CARLA actors."""
    ids = rng.permutation(np.arange(100, 100 + 3 * n, 3))[:n]
    return {int(aid): (float(rng.uniform(-spread, spread)), float(rng.uniform(-spread, spread)),
                       float(rng.uniform(-speed, speed)), float(rng.uniform(-speed, speed)),
                       float(rng.uniform(-np.pi, np.pi)))
            for aid in ids}


def _assert_matches_reference(snapshot, **params):
    for actor_id in snapshot.ids.tolist():
        batched, min_ttc = snapshot.neighbour_features(actor_id, **params)
        reference, ref_min = snapshot._neighbour_features_scalar(actor_id, **params)
        assert len(batched) == len(reference) == params["k"] * 7
        np.testing.assert_allclose(batched, reference, rtol=0, atol=1e-9, err_msg=f"actor {actor_id}")
        assert min_ttc == pytest.approx(ref_min, abs=1e-9) or (min_ttc == ref_min == float("inf"))


@pytest.mark.parametrize("n", [1, 2, 3, 6, 14, 40])
@pytest.mark.parametrize("k", [1, 3, 5, 9])
def test_the_batched_blocks_equal_the_scalar_reference_on_random_scenes(n, k):
    rng = np.random.default_rng(1000 * n + k)
    for radius, spread in ((50.0, 60.0), (15.0, 20.0), (50.0, 200.0)):
        snapshot = _snapshot(_random_scene(rng, n, spread=spread, speed=15.0))
        _assert_matches_reference(snapshot, radius=radius, k=k, max_relative_velocity=40.0, max_ttc=10.0)


def test_ties_stationary_pairs_receding_pairs_and_head_on_pairs_agree_with_the_reference():
    rows = {
        1: (0.0, 0.0, 10.0, 0.0, 0.0),        # ego, driving east
        2: (20.0, 0.0, -10.0, 0.0, np.pi),     # head-on: finite TTC
        3: (0.0, 20.0, 10.0, 0.0, 0.0),        # same velocity: a == 0 -> inf
        4: (-20.0, 0.0, -10.0, 0.0, np.pi),    # receding: both roots negative -> inf
        5: (0.0, -20.0, 0.0, 0.0, 1.0),        # tie in distance with 2, 3 and 4 (row order decides)
        6: (12.0, 16.0, 0.0, 0.0, 0.5),        # exactly 20 m away too (3-4-5), later row
        7: (200.0, 0.0, 0.0, 0.0, 0.0),        # outside the radius
    }
    snapshot = _snapshot(rows)
    _assert_matches_reference(snapshot, **PARAMS)
    block, min_ttc = snapshot.neighbour_features(1, **PARAMS)
    rel_x_by_slot = block[0::7]
    assert rel_x_by_slot == pytest.approx([20.0 / 50, 0.0, -20.0 / 50, 0.0, 12.0 / 50]), "five ties resolve by row order"
    assert min_ttc == pytest.approx((20.0 - 4.0) / 20.0)


def test_more_slots_than_vehicles_pads_with_empty_slots():
    snapshot = _snapshot({1: (0.0, 0.0, 0.0, 0.0, 0.0), 2: (10.0, 0.0, 0.0, 0.0, 0.0)})
    block, min_ttc = snapshot.neighbour_features(1, radius=50.0, k=4, max_relative_velocity=40.0, max_ttc=10.0)
    assert block[:7] == pytest.approx([10.0 / 50, 0.0, 0.0, 0.0, 0.0, 10.0 / 50, 1.0])
    assert block[7:] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 3
    assert min_ttc == float("inf")
    _assert_matches_reference(snapshot, radius=50.0, k=4, max_relative_velocity=40.0, max_ttc=10.0)


def test_an_unknown_actor_and_a_lone_vehicle_get_empty_blocks():
    snapshot = _snapshot({1: (0.0, 0.0, 5.0, 0.0, 0.0)})
    assert snapshot.neighbour_features(1, **PARAMS) == ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 5, float("inf"))
    assert snapshot.neighbour_features(999, **PARAMS) == ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] * 5, float("inf"))


def test_the_pass_runs_once_per_snapshot_and_parameter_set(monkeypatch):
    rng = np.random.default_rng(7)
    snapshot = _snapshot(_random_scene(rng, 12, spread=40.0, speed=10.0))
    computed = []
    original = WorldSnapshot._compute_neighbour_blocks

    def counting(self, *args):
        computed.append(args)
        return original(self, *args)

    monkeypatch.setattr(WorldSnapshot, "_compute_neighbour_blocks", counting)
    for actor_id in snapshot.ids.tolist():
        snapshot.neighbour_features(actor_id, **PARAMS)
    assert len(computed) == 1, "twelve egos, one vectorised pass"
    snapshot.neighbour_features(int(snapshot.ids[0]), **{**PARAMS, "k": 3})
    assert len(computed) == 2, "a different parameter set is its own pass"
    blocks, min_ttc = snapshot.neighbour_blocks(**PARAMS)
    assert blocks.shape == (12, 35) and min_ttc.shape == (12,)


def test_row_lookup_is_by_actor_id_not_position():
    snapshot = _snapshot({42: (0.0, 0.0, 0.0, 0.0, 0.0), 7: (1.0, 0.0, 0.0, 0.0, 0.0)})
    assert snapshot.index_of(7) == 1 and snapshot.index_of(42) == 0 and snapshot.index_of(3) == -1
