"""The committed-exit-lane observation and its feature flag.

Whether a vehicle *sees* the exit lanes its neighbours committed to is
selected by one feature entry (``committed_exit_lanes``): switching it on
widens the state by exactly the per-lane counts and changes nothing else,
so it is a configuration switch, not a rewrite.
"""
from opencda_marl.api.contracts import NormalizedObservation, normalize_observation
from opencda_marl.api.route_intent import committed_exit_lane_occupancy
from opencda_marl.builtin.extractors import SUPPORTED_CUSTOM_FEATURES


def _decided(lane, junction="4", actor=None):
    row = {"junction_id": junction,
           "route_intent": {"selected_exit_lane_idx": lane}}
    if actor is not None:
        row["actor_id"] = actor
    return row


def test_counts_commitments_per_lane():
    others = [_decided(0), _decided(2), _decided(2)]
    assert committed_exit_lane_occupancy(others, n_max=4) == [1.0, 0.0, 2.0, 0.0]


def test_undecided_neighbours_contribute_nothing():
    others = [_decided(1), {"junction_id": "4"}, {}]
    assert committed_exit_lane_occupancy(others, n_max=3) == [0.0, 1.0, 0.0]


def test_only_the_same_junction_counts():
    others = [_decided(0, junction="4"), _decided(1, junction="9")]
    assert committed_exit_lane_occupancy(others, n_max=2, junction_id="4") == [1.0, 0.0]
    # Without a junction filter every commitment counts (single-junction runs).
    assert committed_exit_lane_occupancy(others, n_max=2) == [1.0, 1.0]


def test_the_ego_is_excluded():
    others = [_decided(0, actor="ego"), _decided(0, actor="other")]
    assert committed_exit_lane_occupancy(
        others, n_max=2, exclude_actor="ego") == [1.0, 0.0]


def test_lanes_outside_the_head_are_ignored():
    assert committed_exit_lane_occupancy([_decided(7)], n_max=3) == [0.0, 0.0, 0.0]


def test_the_field_is_canonical_and_survives_normalisation():
    assert "committed_exit_lanes" in NormalizedObservation.__annotations__
    obs = normalize_observation({"speed": 1.0, "committed_exit_lanes": [0.0, 2.0]})
    assert obs["committed_exit_lanes"] == [0.0, 2.0]
    # Absent by default: without published commitments the channel is empty.
    assert normalize_observation({"speed": 1.0})["committed_exit_lanes"] == []


def test_the_channel_is_a_supported_feature():
    assert "committed_exit_lanes" in SUPPORTED_CUSTOM_FEATURES


def test_the_channel_keeps_its_width_when_nobody_has_decided():
    from opencda_marl.builtin.extractors import ObservationExtractor

    features = {"speed": 1, "committed_exit_lanes": 4}
    extractor = ObservationExtractor({"matd3": {"features": features}}, "matd3")
    empty = {"a": normalize_observation({"speed": 3.0})}
    state = extractor.extract(empty)["a"]["ego_state"]
    assert len(state) == 5 and list(state[1:]) == [0.0, 0.0, 0.0, 0.0]


def test_flag_off_and_on_changes_only_the_state_width():
    from opencda_marl.builtin.extractors import ObservationExtractor

    base_features = {"speed": 1, "heading_angle": 1}
    channel_features = dict(base_features, committed_exit_lanes=4)  # width = 4 slots
    obs = {"a": normalize_observation(
        {"speed": 10.0, "heading_angle": 0.5,
         "committed_exit_lanes": [1.0, 0.0, 2.0, 0.0]})}

    flag_off = ObservationExtractor({"matd3": {"features": base_features}}, "matd3")
    flag_on = ObservationExtractor({"matd3": {"features": channel_features}}, "matd3")
    without = flag_off.extract(obs)["a"]["ego_state"]
    with_channel = flag_on.extract(obs)["a"]["ego_state"]

    assert len(with_channel) == len(without) + 4
    # The extra values are exactly the per-lane commitment counts.
    assert list(with_channel[len(without):]) == [1.0, 0.0, 2.0, 0.0]
