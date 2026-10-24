"""A transferred policy must see the same map position on either backend."""
import pytest

from opencda_marl.api.contracts import normalize_observation
from opencda_marl.builtin.extractors import ObservationExtractor


def extractor():
    return ObservationExtractor(
        {"pcmappo": {"features": {
            "rel_x": 1, "rel_y": 1, "position_x": 1, "position_y": 1,
        }}}, "pcmappo")


def test_backend_position_encodings_give_the_same_policy_features():
    relative = {"x": 5.0, "y": -15.0}
    # CARLA retains its legacy relative location dictionary, while publishing
    # the physical map coordinates separately. SUMO publishes its bridged map
    # location and normalize_observation supplies the absolute position fields.
    carla = normalize_observation({
        "position_x": 3.0, "position_y": 115.0,
        "relative_position_to_intersection": relative, "location": relative,
    })
    sumo = normalize_observation({
        "location": {"x": 3.0, "y": 115.0},
        "relative_position_to_intersection": relative,
    })
    states = extractor().extract({"carla": carla, "sumo": sumo})
    assert states["sumo"]["ego_state"].tolist() == pytest.approx([5, -15, 3, 115])
    assert states["carla"]["ego_state"].tolist() == pytest.approx(states["sumo"]["ego_state"].tolist())
    assert carla["location"] == {"x": 5.0, "y": -15.0, "z": 0.0}


def test_zero_world_position_is_not_replaced_by_relative_location():
    states = extractor().extract({"v": {
        "position_x": 0.0, "position_y": 0.0,
        "location": {"x": 18.0, "y": 120.0},
    }})
    assert states["v"]["ego_state"].tolist()[2:] == [0.0, 0.0]


@pytest.mark.parametrize("observation, expected", [
    ({"location": {"x": 3.0, "y": 115.0}}, [3.0, 115.0]),
    ({}, [0.0, 0.0]),
])
def test_position_only_observations_keep_the_existing_fallback(observation, expected):
    assert extractor().extract({"v": observation})["v"]["ego_state"].tolist()[2:] == expected
